from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from xgboost import DMatrix, XGBClassifier

from .dataset import LandUsePatchDataset
from .optical_indices import OPTICAL_INDEX_NAMES
from .text import tokenize


CLASS_NAMES = (
    "Residential",
    "Commercial/office",
    "Industrial/logistics",
    "Institutional/recreation",
    "Transport",
    "Other built-up",
    "Agriculture",
    "Forest/shrub/grass/wetland",
    "Barren",
    "Water",
)

S1_DERIVED_NAMES = (
    "S1_total_power",
    "S1_norm_pol_difference",
    "S1_dual_pol_RVI",
    "S1_intensity_entropy",
)

GROUP_LABELS = {
    "s2": "Sentinel-2",
    "s1": "Sentinel-1",
    "highres": "High-resolution imagery",
    "nighttime": "Nighttime light",
    "building": "Building function/height",
    "roads": "OSM roads",
    "poi": "OSM POI",
    "aoi": "OSM AOI",
    "text": "OSM text",
}

GROUP_COLORS = {
    "s2": "#4477AA",
    "s1": "#66CCEE",
    "highres": "#228833",
    "nighttime": "#CCBB44",
    "building": "#EE6677",
    "roads": "#AA3377",
    "poi": "#BBBBBB",
    "aoi": "#EE8866",
    "text": "#777777",
}


def _short_name(name: str) -> str:
    replacements = {
        "building_": "Bldg_",
        "road_": "Road_",
        "highres_": "HR_",
        "VIIRS_": "NTL_",
        "S2_": "S2_",
        "S1_": "S1_",
        "Text:": "Txt:",
        "POI_": "POI_",
        "AOI_": "AOI_",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result[:46]


def _token_vocabulary(manifest: pd.DataFrame, top_k: int) -> list[str]:
    counts: Counter[str] = Counter()
    train_text = manifest.loc[manifest["split"] == "train", "text"].fillna("")
    for text in train_text.astype(str):
        counts.update(tokenize(text))
    counts.pop(";", None)
    return [token for token, _ in counts.most_common(top_k)]


def _text_vector(text: str, token_to_col: dict[str, int]) -> np.ndarray:
    counts = Counter(tokenize(text))
    values = np.zeros(len(token_to_col), dtype=np.float32)
    for token, count in counts.items():
        col = token_to_col.get(token)
        if col is not None:
            values[col] = min(math.log1p(count) / math.log(10.0), 2.0)
    return values


def _feature_schema(
    image_path: str,
    highres_path: str,
    nighttime_path: str,
    building_path: str,
    roads_path: str,
    text_tokens: list[str],
) -> tuple[list[str], list[str]]:
    with rasterio.open(image_path) as src:
        descriptions = [str(x) for x in src.descriptions]
    s2_names = [name for name in descriptions if name.startswith("S2_")]
    s1_names = [name for name in descriptions if name.startswith("S1_")]
    with rasterio.open(highres_path) as src:
        highres_names = [str(x) for x in src.descriptions]
    with rasterio.open(nighttime_path) as src:
        nighttime_names = [str(x) for x in src.descriptions]
    with rasterio.open(building_path) as src:
        building_names = [str(x) for x in src.descriptions]
    with rasterio.open(roads_path) as src:
        road_names = [str(x) for x in src.descriptions]

    names: list[str] = []
    groups: list[str] = []

    def add(group: str, values: list[str] | tuple[str, ...]) -> None:
        names.extend(values)
        groups.extend([group] * len(values))

    add("s2", s2_names + list(OPTICAL_INDEX_NAMES))
    add("s1", s1_names + list(S1_DERIVED_NAMES))
    add("highres", highres_names)
    add("nighttime", nighttime_names)
    add("building", building_names)
    add("roads", road_names)
    add("poi", [f"POI_category_{i + 1:02d}" for i in range(14)])
    add("aoi", [f"AOI_official_class_{i + 1}" for i in range(6)])
    add("text", [f"Text:{token}" for token in text_tokens])
    return names, groups


def _make_dataset(paths: dict[str, str], split: str, patch_size: int, max_tokens: int) -> LandUsePatchDataset:
    return LandUsePatchDataset(
        image_path=paths["image"],
        label_path=paths["label"],
        manifest_path=paths["manifest"],
        vocab_path=paths["vocab"],
        normalization_path=paths["normalization"],
        patch_size=patch_size,
        max_text_tokens=max_tokens,
        osm_poi_path=paths["poi"],
        osm_aoi_path=paths["aoi"],
        highres_path=paths["highres"],
        nighttime_path=paths["nighttime"],
        building_path=paths["building"],
        road_features_path=paths["roads"],
        split=split,
        augment=False,
    )


def _extract_samples(
    dataset: LandUsePatchDataset,
    token_to_col: dict[str, int],
    rng: np.random.Generator,
    per_class_per_patch: int,
    cap_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for patch_index in range(len(dataset)):
        sample = dataset[patch_index]
        label = sample["label"].numpy().reshape(-1)
        chosen: list[np.ndarray] = []
        for class_id in range(10):
            candidates = np.flatnonzero(label == class_id)
            if candidates.size:
                count = min(per_class_per_patch, candidates.size)
                chosen.append(rng.choice(candidates, count, replace=False))
        if not chosen:
            continue
        index = np.concatenate(chosen)

        def pixels(key: str) -> np.ndarray:
            value = sample[key]
            assert isinstance(value, torch.Tensor)
            return value.numpy().reshape(value.shape[0], -1)[:, index].T.astype(np.float32, copy=False)

        urban = pixels("urban")
        osm = pixels("osm")
        frame_row = dataset.frame.iloc[patch_index]
        text = _text_vector(str(frame_row.text), token_to_col)
        text = np.broadcast_to(text, (index.size, text.size)).astype(np.float32, copy=False)
        x_patch = np.concatenate(
            [
                pixels("s2"),
                pixels("s1"),
                urban[:, :3],
                urban[:, 3:6],
                urban[:, 6:],
                pixels("roads"),
                osm[:, :14],
                osm[:, 14:],
                text,
            ],
            axis=1,
        )
        x_parts.append(x_patch)
        y_parts.append(label[index].astype(np.int32, copy=False))

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    keep_parts: list[np.ndarray] = []
    for class_id in range(10):
        candidates = np.flatnonzero(y == class_id)
        if candidates.size > cap_per_class:
            candidates = rng.choice(candidates, cap_per_class, replace=False)
        keep_parts.append(candidates)
    keep = np.concatenate(keep_parts)
    rng.shuffle(keep)
    return np.ascontiguousarray(x[keep]), np.ascontiguousarray(y[keep])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(10), zero_division=0
    )
    denominator = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    iou = np.divide(np.diag(cm), denominator, out=np.zeros(10, dtype=float), where=denominator > 0)
    summary: dict[str, object] = {
        "OA": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_mIoU": float(iou.mean()),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_test": int(y_true.size),
    }
    table = pd.DataFrame(
        {
            "class_id": np.arange(10),
            "class_name": CLASS_NAMES,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "IoU": iou,
        }
    )
    return summary, table, cm


def _compute_shap(
    model: XGBClassifier,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = model.get_booster().predict(
        DMatrix(x), pred_contribs=True, approx_contribs=False, strict_shape=True
    )
    # Strict multiclass shape: samples x classes x (features + bias).
    if raw.ndim != 3 or raw.shape[1] != 10:
        raise RuntimeError(f"Unexpected XGBoost SHAP shape: {raw.shape}")
    all_classes = raw[:, :, :-1]
    true_class = all_classes[np.arange(y.size), y]
    return all_classes, true_class


def _importance_tables(
    shap_all: np.ndarray,
    feature_names: list[str],
    feature_groups: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_abs = np.abs(shap_all).mean(axis=(0, 1))
    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "modality": feature_groups,
            "mean_abs_SHAP": mean_abs,
        }
    ).sort_values("mean_abs_SHAP", ascending=False, ignore_index=True)
    total = float(mean_abs.sum())
    feature_table["global_share_percent"] = 100.0 * feature_table["mean_abs_SHAP"] / total
    feature_table["rank"] = np.arange(1, len(feature_table) + 1)

    group_rows: list[dict[str, float | int | str]] = []
    class_rows: list[dict[str, float | int | str]] = []
    group_array = np.asarray(feature_groups)
    for group in GROUP_LABELS:
        columns = np.flatnonzero(group_array == group)
        group_total = float(mean_abs[columns].sum())
        group_rows.append(
            {
                "modality": group,
                "modality_label": GROUP_LABELS[group],
                "n_features": int(columns.size),
                "sum_mean_abs_SHAP": group_total,
                "mean_per_feature_SHAP": float(mean_abs[columns].mean()),
                "contribution_percent": 100.0 * group_total / total,
            }
        )
        per_class = np.abs(shap_all[:, :, columns]).mean(axis=0).sum(axis=1)
        for class_id, value in enumerate(per_class):
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "modality": group,
                    "modality_label": GROUP_LABELS[group],
                    "sum_mean_abs_SHAP": float(value),
                }
            )
    group_table = pd.DataFrame(group_rows).sort_values(
        "contribution_percent", ascending=False, ignore_index=True
    )
    group_table["rank"] = np.arange(1, len(group_table) + 1)
    class_table = pd.DataFrame(class_rows)
    class_table["within_class_percent"] = class_table.groupby("class_id")[
        "sum_mean_abs_SHAP"
    ].transform(lambda value: 100.0 * value / value.sum())
    return feature_table, group_table, class_table


def _plot_global(
    feature_table: pd.DataFrame,
    group_table: pd.DataFrame,
    metrics: dict[str, object],
    output_stem: Path,
    top_n: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)

    grouped = group_table.sort_values("contribution_percent")
    colors = [GROUP_COLORS[value] for value in grouped["modality"]]
    axes[0].barh(grouped["modality_label"], grouped["contribution_percent"], color=colors)
    for y_pos, value in enumerate(grouped["contribution_percent"]):
        axes[0].text(value + 0.25, y_pos, f"{value:.1f}%", va="center", fontsize=8)
    axes[0].set_xlabel("Total mean |SHAP| contribution (%)")
    axes[0].set_ylabel("")
    axes[0].set_title("a  Modality-level importance", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)

    top = feature_table.head(top_n).sort_values("mean_abs_SHAP")
    top_colors = [GROUP_COLORS[value] for value in top["modality"]]
    axes[1].barh([_short_name(x) for x in top["feature"]], top["mean_abs_SHAP"], color=top_colors)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_ylabel("")
    axes[1].set_title(f"b  Top {top_n} input features", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "XGBoost–SHAP input importance (spatial hold-out)\n"
        f"OA={metrics['OA']:.3f}, macro-F1={metrics['macro_F1']:.3f}, "
        f"mIoU={metrics['macro_mIoU']:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_beeswarm(
    true_class_shap: np.ndarray,
    x: np.ndarray,
    feature_names: list[str],
    output_stem: Path,
    top_n: int,
) -> None:
    # Signed SHAP values are selected from each sample's true-class logit.
    cmap = LinearSegmentedColormap.from_list("blue_red", ["#2166AC", "#F7F7F7", "#B2182B"])
    shap.summary_plot(
        true_class_shap,
        x,
        feature_names=[_short_name(x) for x in feature_names],
        max_display=top_n,
        plot_type="dot",
        color=cmap,
        show=False,
        plot_size=(9.0, 6.2),
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(
        "Signed SHAP contribution to the true-class logit",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("SHAP value (positive supports the true class)")
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path(args.project_root).resolve()
    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prepared = root / "artifacts/full/prepared"
    cfg_paths = config["paths"]

    def absolute(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else root / path)

    paths = {
        "image": absolute(cfg_paths["image"]),
        "label": str(prepared / "labels_10class.tif"),
        "manifest": str(prepared / "manifest_hier_v2.csv"),
        "vocab": str(prepared / "vocab_hier_v2.json"),
        "normalization": str(prepared / "normalization_hier_v2.json"),
        "poi": absolute(cfg_paths["osm_poi_raster"]),
        "aoi": absolute(cfg_paths["osm_aoi_raster"]),
        "highres": absolute(cfg_paths["highres_aligned"]),
        "nighttime": absolute(cfg_paths["nighttime_aligned"]),
        "building": absolute(cfg_paths["building_aligned"]),
        "roads": absolute(cfg_paths["roads_aligned"]),
    }
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(paths["manifest"])
    text_tokens = _token_vocabulary(manifest, args.text_features)
    token_to_col = {token: index for index, token in enumerate(text_tokens)}
    feature_names, feature_groups = _feature_schema(
        paths["image"],
        paths["highres"],
        paths["nighttime"],
        paths["building"],
        paths["roads"],
        text_tokens,
    )

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    split_caps = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    for split_index, split in enumerate(("train", "val", "test")):
        print(f"Extracting spatial {split} samples...", flush=True)
        dataset = _make_dataset(
            paths,
            split,
            int(config["data"]["patch_size"]),
            int(config["data"]["max_text_tokens"]),
        )
        arrays[split] = _extract_samples(
            dataset,
            token_to_col,
            np.random.default_rng(args.seed + split_index),
            args.per_class_per_patch,
            split_caps[split],
        )
        print(f"  {split}: X={arrays[split][0].shape}, y={arrays[split][1].shape}", flush=True)

    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    if x_train.shape[1] != len(feature_names):
        raise RuntimeError(f"Feature schema mismatch: data={x_train.shape[1]}, names={len(feature_names)}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=10,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        gamma=0.0,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    print("Training XGBoost surrogate...", flush=True)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=25)
    model.save_model(output_dir / "xgboost_surrogate.json")

    y_pred = model.predict(x_test).astype(np.int32)
    metrics, class_table, cm = _metrics(y_test, y_pred)
    metrics.update(
        {
            "best_iteration": int(model.best_iteration),
            "n_features": int(x_train.shape[1]),
            "n_train": int(y_train.size),
            "n_val": int(y_val.size),
            "sampling": "class-balanced pixels sampled only within the original spatial train/val/test patch split",
            "interpretation_scope": (
                "XGBoost is an independently trained surrogate/feature-audit classifier against BLU labels. "
                "Its SHAP values quantify predictive information in the supplied modalities; they are not "
                "native attributions of the hierarchical neural network's internal activations."
            ),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    class_table.to_csv(output_dir / "class_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    shap_count = min(args.shap_samples, y_test.size)
    rng = np.random.default_rng(args.seed + 100)
    shap_index = rng.choice(y_test.size, shap_count, replace=False)
    print(f"Computing exact TreeSHAP values for {shap_count} spatial test samples...", flush=True)
    shap_all, true_class_shap = _compute_shap(model, x_test[shap_index], y_test[shap_index])
    feature_table, group_table, class_group_table = _importance_tables(
        shap_all, feature_names, feature_groups
    )
    feature_table.to_csv(output_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    group_table.to_csv(output_dir / "shap_modality_importance.csv", index=False, encoding="utf-8-sig")
    class_group_table.to_csv(
        output_dir / "shap_class_modality_importance.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {"feature": feature_names, "modality": feature_groups}
    ).to_csv(output_dir / "feature_schema.csv", index=False, encoding="utf-8-sig")

    _plot_global(
        feature_table,
        group_table,
        metrics,
        output_dir / "xgboost_shap_global_importance",
        args.top_features,
    )
    _plot_beeswarm(
        true_class_shap,
        x_test[shap_index],
        feature_names,
        output_dir / "xgboost_shap_beeswarm",
        args.top_features,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(group_table.to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XGBoost-TreeSHAP audit of all hierarchical-model inputs")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="config_release.yaml")
    parser.add_argument(
        "--output-dir",
        default="outputs/full/final_height_experiment/xgboost_shap_importance",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-features", type=int, default=32)
    parser.add_argument("--per-class-per-patch", type=int, default=24)
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--val-per-class", type=int, default=1200)
    parser.add_argument("--test-per-class", type=int, default=1200)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--shap-samples", type=int, default=5000)
    parser.add_argument("--top-features", type=int, default=20)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
