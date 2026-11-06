from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import load_config
from .model_shap_importance_8class import (
    CenterClassWrapper,
    _bootstrap_mean_ci,
    _feature_schema,
    _input_shap_values,
    _project_text,
    _select_centres,
)
from .predict_experiment_models_8class import make_ours
from .train_comparison import make_dataset


CLASS_NAMES = (
    "住宅", "工业仓储", "公共机构与休闲", "交通设施",
    "其他建设用地", "农业", "林地/灌丛/草地/湿地", "水体",
)
GROUP_LABELS = {
    "s2": "Sentinel-2",
    "s1": "Sentinel-1",
    "highres": "高分辨率影像",
    "nighttime": "夜间灯光",
    "building": "建筑特征",
    "roads": "OSM道路",
    "poi": "OSM POI",
    "aoi": "OSM AOI",
}
GROUP_COLORS = {
    "s2": "#4477AA", "s1": "#66CCEE", "highres": "#228833",
    "nighttime": "#CCBB44", "building": "#EE6677", "roads": "#AA3377",
    "poi": "#999999", "aoi": "#EE8866",
}


def _recenter_dataset(dataset, centres: list[dict[str, int | str]], patch_size: int):
    """Create full-size windows centred on the selected labelled pixels.

    The OSM text remains the text belonging to the source manifest patch.  It
    is intentionally reported separately from raster attributions because it
    is a projected patch-level latent representation.
    """
    half = patch_size // 2
    rows: list[pd.Series] = []
    recentered: list[dict[str, int | str]] = []
    for new_index, centre in enumerate(centres):
        source_index = int(centre["patch_index"])
        row = dataset.frame.iloc[source_index].copy()
        row["row_off"] = int(row["row_off"]) + int(centre["row"]) - half
        row["col_off"] = int(row["col_off"]) + int(centre["col"]) - half
        row["patch_id"] = (
            f"{row['patch_id']}_eg_r{int(centre['row'])}_c{int(centre['col'])}"
        )
        rows.append(row)
        item = dict(centre)
        item.update({"patch_index": new_index, "row": half, "col": half,
                     "class_name": CLASS_NAMES[int(centre["class_id"])]})
        recentered.append(item)
    clone = copy.copy(dataset)
    clone.frame = pd.DataFrame(rows).reset_index(drop=True)
    clone.patch_size = patch_size
    clone._sources = {}
    return clone, recentered


def _pack_full(dataset, centres, model, device):
    keys = ("s2", "s1", "urban", "roads", "osm")
    parts: list[list[torch.Tensor]] = [[] for _ in range(6)]
    records: list[dict[str, object]] = []
    for centre in centres:
        sample = dataset[int(centre["patch_index"])]
        for index, key in enumerate(keys):
            parts[index].append(sample[key].float())
        parts[5].append(_project_text(model, sample, device).cpu()[0])
        records.append(dict(centre))
    return [torch.stack(values).to(device) for values in parts], pd.DataFrame(records)


def _aggregate_window(shap_inputs, input_tensors, window_size: int):
    signed_parts, absolute_parts, value_parts, complete_signed = [], [], [], []
    for shap_value, tensor in zip(shap_inputs, input_tensors):
        values = tensor.detach().float().cpu().numpy()
        if shap_value.ndim == 5 and shap_value.shape[-1] == 1:
            shap_value = shap_value[..., 0]
        if shap_value.ndim == 4:
            height, width = shap_value.shape[-2:]
            r0 = (height - window_size) // 2
            c0 = (width - window_size) // 2
            view = shap_value[:, :, r0:r0 + window_size, c0:c0 + window_size]
            value_view = values[:, :, r0:r0 + window_size, c0:c0 + window_size]
            signed_parts.append(view.sum(axis=(2, 3)))
            absolute_parts.append(np.abs(view).sum(axis=(2, 3)))
            value_parts.append(value_view.mean(axis=(2, 3)))
            complete_signed.append(shap_value.sum(axis=(2, 3)))
        elif shap_value.ndim == 3 and shap_value.shape[-1] == 1:
            view = shap_value[..., 0]
            signed_parts.append(view)
            absolute_parts.append(np.abs(view))
            value_parts.append(values)
            complete_signed.append(view)
        elif shap_value.ndim == 2:
            signed_parts.append(shap_value)
            absolute_parts.append(np.abs(shap_value))
            value_parts.append(values)
            complete_signed.append(shap_value)
        else:
            raise RuntimeError(f"Unsupported SHAP tensor shape: {shap_value.shape}")
    return (
        np.concatenate(signed_parts, axis=1),
        np.concatenate(absolute_parts, axis=1),
        np.concatenate(value_parts, axis=1),
        np.concatenate(complete_signed, axis=1),
    )


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def _plot_raster(modality: pd.DataFrame, class_table: pd.DataFrame, output: Path) -> None:
    _style()
    order = list(GROUP_LABELS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.65), constrained_layout=True)
    plot = modality.set_index("group").loc[order].sort_values("importance_percent")
    y = np.arange(len(plot))
    ax1.barh(y, plot.importance_percent, color=[GROUP_COLORS[x] for x in plot.index])
    ax1.errorbar(
        plot.importance_percent, y,
        xerr=np.vstack([
            plot.importance_percent - plot.ci_low_percent,
            plot.ci_high_percent - plot.importance_percent,
        ]), fmt="none", ecolor="#333333", capsize=2, lw=0.7,
    )
    ax1.set_yticks(y, [GROUP_LABELS[x] for x in plot.index])
    ax1.set_xlabel("栅格模态内部归一化贡献（%）")
    ax1.set_title("a  栅格模态归因（95% bootstrap CI）", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#dddddd", lw=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    matrix = class_table.pivot(index="class_name", columns="group", values="row_percent")
    matrix = matrix.loc[list(CLASS_NAMES), order]
    image = ax2.imshow(matrix.values, cmap="YlGnBu", vmin=0, aspect="auto")
    ax2.set_xticks(np.arange(len(order)), [GROUP_LABELS[x] for x in order], rotation=35, ha="right")
    ax2.set_yticks(np.arange(8), CLASS_NAMES)
    ax2.set_title("b  类别内栅格模态构成", loc="left", fontweight="bold")
    for r in range(8):
        for c in range(len(order)):
            value = matrix.iat[r, c]
            ax2.text(c, r, f"{value:.1f}", ha="center", va="center", fontsize=5.4,
                     color="white" if value > 0.6 * image.norm.vmax else "#222222")
    cb = fig.colorbar(image, ax=ax2, fraction=0.045, pad=0.025)
    cb.set_label("类别内贡献（%）")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"expected_gradients_raster_modalities.{suffix}",
                    dpi=400 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_text(text_class: pd.DataFrame, output: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    y = np.arange(len(text_class))
    ax.barh(y, text_class.mean_abs_shap, color="#555555")
    ax.errorbar(
        text_class.mean_abs_shap, y,
        xerr=np.vstack([
            text_class.mean_abs_shap - text_class.ci_low,
            text_class.ci_high - text_class.mean_abs_shap,
        ]), fmt="none", ecolor="#222222", capsize=2, lw=0.7,
    )
    ax.set_yticks(y, text_class.class_name)
    ax.invert_yaxis()
    ax.set_xlabel("16维文本潜变量的平均绝对归因（独立尺度）")
    ax.set_title("OSM文本潜变量归因（不与栅格模态归一化）", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"expected_gradients_text_separate.{suffix}",
                    dpi=400 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_hierarchical.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--background-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--aggregation-window", type=int, default=49)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--explainer-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.aggregation_window % 2 != 1 or args.aggregation_window > args.patch_size:
        raise ValueError("aggregation-window must be odd and no larger than patch-size")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config.resolve())
    modalities = {"s2", "s1", "highres", "nighttime", "building", "roads", "poi", "aoi", "text"}
    train = make_dataset(cfg, "train", False, modalities)
    test = make_dataset(cfg, "test", False, modalities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_ours(cfg, test, args.checkpoint.resolve()).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    selection_margin = args.aggregation_window // 2
    test_centres = _select_centres(test, rng, selection_margin, args.samples_per_class)
    train_per_class = max(1, math.ceil(args.background_size / 8))
    background_centres = _select_centres(train, rng, selection_margin, train_per_class)[:args.background_size]
    test_recentred, test_centres = _recenter_dataset(test, test_centres, args.patch_size)
    train_recentred, background_centres = _recenter_dataset(train, background_centres, args.patch_size)
    background_inputs, _ = _pack_full(train_recentred, background_centres, model, device)
    test_inputs, samples = _pack_full(test_recentred, test_centres, model, device)

    feature_names, groups = _feature_schema(test)
    n_features = len(feature_names)
    signed_all = np.zeros((len(samples), n_features), dtype=np.float64)
    absolute_all = np.zeros_like(signed_all)
    values_all = np.zeros_like(signed_all)
    complete_signed_all = np.zeros_like(signed_all)
    output_values = np.zeros(len(samples), dtype=np.float64)
    expected_values = np.zeros(len(samples), dtype=np.float64)

    for class_id, class_name in enumerate(CLASS_NAMES):
        indices = np.flatnonzero(samples.class_id.to_numpy() == class_id)
        foreground = [value[indices] for value in test_inputs]
        wrapper = CenterClassWrapper(model, class_id).to(device).eval()
        explainer = shap.GradientExplainer(
            wrapper, background_inputs, batch_size=args.explainer_batch_size
        )
        result = explainer.shap_values(
            foreground, nsamples=args.nsamples, rseed=args.seed + class_id
        )
        shap_inputs = _input_shap_values(result)
        signed, absolute, values, complete_signed = _aggregate_window(
            shap_inputs, foreground, args.aggregation_window
        )
        signed_all[indices], absolute_all[indices] = signed, absolute
        values_all[indices], complete_signed_all[indices] = values, complete_signed
        with torch.no_grad():
            output_values[indices] = wrapper(*foreground).squeeze(1).cpu().numpy()
            expected_values[indices] = wrapper(*background_inputs).mean().item()
        print(f"Expected gradients {class_id + 1}/8: {class_name}", flush=True)
        del explainer, wrapper, result, shap_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    samples["target_log_probability"] = output_values
    samples["background_expected_log_probability"] = expected_values
    samples["complete_sum_signed_shap"] = complete_signed_all.sum(axis=1)
    samples["additivity_residual"] = (
        output_values - expected_values - complete_signed_all.sum(axis=1)
    )
    samples.to_csv(output / "expected_gradients_samples.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "expected_gradients_values.npz", signed_shap=signed_all,
        absolute_shap=absolute_all, complete_signed_shap=complete_signed_all,
        feature_values=values_all, feature_names=np.asarray(feature_names),
        groups=np.asarray(groups),
    )

    groups_array = np.asarray(groups)
    raster_mask = groups_array != "text"
    raster_mean = absolute_all[:, raster_mask].mean(axis=0)
    raster_total = raster_mean.sum()
    feature_table = pd.DataFrame({
        "feature_index": np.flatnonzero(raster_mask),
        "feature": np.asarray(feature_names)[raster_mask],
        "group": groups_array[raster_mask],
        "mean_abs_shap": raster_mean,
        "raster_only_percent": 100.0 * raster_mean / raster_total,
        "mean_signed_shap": signed_all[:, raster_mask].mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    feature_table.to_csv(output / "expected_gradients_raster_features.csv", index=False, encoding="utf-8-sig")

    group_order = list(GROUP_LABELS)
    group_matrix = np.column_stack([
        absolute_all[:, groups_array == group].sum(axis=1) for group in group_order
    ])
    group_mean = group_matrix.mean(axis=0)
    ci_low, ci_high = _bootstrap_mean_ci(group_matrix, np.random.default_rng(args.seed + 1000))
    group_total = group_mean.sum()
    modality = pd.DataFrame({
        "group": group_order,
        "modality": [GROUP_LABELS[x] for x in group_order],
        "n_channels": [int((groups_array == x).sum()) for x in group_order],
        "mean_abs_shap": group_mean, "ci_low": ci_low, "ci_high": ci_high,
        "importance_percent": 100.0 * group_mean / group_total,
        "ci_low_percent": 100.0 * ci_low / group_total,
        "ci_high_percent": 100.0 * ci_high / group_total,
    })
    modality.to_csv(output / "expected_gradients_raster_modalities.csv", index=False, encoding="utf-8-sig")

    class_rows, text_rows = [], []
    text_values = absolute_all[:, groups_array == "text"].sum(axis=1)
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = samples.class_id.to_numpy() == class_id
        means = group_matrix[mask].mean(axis=0)
        for group, value in zip(group_order, means):
            class_rows.append({"class_id": class_id, "class_name": class_name,
                               "group": group, "mean_abs_shap": value,
                               "row_percent": 100.0 * value / means.sum()})
        class_text = text_values[mask]
        boot = np.asarray([
            class_text[rng.integers(0, len(class_text), len(class_text))].mean()
            for _ in range(2000)
        ])
        text_rows.append({"class_id": class_id, "class_name": class_name,
                          "mean_abs_shap": class_text.mean(),
                          "ci_low": np.quantile(boot, 0.025),
                          "ci_high": np.quantile(boot, 0.975)})
    class_table = pd.DataFrame(class_rows)
    text_table = pd.DataFrame(text_rows)
    class_table.to_csv(output / "expected_gradients_raster_by_class.csv", index=False, encoding="utf-8-sig")
    text_table.to_csv(output / "expected_gradients_text_separate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "text_dimension": np.asarray(feature_names)[groups_array == "text"],
        "mean_abs_shap": absolute_all[:, groups_array == "text"].mean(axis=0),
        "mean_signed_shap": signed_all[:, groups_array == "text"].mean(axis=0),
    }).to_csv(output / "expected_gradients_text_dimensions.csv", index=False, encoding="utf-8-sig")

    _plot_raster(modality, class_table, output)
    _plot_text(text_table, output)
    metadata = {
        "method": "SHAP GradientExplainer implementation of Expected Gradients",
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "input_patch_size": args.patch_size,
        "raster_attribution_aggregation_window": args.aggregation_window,
        "samples_per_class": args.samples_per_class,
        "n_test_samples": len(samples),
        "background_size": args.background_size,
        "nsamples": args.nsamples,
        "explainer_batch_size": args.explainer_batch_size,
        "explained_output": "final gated target-class centre-pixel log probability",
        "raster_normalization": "79 raster channels only; sums to 100 percent",
        "text_reporting": "16-D projected text latent reported separately and not normalized with raster inputs",
        "text_scope": "text retained from the source 128x128 manifest patch",
        "additivity_median_absolute_residual": float(np.median(np.abs(samples.additivity_residual))),
        "additivity_mean_absolute_residual": float(np.mean(np.abs(samples.additivity_residual))),
    }
    (output / "expected_gradients_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
