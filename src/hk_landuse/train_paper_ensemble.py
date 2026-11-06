from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .labels import CLASS_NAMES, COARSE_CLASS_NAMES, LEAF_CLASS_NAMES


TASKS = {
    "coarse4": (COARSE_CLASS_NAMES, lambda d: np.ones(len(d), dtype=bool), lambda d: d.coarse_target.to_numpy()),
    "built6": (LEAF_CLASS_NAMES[:6], lambda d: d.leaf_target.to_numpy() < 6, lambda d: d.leaf_target.to_numpy()),
    "natural8": (LEAF_CLASS_NAMES[6:], lambda d: d.leaf_target.to_numpy() >= 6, lambda d: d.leaf_target.to_numpy() - 6),
    "leaf14": (LEAF_CLASS_NAMES, lambda d: np.ones(len(d), dtype=bool), lambda d: d.leaf_target.to_numpy()),
    "official10": (CLASS_NAMES, lambda d: np.ones(len(d), dtype=bool), lambda d: d.official_target.to_numpy()),
}


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "s2": [c for c in columns if c.startswith("s2_")],
        "s1": [c for c in columns if c.startswith("s1_")],
        "highres": [c for c in columns if c.startswith("highres_")],
        "poi": [c for c in columns if c.startswith("poi_tfidf_")],
        "aoi": [c for c in columns if c.startswith("aoi_")],
        "nightlight": [c for c in columns if c.startswith("ntl_")],
        "building": [
            c for c in columns if c.startswith("building_") and c != "building_nightlight_proxy"
        ],
        "nightlight_building_proxy": ["building_nightlight_proxy"],
    }
    return groups


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    labels = np.arange(len(names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    union = support + predicted - tp
    precision = np.divide(tp, predicted, out=np.full_like(tp, np.nan), where=predicted > 0)
    recall = np.divide(tp, support, out=np.full_like(tp, np.nan), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.full_like(tp, np.nan), where=(precision + recall) > 0)
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    present = support > 0
    normalized = np.divide(
        matrix,
        support[:, None],
        out=np.zeros_like(matrix, dtype=float),
        where=support[:, None] > 0,
    )
    return {
        "overall_accuracy": float(tp.sum() / max(matrix.sum(), 1)),
        "macro_precision_present": float(np.nanmean(precision[present])),
        "macro_recall_present": float(np.nanmean(recall[present])),
        "macro_f1_present": float(np.nanmean(f1[present])),
        "macro_iou_present": float(np.nanmean(iou[present])),
        "per_class": {
            name: {
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": None if np.isnan(recall[i]) else float(recall[i]),
                "f1": None if np.isnan(f1[i]) else float(f1[i]),
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
            }
            for i, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_row_normalized": normalized.tolist(),
    }


def group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    names: list[str],
    seed: int,
    repetitions: int = 500,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = {"overall_accuracy": [], "macro_iou_present": []}
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        metric = classification_metrics(y_true[indices], y_pred[indices], names)
        for key in values:
            values[key].append(metric[key])
    return {key: np.percentile(value, (2.5, 97.5)).tolist() for key, value in values.items()}


def extra_trees(cfg: dict, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(cfg["ensemble"].get("n_estimators", 500)),
        max_depth=cfg["ensemble"].get("max_depth"),
        min_samples_leaf=int(cfg["ensemble"].get("min_samples_leaf", 2)),
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=int(cfg["ensemble"].get("n_jobs", -1)),
        random_state=seed,
    )


def stacking_model(cfg: dict, y: np.ndarray, groups: np.ndarray, seed: int):
    estimators = [
        ("extra", extra_trees(cfg, seed)),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed + 1,
            ),
        ),
        (
            "hist_gradient",
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.06,
                    l2_regularization=1.0,
                    random_state=seed + 2,
                ),
            ),
        ),
    ]
    # XGBoost 1.7 cannot fit a spatial fold when that fold's training subset
    # misses an intermediate rare-class label.  Random folds would hide this
    # issue through spatial leakage, so the spatially valid ensemble omits it.
    folds = min(int(cfg["ensemble"].get("stacking_folds", 5)), len(np.unique(groups)))
    splits = list(GroupKFold(folds).split(np.zeros(len(y)), y, groups))
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final,
        cv=splits,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def task_arrays(frame: pd.DataFrame, task: str, features: list[str], split: str):
    names, mask_fn, target_fn = TASKS[task]
    part = frame[frame.split == split].copy()
    mask = mask_fn(part)
    part = part.loc[mask]
    return (
        part[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
        target_fn(part).astype(int),
        part.group.to_numpy(),
        part.patch_id.to_numpy(),
        names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style block ensemble and modality ablations.")
    parser.add_argument("--config", default="config_paper_replication.yaml")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["data"]["seed"])
    output = Path(cfg["paths"]["output_dir"])
    frame = pd.read_csv(output / "block_features.csv")
    frame = frame[frame.leaf_purity >= float(cfg["data"].get("min_patch_purity", 0.25))].copy()
    groups = feature_groups(frame.columns.tolist())
    sequences = {
        "S2": ["s2"],
        "highres": ["highres"],
        "S2_highres": ["s2", "highres"],
        "plus_POI": ["s2", "highres", "poi"],
        "plus_AOI": ["s2", "highres", "poi", "aoi"],
        "plus_NTL": ["s2", "highres", "poi", "aoi", "nightlight"],
        "plus_building": ["s2", "highres", "poi", "aoi", "nightlight", "building", "nightlight_building_proxy"],
        "plus_S1_all": list(groups),
    }
    if not args.skip_ablations:
        ablations: dict[str, dict] = {}
        for task in TASKS:
            ablations[task] = {}
            for sequence, group_names in sequences.items():
                features = [column for group in group_names for column in groups[group]]
                x_train, y_train, _, _, names = task_arrays(frame, task, features, "train")
                model = make_pipeline(SimpleImputer(strategy="median"), extra_trees(cfg, seed))
                model.fit(x_train, y_train)
                ablations[task][sequence] = {}
                for split in ("val", "test"):
                    x, y, _, _, _ = task_arrays(frame, task, features, split)
                    prediction = model.predict(x)
                    ablations[task][sequence][split] = classification_metrics(y, prediction, list(names))
                print(
                    f"task={task:10s} features={sequence:14s} "
                    f"test_OA={ablations[task][sequence]['test']['overall_accuracy']:.4f} "
                    f"test_mIoU={ablations[task][sequence]['test']['macro_iou_present']:.4f}",
                    flush=True,
                )
        (output / "ablation_metrics.json").write_text(
            json.dumps(ablations, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    all_features = [column for columns in groups.values() for column in columns]
    final_results: dict[str, dict] = {}
    predictions = []
    for task in TASKS:
        x_train, y_train, train_groups, _, names = task_arrays(frame, task, all_features, "train")
        model = stacking_model(cfg, y_train, train_groups, seed)
        model.fit(x_train, y_train)
        final_results[task] = {}
        for split in ("val", "test"):
            x, y, split_groups, patch_ids, _ = task_arrays(frame, task, all_features, split)
            prediction = model.predict(x)
            probabilities = model.predict_proba(x)
            metrics = classification_metrics(y, prediction, list(names))
            if split == "test":
                metrics["spatial_group_bootstrap_95ci"] = group_bootstrap_ci(
                    y, prediction, split_groups, list(names), seed
                )
            final_results[task][split] = metrics
            for patch_id, truth, pred, probability in zip(patch_ids, y, prediction, probabilities):
                predictions.append(
                    {
                        "task": task,
                        "split": split,
                        "patch_id": patch_id,
                        "target": int(truth),
                        "prediction": int(pred),
                        "probabilities": json.dumps(probability.tolist()),
                    }
                )
        print(
            f"STACK task={task:10s} test_OA={final_results[task]['test']['overall_accuracy']:.4f} "
            f"test_mIoU={final_results[task]['test']['macro_iou_present']:.4f}",
            flush=True,
        )
        (output / "stacking_metrics.json").write_text(
            json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "stacking_metrics.json").write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(predictions).to_csv(output / "stacking_predictions.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
