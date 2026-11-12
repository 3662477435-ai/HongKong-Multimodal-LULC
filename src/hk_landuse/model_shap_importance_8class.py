from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
from torch.nn import functional as F

from .config import load_config
from .optical_indices import OPTICAL_INDEX_NAMES
from .predict_experiment_models_8class import make_ours
from .task8 import leaf_target_to_official8
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
    "building": "建筑功能与高度",
    "roads": "OSM道路",
    "poi": "OSM POI",
    "aoi": "OSM AOI",
    "text": "OSM文本",
}

GROUP_COLORS = {
    "s2": "#4477AA",
    "s1": "#66CCEE",
    "highres": "#228833",
    "nighttime": "#CCBB44",
    "building": "#EE6677",
    "roads": "#AA3377",
    "poi": "#999999",
    "aoi": "#EE8866",
    "text": "#555555",
}

S1_DERIVED_NAMES = (
    "S1_total_power",
    "S1_normalized_pol_difference",
    "S1_dual_pol_RVI",
    "S1_dual_pol_entropy",
)


class CenterClassWrapper(nn.Module):
    """Expose one class log-probability at the crop centre to SHAP.

    The sixth input is the trained network's continuous 16-D projected text
    representation.  This avoids taking gradients with respect to discrete
    token IDs while preserving the actual downstream computation.
    """

    def __init__(self, model: nn.Module, class_id: int) -> None:
        super().__init__()
        self.model = model
        self.class_id = int(class_id)

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        urban: torch.Tensor,
        roads: torch.Tensor,
        osm: torch.Tensor,
        text_projected: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        text_map = text_projected[:, :, None, None].expand(
            -1, -1, s2.shape[-2], s2.shape[-1]
        )
        fused = model.base.fusion(
            torch.cat([s2, s1, urban, roads, osm, text_map], dim=1)
        )
        segmenter = model._base_segmenter()
        features = segmenter.encode(fused)
        base_logits = segmenter.logits_from_features(features, s2.shape[-2:])
        semantic_feature = torch.cat(
            [
                features[0],
                model._semantic_feature(model.semantic_osm, osm, features[0]),
                model._semantic_feature(model.semantic_urban, urban, features[0]),
                model._semantic_feature(model.semantic_roads, roads, features[0]),
            ],
            dim=1,
        )
        direct_logits = base_logits + F.softplus(model.semantic_residual_scale) * model.semantic_residual(
            semantic_feature
        )
        hierarchy_feature = model._hierarchy_feature(features)
        hierarchy_log_probs = model._route(
            model.coarse_head(hierarchy_feature),
            model.built_head(hierarchy_feature),
            model.natural_head(hierarchy_feature),
        )
        direct_log_probs = F.log_softmax(direct_logits, dim=1)
        gate = torch.sigmoid(model.hierarchy_gate_logit)
        combined = torch.logaddexp(
            direct_log_probs
            + torch.log((1.0 - gate).clamp_min(1e-6))[None, :, None, None],
            hierarchy_log_probs
            + torch.log(gate.clamp_min(1e-6))[None, :, None, None],
        )
        combined = combined - torch.logsumexp(combined, dim=1, keepdim=True)
        row, col = combined.shape[-2] // 2, combined.shape[-1] // 2
        return combined[:, self.class_id, row, col].unsqueeze(1)


def _feature_schema(dataset) -> tuple[list[str], list[str]]:
    with rasterio.open(dataset.image_path) as source:
        descriptions = [str(value) for value in source.descriptions]
    s2_names = [descriptions[index] for index in dataset.s2_indices] + list(OPTICAL_INDEX_NAMES)
    s1_names = [descriptions[index] for index in dataset.s1_indices] + list(S1_DERIVED_NAMES)
    with rasterio.open(dataset.highres_path) as source:
        highres_names = [str(value) for value in source.descriptions]
    with rasterio.open(dataset.nighttime_path) as source:
        nighttime_names = [str(value) for value in source.descriptions]
    with rasterio.open(dataset.building_path) as source:
        building_names = [str(value) for value in source.descriptions]
    with rasterio.open(dataset.road_features_path) as source:
        road_names = [str(value) for value in source.descriptions]
    with rasterio.open(dataset.osm_poi_path) as source:
        poi_names = [str(value).split("|")[0] for value in source.descriptions]
    aoi_names = [f"AOI_function_class_{index}" for index in range(1, 7)]
    text_names = [f"OSM_text_latent_{index:02d}" for index in range(1, 17)]

    names: list[str] = []
    groups: list[str] = []

    def add(group: str, values: list[str] | tuple[str, ...]) -> None:
        names.extend(values)
        groups.extend([group] * len(values))

    add("s2", s2_names)
    add("s1", s1_names)
    add("highres", highres_names)
    add("nighttime", nighttime_names)
    add("building", building_names)
    add("roads", road_names)
    add("poi", poi_names)
    add("aoi", aoi_names)
    add("text", text_names)
    return names, groups


def _candidate_centres(dataset, rng: np.random.Generator, margin: int) -> dict[int, list[tuple[int, int, int]]]:
    candidates = {class_id: [] for class_id in range(8)}
    for patch_index in rng.permutation(len(dataset)):
        sample = dataset[int(patch_index)]
        target = leaf_target_to_official8(sample["label"]).numpy()
        interior = target[margin:-margin, margin:-margin]
        for class_id in range(8):
            positions = np.argwhere(interior == class_id)
            if positions.size:
                row, col = positions[rng.integers(0, len(positions))]
                candidates[class_id].append(
                    (int(patch_index), int(row + margin), int(col + margin))
                )
    return candidates


def _select_centres(
    dataset,
    rng: np.random.Generator,
    margin: int,
    per_class: int,
) -> list[dict[str, int | str]]:
    candidates = _candidate_centres(dataset, rng, margin)
    selected: list[dict[str, int | str]] = []
    for class_id in range(8):
        values = candidates[class_id]
        if len(values) < per_class:
            raise RuntimeError(
                f"Class {class_id} has only {len(values)} eligible patches; requested {per_class}"
            )
        chosen = rng.choice(len(values), size=per_class, replace=False)
        for position in chosen:
            patch_index, row, col = values[int(position)]
            selected.append(
                {
                    "patch_index": patch_index,
                    "patch_id": str(dataset.frame.iloc[patch_index].patch_id),
                    "row": row,
                    "col": col,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                }
            )
    return selected


def _project_text(model: nn.Module, sample: dict[str, object], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return model.base.text_projection(
            model.base.text_encoder(
                sample["input_ids"][None].to(device),
                sample["text_mask"][None].to(device),
            )
        ).float()


def _pack_centres(
    dataset,
    centres: list[dict[str, int | str]],
    model: nn.Module,
    device: torch.device,
    crop_size: int,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    radius = crop_size // 2
    keys = ("s2", "s1", "urban", "roads", "osm")
    parts: list[list[torch.Tensor]] = [[] for _ in range(6)]
    records: list[dict[str, object]] = []
    cache: dict[int, dict[str, object]] = {}
    text_cache: dict[int, torch.Tensor] = {}
    for centre in centres:
        patch_index = int(centre["patch_index"])
        sample = cache.setdefault(patch_index, dataset[patch_index])
        row, col = int(centre["row"]), int(centre["col"])
        row_slice = slice(row - radius, row + radius + 1)
        col_slice = slice(col - radius, col + radius + 1)
        for input_index, key in enumerate(keys):
            tensor = sample[key][:, row_slice, col_slice].float()
            parts[input_index].append(tensor)
        if patch_index not in text_cache:
            text_cache[patch_index] = _project_text(model, sample, device).cpu()[0]
        parts[5].append(text_cache[patch_index])
        records.append(dict(centre))
    packed = [torch.stack(values).to(device) for values in parts]
    return packed, pd.DataFrame(records)


def _input_shap_values(result: object, n_inputs: int = 6) -> list[np.ndarray]:
    # SHAP <=0.48 returns [input] for a scalar output, while some versions
    # wrap it once more as [output][input].
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list) or len(result) != n_inputs:
        raise RuntimeError(f"Unexpected GradientExplainer output structure: {type(result)}")
    return [np.asarray(value) for value in result]


def _aggregate_shap(
    shap_inputs: list[np.ndarray],
    input_tensors: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signed_parts: list[np.ndarray] = []
    absolute_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for shap_value, tensor in zip(shap_inputs, input_tensors):
        values = tensor.detach().float().cpu().numpy()
        if shap_value.ndim == 5 and shap_value.shape[-1] == 1:
            shap_value = shap_value[..., 0]
        if shap_value.ndim == 4:
            signed_parts.append(shap_value.sum(axis=(2, 3)))
            absolute_parts.append(np.abs(shap_value).sum(axis=(2, 3)))
            value_parts.append(values.mean(axis=(2, 3)))
        elif shap_value.ndim == 3 and shap_value.shape[-1] == 1:
            shap_value = shap_value[..., 0]
            signed_parts.append(shap_value)
            absolute_parts.append(np.abs(shap_value))
            value_parts.append(values)
        elif shap_value.ndim == 2:
            signed_parts.append(shap_value)
            absolute_parts.append(np.abs(shap_value))
            value_parts.append(values)
        else:
            raise RuntimeError(f"Unsupported SHAP tensor shape: {shap_value.shape}")
    return (
        np.concatenate(signed_parts, axis=1),
        np.concatenate(absolute_parts, axis=1),
        np.concatenate(value_parts, axis=1),
    )


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, repeats: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    draws = np.empty((repeats, values.shape[1]), dtype=np.float64)
    for index in range(repeats):
        draws[index] = values[rng.integers(0, n, size=n)].mean(axis=0)
    return np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _short_name(value: str) -> str:
    replacements = {
        "building_": "建筑_",
        "road_": "道路_",
        "highres_": "高分_",
        "VIIRS_": "夜光_",
        "OSM_text_latent_": "文本潜变量_",
        "AOI_function_class_": "AOI功能类_",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value[:36]


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tif"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def _plot_global(
    feature_table: pd.DataFrame,
    modality_table: pd.DataFrame,
    class_modality: pd.DataFrame,
    output_dir: Path,
) -> None:
    _style()
    # Nature-style double-column figure (183 mm): two equally weighted panels.
    fig = plt.figure(figsize=(7.20, 4.05), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1), wspace=0.10)
    ax_mod = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    modality_plot = modality_table.sort_values("importance_percent")
    y = np.arange(len(modality_plot))
    colors = [GROUP_COLORS[group] for group in modality_plot["group"]]
    lower = modality_plot["importance_percent"] - modality_plot["ci_low_percent"]
    upper = modality_plot["ci_high_percent"] - modality_plot["importance_percent"]
    ax_mod.barh(y, modality_plot["importance_percent"], color=colors, height=0.68, edgecolor="white")
    ax_mod.errorbar(
        modality_plot["importance_percent"], y,
        xerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2, lw=0.8,
    )
    for y_value, value in zip(y, modality_plot["importance_percent"]):
        ax_mod.text(value + 0.8, y_value, f"{value:.1f}", va="center", ha="left", fontsize=6.4)
    ax_mod.set_yticks(y, modality_plot["modality"])
    ax_mod.set_xlabel("归一化平均 |SHAP|（%）")
    ax_mod.set_xlim(0, max(56.0, float(modality_plot["ci_high_percent"].max()) + 3.0))
    ax_mod.set_title("a  模态总体贡献（95% bootstrap CI）", loc="left", fontweight="bold", fontsize=9)
    ax_mod.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax_mod.spines[["top", "right"]].set_visible(False)

    group_order = list(GROUP_LABELS)
    matrix = class_modality.pivot(index="class_name", columns="group", values="row_percent").loc[
        list(CLASS_NAMES), group_order
    ]
    image = ax_heat.imshow(
        matrix.values,
        cmap="YlGnBu",
        vmin=0,
        vmax=max(35, float(matrix.values.max())),
        aspect="auto",
    )
    short_group_labels = ("S2", "S1", "高分", "夜光", "建筑", "道路", "POI", "AOI", "文本")
    ax_heat.set_xticks(np.arange(len(group_order)), short_group_labels, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(8), CLASS_NAMES)
    ax_heat.set_title("b  各类别内部的模态贡献构成", loc="left", fontweight="bold", fontsize=9)
    ax_heat.tick_params(axis="both", labelsize=6.3, length=2.5)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iat[row, col]
            ax_heat.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=5.8,
                         color="white" if value > 0.60 * image.norm.vmax else "#222222")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.025)
    colorbar.set_label("类内贡献（%）")
    colorbar.ax.tick_params(labelsize=6.3, length=2.5)
    colorbar.set_label("类内贡献（%）", fontsize=7)
    _save_figure(fig, output_dir / "model_shap_global_importance_nature")
    plt.close(fig)


def _plot_beeswarm(
    signed: np.ndarray,
    values: np.ndarray,
    feature_table: pd.DataFrame,
    feature_names: list[str],
    groups: list[str],
    output_dir: Path,
    seed: int,
) -> None:
    _style()
    rng = np.random.default_rng(seed)
    top_indices = feature_table.nlargest(20, "mean_abs_shap")["feature_index"].to_numpy(dtype=int)
    top_indices = top_indices[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 7.0), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("feature_value", ["#2C7BB6", "#F7F7F7", "#D7191C"])
    for row, feature_index in enumerate(top_indices):
        shap_values = signed[:, feature_index]
        feature_values = values[:, feature_index]
        finite = np.isfinite(feature_values)
        lo, hi = np.quantile(feature_values[finite], [0.05, 0.95]) if finite.any() else (0.0, 1.0)
        scale = np.clip((feature_values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        jitter = rng.uniform(-0.22, 0.22, size=len(shap_values))
        ax.scatter(shap_values, row + jitter, c=scale, cmap=cmap, vmin=0, vmax=1,
                   s=16, alpha=0.78, linewidths=0)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(np.arange(len(top_indices)), [_short_name(feature_names[index]) for index in top_indices])
    ax.set_xlabel("聚合SHAP值（对目标类别中心像元 log-probability 的贡献）")
    ax.set_title("本文模型的特征贡献方向与样本差异", fontweight="bold")
    ax.grid(axis="x", color="#E2E2E2", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["低", "高"])
    colorbar.set_label("归一化特征值")
    _save_figure(fig, output_dir / "model_shap_beeswarm")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="GradientSHAP for the trained eight-class hierarchical multimodal FCNN")
    parser.add_argument("--config", type=Path, default=Path("config_release.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=12)
    parser.add_argument("--background-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=65)
    parser.add_argument("--nsamples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.crop_size % 2 != 1:
        raise ValueError("crop-size must be odd")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config.resolve())
    modalities = {"s2", "s1", "highres", "nighttime", "building", "roads", "poi", "aoi", "text"}
    train_dataset = make_dataset(cfg, "train", False, modalities)
    test_dataset = make_dataset(cfg, "test", False, modalities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_ours(cfg, test_dataset, args.checkpoint.resolve()).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    radius = args.crop_size // 2
    test_centres = _select_centres(test_dataset, rng, radius, args.samples_per_class)
    train_per_class = max(1, math.ceil(args.background_size / 8))
    background_centres = _select_centres(train_dataset, rng, radius, train_per_class)[: args.background_size]
    background_inputs, background_frame = _pack_centres(
        train_dataset, background_centres, model, device, args.crop_size
    )
    test_inputs, sample_frame = _pack_centres(
        test_dataset, test_centres, model, device, args.crop_size
    )
    feature_names, groups = _feature_schema(test_dataset)
    if len(feature_names) != sum(value.shape[1] for value in test_inputs):
        raise RuntimeError("Feature schema does not match model inputs")

    signed_all = np.zeros((len(sample_frame), len(feature_names)), dtype=np.float64)
    absolute_all = np.zeros_like(signed_all)
    values_all = np.zeros_like(signed_all)
    output_values = np.zeros(len(sample_frame), dtype=np.float64)
    expected_values = np.zeros(len(sample_frame), dtype=np.float64)

    for class_id, class_name in enumerate(CLASS_NAMES):
        indices = np.flatnonzero(sample_frame["class_id"].to_numpy() == class_id)
        class_inputs = [value[indices] for value in test_inputs]
        wrapper = CenterClassWrapper(model, class_id).to(device).eval()
        explainer = shap.GradientExplainer(wrapper, background_inputs, batch_size=8)
        result = explainer.shap_values(class_inputs, nsamples=args.nsamples, rseed=args.seed + class_id)
        shap_inputs = _input_shap_values(result)
        signed, absolute, values = _aggregate_shap(shap_inputs, class_inputs)
        signed_all[indices] = signed
        absolute_all[indices] = absolute
        values_all[indices] = values
        with torch.no_grad():
            output_values[indices] = wrapper(*class_inputs).squeeze(1).cpu().numpy()
            expected = wrapper(*background_inputs).mean().item()
            expected_values[indices] = expected
        print(
            f"SHAP {class_id + 1}/8 complete: {class_name}, n={len(indices)}, "
            f"mean|SHAP|={absolute.mean():.6f}",
            flush=True,
        )
        del explainer, wrapper, result, shap_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sample_frame["target_log_probability"] = output_values
    sample_frame["background_expected_log_probability"] = expected_values
    sample_frame["sum_signed_shap"] = signed_all.sum(axis=1)
    sample_frame["additivity_residual"] = output_values - expected_values - signed_all.sum(axis=1)
    sample_frame.to_csv(output_dir / "model_shap_samples.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "model_shap_values.npz",
        signed_shap=signed_all,
        absolute_shap=absolute_all,
        feature_values=values_all,
        feature_names=np.asarray(feature_names),
        groups=np.asarray(groups),
    )

    mean_abs = absolute_all.mean(axis=0)
    total = mean_abs.sum()
    feature_table = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names)),
            "feature": feature_names,
            "group": groups,
            "modality": [GROUP_LABELS[group] for group in groups],
            "mean_abs_shap": mean_abs,
            "importance_percent": 100.0 * mean_abs / total,
            "mean_signed_shap": signed_all.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    feature_table.to_csv(output_dir / "model_shap_feature_importance.csv", index=False, encoding="utf-8-sig")

    group_order = list(GROUP_LABELS)
    group_matrix = np.column_stack(
        [absolute_all[:, np.asarray(groups) == group].sum(axis=1) for group in group_order]
    )
    group_mean = group_matrix.mean(axis=0)
    ci_low, ci_high = _bootstrap_mean_ci(group_matrix, np.random.default_rng(args.seed + 1000))
    group_total = group_mean.sum()
    modality_table = pd.DataFrame(
        {
            "group": group_order,
            "modality": [GROUP_LABELS[group] for group in group_order],
            "n_channels": [groups.count(group) for group in group_order],
            "mean_abs_shap": group_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "importance_percent": 100.0 * group_mean / group_total,
            "ci_low_percent": 100.0 * ci_low / group_total,
            "ci_high_percent": 100.0 * ci_high / group_total,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    modality_table.to_csv(output_dir / "model_shap_modality_importance.csv", index=False, encoding="utf-8-sig")

    class_rows: list[dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        indices = sample_frame["class_id"].to_numpy() == class_id
        means = group_matrix[indices].mean(axis=0)
        denominator = means.sum()
        for group, value in zip(group_order, means):
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "group": group,
                    "modality": GROUP_LABELS[group],
                    "mean_abs_shap": value,
                    "row_percent": 100.0 * value / denominator,
                }
            )
    class_modality = pd.DataFrame(class_rows)
    class_modality.to_csv(output_dir / "model_shap_class_modality_importance.csv", index=False, encoding="utf-8-sig")

    _plot_global(feature_table, modality_table, class_modality, output_dir)
    _plot_beeswarm(signed_all, values_all, feature_table, feature_names, groups, output_dir, args.seed)

    metadata = {
        "method": "SHAP GradientExplainer (expected gradients) on the trained hierarchical multimodal FCNN",
        "model_checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "classes": list(CLASS_NAMES),
        "n_test_samples": int(len(sample_frame)),
        "samples_per_class": int(args.samples_per_class),
        "background_size": int(args.background_size),
        "crop_size": int(args.crop_size),
        "nsamples": int(args.nsamples),
        "feature_count": int(len(feature_names)),
        "raster_channel_count": 79,
        "text_latent_dimensions": 16,
        "explained_output": "target-class centre-pixel log probability",
        "spatial_aggregation": "signed SHAP summed over crop pixels; importance is sum of absolute pixel-level SHAP",
        "text_interpretation": "16-D continuous projected OSM text representation; not token-level SHAP",
        "additivity_median_absolute_residual": float(np.median(np.abs(sample_frame["additivity_residual"]))),
        "additivity_mean_absolute_residual": float(np.mean(np.abs(sample_frame["additivity_residual"]))),
        "warning": "Model attribution quantifies reliance under the empirical background distribution; it is not a causal effect.",
    }
    (output_dir / "model_shap_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    qc = {
        "files_exist": {
            name: (output_dir / name).exists()
            for name in (
                "model_shap_global_importance.png", "model_shap_global_importance.pdf",
                "model_shap_global_importance.svg", "model_shap_global_importance.tif",
                "model_shap_beeswarm.png", "model_shap_beeswarm.pdf",
                "model_shap_beeswarm.svg", "model_shap_beeswarm.tif",
                "model_shap_feature_importance.csv", "model_shap_modality_importance.csv",
                "model_shap_class_modality_importance.csv", "model_shap_values.npz",
            )
        },
        "all_values_finite": bool(np.isfinite(signed_all).all() and np.isfinite(absolute_all).all()),
        "modality_percent_sum": float(modality_table["importance_percent"].sum()),
        "class_percent_sums": {
            name: float(group["row_percent"].sum())
            for name, group in class_modality.groupby("class_name", sort=False)
        },
        "figure_contract": {
            "claim": "The trained network's relative reliance on modalities and channels differs across the eight mapped classes.",
            "evidence": ["global modality importance", "top-channel importance", "class-specific modality composition", "sample SHAP distributions"],
            "archetype": "quantitative composite grid plus beeswarm",
        },
    }
    (output_dir / "model_shap_figure_contract_and_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
