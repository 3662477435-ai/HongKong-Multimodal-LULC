from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


MODEL_PATHS = {
    "本文模型": "ours_fcnn_hierarchy_semantic_final_seed42",
    "CNN": "cnn_full_seed42",
    "FCNN": "fcnn_full_seed42",
    "Transformer": "transformer_full_seed42",
    "FCN-ResNet50": "fcn_resnet50_full_seed42",
    "VGG-UNet-Transformer": "vgg_unet_transformer_full_seed42",
}
CLASS_NAMES = ("住宅", "工业仓储", "公共机构与休闲", "交通设施", "其他建设用地", "农业", "林地等", "水体")
METRICS = {"OA": "overall_accuracy", "mIoU": "macro_iou_present", "F1": "macro_f1_present"}


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["test"]["official8"]


def from_matrix(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = support + predicted - tp
    precision = np.divide(tp, predicted, out=np.zeros(8), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(8), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=(precision + recall) > 0)
    iou = np.divide(tp, union, out=np.zeros(8), where=union > 0)
    return {"OA": float(tp.sum() / matrix.sum()), "mIoU": float(iou.mean()), "F1": float(f1.mean())}


def group_ci(metric: dict, draws: int, seed: int) -> dict[str, tuple[float, float]]:
    groups = metric["spatial_group_confusions"]
    matrices = np.asarray([groups[key] for key in sorted(groups)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = {key: np.empty(draws) for key in METRICS}
    for index in range(draws):
        sampled = rng.integers(0, len(matrices), size=len(matrices))
        row = from_matrix(matrices[sampled].sum(axis=0))
        for key in METRICS:
            values[key][index] = row[key]
    return {key: tuple(np.percentile(value, [2.5, 97.5])) for key, value in values.items()}


def paired_exact(a: dict, b: dict, key: str) -> tuple[float, float]:
    ga = a["spatial_group_confusions"]
    gb = b["spatial_group_confusions"]
    groups = sorted(set(ga) & set(gb))
    aa = np.asarray([ga[group] for group in groups], dtype=np.int64)
    bb = np.asarray([gb[group] for group in groups], dtype=np.int64)
    observed = from_matrix(aa.sum(axis=0))[key] - from_matrix(bb.sum(axis=0))[key]
    exceed = 0
    total = 2 ** len(groups)
    for bits in itertools.product((False, True), repeat=len(groups)):
        swap = np.asarray(bits)
        pa = np.where(swap[:, None, None], bb, aa).sum(axis=0)
        pb = np.where(swap[:, None, None], aa, bb).sum(axis=0)
        difference = from_matrix(pa)[key] - from_matrix(pb)[key]
        exceed += abs(difference) >= abs(observed) - 1e-15
    return observed, exceed / total


def values(metric: dict) -> dict[str, float]:
    return {label: float(metric[field]) for label, field in METRICS.items()}


def per_class(metric: dict) -> list[dict]:
    rows = []
    source = list(metric["per_class"].values())
    for name, item in zip(CLASS_NAMES, source):
        rows.append({"类别": name, "Precision": item["precision"], "Recall": item["recall"], "F1": item["f1"], "IoU": item["iou"], "支持像元": item["support"]})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def configure_plot() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    repo = args.repo.resolve()
    old = repo / "outputs/full/eight_class_20260725"
    root = repo / "outputs/full/eight_class_review_revision_20260801"
    result_dir = root / "final_results"
    figure_dir = root / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    full = load(old / MODEL_PATHS["本文模型"] / "metrics.json")
    structural_paths = {
        "完整模型": old / MODEL_PATHS["本文模型"] / "metrics.json",
        "无层级分类": root / "structural_no_hierarchy_seed42/metrics.json",
        "无语义残差": root / "structural_no_residual_seed42/metrics.json",
        "共享门控（非类别门控）": root / "structural_shared_gate_seed42/metrics.json",
    }
    structural_rows, structural_class_rows = [], []
    full_class = per_class(full)
    for index, (label, path) in enumerate(structural_paths.items()):
        metric = load(path)
        row = {"结构设置": label, **values(metric)}
        for key in METRICS:
            difference, p = paired_exact(full, metric, key) if index else (0.0, 1.0)
            row[f"相对完整模型_{key}"] = values(metric)[key] - values(full)[key]
            row[f"完整减当前_{key}"] = difference
            row[f"配对置换p_{key}"] = p
        structural_rows.append(row)
        for current, reference in zip(per_class(metric), full_class):
            structural_class_rows.append({"结构设置": label, **current, "相对完整模型_IoU": current["IoU"] - reference["IoU"]})
    # The three structural hypotheses form one family for each metric.  Report
    # both raw exact-permutation p values and Holm-adjusted values; conclusions
    # in the manuscript are always based on the latter.
    for key in METRICS:
        comparison_rows = structural_rows[1:]
        adjusted_values = holm([row[f"配对置换p_{key}"] for row in comparison_rows])
        for row, adjusted in zip(comparison_rows, adjusted_values):
            row[f"Holm校正p_{key}"] = adjusted
            row[f"Holm_0.05显著_{key}"] = adjusted < 0.05
    write_csv(result_dir / "table_structural_component_ablation.csv", structural_rows)
    write_csv(result_dir / "table_structural_component_per_class.csv", structural_class_rows)

    osm_paths = {
        "全模态": structural_paths["完整模型"],
        "去除POI": root / "ours_no_poi_seed42/metrics.json",
        "去除AOI": root / "ours_no_aoi_seed42/metrics.json",
        "去除道路": root / "ours_no_roads_seed42/metrics.json",
        "去除OSM文本": root / "ours_no_text_seed42/metrics.json",
        "联合去除POI/AOI/道路/文本": old / "ours_hfcnn_no_osm_roads_seed42/metrics.json",
    }
    osm_rows, osm_class_rows = [], []
    for index, (label, path) in enumerate(osm_paths.items()):
        metric = load(path)
        row = {"OSM设置": label, **values(metric)}
        for key in METRICS:
            difference, p = paired_exact(full, metric, key) if index else (0.0, 1.0)
            row[f"相对全模态_{key}"] = values(metric)[key] - values(full)[key]
            row[f"配对置换p_{key}"] = p
        osm_rows.append(row)
        for current, reference in zip(per_class(metric), full_class):
            osm_class_rows.append({"OSM设置": label, **current, "相对全模态_IoU": current["IoU"] - reference["IoU"]})
    # The five OSM removals are a second, separate multiplicity family.
    for key in METRICS:
        comparison_rows = osm_rows[1:]
        adjusted_values = holm([row[f"配对置换p_{key}"] for row in comparison_rows])
        for row, adjusted in zip(comparison_rows, adjusted_values):
            row[f"Holm校正p_{key}"] = adjusted
            row[f"Holm_0.05显著_{key}"] = adjusted < 0.05
    write_csv(result_dir / "table_osm_separate_ablation.csv", osm_rows)
    write_csv(result_dir / "table_osm_separate_ablation_per_class.csv", osm_class_rows)

    seed_rows, seed_summary = [], []
    for model, old_name in MODEL_PATHS.items():
        runs = []
        for seed in (42, 43, 44):
            path = old / old_name / "metrics.json" if seed == 42 else root / (("ours" if model == "本文模型" else old_name.split("_full_seed42")[0]) + f"_full_seed{seed}") / "metrics.json"
            metric = load(path)
            row = {"模型": model, "随机种子": seed, **values(metric)}
            seed_rows.append(row)
            runs.append(row)
        summary = {"模型": model, "独立训练次数": len(runs)}
        for key in METRICS:
            array = np.asarray([row[key] for row in runs])
            mean, sd = float(array.mean()), float(array.std(ddof=1))
            half = 4.302652729 * sd / math.sqrt(len(array))
            summary.update({f"{key}_均值": mean, f"{key}_SD": sd, f"{key}_t95CI下限": mean - half, f"{key}_t95CI上限": mean + half})
        seed_summary.append(summary)
    write_csv(result_dir / "table_multiseed_runs.csv", seed_rows)
    write_csv(result_dir / "table_multiseed_summary.csv", seed_summary)
    seed_frame = pd.DataFrame(seed_rows)
    seed_significance = []
    for key in METRICS:
        ours = seed_frame.loc[seed_frame["模型"] == "本文模型"].sort_values("随机种子")[key].to_numpy()
        temporary, raw_p = [], []
        for model in [name for name in MODEL_PATHS if name != "本文模型"]:
            baseline = seed_frame.loc[seed_frame["模型"] == model].sort_values("随机种子")[key].to_numpy()
            difference = ours - baseline
            test = ttest_rel(ours, baseline)
            mean = float(difference.mean())
            sd = float(difference.std(ddof=1))
            half = 4.302652729 * sd / math.sqrt(len(difference))
            row = {
                "指标": key, "比较": f"本文模型 vs {model}", "配对种子数": 3,
                "本文减基线_均值": mean, "差值_SD": sd,
                "差值_t95CI下限": mean - half, "差值_t95CI上限": mean + half,
                "配对t检验p": float(test.pvalue),
            }
            temporary.append(row)
            raw_p.append(float(test.pvalue))
        for row, adjusted in zip(temporary, holm(raw_p)):
            row["Holm校正p"] = adjusted
            row["Holm_0.05显著"] = adjusted < 0.05
            seed_significance.append(row)
    write_csv(result_dir / "table_multiseed_paired_significance.csv", seed_significance)

    spatial_paths = {
        "5.12 km（原协议）": structural_paths["完整模型"],
        "2.56 km空间块": root / "ours_block256_seed42/metrics.json",
        "10.24 km空间块": root / "ours_block1024_seed42/metrics.json",
        "5.12 km + 1.28 km缓冲": root / "ours_block512_buffer128_seed42/metrics.json",
        "西部留区验证": root / "ours_loro_west_seed42/metrics.json",
    }
    spatial_rows = []
    for index, (label, path) in enumerate(spatial_paths.items()):
        metric = load(path)
        ci = group_ci(metric, args.bootstrap, 20260801 + index)
        row = {"空间验证协议": label, "测试空间组数": len(metric["spatial_group_confusions"]), **values(metric)}
        for key in METRICS:
            row[f"{key}_空间bootstrap95CI下限"], row[f"{key}_空间bootstrap95CI上限"] = ci[key]
        spatial_rows.append(row)
    write_csv(result_dir / "table_spatial_robustness.csv", spatial_rows)

    manifest = pd.read_csv(repo / "artifacts/full/prepared/manifest_hier_v2.csv")
    class_columns = [f"pixels_class_{value}" for value in (0, 2, 3, 4, 5, 6, 7, 9)]
    composition = manifest[class_columns].to_numpy(dtype=float)
    composition /= np.maximum(composition.sum(axis=1, keepdims=True), 1.0)
    centres = np.c_[
        manifest.col_off.to_numpy() + manifest.width.to_numpy() / 2,
        manifest.row_off.to_numpy() + manifest.height.to_numpy() / 2,
    ] * 0.01  # 10-m grid coordinates -> km
    first, second = np.triu_indices(len(manifest), 1)
    distances = np.sqrt(((centres[first] - centres[second]) ** 2).sum(axis=1))
    standard = (composition - composition.mean(axis=0)) / np.where(
        composition.std(axis=0) > 0, composition.std(axis=0), 1.0
    )
    products = standard[first] * standard[second]
    correlogram_rows = []
    for lower in np.arange(0.0, 25.6, 1.28):
        selected = (distances >= lower) & (distances < lower + 1.28)
        if selected.any():
            correlogram_rows.append({
                "距离下限_km": lower,
                "距离上限_km": lower + 1.28,
                "距离中点_km": lower + 0.64,
                "patch对数": int(selected.sum()),
                "八类组成平均标准化相关": float(np.nanmean(products[selected])),
            })
    write_csv(result_dir / "table_spatial_label_composition_correlogram.csv", correlogram_rows)

    configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    struct_plot = structural_rows[1:]
    axes[0].barh([row["结构设置"] for row in struct_plot], [-100 * row["相对完整模型_mIoU"] for row in struct_plot], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_xlabel("移除/替换后 mIoU 下降（百分点）")
    axes[0].set_title("a  结构组成贡献", loc="left", fontweight="bold")
    osm_plot = osm_rows[1:]
    axes[1].barh([row["OSM设置"] for row in osm_plot], [-100 * row["相对全模态_mIoU"] for row in osm_plot], color="#4C78A8")
    axes[1].set_xlabel("mIoU 下降（百分点）")
    axes[1].set_title("b  OSM信息单项与联合消融", loc="left", fontweight="bold")
    fig.savefig(figure_dir / "figure_structural_osm_ablation.png", dpi=400, bbox_inches="tight")
    fig.savefig(figure_dir / "figure_structural_osm_ablation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8), constrained_layout=True)
    x = np.arange(len(seed_summary))
    for key, color, marker in (("OA", "#4C78A8", "o"), ("mIoU", "#F58518", "s"), ("F1", "#54A24B", "^")):
        mean = np.asarray([row[f"{key}_均值"] for row in seed_summary])
        sd = np.asarray([row[f"{key}_SD"] for row in seed_summary])
        axes[0].errorbar(x, mean, yerr=sd, fmt=marker, color=color, capsize=3, label=key)
    axes[0].set_xticks(x, [row["模型"] for row in seed_summary], rotation=25, ha="right")
    axes[0].set_ylabel("三次独立训练：均值 ± SD")
    axes[0].set_title("a  训练随机性稳定性", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    sp = pd.DataFrame(spatial_rows)
    width = 0.24
    for offset, key, color in ((-width, "OA", "#4C78A8"), (0, "mIoU", "#F58518"), (width, "F1", "#54A24B")):
        axes[1].bar(np.arange(len(sp)) + offset, sp[key], width, label=key, color=color)
    axes[1].set_xticks(np.arange(len(sp)), sp["空间验证协议"], rotation=25, ha="right")
    axes[1].set_ylabel("精度")
    axes[1].set_title("b  空间协议敏感性", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    corr = pd.DataFrame(correlogram_rows)
    axes[2].plot(corr["距离中点_km"], corr["八类组成平均标准化相关"], marker="o", ms=3, color="#4C78A8")
    axes[2].axhline(0.1, color="#E45756", ls="--", lw=1, label="经验阈值 0.1")
    axes[2].axvline(5.12, color="#777777", ls=":", lw=1, label="原空间块 5.12 km")
    axes[2].set_xlabel("patch中心距离（km）")
    axes[2].set_ylabel("八类组成平均标准化相关")
    axes[2].set_title("c  标签组成空间相关尺度", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=7)
    fig.savefig(figure_dir / "figure_seed_spatial_robustness.png", dpi=400, bbox_inches="tight")
    fig.savefig(figure_dir / "figure_seed_spatial_robustness.pdf", bbox_inches="tight")
    plt.close(fig)

    audit = {
        "complete": True,
        "bootstrap_repetitions": args.bootstrap,
        "structural_rows": len(structural_rows),
        "osm_rows": len(osm_rows),
        "multiseed_runs": len(seed_rows),
        "spatial_protocols": len(spatial_rows),
        "tables": sorted(str(path) for path in result_dir.glob("*.csv")),
        "figures": sorted(str(path) for path in figure_dir.glob("figure_*")),
    }
    (root / "FINAL_REVIEWER_REVISION_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
