from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("NA")
            elif column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overall = pd.read_csv(
        args.results_dir / "table_overall_metrics_with_ci.csv", encoding="utf-8-sig"
    )
    per_class = pd.read_csv(
        args.results_dir / "table_per_class_metrics_with_ci.csv", encoding="utf-8-sig"
    )
    significance = pd.read_csv(
        args.results_dir / "table_paired_significance.csv", encoding="utf-8-sig"
    )
    seed_summary = pd.read_csv(
        args.results_dir / "table_training_seed_summary.csv", encoding="utf-8-sig"
    )
    ablation = pd.read_csv(
        args.results_dir / "table_ablation_all_models.csv", encoding="utf-8-sig"
    )
    support = pd.read_csv(
        args.results_dir / "table_dataset_support_8class.csv", encoding="utf-8-sig"
    )
    best = overall.loc[overall["macro_mIoU"].idxmax()]
    ours_classes = per_class.loc[per_class["模型"] == "本文模型"]
    strongest = ours_classes.loc[ours_classes["IoU"].idxmax()]
    weakest = ours_classes.loc[ours_classes["IoU"].idxmin()]

    overall_display = overall[[
        "模型", "OA", "OA_95CI下限", "OA_95CI上限",
        "macro_mIoU", "mIoU_95CI下限", "mIoU_95CI上限",
        "macro_F1", "F1_95CI下限", "F1_95CI上限",
    ]].copy()
    for metric, low, high, output in (
        ("OA", "OA_95CI下限", "OA_95CI上限", "OA（95% CI）"),
        ("macro_mIoU", "mIoU_95CI下限", "mIoU_95CI上限", "macro mIoU（95% CI）"),
        ("macro_F1", "F1_95CI下限", "F1_95CI上限", "macro-F1（95% CI）"),
    ):
        overall_display[output] = overall_display.apply(
            lambda row: f"{row[metric]:.4f} [{row[low]:.4f}, {row[high]:.4f}]",
            axis=1,
        )
    overall_display = overall_display[[
        "模型", "OA（95% CI）", "macro mIoU（95% CI）", "macro-F1（95% CI）"
    ]]

    per_class_display = per_class[[
        "模型", "类别", "测试像元数", "Precision", "Recall", "F1", "IoU",
        "IoU_95CI下限", "IoU_95CI上限",
    ]].copy()
    per_class_display["IoU（95% CI）"] = per_class_display.apply(
        lambda row: (
            f"{row['IoU']:.4f} "
            f"[{row['IoU_95CI下限']:.4f}, {row['IoU_95CI上限']:.4f}]"
        ),
        axis=1,
    )
    per_class_display = per_class_display[[
        "模型", "类别", "测试像元数", "Precision", "Recall", "F1", "IoU（95% CI）"
    ]]

    lines = [
        "# 八类重新训练实验完整结果",
        "",
        "## 实验定义",
        "",
        "商业办公和荒地在训练前即从标签、采样、损失、验证、测试和制图掩膜中排除；其余类别重新编号为连续8类。所有模型使用相同的空间训练/验证/测试划分和相同全模态输入。主结果采用预先固定的seed 42，检查点仅依据验证集macro mIoU选择。",
        "",
        "测试不确定性由14个独立空间组的5,000次配对块bootstrap估计。模型差异使用全部2^14种空间组标签交换的精确配对置换检验，并在每个指标内对五个基线进行Holm校正。已有的独立训练随机种子另作描述性稳定性补充；主置信区间不把空间相关像元当作独立样本。",
        "",
        "## 主要结论",
        "",
        (
            f"当前8类实验中，macro mIoU最高的模型为{best['模型']} "
            f"（OA={best['OA']:.4f}，macro mIoU={best['macro_mIoU']:.4f}，"
            f"macro-F1={best['macro_F1']:.4f}）。"
        ),
        (
            f"本文模型逐类IoU最高的是{strongest['类别']}（{strongest['IoU']:.4f}），"
            f"最低的是{weakest['类别']}（{weakest['IoU']:.4f}）。"
        ),
        "",
        "## 表1  六种全模态模型总体结果",
        "",
        markdown_table(overall_display),
        "",
        "## 表2  逐模型逐类别指标",
        "",
        markdown_table(
            per_class_display,
            {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"},
        ),
        "",
        "## 表3  配对显著性检验",
        "",
        markdown_table(
            significance[[
                "指标", "比较", "差值_本文减基线", "差值95CI下限",
                "差值95CI上限", "精确配对置换p值", "Holm校正p值",
                "Holm_0.05显著",
            ]],
            {
                "差值_本文减基线": "{:.5f}",
                "差值95CI下限": "{:.5f}",
                "差值95CI上限": "{:.5f}",
                "精确配对置换p值": "{:.6f}",
                "Holm校正p值": "{:.6f}",
            },
        ),
        "",
        "## 表4  可用独立训练随机种子的稳定性补充",
        "",
        markdown_table(seed_summary, {
            column: "{:.5f}"
            for column in seed_summary.columns
            if column not in ("模型", "训练随机种子数")
        }),
        "",
        "## 表5  八类样本支持",
        "",
        markdown_table(support),
        "",
        "## 表6  全模型消融结果",
        "",
        markdown_table(
            ablation,
            {
                "OA": "{:.4f}", "macro_mIoU": "{:.4f}", "macro_F1": "{:.4f}",
                "相对全模态_OA变化": "{:+.4f}",
                "相对全模态_mIoU变化": "{:+.4f}",
                "相对全模态_F1变化": "{:+.4f}",
            },
        ),
        "",
        "## 解释边界",
        "",
        "- 8类结果与原10类结果属于不同任务，不能仅因OA或mIoU提高就断言模型本身获得同等幅度的结构性改进。",
        "- 空间bootstrap区间反映当前香港测试空间组的采样不确定性；三随机种子表才反映部分初始化和优化随机性。",
        "- 显著性结论以Holm校正后的p值为准，而不是只看点估计大小或未校正p值。",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
