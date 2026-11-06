from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


MODEL_ORDER = (
    "本文分层多模态模型",
    "CNN",
    "FCNN",
    "Transformer",
    "FCN-ResNet50",
    "VGG-UNet-Transformer",
)
VARIANT_ORDER = (
    "full",
    "s2",
    "no_osm_roads",
    "no_building",
    "no_s1",
    "no_nighttime",
    "no_highres",
)
FEATURES = ("S2", "S1", "高分影像", "夜间灯光", "建筑特征", "POI/AOI/道路")
FEATURE_STATE = {
    "full": (1, 1, 1, 1, 1, 1),
    "s2": (1, 0, 0, 0, 0, 0),
    "no_osm_roads": (1, 1, 1, 1, 1, 0),
    "no_building": (1, 1, 1, 1, 0, 1),
    "no_s1": (1, 0, 1, 1, 1, 1),
    "no_nighttime": (1, 1, 1, 0, 1, 1),
    "no_highres": (1, 1, 0, 1, 1, 1),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["模型"], row["变体代码"]): row for row in rows}
    missing = [
        (model, variant)
        for model in MODEL_ORDER
        for variant in VARIANT_ORDER
        if (model, variant) not in indexed
    ]
    if missing:
        raise ValueError(f"CSV缺少实验行：{missing}")
    return indexed


def cell(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str = "",
    *,
    facecolor: str = "white",
    edgecolor: str = "#7D838E",
    linewidth: float = 0.55,
    color: str = "#1E2530",
    fontsize: float = 7.2,
    weight: str = "normal",
    ha: str = "center",
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y), width, height,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        )
    )
    if text:
        padding = 0.012 * width if ha == "left" else 0.0
        ax.text(
            x + (padding if ha == "left" else width / 2),
            y + height / 2,
            text,
            ha=ha,
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            linespacing=1.15,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="将扩展模态消融CSV绘制成论文表格。")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    indexed = read_rows(args.csv)

    column_labels = (
        "模型", "组合", *FEATURES, "OA (%)", "mIoU", "mIoU下降",
    )
    widths = (1.90, 0.70, 0.52, 0.52, 0.72, 0.72, 0.80, 1.20, 0.85, 0.78, 0.92)
    total_width = sum(widths)
    x_edges = [0.0]
    for width in widths:
        x_edges.append(x_edges[-1] + width)

    header_height = 0.62
    row_height = 0.40
    footnote_height = 1.05
    row_count = len(MODEL_ORDER) * len(VARIANT_ORDER)
    table_height = header_height * 2 + row_height * row_count
    total_height = table_height + footnote_height
    fig, ax = plt.subplots(figsize=(13.2, 17.2), constrained_layout=False)
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    dark = "#656B78"
    header_edge = "#515865"
    top_y = total_height - header_height
    # Two-row header: model/combination and metrics span vertically; feature spans horizontally.
    for index in (0, 1, 8, 9, 10):
        label = column_labels[index]
        cell(
            ax, x_edges[index], top_y - header_height, widths[index], header_height * 2,
            label, facecolor=dark, edgecolor=header_edge, linewidth=0.7,
            color="white", fontsize=8.3, weight="semibold",
        )
    feature_x = x_edges[2]
    feature_width = sum(widths[2:8])
    cell(
        ax, feature_x, top_y, feature_width, header_height,
        "输入特征", facecolor=dark, edgecolor=header_edge, linewidth=0.7,
        color="white", fontsize=8.5, weight="semibold",
    )
    for index in range(2, 8):
        cell(
            ax, x_edges[index], top_y - header_height, widths[index], header_height,
            column_labels[index], facecolor=dark, edgecolor=header_edge, linewidth=0.7,
            color="white", fontsize=7.4, weight="semibold",
        )

    current_y = top_y - header_height - row_height
    model_summaries = []
    for model_index, model in enumerate(MODEL_ORDER):
        group_top = current_y + row_height
        group_height = row_height * len(VARIANT_ORDER)
        group_bottom = group_top - group_height
        model_face = "#EEF2F7" if model_index % 2 == 0 else "#F6F7F9"
        model_label = model.replace("VGG-UNet-Transformer", "VGG-UNet-\nTransformer")
        cell(
            ax, x_edges[0], group_bottom, widths[0], group_height,
            model_label, facecolor=model_face, edgecolor="#69717E", linewidth=0.8,
            fontsize=7.7, weight="semibold",
        )
        model_rows = [indexed[(model, variant)] for variant in VARIANT_ORDER]
        best_oa = max(float(row["测试OA"]) for row in model_rows)
        best_miou = max(float(row["测试mIoU"]) for row in model_rows)
        for combination, variant in enumerate(VARIANT_ORDER, start=1):
            row = indexed[(model, variant)]
            row_face = "#E7F0F8" if variant == "full" else (
                "#FAFBFC" if combination % 2 == 0 else "white"
            )
            cell(ax, x_edges[1], current_y, widths[1], row_height, str(combination), facecolor=row_face)
            for feature_offset, present in enumerate(FEATURE_STATE[variant], start=2):
                cell(
                    ax, x_edges[feature_offset], current_y, widths[feature_offset], row_height,
                    "√" if present else "", facecolor=row_face,
                    fontsize=10.3, color="#245E8C", weight="semibold",
                )
            oa = float(row["测试OA"])
            miou = float(row["测试mIoU"])
            drop = float(row["相对全模态mIoU下降"])
            cell(
                ax, x_edges[8], current_y, widths[8], row_height, f"{oa * 100:.2f}",
                facecolor=row_face, weight="bold" if abs(oa - best_oa) < 1e-12 else "normal",
            )
            cell(
                ax, x_edges[9], current_y, widths[9], row_height, f"{miou:.4f}",
                facecolor=row_face, weight="bold" if abs(miou - best_miou) < 1e-12 else "normal",
            )
            drop_color = "#B64040" if drop > 0.005 else ("#2B6F55" if drop < -0.005 else "#56606D")
            cell(
                ax, x_edges[10], current_y, widths[10], row_height,
                "—" if variant == "full" else f"{drop:+.4f}",
                facecolor=row_face, color=drop_color,
            )
            current_y -= row_height
        ax.plot(
            [0, total_width], [group_bottom, group_bottom],
            color="#515865", linewidth=1.0, solid_capstyle="butt",
        )
        model_summaries.append({"model": model, "best_OA": best_oa, "best_mIoU": best_miou})

    footnote_y = 0.66
    ax.text(
        0.02,
        footnote_y,
        "组合：1 全部模态；2 仅S2；3 去POI/AOI/道路及OSM文本；4 去建筑物特征；"
        "5 去S1；6 去夜间灯光；7 去高分辨率影像。",
        ha="left", va="center", fontsize=7.0, color="#303844",
    )
    ax.text(
        0.02,
        0.28,
        "注：mIoU下降 = 全模态mIoU − 当前组合mIoU；负值表示该消融在单次训练中略高于全模态。"
        "加粗表示同一模型七种组合中的最高值。",
        ha="left", va="center", fontsize=6.8, color="#56606D",
    )

    stem = args.output_dir / "table_six_models_extended_modality_ablation"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        stem.with_suffix(".tif"), dpi=300, bbox_inches="tight", pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    with Image.open(stem.with_suffix(".png")) as preview:
        preview_qc = {
            "size_pixels": list(preview.size),
            "dpi": list(preview.info.get("dpi", ())),
            "mode": preview.mode,
        }
    report = {
        "status": "passed",
        "source_csv": str(args.csv),
        "rows": row_count,
        "models": len(MODEL_ORDER),
        "combinations_per_model": len(VARIANT_ORDER),
        "preview": preview_qc,
        "model_best_values": model_summaries,
        "outputs": [str(stem.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg", ".tif")],
    }
    (args.output_dir / "table_six_models_extended_modality_ablation_qc.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
