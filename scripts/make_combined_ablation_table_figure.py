from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


HEADER = "#626A78"
GRID = "#B8C1CC"
FULL = "#DFEAF4"
ALT = "#F4F7FA"
STRUCT = "#E7F1E9"
OSM = "#F5EEE4"
NEG = "#C83D3D"
POS = "#248653"
TEXT = "#1E252B"


def draw_table(ax, columns, widths, rows, *, font_size=5.1, header_size=5.2, row_colors=None):
    ax.set_axis_off()
    x = [0.0]
    for width in widths:
        x.append(x[-1] + width)
    nrows = len(rows) + 1
    height = 1.0 / nrows

    for col, label in enumerate(columns):
        ax.add_patch(Rectangle((x[col], 1 - height), widths[col], height,
                               facecolor=HEADER, edgecolor=GRID, linewidth=0.55))
        ax.text(x[col] + widths[col] / 2, 1 - height / 2, label,
                ha="center", va="center", fontsize=header_size,
                fontweight="bold", color="white")

    for row_idx, row in enumerate(rows, start=1):
        y = 1 - (row_idx + 1) * height
        fill = row_colors[row_idx - 1] if row_colors else "white"
        for col, value in enumerate(row):
            ax.add_patch(Rectangle((x[col], y), widths[col], height,
                                   facecolor=fill, edgecolor=GRID, linewidth=0.38))
            color = TEXT
            if columns[col] == "ΔmIoU":
                text = str(value)
                color = NEG if text.startswith("-") else POS if text.startswith("+") else TEXT
            weight = "bold" if row_idx == 1 or (len(row) > 2 and str(row[2]) in {"全部模态", "完整模型", "全模态"}) else "normal"
            ax.text(x[col] + widths[col] / 2, y + height / 2, str(value),
                    ha="center", va="center", fontsize=font_size,
                    fontweight=weight, color=color, clip_on=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def fmt_delta(value: float) -> str:
    return "—" if abs(value) < 0.00005 else f"{value:+.4f}"


def fmt_p(value: float) -> str:
    if value == 1.0:
        return "—"
    return "<0.001" if value < 0.001 else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality-csv", type=Path, required=True)
    parser.add_argument("--structural-csv", type=Path, required=True)
    parser.add_argument("--osm-csv", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    modality = pd.read_csv(args.modality_csv)
    structural = pd.read_csv(args.structural_csv)
    osm = pd.read_csv(args.osm_csv)

    checks = {
        "full": ["●"] * 6,
        "s2": ["●", "", "", "", "", ""],
        "no_osm_roads": ["●", "●", "●", "●", "●", ""],
        "no_building": ["●", "●", "●", "●", "", "●"],
        "no_s1": ["●", "", "●", "●", "●", "●"],
        "no_nighttime": ["●", "●", "●", "", "●", "●"],
        "no_highres": ["●", "●", "", "●", "●", "●"],
    }
    a_rows = []
    a_colors = []
    for model_index, (model, group) in enumerate(modality.groupby("模型", sort=False)):
        for combination, (_, item) in enumerate(group.iterrows(), start=1):
            a_rows.append([
                model,
                combination,
                *checks[item["变体代码"]],
                f"{100 * float(item['OA']):.2f}",
                f"{float(item['macro_mIoU']):.4f}",
                fmt_delta(float(item["相对全模态_mIoU变化"])),
            ])
            if item["变体代码"] == "full":
                a_colors.append(FULL)
            elif model_index % 2:
                a_colors.append(ALT)
            else:
                a_colors.append("white")

    b_rows = []
    for _, item in structural.iterrows():
        setting = item["结构设置"]
        config = {
            "完整模型": "层级+语义残差+类别门控",
            "无层级分类": "−层级分类",
            "无语义残差": "−语义残差",
            "共享门控（非类别门控）": "类别门控→共享门控",
        }[setting]
        b_rows.append([
            setting, config, f"{100 * float(item['OA']):.2f}", f"{float(item['mIoU']):.4f}",
            f"{float(item['F1']):.4f}", fmt_delta(float(item["相对完整模型_mIoU"])),
            fmt_p(float(item["配对置换p_mIoU"])),
        ])

    c_rows = []
    for _, item in osm.iterrows():
        setting = item["OSM设置"]
        config = "完整OSM输入" if setting == "全模态" else setting.replace("去除", "−")
        c_rows.append([
            setting, config, f"{100 * float(item['OA']):.2f}", f"{float(item['mIoU']):.4f}",
            f"{float(item['F1']):.4f}", fmt_delta(float(item["相对全模态_mIoU"])),
            fmt_p(float(item["配对置换p_mIoU"])),
        ])

    fig = plt.figure(figsize=(7.2, 10.6), facecolor="white")
    fig.text(0.04, 0.985, "A  六种模型的模态组消融", fontsize=7.2, fontweight="bold", va="top", color=TEXT)
    ax_a = fig.add_axes([0.04, 0.285, 0.92, 0.675])
    draw_table(
        ax_a,
        ["模型", "组合", "S2", "S1", "高分影像", "夜间灯光", "建筑特征", "POI/AOI/道路/文本", "OA (%)", "mIoU", "ΔmIoU"],
        [0.160, 0.050, 0.045, 0.045, 0.073, 0.073, 0.073, 0.119, 0.090, 0.090, 0.082],
        a_rows,
        font_size=4.75,
        header_size=4.75,
        row_colors=a_colors,
    )

    fig.text(0.04, 0.265, "B  本文模型的结构组成消融", fontsize=7.0, fontweight="bold", va="top", color=TEXT)
    ax_b = fig.add_axes([0.04, 0.175, 0.92, 0.078])
    draw_table(
        ax_b,
        ["结构设置", "配置说明", "OA (%)", "mIoU", "macro-F1", "ΔmIoU", "mIoU配对p"],
        [0.220, 0.270, 0.100, 0.100, 0.110, 0.095, 0.105],
        b_rows,
        font_size=5.0,
        header_size=5.0,
        row_colors=[STRUCT] + ["white"] * (len(b_rows) - 1),
    )

    fig.text(0.04, 0.155, "C  本文模型的OSM单项与联合消融", fontsize=7.0, fontweight="bold", va="top", color=TEXT)
    ax_c = fig.add_axes([0.04, 0.030, 0.92, 0.113])
    draw_table(
        ax_c,
        ["OSM设置", "配置说明", "OA (%)", "mIoU", "macro-F1", "ΔmIoU", "mIoU配对p"],
        [0.260, 0.230, 0.100, 0.100, 0.110, 0.095, 0.105],
        c_rows,
        font_size=5.0,
        header_size=5.0,
        row_colors=[OSM] + ["white"] * (len(c_rows) - 1),
    )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {ext: Path(str(args.output_stem) + ext) for ext in (".png", ".tif", ".pdf", ".svg")}
    fig.savefig(paths[".png"], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[".tif"], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[".pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[".svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(paths[".png"])


if __name__ == "__main__":
    main()
