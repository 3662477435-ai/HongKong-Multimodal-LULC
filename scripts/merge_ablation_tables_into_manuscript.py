from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
from docx import Document
from docx.document import Document as DocumentType
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_blocks(document: DocumentType):
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def has_drawing(paragraph: Paragraph) -> bool:
    return bool(paragraph._p.xpath(".//w:drawing"))


def find_paragraph(document: DocumentType, prefix: str) -> Paragraph:
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise ValueError(f"Paragraph not found: {prefix}")


def remove_paragraph(paragraph: Paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)


def remove_table(table: Table) -> None:
    element = table._element
    element.getparent().remove(element)


def insert_paragraph_before(anchor: Paragraph, text: str = "") -> Paragraph:
    new_p = OxmlElement("w:p")
    anchor._p.addprevious(new_p)
    paragraph = Paragraph(new_p, anchor._parent)
    if text:
        paragraph.add_run(text)
    return paragraph


def insert_table_before(document: DocumentType, anchor: Paragraph, rows: int, cols: int) -> Table:
    temp = document.add_table(rows=rows, cols=cols)
    anchor._p.addprevious(temp._tbl)
    return temp


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=55, start=55, bottom=55, end=55) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def fmt_metric(value: float, percent: bool = False) -> str:
    return f"{100.0 * value:.2f}" if percent else f"{value:.4f}"


def fmt_delta(value: float) -> str:
    if abs(value) < 0.00005:
        return "—"
    return f"{value:+.4f}"


def fmt_p(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if value == 1.0:
        return "—"
    return "<0.001" if value < 0.001 else f"{value:.4f}"


def build_combined_rows(modality: pd.DataFrame, structural: pd.DataFrame, osm: pd.DataFrame):
    rows: list[dict[str, str]] = []
    for _, item in modality.iterrows():
        code = item["变体代码"]
        config = {
            "full": "S2+S1+高分+夜光+建筑+OSM",
            "s2": "仅S2",
            "no_osm_roads": "全模态−POI/AOI/道路/文本",
            "no_building": "全模态−建筑功能与高度",
            "no_s1": "全模态−S1",
            "no_nighttime": "全模态−夜间灯光",
            "no_highres": "全模态−高分辨率影像",
        }[code]
        rows.append({
            "part": "A 模态组",
            "model": item["模型"],
            "setting": item["变体"],
            "config": config,
            "oa": fmt_metric(float(item["OA"]), percent=True),
            "miou": fmt_metric(float(item["macro_mIoU"])),
            "f1": fmt_metric(float(item["macro_F1"])),
            "delta": fmt_delta(float(item["相对全模态_mIoU变化"])),
            "p": "—",
        })

    for _, item in structural.iterrows():
        setting = item["结构设置"]
        config = {
            "完整模型": "层级分类+语义残差+类别门控",
            "无层级分类": "−层级分类",
            "无语义残差": "−语义残差",
            "共享门控（非类别门控）": "类别门控→共享门控",
        }[setting]
        rows.append({
            "part": "B 结构",
            "model": "本文模型",
            "setting": setting,
            "config": config,
            "oa": fmt_metric(float(item["OA"]), percent=True),
            "miou": fmt_metric(float(item["mIoU"])),
            "f1": fmt_metric(float(item["F1"])),
            "delta": fmt_delta(float(item["相对完整模型_mIoU"])),
            "p": fmt_p(item["配对置换p_mIoU"]),
        })

    for _, item in osm.iterrows():
        setting = item["OSM设置"]
        rows.append({
            "part": "C OSM",
            "model": "本文模型",
            "setting": setting,
            "config": "完整OSM输入" if setting == "全模态" else setting.replace("去除", "−"),
            "oa": fmt_metric(float(item["OA"]), percent=True),
            "miou": fmt_metric(float(item["mIoU"])),
            "f1": fmt_metric(float(item["F1"])),
            "delta": fmt_delta(float(item["相对全模态_mIoU"])),
            "p": fmt_p(item["配对置换p_mIoU"]),
        })
    return rows


def write_combined_csv(rows: list[dict[str, str]], path: Path) -> None:
    rename = {
        "part": "分区",
        "model": "模型",
        "setting": "变体/设置",
        "config": "配置说明",
        "oa": "OA(%)",
        "miou": "mIoU",
        "f1": "macro-F1",
        "delta": "ΔmIoU",
        "p": "mIoU配对p",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).rename(columns=rename).to_csv(path, index=False, encoding="utf-8-sig")


def style_combined_table(table: Table, rows: list[dict[str, str]]) -> None:
    headers = ["分区", "模型", "变体／设置", "配置说明", "OA (%)", "mIoU", "macro-F1", "ΔmIoU", "mIoU配对p"]
    widths = [0.48, 0.77, 1.08, 1.48, 0.58, 0.57, 0.66, 0.64, 0.72]
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"

    for col, (text, width) in enumerate(zip(headers, widths)):
        cell = table.rows[0].cells[col]
        cell.text = text
        cell.width = Inches(width)
        set_cell_shading(cell, "626A78")
        set_cell_margins(cell, 70, 45, 70, 45)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(6.8)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
    set_repeat_table_header(table.rows[0])

    section_fills = {"A 模态组": "E8EEF5", "B 结构": "EAF3EC", "C OSM": "F5EEE6"}
    previous_part = None
    previous_model = None
    for row_index, item in enumerate(rows, start=1):
        row = table.rows[row_index]
        values = [item[k] for k in ("part", "model", "setting", "config", "oa", "miou", "f1", "delta", "p")]
        for col, (value, width) in enumerate(zip(values, widths)):
            cell = row.cells[col]
            cell.text = value
            cell.width = Inches(width)
            set_cell_margins(cell, 42, 42, 42, 42)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col in (1, 2, 3) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 0.9
            for run in p.runs:
                run.font.name = "Arial"
                run.font.size = Pt(6.5)
                if item["setting"] in ("全部模态", "完整模型", "全模态"):
                    run.font.bold = True

        if item["part"] != previous_part:
            for cell in row.cells:
                set_cell_shading(cell, section_fills[item["part"]])
            for run in row.cells[0].paragraphs[0].runs:
                run.font.bold = True
        elif item["model"] != previous_model and item["part"] == "A 模态组":
            for cell in row.cells:
                set_cell_shading(cell, "F4F6F8")

        delta = item["delta"]
        if delta.startswith("-"):
            for run in row.cells[7].paragraphs[0].runs:
                run.font.color.rgb = RGBColor(183, 54, 54)
        elif delta.startswith("+"):
            for run in row.cells[7].paragraphs[0].runs:
                run.font.color.rgb = RGBColor(35, 120, 75)
        set_cant_split(row)
        previous_part = item["part"]
        previous_model = item["model"]

    tbl_pr = table._tbl.tblPr
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")


def replace_text_preserving_style(paragraph: Paragraph, old: str, new: str) -> None:
    if old not in paragraph.text:
        return
    paragraph.text = paragraph.text.replace(old, new)


def merge_into_document(source: Path, output: Path, rows: list[dict[str, str]], table_image: Path) -> None:
    document = Document(source)

    caption3 = find_paragraph(document, "表3")
    section_intro = find_paragraph(document, "本节依次报告")
    heading42_1 = find_paragraph(document, "4.2.1")
    blocks = list(iter_blocks(document))
    start = next(i for i, block in enumerate(blocks) if getattr(block, "_element", None) is section_intro._element)
    end = next(i for i, block in enumerate(blocks) if getattr(block, "_element", None) is heading42_1._element)
    old_table_image = next(
        block for block in blocks[start + 1:end]
        if isinstance(block, Paragraph) and has_drawing(block)
    )
    for child in list(old_table_image._p):
        if child.tag != qn("w:pPr"):
            old_table_image._p.remove(child)
    old_table_image.alignment = WD_ALIGN_PARAGRAPH.CENTER
    old_table_image.add_run().add_picture(str(table_image), width=Inches(5.25))

    caption3.text = (
        "表3  六种模型的模态组消融及本文模型的结构与OSM单项消融汇总。"
        "A，模态组消融；B，结构组成消融；C，OSM信息单项与联合消融。"
        "ΔmIoU为当前设置相对各分区完整基准的变化；A区未实施配对置换检验，B、C区p值来自测试空间组配对置换检验。"
    )

    table4_caption = find_paragraph(document, "表4")
    table5_caption = find_paragraph(document, "表5")
    table4 = next(block for block in iter_blocks(document) if isinstance(block, Table) and block.rows[0].cells[0].text == "结构设置")
    table5 = next(block for block in iter_blocks(document) if isinstance(block, Table) and block.rows[0].cells[0].text == "OSM设置")
    remove_table(table4)
    remove_table(table5)
    remove_paragraph(table4_caption)
    remove_paragraph(table5_caption)

    replace_text_preserving_style(find_paragraph(document, "完整模型测试"), "（表4）", "（表3B）")
    osm_para = find_paragraph(document, "四种OSM信息")
    if "表3C" not in osm_para.text:
        replace_text_preserving_style(osm_para, "。联合删除", "（表3C）。联合删除")

    seed_para = find_paragraph(document, "三个独立种子下")
    replace_text_preserving_style(seed_para, "表6", "表4")
    find_paragraph(document, "表6").text = "表4  六种模型三个独立随机种子的训练稳定性（均值±SD）。"
    spatial_para = find_paragraph(document, "八类标签组成")
    replace_text_preserving_style(spatial_para, "表7", "表5")
    find_paragraph(document, "表7").text = "表5  本文模型在不同空间验证协议下的结果及空间块bootstrap区间。"

    document.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modality-csv", type=Path, required=True)
    parser.add_argument("--structural-csv", type=Path, required=True)
    parser.add_argument("--osm-csv", type=Path, required=True)
    parser.add_argument("--combined-csv", type=Path, required=True)
    parser.add_argument("--table-image", type=Path, required=True)
    args = parser.parse_args()

    modality = pd.read_csv(args.modality_csv)
    structural = pd.read_csv(args.structural_csv)
    osm = pd.read_csv(args.osm_csv)
    rows = build_combined_rows(modality, structural, osm)
    if len(rows) != 52:
        raise RuntimeError(f"Expected 52 combined rows, got {len(rows)}")
    write_combined_csv(rows, args.combined_csv)
    merge_into_document(args.source, args.output, rows, args.table_image)
    print(args.output)
    print(args.combined_csv)


if __name__ == "__main__":
    main()
