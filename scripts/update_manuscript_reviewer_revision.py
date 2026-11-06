from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


def find_paragraph(document: Document, prefix: str):
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise KeyError(prefix)


def replace(document: Document, prefix: str, text: str) -> None:
    paragraph = find_paragraph(document, prefix)
    paragraph.text = text
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.line_spacing = 1.5


def replace_suffix(document: Document, marker: str, text: str) -> None:
    """Replace a sentence suffix beginning at marker while preserving earlier text."""
    for paragraph in document.paragraphs:
        index = paragraph.text.find(marker)
        if index >= 0:
            paragraph.text = paragraph.text[:index] + text
            paragraph.paragraph_format.first_line_indent = Cm(0.74)
            paragraph.paragraph_format.line_spacing = 1.5
            return
    raise KeyError(marker)


def move_before(anchor, element) -> None:
    anchor._p.addprevious(element)


def add_paragraph_before(document: Document, anchor, text: str, *, heading: bool = False, caption: bool = False):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    if heading:
        run.bold = True
        run.font.size = Pt(12)
        paragraph.paragraph_format.space_before = Pt(8)
        paragraph.paragraph_format.space_after = Pt(4)
    elif caption:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(4)
        paragraph.paragraph_format.space_after = Pt(3)
    else:
        paragraph.paragraph_format.first_line_indent = Cm(0.74)
        paragraph.paragraph_format.line_spacing = 1.5
    move_before(anchor, paragraph._p)
    return paragraph


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    properties.append(element)


def add_table_before(document: Document, anchor, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for index, value in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = value
        shade(cell, "D9E2F3")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(8)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = str(value)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cells[index].paragraphs[0].runs:
                run.font.size = Pt(8)
    move_before(anchor, table._tbl)


def add_figure_before(document: Document, anchor, image: Path, caption: str, width_cm: float = 16.2) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image), width=Cm(width_cm))
    move_before(anchor, paragraph._p)
    add_paragraph_before(document, anchor, caption, caption=True)


def f4(value: float) -> str:
    return f"{value:.4f}"


def pvalue(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"={value:.4f}"


def mean_sd(row: pd.Series, key: str) -> str:
    return f"{row[f'{key}_均值']:.4f} ± {row[f'{key}_SD']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    results = args.experiment_root / "final_results"
    figures = args.experiment_root / "figures"
    required = [
        results / "table_structural_component_ablation.csv",
        results / "table_osm_separate_ablation.csv",
        results / "table_multiseed_summary.csv",
        results / "table_multiseed_paired_significance.csv",
        results / "table_spatial_robustness.csv",
        results / "table_spatial_label_composition_correlogram.csv",
        figures / "figure_structural_osm_ablation.png",
        figures / "figure_seed_spatial_robustness.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Reviewer-revision results are incomplete:\n" + "\n".join(missing))

    structure = pd.read_csv(required[0])
    osm = pd.read_csv(required[1])
    seeds = pd.read_csv(required[2])
    seed_tests = pd.read_csv(required[3])
    spatial = pd.read_csv(required[4])
    correlogram = pd.read_csv(required[5])
    full = structure.iloc[0]
    no_hierarchy = structure.loc[structure["结构设置"] == "无层级分类"].iloc[0]
    no_residual = structure.loc[structure["结构设置"] == "无语义残差"].iloc[0]
    shared_gate = structure.loc[structure["结构设置"].str.startswith("共享门控")].iloc[0]
    ours_seed = seeds.loc[seeds["模型"] == "本文模型"].iloc[0]
    loro = spatial.loc[spatial["空间验证协议"] == "西部留区验证"].iloc[0]
    buffered = spatial.loc[spatial["空间验证协议"].str.contains("缓冲")].iloc[0]
    first_low = correlogram.loc[correlogram["八类组成平均标准化相关"] < 0.1].iloc[0]
    individual = osm.iloc[1:5].copy()
    individual["mIoU下降"] = -individual["相对全模态_mIoU"]
    largest = individual.sort_values("mIoU下降", ascending=False).iloc[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.output)
    document = Document(args.output)

    abstract = (
        "摘要：高密度城市土地利用/土地覆盖制图需要同时识别自然地表的物理属性与建成区的使用功能，"
        "但单一遥感影像难以区分光谱相似的城市功能类别，开放地理数据的空间异质性和类别不平衡也会降低模型泛化能力。"
        "本文面向香港全域10 m输出网格，构建融合Sentinel-1/2、高分辨率影像、VIIRS夜间灯光、建筑功能与高度、"
        "OSM道路、POI、AOI及文本语义的层级多模态FCNN。模型设置直接分类与条件层级分类两条路径，通过语义残差强化城市功能信息，"
        "并利用类别级软路由在概率域融合两类预测。固定空间测试集上，种子42模型取得"
        f"{full['OA']:.4f}的总体精度、{full['mIoU']:.4f}的macro mIoU和{full['F1']:.4f}的macro-F1。"
        f"三个独立随机种子下，本文模型的macro mIoU为{ours_seed['mIoU_均值']:.4f}±{ours_seed['mIoU_SD']:.4f}。"
        f"去除层级分类、语义残差以及将类别门控替换为共享门控后，macro mIoU分别变化至{no_hierarchy['mIoU']:.4f}、"
        f"{no_residual['mIoU']:.4f}和{shared_gate['mIoU']:.4f}，从而区分了三个结构组成的条件边际贡献。"
        f"OSM单项消融显示，{largest['OSM设置']}造成最大的单项mIoU下降（{largest['mIoU下降']:.4f}），"
        "但其幅度小于联合删除全部OSM信息，说明联合消融不能归因于某一单独来源。"
        f"在西部leave-region-out验证中，模型取得OA={loro['OA']:.4f}、mIoU={loro['mIoU']:.4f}，"
        "表明跨区域泛化弱于固定随机空间组测试。GradientSHAP结果进一步显示光学观测仍是主要归因来源，而道路和AOI对特定城市功能类别提供补充。"
        "这些结果同时支持层级结构和多源语义的增益，并揭示其训练随机性、空间尺度和跨区域泛化边界。"
    )
    replace(document, "摘要：", abstract)

    replace(document, "模型采用“全模态FCNN预训练", (
        "模型采用“全模态FCNN预训练—层级分支训练—语义残差与门控训练—联合微调”的分阶段策略。"
        "这样可以首先获得稳定的全模态像元分类器，再逐步引入层级结构和语义修正，避免所有随机初始化模块同时训练导致优化不稳定。"
        "审稿补充实验中的所有结构和输入变体均从各自同模态FCNN检查点初始化，并沿用相同的阶段顺序、验证集选择准则和测试掩膜。"
    ))
    replace(document, "BLU标签采用最近邻方法", (
        "BLU标签采用最近邻方法重投影至10 m输出参考网格。BLU值为0的像元均转换为ignore index 255，不参与交叉熵、Dice损失、"
        "验证指标和测试指标计算，并在最终分类结果中保持为0。研究区域按照128×128像元裁剪为patch，每个patch约为1.28 km×1.28 km。"
        "主实验按照512×512像元（5.12 km）空间块对patch分组，再以空间组为单位划分训练集、验证集和测试集。为检验这一尺度选择，"
        "本文补充计算八类标签组成的距离分箱标准化相关，并进一步比较2.56 km、5.12 km和10.24 km空间块、1.28 km排除缓冲区及"
        "西部leave-region-out协议。每个新协议均仅用其训练子集重新估计通道归一化参数和OSM文本词表，避免验证区或测试区信息进入预处理。"
    ))
    replace(document, "语义残差模块采用3×10-4", (
        "语义残差模块采用3×10-4的学习率，类别门控采用2×10-2的学习率，该阶段最多训练8个epoch。"
        "较大的门控学习率使八个类别能够较快调整直接路径和层级路径的相对权重，而零初始化的语义残差保证优化从FCNN基线附近开始。"
    ))
    replace(document, "联合训练时，新加入的层级模块", (
        "联合训练时，新加入的层级模块、语义残差和门控模块使用10-4的学习率；已预训练的全模态FCNN、文本Transformer和输入融合层使用10-5学习率。"
        "所有参数使用AdamW优化，权重衰减为10-4，梯度范数裁剪阈值为5.0，并采用bfloat16自动混合精度。联合阶段最多训练12个epoch，"
        "以验证集macro mIoU作为唯一检查点选择指标并采用早停。种子42完整模型的最佳检查点出现在联合训练第3个epoch。"
        "测试集只在每次独立训练的模型与检查点确定后评价，不参与超参数选择。"
    ))

    anchor_methods = find_paragraph(document, "3.4.")
    add_paragraph_before(document, anchor_methods, "3.3.6 审稿补充实验设计", heading=True)
    add_paragraph_before(document, anchor_methods, (
        "为区分层级分类、多模态语义残差和类别级门控的贡献，本文采用一次只改变一个结构组成的重训练消融。"
        "“无层级分类”令最终输出等于直接分类概率；“无语义残差”令直接logit等于全模态FCNN的原始logit；"
        "“共享门控”用一个标量替代八个类别独立门控。三个变体均保持相同输入、FCNN初始化、损失、训练轮数与空间划分。"
        "完整模型与相应变体的差异因此表示该组成在其余模块存在条件下的边际效应，但不被解释为彼此独立可加的因果贡献。"
    ))
    add_paragraph_before(document, anchor_methods, (
        "训练稳定性实验对本文模型及五个全模态基线分别使用随机种子42、43和44独立训练三次，报告均值、样本标准差和基于t分布（df=2）的95%区间。"
        "模型间训练稳定性差异采用相同种子配对t检验，并在每个指标内对五个基线实施Holm校正。由于仅有三次独立训练，相关p值作为探索性证据，"
        "不能替代更多重复。OSM实验分别删除POI、AOI、道路和文本并重新训练FCNN与层级扩展，同时保留原联合删除设置，以避免把四种信息的联合效应误归于单一来源。"
    ))
    add_paragraph_before(document, anchor_methods, (
        "空间稳健性实验首先以1.28 km为距离步长计算所有patch八类标签比例的标准化空间相关函数；随后在2.56 km和10.24 km块尺度下重新划分并训练，"
        "并与原5.12 km协议比较。缓冲实验删除距验证或测试patch边缘不足1.28 km的训练patch。leave-region-out实验将香港最西侧18% patch作为完整测试区域，"
        "以相反方向地理尾部作为验证区、其余区域作为训练区；该方向是在模型训练前按八类测试支持完整且类别组成偏差较小的预设准则选定。"
    ))

    replace(document, "为考虑相邻像元之间的空间自相关", (
        "主结果不将单个像元作为独立统计单元，而以512×512像元空间组作为重抽样单元。针对主测试集14个空间组实施5,000次有放回bootstrap，"
        "计算OA、macro mIoU、macro-F1和逐类指标的百分位95%置信区间。该区间描述现有测试区域之间的空间变异。训练随机性则由三个独立种子的"
        "均值、样本标准差及t分布95%区间单独表征，二者不相互替代。对不同空间协议的结果也分别按其测试空间组实施5,000次块bootstrap。"
    ))
    replace(document, "在比较本文模型与五种基线时", (
        "在固定主测试集上比较本文模型与五种基线时，采用测试空间组为配对单元的双侧精确标签交换检验，并在每个指标内用Holm方法校正五次比较。"
        "结构和OSM消融同样在共享测试空间组上实施配对检验。多随机种子比较使用种子配对t检验并作Holm校正。显著性阈值为校正后p<0.05；"
        "同时报告差值及置信区间，避免仅依据p值解释实际效应幅度。"
    ))

    replace(document, "4.2.模型消融实验", "4.2.模型消融与稳健性实验")
    replace(document, "为量化不同数据源在各网络结构中的作用", (
        "本节依次报告原有模态组消融、本文模型结构组成消融、OSM单项消融、三随机种子稳定性以及空间尺度、缓冲区和leave-region-out验证。"
        "所有可训练变体均重新训练，并仅依据各自验证集macro mIoU选择检查点；测试集不参与模型选择。原有模态组消融仍包括全部模态、仅Sentinel-2、"
        "联合去除POI/AOI/道路/文本、去除建筑、去除Sentinel-1、去除夜间灯光和去除高分辨率影像七种设置。"
    ))
    caption_table3 = find_paragraph(document, "表3")
    add_paragraph_before(document, caption_table3, "4.2.1 模态组消融", heading=True)
    replace(document, "本文模型从仅Sentinel-2", (
        "本文模型从仅Sentinel-2扩展到全部模态后，OA由0.8253提高至0.8789，macro mIoU由0.4392提高至0.5946，macro-F1由0.5874提高至0.7338。"
        "联合去除POI、AOI、道路和OSM文本时，OA、macro mIoU和macro-F1分别下降0.0319、0.1111和0.1034。由于这一设置同步删除四种相关信息，"
        "它只能说明OSM语义—道路组合的联合贡献，不能用于判断POI、AOI、道路或文本中哪一种最不可替代；相应单项消融见4.2.3。"
        "其余模态对macro mIoU的影响为：夜间灯光下降0.0165、建筑功能与高度下降0.0151、高分辨率影像下降0.0149，Sentinel-1下降0.0031。"
    ))
    replace_suffix(document, "综合六种结构，最稳定的结论", (
        "综合六种结构，可重复观察到的现象是多源输入整体优于仅Sentinel-2，且联合删除OSM语义与道路后六种结构均发生明显退化。"
        "但联合消融不区分各OSM组成，且部分模态在个别网络上呈非单调变化，因此下文通过单项重训练和多随机种子结果进一步限定解释范围。"
    ))

    anchor_results = find_paragraph(document, "4.3.输入要素重要性")
    add_paragraph_before(document, anchor_results, "4.2.2 结构组成消融", heading=True)
    add_paragraph_before(document, anchor_results, (
        f"完整模型测试OA、mIoU和macro-F1分别为{full['OA']:.4f}、{full['mIoU']:.4f}和{full['F1']:.4f}（表4）。"
        f"去除层级分类后，三项指标变为{no_hierarchy['OA']:.4f}、{no_hierarchy['mIoU']:.4f}和{no_hierarchy['F1']:.4f}，"
        f"mIoU下降{full['mIoU']-no_hierarchy['mIoU']:.4f}（配对置换p{pvalue(no_hierarchy['配对置换p_mIoU'])}）。"
        f"去除语义残差后mIoU为{no_residual['mIoU']:.4f}，下降{full['mIoU']-no_residual['mIoU']:.4f}（p{pvalue(no_residual['配对置换p_mIoU'])}）；"
        f"将类别级门控替换为共享门控后mIoU为{shared_gate['mIoU']:.4f}，变化{shared_gate['mIoU']-full['mIoU']:+.4f}（p{pvalue(shared_gate['配对置换p_mIoU'])}）。"
        "这些结果分别量化三个结构组成在其余模块保持时的条件边际效应；若共享门控未降低精度，则不再把类别特异门控表述为已得到独立性能支持。"
    ))
    add_paragraph_before(document, anchor_results, "表4  本文模型结构组成消融结果。", caption=True)
    add_table_before(document, anchor_results,
        ["结构设置", "OA", "mIoU", "macro-F1", "ΔmIoU", "mIoU配对p"],
        [[row["结构设置"], f4(row["OA"]), f4(row["mIoU"]), f4(row["F1"]), f"{row['相对完整模型_mIoU']:+.4f}", pvalue(row["配对置换p_mIoU"])] for _, row in structure.iterrows()]
    )
    add_paragraph_before(document, anchor_results, "4.2.3 OSM信息单项消融", heading=True)
    joint = osm.iloc[-1]
    add_paragraph_before(document, anchor_results, (
        f"四种OSM信息分别删除并重新训练后，最大的单项mIoU退化来自“{largest['OSM设置']}”，下降{largest['mIoU下降']:.4f}；"
        + "，".join(f"{row['OSM设置']}变化{row['相对全模态_mIoU']:+.4f}" for _, row in individual.iterrows())
        + f"。联合删除POI、AOI、道路与文本时mIoU下降{-joint['相对全模态_mIoU']:.4f}。"
        "联合下降明显大于任一单项并不意味着四者贡献可线性相加，因为POI、AOI、道路和文本在空间与语义上相关，模型也会在重训练时利用剩余模态进行补偿。"
        "因此，本文仅将最大单项下降解释为当前模型和数据条件下较强的边际证据，不再使用‘最不可替代’的绝对表述。"
    ))
    add_paragraph_before(document, anchor_results, "表5  本文模型OSM信息单项与联合消融结果。", caption=True)
    add_table_before(document, anchor_results,
        ["OSM设置", "OA", "mIoU", "macro-F1", "ΔmIoU", "mIoU配对p"],
        [[row["OSM设置"], f4(row["OA"]), f4(row["mIoU"]), f4(row["F1"]), f"{row['相对全模态_mIoU']:+.4f}", pvalue(row["配对置换p_mIoU"])] for _, row in osm.iterrows()]
    )
    add_figure_before(document, anchor_results, figures / "figure_structural_osm_ablation.png", "图4  本文模型的结构组成与OSM信息消融。a，移除层级分类、语义残差或以共享门控替代类别门控后的mIoU变化；b，POI、AOI、道路、文本单项及联合删除后的mIoU变化。")

    add_paragraph_before(document, anchor_results, "4.2.4 多随机种子训练稳定性", heading=True)
    fcnn_seed = seeds.loc[seeds["模型"] == "FCNN"].iloc[0]
    transformer_seed = seeds.loc[seeds["模型"] == "Transformer"].iloc[0]
    ours_vs_fcnn = seed_tests.loc[(seed_tests["指标"] == "mIoU") & seed_tests["比较"].str.endswith("FCNN")].iloc[0]
    add_paragraph_before(document, anchor_results, (
        f"三个独立种子下，本文模型OA、mIoU和macro-F1为{mean_sd(ours_seed,'OA')}、{mean_sd(ours_seed,'mIoU')}和{mean_sd(ours_seed,'F1')}（均值±SD；表6）。"
        f"FCNN的mIoU为{mean_sd(fcnn_seed,'mIoU')}，Transformer为{mean_sd(transformer_seed,'mIoU')}。"
        f"本文模型相对FCNN的种子配对mIoU平均差为{ours_vs_fcnn['本文减基线_均值']:+.4f}，95% t区间为"
        f"{ours_vs_fcnn['差值_t95CI下限']:.4f}至{ours_vs_fcnn['差值_t95CI上限']:.4f}，Holm校正p={ours_vs_fcnn['Holm校正p']:.4f}。"
        "由于n=3时t区间较宽、显著性检验功效有限，本研究将多种子结果主要用于验证排序和训练波动，不把未显著差异解释为模型等效。"
    ))
    add_paragraph_before(document, anchor_results, "表6  六种模型三个独立随机种子的训练稳定性（均值±SD）。", caption=True)
    add_table_before(document, anchor_results,
        ["模型", "训练次数", "OA", "mIoU", "macro-F1"],
        [[row["模型"], str(int(row["独立训练次数"])), mean_sd(row,"OA"), mean_sd(row,"mIoU"), mean_sd(row,"F1")] for _, row in seeds.iterrows()]
    )

    add_paragraph_before(document, anchor_results, "4.2.5 空间相关尺度、缓冲区与留区验证", heading=True)
    corr_at_5 = correlogram.loc[(correlogram["距离下限_km"] >= 5.12) & (correlogram["距离下限_km"] < 6.4)].iloc[0]
    add_paragraph_before(document, anchor_results, (
        f"八类标签组成的平均标准化空间相关由0–1.28 km距离箱的{correlogram.iloc[0]['八类组成平均标准化相关']:.3f}降至"
        f"5.12–6.40 km的{corr_at_5['八类组成平均标准化相关']:.3f}，并在{first_low['距离下限_km']:.2f}–{first_low['距离上限_km']:.2f} km首次低于0.1（图5c）。"
        "这说明原5.12 km块已经覆盖大部分局地相关范围，但仍不能完全消除块边界附近的相关性，因此需要缓冲和更严格留区验证。"
        f"加入1.28 km排除缓冲后，本文模型OA、mIoU和macro-F1为{buffered['OA']:.4f}、{buffered['mIoU']:.4f}和{buffered['F1']:.4f}；"
        f"西部leave-region-out时相应指标为{loro['OA']:.4f}、{loro['mIoU']:.4f}和{loro['F1']:.4f}（表7）。"
        "跨协议指标变化同时受训练样本减少、测试区类别组成和区域分布偏移影响，因而用于界定空间泛化边界，而不是将不同测试集上的数值作配对显著性比较。"
    ))
    table7_caption = add_paragraph_before(
        document,
        anchor_results,
        "表7  本文模型在不同空间验证协议下的结果及空间块bootstrap区间。",
        caption=True,
    )
    table7_caption.paragraph_format.page_break_before = True
    add_table_before(document, anchor_results,
        ["空间协议", "测试组", "OA（95% CI）", "mIoU（95% CI）", "F1（95% CI）"],
        [[row["空间验证协议"], str(int(row["测试空间组数"])),
          f"{row['OA']:.4f} ({row['OA_空间bootstrap95CI下限']:.4f}–{row['OA_空间bootstrap95CI上限']:.4f})",
          f"{row['mIoU']:.4f} ({row['mIoU_空间bootstrap95CI下限']:.4f}–{row['mIoU_空间bootstrap95CI上限']:.4f})",
          f"{row['F1']:.4f} ({row['F1_空间bootstrap95CI下限']:.4f}–{row['F1_空间bootstrap95CI上限']:.4f})"] for _, row in spatial.iterrows()]
    )
    add_figure_before(document, anchor_results, figures / "figure_seed_spatial_robustness.png", "图5  训练与空间稳健性。a，六种模型三个随机种子的均值±SD；b，本文模型在不同空间验证协议下的总体指标；c，八类标签组成随patch中心距离变化的平均标准化相关。")

    replace(document, "图4  本文模型的GradientSHAP蜂群图", "图6  本文模型的GradientSHAP蜂群图。横轴为各输入通道在49×49邻域内聚合后对真实目标类别中心像元log-probability的贡献，颜色表示归一化特征值高低。")
    replace(document, "蜂群分布（图4）", find_paragraph(document, "蜂群分布（图4）").text.replace("图4", "图6"))
    replace(document, "图5  本文模型的GradientSHAP模态归因", "图7  本文模型的GradientSHAP模态归因。a，九组输入模态的归一化平均绝对SHAP贡献及95% bootstrap置信区间；b，八个类别内部的模态贡献构成。")
    replace(document, "按80个分层测试样本的mean |SHAP|汇总（图5a）", find_paragraph(document, "按80个分层测试样本的mean |SHAP|汇总（图5a）").text.replace("图5a", "图7a").replace("图5b", "图7b"))

    replace(document, "在相同多模态输入、数据划分和评价流程下", (
        f"在相同多模态输入、数据划分和评价流程下，种子42的本文模型取得最高OA、macro mIoU和macro-F1点估计。"
        f"结构消融进一步表明，去除层级分类和语义残差使mIoU分别下降{full['mIoU']-no_hierarchy['mIoU']:.4f}和{full['mIoU']-no_residual['mIoU']:.4f}；"
        f"共享门控相对完整类别门控的mIoU变化为{shared_gate['mIoU']-full['mIoU']:+.4f}。"
        "因此，层级分类与语义残差的贡献可以在当前结构内被分别观测，而类别级门控是否优于更简洁共享门控须以实际消融方向为准。"
        f"三个随机种子下本文模型mIoU为{ours_seed['mIoU_均值']:.4f}±{ours_seed['mIoU_SD']:.4f}，"
        "说明单一最佳种子点估计不能完全代表训练波动。模型相对FCNN和Transformer的优势仍应结合配对种子区间和空间配对检验谨慎解释。"
    ))
    replace(document, "消融实验与GradientSHAP从互补角度", (
        f"消融实验与GradientSHAP从互补角度揭示不同数据源的作用。全部模态相较仅使用Sentinel-2使本文模型macro mIoU提高0.1554；"
        f"OSM单项重训练中，{largest['OSM设置']}产生最大的单项下降（{largest['mIoU下降']:.4f}），"
        f"而联合去除POI、AOI、道路和文本下降{-joint['相对全模态_mIoU']:.4f}。"
        "该联合效应反映四种相关城市语义和网络信息的整体作用，不证明其中任一单项‘最不可替代’。道路、AOI、POI及文本在覆盖范围和语义粒度上互补，"
        "单项删除后模型可利用其余来源补偿，因此联合下降不应分解为线性可加贡献。"
    ))
    replace(document, "模型评价结果还说明", (
        f"模型评价结果说明，空间支持度、分块尺度和区域留出方式都会影响精度解释。标签组成相关在{first_low['距离下限_km']:.2f}–{first_low['距离上限_km']:.2f} km首次低于0.1，"
        "说明原5.12 km空间块大体覆盖局地相关范围但并非严格独立。2.56 km与10.24 km重划分检验了尺度敏感性，1.28 km缓冲进一步减少边界邻近样本，"
        f"西部留区验证的mIoU为{loro['mIoU']:.4f}，直接揭示区域分布偏移下的泛化水平。"
        "因此，主测试集的空间bootstrap用于量化当前测试组之间的变异，多随机种子用于量化优化波动，leave-region-out用于考察跨区域外推，三者回答不同问题。"
    ))
    replace(document, "本研究仍存在若干局限", (
        "本研究仍存在若干局限。首先，三随机种子只能提供初步训练稳定性证据，基于df=2的置信区间较宽，仍需更多独立重复。其次，leave-region-out只覆盖香港西部一个地理方向，"
        "尚不能等同于跨城市或跨时相验证。再次，BLU标签、卫星影像、OSM、建筑功能与高度数据并非完全同日，开放地理数据的完整性和更新频率也具有空间差异。"
        "高分辨率影像拼接差异以及山地SAR阴影与叠掩仍可能形成局地伪特征。最后，GradientSHAP基于80个类别均衡样本、8个训练背景样本和有限积分次数，"
        "其百分比受背景分布、邻域尺度、相关特征及模态通道数影响。后续研究应开展多方向或行政区leave-one-region-out、跨城市和跨时相验证，并扩大解释样本与背景分布。"
    ))
    replace(document, "本文面向香港全域10 m土地利用/土地覆盖制图", (
        f"本文面向香港全域10 m输出网格的八类土地利用/土地覆盖制图，构建融合Sentinel-2、Sentinel-1、高分辨率影像、VIIRS夜间灯光、建筑功能与高度及OSM道路、POI、AOI和文本语义的层级多模态FCNN。"
        f"固定主测试集的种子42结果为OA={full['OA']:.4f}、macro mIoU={full['mIoU']:.4f}和macro-F1={full['F1']:.4f}；三个独立种子的mIoU为{ours_seed['mIoU_均值']:.4f}±{ours_seed['mIoU_SD']:.4f}。"
        f"结构消融显示，去除层级分类和语义残差分别使mIoU下降{full['mIoU']-no_hierarchy['mIoU']:.4f}和{full['mIoU']-no_residual['mIoU']:.4f}，"
        f"共享门控相对类别门控的变化为{shared_gate['mIoU']-full['mIoU']:+.4f}。"
        f"OSM单项消融中{largest['OSM设置']}的边际下降最大（{largest['mIoU下降']:.4f}），但联合删除四类OSM信息的下降更大，证明联合消融不能归因于单一来源。"
        f"西部leave-region-out的mIoU为{loro['mIoU']:.4f}，说明固定空间组测试仍可能高估跨区域外推能力。"
        "总体而言，层级结构和城市语义在当前香港数据上提供了可量化增益，但其效应幅度、训练稳定性和空间可迁移性必须分别报告。"
    ))

    document.save(args.output)
    audit = args.output.with_suffix(".audit.txt")
    audit.write_text(
        "Source preserved: " + str(args.source) + "\n"
        "Output: " + str(args.output) + "\n"
        "Inserted: structural ablation, separate OSM ablation, three-seed stability, spatial scale/buffer/LORO.\n"
        f"Paragraphs={len(document.paragraphs)}, tables={len(document.tables)}, inline_shapes={len(document.inline_shapes)}\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
