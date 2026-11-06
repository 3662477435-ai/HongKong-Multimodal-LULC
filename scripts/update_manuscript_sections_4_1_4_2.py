from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


SECTION_41 = {
    155: (
        "在排除商业办公和荒地后，六种模型均在同一八类空间独立测试集上进行评价。"
        "测试集包含125个128×128像元patch，归属于14个互不重叠的5.12 km空间组；"
        "以下点估计均来自随机种子42、仅依据验证集macro mIoU选取的最佳检查点。"
        "为刻画测试区域之间的空间异质性，本文以空间组为抽样单元进行5,000次有放回bootstrap，"
        "并报告百分位95%置信区间。"
    ),
    158: (
        "六种模型的全香港制图结果见图2。各模型均恢复了大面积林地／灌丛／草地／湿地及沿主要城镇带分布的建设类别，"
        "但在工业仓储、其他建设用地、农业以及水陆交界处呈现出不同程度的局部差异。"
        "本文模型、FCNN和Transformer的制图结果较为接近，FCN-ResNet50和VGG-UNet-Transformer的类别错分更为集中；"
        "这一空间表现与总体指标和逐类指标的排序基本一致。"
    ),
    161: (
        "在统一全模态输入、数据划分和评价流程下，本文模型取得最高的总体点估计：OA为0.8789"
        "（95% CI：0.8429–0.9089），macro mIoU为0.5946（0.4669–0.6176），macro-F1为0.7338"
        "（0.6054–0.7523）（表1）。FCNN、Transformer和CNN的macro mIoU分别为0.5796、0.5775和0.5698，"
        "而VGG-UNet-Transformer和FCN-ResNet50分别为0.4255和0.4239。相对于性能最接近的FCNN，"
        "本文模型的OA、macro mIoU和macro-F1分别提高0.0056、0.0150和0.0126；相对于CNN则分别提高"
        "0.0111、0.0248和0.0215。"
    ),
    162: (
        "基于14个空间测试组的精确配对置换检验并对五个基线实施Holm校正后，本文模型相对于CNN和FCNN的"
        "macro mIoU优势仍达到0.05显著性水平（校正后p分别为0.0098和0.0421），macro-F1差异亦显著"
        "（p分别为0.0103和0.0406）。相对于FCN-ResNet50，OA、macro mIoU和macro-F1差异均显著"
        "（校正后p分别为0.0024、0.0061和0.0049）。然而，本文模型与Transformer之间三项指标的差异均未显著，"
        "与VGG-UNet-Transformer的差异在Holm校正后也未达到0.05水平。因此，结果支持本文模型在当前空间划分上"
        "具有最高点估计及相对CNN、FCNN的类别均衡优势，但不能据此宣称其显著优于所有基线。"
    ),
    164: (
        "逐类结果进一步揭示了OA与macro mIoU之间的差异（表2）。林地／灌丛／草地／湿地占测试有效像元的"
        "71.8%，本文模型在该类取得0.9074的IoU和0.9515的F1；水体的IoU和F1分别为0.7589和0.8630。"
        "公共机构与休闲、交通设施和住宅的IoU分别为0.6034、0.6009和0.5703，工业仓储、其他建设用地及农业"
        "则分别为0.4921、0.4256和0.3981。与五种基线相比，本文模型在住宅、交通设施、其他建设用地、农业以及"
        "林地／灌丛／草地／湿地五类上取得最高IoU；Transformer在工业仓储、公共机构与休闲和水体三类上略高。"
        "这表明本文模型的总体优势主要来自多个类别的共同改善，而不是所有单类均占优。"
    ),
    167: (
        "行归一化混淆矩阵（图3）显示，本文模型对林地／灌丛／草地／湿地、水体、公共机构与休闲、交通设施和住宅的"
        "召回率分别为0.9430、0.8288、0.7689、0.7465和0.7279。主要误差路径集中在相近地表或功能类别之间："
        "27.97%的农业像元被分为林地／灌丛／草地／湿地，17.58%的工业仓储被分为其他建设用地，12.29%的水体"
        "被分为林地／灌丛／草地／湿地，10.22%的其他建设用地被分为林地／灌丛／草地／湿地，另有9.37%的住宅"
        "被分为工业仓储。上述混淆解释了农业、其他建设用地和工业仓储的IoU仍明显低于多数类别。"
    ),
    168: (
        "置信区间还表明，不同类别的空间稳定性并不一致。林地／灌丛／草地／湿地的IoU 95% CI为0.8767–0.9292，"
        "而仅分布于9个测试空间组的水体，其IoU点估计虽为0.7589，区间却扩展至0.0032–0.8278；工业仓储仅覆盖"
        "8个空间组，其IoU区间为0.2814–0.6060。因而，少数空间组内高度集中的类别应同时结合点估计、空间支持度"
        "和置信区间解释。需要强调的是，这些bootstrap区间衡量的是已抽取空间组之间的变异，并不包含优化器或随机初始化"
        "造成的训练波动；目前每个模型只有一个完整训练种子，尚不能据此评价多次独立训练的稳定性。"
    ),
}


SECTION_42_OPENING = (
    "为量化不同数据源在各网络结构中的作用，本文对本文模型及五种基线分别运行全部模态、仅Sentinel-2、"
    "去除POI/AOI/道路与OSM文本、去除建筑功能与高度、去除Sentinel-1、去除夜间灯光和去除高分辨率影像七种设置，"
    "共形成42个模型—输入组合。所有变体沿用相同的空间划分、标签掩膜和评价指标，并仅依据验证集macro mIoU选择检查点。"
    "表3中的变化量均以同一模型的全部模态结果为参照；由于各变体目前仅完成一个训练种子，这些差值用于描述效应幅度，"
    "不作为统计显著性证据。"
)


SECTION_42_PARAGRAPHS = [
    (
        "本文模型从仅Sentinel-2扩展到全部模态后，OA由0.8253提高至0.8789，macro mIoU由0.4392提高至0.5946，"
        "macro-F1由0.5874提高至0.7338，绝对增幅分别为0.0536、0.1554和0.1464（表3）。在逐项移除实验中，"
        "去除POI/AOI/道路与OSM文本造成最大退化，OA、macro mIoU和macro-F1分别下降0.0319、0.1111和0.1034。"
        "其余模态对macro mIoU的影响依次为夜间灯光下降0.0165、建筑功能与高度下降0.0151、高分辨率影像下降0.0149，"
        "以及Sentinel-1下降0.0031。由此可见，在本文模型内，城市语义和道路上下文提供了最主要的非光学增益，"
        "夜间活动、建筑形态和高分辨率空间细节提供了幅度接近的补充信息，而Sentinel-1在八类汇总指标上的边际变化较小。"
    ),
    (
        "这种趋势在三种通用早期融合基线中同样出现。相对于仅Sentinel-2，全部模态使CNN、FCNN和Transformer的"
        "macro mIoU分别提高0.1662、0.1422和0.1832；去除POI/AOI/道路与OSM文本则分别使其下降0.1222、0.1015"
        "和0.1245。FCN-ResNet50和VGG-UNet-Transformer从仅Sentinel-2扩展到全部模态后，macro mIoU也分别提高"
        "0.0802和0.0810，去除OSM语义与道路信息后分别下降0.0770和0.0751。因而，OSM相关输入的贡献并非只出现在"
        "本文的层级结构中，而是在六种模型上均表现为正向的类别均衡增益。"
    ),
    (
        "其他模态的效应具有更明显的结构依赖性。去除夜间灯光使FCN-ResNet50的macro mIoU下降0.1261，明显高于其余模型；"
        "去除高分辨率影像使CNN、FCNN、Transformer和FCN-ResNet50分别下降0.0260、0.0198、0.0312和0.0606。"
        "去除建筑功能与高度时，上述四个基线的macro mIoU下降0.0357、0.0285、0.0457和0.0278。"
        "相反，VGG-UNet-Transformer在去除建筑特征或高分辨率影像后，macro mIoU分别提高0.0545和0.0192，"
        "去除Sentinel-1后也小幅提高0.0028，因此其全部模态配置并非该结构的最优输入组合。该非单调现象说明，"
        "消融差值不仅受信息量影响，也受输入冗余、适配层压缩和训练随机性的共同影响；它不能被解释为相应数据源本身具有负贡献。"
        "综合六种结构，最稳定的结论是多源输入整体优于仅Sentinel-2，且POI、AOI、道路与OSM文本构成最一致的增益来源。"
    ),
]


def _copy_run_format(source, target) -> None:
    if source is not None and source._r.rPr is not None:
        target._r.insert(0, deepcopy(source._r.rPr))


def set_paragraph_text(paragraph, text: str, format_source=None) -> None:
    source_run = None
    if paragraph.runs:
        source_run = paragraph.runs[0]
    elif format_source is not None and format_source.runs:
        source_run = format_source.runs[0]
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    run = paragraph.add_run(text)
    _copy_run_format(source_run, run)


def insert_formatted_paragraph_before(target, text: str, template):
    paragraph = target.insert_paragraph_before()
    if paragraph._p.pPr is not None:
        paragraph._p.remove(paragraph._p.pPr)
    if template._p.pPr is not None:
        paragraph._p.insert(0, deepcopy(template._p.pPr))
    run = paragraph.add_run(text)
    _copy_run_format(template.runs[0] if template.runs else None, run)
    return paragraph


def replace_embedded_picture(document: Document, paragraph, image_path: Path) -> None:
    blips = paragraph._p.xpath(".//a:blip")
    if len(blips) != 1:
        raise RuntimeError(f"Expected one embedded image, found {len(blips)}")
    relationship_id = blips[0].get(qn("r:embed"))
    image_part = document.part.related_parts[relationship_id]
    image_part._blob = image_path.read_bytes()


def remove_paragraph(paragraph) -> None:
    parent = paragraph._element.getparent()
    parent.remove(paragraph._element)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_docx", type=Path)
    parser.add_argument("output_docx", type=Path)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()

    document = Document(args.input_docx)
    original = document.paragraphs
    if original[154].text.strip() != "4.1.模型结果及对比":
        raise RuntimeError("Section 4.1 anchor was not found at the expected location")
    if original[176].text.strip() != "4.2.模型消融实验":
        raise RuntimeError("Section 4.2 anchor was not found at the expected location")
    if original[181].text.strip() != "4.3.输入要素重要性":
        raise RuntimeError("Section 4.3 anchor was not found at the expected location")

    body_template = original[155]
    for index, text in SECTION_41.items():
        set_paragraph_text(original[index], text, body_template)

    set_paragraph_text(
        original[163],
        "表2  本文模型在空间独立测试集上的逐类IoU与F1。",
        original[160],
    )
    set_paragraph_text(original[177], SECTION_42_OPENING, body_template)
    set_paragraph_text(
        original[179],
        "表3  本文模型及五种基线的扩展模态消融结果。全部变体使用相同空间划分。",
        original[160],
    )
    set_paragraph_text(original[180], SECTION_42_PARAGRAPHS[0], body_template)
    for text in SECTION_42_PARAGRAPHS[1:]:
        insert_formatted_paragraph_before(original[181], text, body_template)

    # Remove unused placeholder paragraphs between the completed 4.1 text and 4.2.
    for index in range(169, 176):
        if not original[index].text.strip() and not original[index]._p.xpath(".//w:drawing"):
            remove_paragraph(original[index])

    replace_embedded_picture(
        document,
        original[156],
        args.figures_dir / "figure_8class_six_model_maps_2x3.png",
    )
    replace_embedded_picture(
        document,
        original[165],
        args.figures_dir / "figure_8class_confusion_matrices_2x3.png",
    )
    replace_embedded_picture(
        document,
        original[178],
        args.figures_dir / "table_8class_extended_modality_ablation.png",
    )

    args.output_docx.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output_docx)
    print(args.output_docx)


if __name__ == "__main__":
    main()
