from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches


def replace_in_run(paragraph, old: str, new: str) -> None:
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            return
    raise ValueError(f"Text not found in paragraph: {old[:80]}")


def set_single_run_text(paragraph, text: str) -> None:
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def replace_picture(paragraph, image_path: Path, width_inches: float) -> None:
    paragraph.clear()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image_path), width=Inches(width_inches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--beeswarm", type=Path, required=True)
    parser.add_argument("--global-figure", type=Path, required=True)
    args = parser.parse_args()

    document = Document(args.input)
    paragraphs = document.paragraphs

    # Abstract: retain the original structure and update only stale empirical claims.
    abstract = paragraphs[1]
    replace_in_run(
        abstract,
        "空间独立测试表明，本文模型取得0.8554的总体精度、0.4753的macro mIoU和0.6226的macro-F1，在六种统一全模态输入的模型中为最高值，但相对FCNN的优势分别仅为0.0014和0.0036。相较仅使用Sentinel-2，全部模态使本文模型的macro mIoU提高0.1411；去除POI/AOI/道路或高分辨率影像分别使macro mIoU下降0.0549和0.0396。独立XGBoost–SHAP特征审计中，Sentinel-2、OSM文本和道路的总绝对SHAP贡献占比分别为32.0%、23.4%和12.3%。",
        "空间独立测试表明，本文模型取得0.8789的总体精度、0.5946的macro mIoU和0.7338的macro-F1，在六种统一全模态输入的模型中均为最高值；相较全模态FCNN，三项指标分别提高0.0056、0.0150和0.0126。相较仅使用Sentinel-2，全部模态使本文模型的macro mIoU提高0.1554；去除POI/AOI/道路使macro mIoU下降0.1111。基于最终网络的GradientSHAP归因中，Sentinel-2、高分辨率影像、Sentinel-1、道路和AOI的总绝对SHAP贡献占比分别为47.5%、12.8%、11.6%、9.4%和9.0%。",
    )

    replace_in_run(
        paragraphs[7],
        "并通过模态消融、分类别混淆和XGBoost–SHAP代理审计检验不同数据源的贡献。",
        "并通过模态消融、分类别混淆和基于最终网络的GradientSHAP归因检验不同数据源的贡献。",
    )

    set_single_run_text(
        paragraphs[177],
        "为解释最终分类网络实际使用输入信息的方式，本文直接对测试精度最高的层级多模态FCNN检查点实施GradientSHAP（expected gradients），不再训练独立代理分类器。归因样本从八类空间独立测试区中分层抽取，每类10个中心像元，共80个样本；背景分布由8个训练样本构成。对每个样本截取49×49像元邻域，并以真实目标类别中心像元的log-probability作为解释输出，采用256次期望梯度积分。79个栅格输入通道的像素级SHAP绝对值在邻域内求和，OSM文本则在模型内部16维连续语义投影上计算归因，因此文本结果代表潜在语义表示而非单个token。模态均值的95%置信区间通过对80个测试样本进行2,000次bootstrap获得。",
    )
    replace_picture(paragraphs[178], args.beeswarm, 5.70)
    set_single_run_text(
        paragraphs[179],
        "图4  本文模型的GradientSHAP蜂群图。横轴为各输入通道在49×49邻域内聚合后对真实目标类别中心像元log-probability的贡献，颜色表示归一化特征值高低。",
    )
    set_single_run_text(
        paragraphs[180],
        "蜂群分布（图4）显示，高分辨率绿色和红色通道、Sentinel-2红边1、蓝光、SWIR2、NDVI及Sentinel-1 VV等特征具有较宽的贡献范围；AOI功能类4、道路中心线与道路面掩膜也进入全局前20位。同一特征在不同类别和局地样本中可同时出现正、负贡献，例如高分辨率绿色通道和红边1均跨越零值两侧，表明模型依赖的是多特征条件组合，而不是可直接转化为统一阈值的单变量关系。",
    )
    replace_picture(paragraphs[181], args.global_figure, 5.70)
    set_single_run_text(
        paragraphs[182],
        "图5  本文模型的GradientSHAP模态归因。a，九组输入模态的归一化平均绝对SHAP贡献及95% bootstrap置信区间；b，八个类别内部的模态贡献构成。",
    )
    set_single_run_text(
        paragraphs[183],
        "按80个分层测试样本的mean |SHAP|汇总（图5a），Sentinel-2占47.5%（95% CI：43.4%–51.6%），高分辨率影像占12.8%，Sentinel-1占11.6%，OSM道路占9.4%，OSM AOI占9.0%，夜间灯光占3.4%，OSM文本潜在表示占2.9%，建筑功能与高度占2.3%，POI计数栅格占0.9%。类别内部归因（图5b）进一步显示，交通设施对高分辨率影像和道路的依赖分别升至17.1%和15.8%，住宅与工业仓储对AOI的依赖分别为12.3%和12.8%，而林地等与水体对Sentinel-2的依赖分别达到53.8%和54.2%。这些差异与各模态的物理或语义职责一致，但归因百分比仍受背景分布、通道数量及相关特征共同影响；单通道贡献方向及样本差异另见图4。",
    )
    set_single_run_text(
        paragraphs[186],
        "消融与本文模型自身的GradientSHAP归因从不同角度支持了模态互补。Sentinel-2在SHAP总贡献中占47.5%，且仅使用Sentinel-2时macro mIoU由0.5946降至0.4392，说明光学信息构成物理地表识别基础。去除POI/AOI/道路后macro mIoU下降0.1111；对应地，道路与AOI在模型SHAP中合计贡献18.5%，交通设施的道路类内贡献达到15.8%，表明城市语义和交通结构对建成区功能细分关键。去除高分辨率影像、夜间灯光、建筑特征和Sentinel-1时，macro mIoU分别下降0.0149、0.0165、0.0151和0.0031；模型SHAP中相应模态贡献分别为12.8%、3.4%、2.3%和11.6%。二者数值不应直接对应：重训练消融测量缺失整个模态后的性能变化，而GradientSHAP测量固定模型相对于经验背景分布的局部依赖。",
    )
    set_single_run_text(
        paragraphs[189],
        "本研究仍有四项边界。第一，实验仅覆盖香港和一个近时相数据组合，尚未进行跨城市、跨季节或跨年份迁移。第二，BLU与遥感、OSM及建筑数据可能存在时间差，AOI/POI覆盖和标签完整性也具有空间异质性。第三，高分辨率影像拼接缝、山地SAR阴影与叠掩可能形成局地伪特征，尽管输出已限制在BLU有效范围，输入质量问题仍不能由掩膜完全消除。第四，GradientSHAP基于80个分层中心像元、8个经验背景和49×49邻域，相关通道、模态通道数及背景选择均会影响组级归因；文本归因仅对应16维潜在表示，不能解释为具体OSM词元的贡献。因此，SHAP结果反映固定网络在当前数据分布下的统计依赖，而非模态的因果效应。",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)


if __name__ == "__main__":
    main()
