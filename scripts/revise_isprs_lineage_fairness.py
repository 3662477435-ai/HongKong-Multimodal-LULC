"""Create a minimally edited manuscript copy for the lineage/fairness revision.

The source DOCX is preserved.  Numerical additions that require new training
remain explicitly scoped to the accompanying experiment scripts until those
runs complete.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


SRC = Path(r"<third_party_sources>\融合多源遥感与城市语义的条件层级概率融合香港土地利用土地覆盖制图_城市上下文残差与条件层级概率修订版.docx")
DST = Path(r"<third_party_sources>\融合多源遥感与城市语义的条件层级概率融合香港土地利用土地覆盖制图_数据谱系与公平比较扩展版.docx")


def replace(paragraph, text: str) -> None:
    rpr = deepcopy(paragraph.runs[0]._r.rPr) if paragraph.runs and paragraph.runs[0]._r.rPr is not None else None
    paragraph.clear()
    run = paragraph.add_run(text)
    if rpr is not None:
        run._r.insert(0, rpr)


def insert_after(paragraph, element) -> None:
    """Place a newly created paragraph or table immediately after a paragraph."""
    paragraph._p.addnext(element)


def add_building_lineage_table(doc: Document, anchor) -> None:
    """Add the single-seed building lineage ablation as a transparent table."""
    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.add_run("表S5  建筑功能与建筑形态/高度的数据谱系拆分消融（种子42）").bold = True
    insert_after(anchor, caption._p)

    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["建筑输入", "保留通道", "OA", "macro mIoU", "macro-F1"]
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    header_properties = table.rows[0]._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    header_properties.append(header)
    rows = [
        ("完整建筑输入", "9功能 + 5形态/高度", "0.8789", "0.5946", "0.7338"),
        ("去建筑功能", "5形态/高度", "0.8786", "0.5956", "0.7342"),
        ("去形态/高度", "9功能", "0.8744", "0.5853", "0.7253"),
        ("去全部建筑", "无（14通道均置零）", "0.8762", "0.5916", "0.7312"),
    ]
    for row_values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row_values):
            cell.text = value
    insert_after(caption, table._tbl)

    note = doc.add_paragraph(
        "注：所有变体保持95→96→48共同特征转码头、空间阻隔划分、采样、损失和训练流程不变；"
        "结果均为种子42的单次重训练，未用于独立显著性结论。相对完整输入，去建筑功能的mIoU变化为+0.0010，"
        "去形态/高度为-0.0093，去全部建筑为-0.0030。"
    )
    table._tbl.addnext(note._p)


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    shutil.copy2(SRC, DST)
    doc = Document(DST)

    replace(doc.paragraphs[2], (
        "高密度城市土地利用/土地覆盖制图需要同时识别自然地表物理属性与建成区使用功能，但单一遥感影像难以区分光谱相似的城市功能类别，"
        "开放地理数据的空间异质性、数据谱系相关性和类别不平衡也会降低模型泛化能力。本文面向香港全域10 m输出网格，构建融合Sentinel-1/2、"
        "高分辨率影像、VIIRS夜间灯光、建筑功能与高度、OSM道路、POI、AOI及文本语义的多模态FCNN。模型以直接分类路径保持逐像元分割，"
        "以城市上下文残差将局地POI、AOI、道路、建筑和社会经济信息再次注入类别logit，并以条件层级概率组织建成区、自然地表和水体之间的类别关系，"
        "最后通过类别级全局概率门控融合两条路径。空间阻隔主测试集上，种子42模型取得0.8789的总体精度、0.5946的macro mIoU和0.7338的macro-F1；"
        "同等47 epoch训练轮数下，FCNN的OA为0.8794、macro mIoU为0.5885、macro-F1为0.7295。因此，本文模型的主证据是类别均衡指标的最高点估计，"
        "而非所有总体指标均优。经结构消融的Holm族内校正，去除城市上下文残差后的macro mIoU下降0.0210仍达到0.05水平；"
        "去除条件层级分类和以共享门控替代类别门控的差异未达校正后显著性。完整128×128输入的Expected Gradients结果、"
        "建筑功能与建筑形态拆分消融及直接48通道强基线进一步用于界定数据谱系、概率组织和网络结构的作用边界。"
    ))
    replace(doc.paragraphs[48], (
        "建筑输入包括9个功能通道和5个形态通道。功能类别为住宅、商业、工业、教育、医疗、行政、公共服务、体育文化和未知功能；"
        "形态通道为建筑覆盖率、平均高度、最大高度、容积率代理和高度极差。建筑高度仅将有限且h>0的值视为有效；无效、缺失或非正高度以全部有效建筑高度的中位数10.2912 m填补，"
        "随后裁剪至1–300 m，原始有效数据最大值为278.1048 m。两套建筑数据经几何修复、合并、研究区裁剪和功能匹配后，共保留335,041栋建筑，"
        "其中220,269栋匹配到建筑功能。建筑功能产品不与BLU监督标签同源，但也不应被表述为与OSM、高分辨率Google Earth影像、夜间灯光或建筑高度统计独立："
        "SinoBF-1的原始功能标签由OSM land-use/AOI构建，且其制图使用了高分辨率光学影像、夜间灯光和建筑高度等信息[10]。"
        "因此，本文将其视为派生的外部城市语义特征，而不是独立观测；新增消融分别置零9个功能通道、5个建筑形态/高度通道及其全部14通道，"
        "并保持95→96→48共同特征转码头不变，以量化两组相关信息的条件边际作用。"
    ))
    replace(doc.paragraphs[121], "3.3. 模型训练与空间阻隔验证")
    replace(doc.paragraphs[127], (
        "图 空间阻隔划分下的八类土地利用训练、验证和测试像元分布。a，128×128像元patch按5.12 km空间块分配到训练、验证和测试集；"
        "空间块用于降低而非宣称完全消除空间泄漏。b，三套数据中八类有效标签像元的类别组成。c，各类别的像元支持量；颜色表示log10像元数。"
        "测试集包含1,518,543个有效标签像元，所有分类精度指标均在这些像元上逐像元计算。"
    ))
    replace(doc.paragraphs[195], (
        "除总体指标外，本文还报告逐类IoU、F1、生产者精度、用户精度和行归一化混淆矩阵。行归一化混淆矩阵以真实类别为分母，用于识别住宅—工业仓储、"
        "农业—林地以及水体—植被等主要误差路径。为补充面积型指标对边界误差不敏感的问题[26]，本文在每个测试patch内分别从预测与BLU参考标签提取一像元形态梯度边界，"
        "将每一边界以1像元（10 m）膨胀作为容忍区，计算每类别边界精度和召回率及其调和平均，再对有支持的八类取宏平均。"
        "该实现报告的是Boundary F1；文献[26]用于说明边界敏感评价的必要性，而不是作为本实现的同名定义。"
    ))
    replace(doc.paragraphs[197], (
        "在固定主测试集上比较本文模型与五种原始基线时，采用测试空间组为配对单元的双侧精确标签交换检验，并在每个指标内用Holm方法校正五次比较。"
        "结构消融的三项比较和OSM消融的五项比较分别构成独立的多重比较族，均报告原始与Holm校正p值，且显著性结论以校正后p<0.05为准。"
        "多随机种子比较使用种子配对t检验并作Holm校正；同时报告差值及置信区间，避免仅依据p值解释实际效应幅度。"
    ))
    replace(doc.paragraphs[201], (
        "六种模型均在同一八类空间阻隔测试集上进行评价。测试集包含125个128×128像元patch，归属于14个互不重叠的5.12 km空间组；"
        "以下点估计均来自随机种子42、仅依据验证集macro mIoU选取的最佳检查点。空间组成相关在6.40–7.68 km距离箱才首次低于0.1，"
        "因此5.12 km主协议被称为空间阻隔而非完全空间独立；1.28 km缓冲和西部leave-region-out结果作为更严格的稳健性验证。"
        "为刻画主测试区域之间的异质性，本文以空间组为抽样单元进行5,000次有放回bootstrap，并报告百分位95%置信区间。"
    ))
    replace(doc.paragraphs[221], (
        "本节依次报告原有模态组消融、本文模型结构组成消融、OSM单项消融、建筑功能与形态/高度拆分消融、三随机种子稳定性以及空间尺度、"
        "缓冲区和leave-region-out验证。所有可训练变体均重新训练，并仅依据各自验证集macro mIoU选择检查点；测试集不参与模型选择。"
        "建筑拆分消融保持9+5建筑通道的输入几何，但分别置零功能通道、形态/高度通道或全部建筑通道，避免因改变共同95→96→48转码头而混淆信息来源与模型容量。"
    ))
    replace(doc.paragraphs[230], (
        "完整模型测试OA、mIoU和macro-F1分别为0.8789、0.5946和0.7338（表6）。去除条件层级概率组织后，mIoU下降0.0093；"
        "去除城市上下文残差后mIoU下降0.0210；将类别级概率融合替换为共享门控后mIoU变化为-0.0040。三项结构比较在同一mIoU族内进行Holm校正后，"
        "仅城市上下文残差的退化仍达到0.05显著性（原始p=0.0140，Holm p=0.0420）。因此，城市上下文残差是当前数据上得到最强统计支持的结构贡献；"
        "条件层级概率和类别级概率融合的证据应理解为方向一致的结构化组织贡献，而非其单独带来决定性性能提升的证明。"
        "进一步地，冻结最终检查点的路径分解、类别级门控参数和概率校准指标在表S4中报告，以检验条件概率组织的实际使用程度和概率质量。"
    ))
    replace(doc.paragraphs[233], (
        "四种OSM信息分别删除并重新训练后，最大的单项mIoU退化来自去除AOI（-0.0656，原始p=0.0127）；去除POI、道路和OSM文本的变化分别为"
        "-0.0025、-0.0185和-0.0201。联合删除POI、AOI、道路与文本时mIoU下降0.1111（原始p=0.0083）。"
        "在五项OSM比较的Holm校正后，联合移除仍达到0.05显著性（Holm p=0.0415），而AOI的校正p约为0.0508，未跨越0.05阈值。"
        "因此，AOI应表述为最大单项点估计和最强原始单项证据，而非已被多重校正确认的独立显著贡献。由于各信息在空间与语义上相关，"
        "联合下降不应线性分摊为各自独立贡献。"
    ))
    replace(doc.paragraphs[245], (
        "为探索最终分类网络对输入信息的局部依赖，本文对种子42的层级多模态FCNN检查点实施Expected Gradients。解释目标统一为中心像元真实类别的"
        "最终融合log-probability。程序始终以完整128×128 patch输入模型，仅在归因汇总时统计中心49×49区域；解释样本从八类测试区中每类抽取10个，"
        "共80个，背景分布由8个训练样本构成，nsamples=256。79个栅格通道在原始输入空间内归一化汇总，16维OSM文本潜变量不与栅格归因强行归一化，"
        "而作为patch尺度上下文的独立结果报告。该类别均衡解释样本不代表香港真实类别面积分布。"
    ))
    replace(doc.paragraphs[259], (
        "公平性补充实验改变了本文对主结果的表述边界。原20 epoch基线下，种子42的本文模型相对FCNN、CNN和FCN-ResNet50等模型具有更高点估计；"
        "但FCNN延长至47 epoch后，其OA达到0.8794，略高于本文模型0.8789。本文模型的优势应限定为macro mIoU和macro-F1等类别均衡指标上的最高点估计，"
        "而非所有应用目标下的绝对最优。47 epoch控制的是训练轮数而非严格等GPU时间或等优化步；直接48通道预训练强基线与本文模型使用相同的"
        "95→96→48特征转码头、数据划分、采样、增强、损失和验证选择规则，以进一步检验该结论是否取决于较弱的历史基线。"
    ))
    replace(doc.paragraphs[262], (
        "消融与Expected Gradients回答的是不同问题。Expected Gradients衡量给定模型、给定样本和局部线性路径下的相对敏感性；重训练消融衡量删除某一数据源后模型重新优化的性能变化。"
        "由于SinoBF-1建筑功能是在OSM land-use/AOI监督并结合高分辨率影像、夜间灯光和建筑高度构建，其功能通道与这些输入存在数据谱系相关，"
        "不能将“去建筑”的退化解释为独立模态贡献。本文通过功能与形态/高度拆分重训练，并将结果限定为给定其他辅助数据存在条件下的边际变化。"
        "单种子建筑谱系拆分显示，去除9个建筑功能通道后macro mIoU为0.5956（完整输入为0.5946），去除5个形态/高度通道后为0.5853，"
        "全部14个建筑通道置零后为0.5916。该非单调模式说明这些高度相关特征存在冗余和优化波动；它不支持把建筑功能产品当作独立信息源或把任何一项变化解释为因果贡献。"
        "同样，OSM语义和道路信息的单项或联合消融影响反映其在当前特征集合中的条件作用，而非互不重叠的因果归因。"
    ))
    replace(doc.paragraphs[265], (
        "模型评价结果说明，空间支持度、分块尺度和区域留出方式都会影响精度解释。标签组成相关在6.40–7.68 km首次低于0.1，"
        "说明原5.12 km空间块大体覆盖局地相关范围但并非严格独立。2.56 km与10.24 km重划分检验了尺度敏感性，1.28 km缓冲进一步减少边界邻近样本，"
        "西部留区验证直接揭示区域分布偏移下的泛化水平。因此，本文将主测试集称为空间分块或空间阻隔测试，而不将其描述为完全空间独立验证。"
    ))
    add_building_lineage_table(doc, doc.paragraphs[233])
    doc.save(DST)
    print(DST)


if __name__ == "__main__":
    main()
