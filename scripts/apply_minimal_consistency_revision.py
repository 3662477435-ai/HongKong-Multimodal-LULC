"""Apply a minimal, evidence-bounded consistency revision to the current DOCX."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


SRC = Path(r"<third_party_sources>\融合多源遥感与城市语义的条件层级概率融合香港土地利用土地覆盖制图_数据谱系与公平比较扩展版.docx")
DST = Path(r"<third_party_sources>\融合多源遥感与城市语义的条件层级概率融合香港土地利用土地覆盖制图_最小一致性修订版.docx")


def replace(paragraph, text: str) -> None:
    rpr = deepcopy(paragraph.runs[0]._r.rPr) if paragraph.runs and paragraph.runs[0]._r.rPr is not None else None
    paragraph.clear()
    run = paragraph.add_run(text)
    if rpr is not None:
        run._r.insert(0, rpr)


def find_paragraph(doc: Document, needle: str):
    for paragraph in doc.paragraphs:
        if needle in paragraph.text:
            return paragraph
    raise KeyError(needle)


def mark_header(row) -> None:
    pr = row._tr.get_or_add_trPr()
    flag = OxmlElement("w:tblHeader")
    flag.set(qn("w:val"), "true")
    pr.append(flag)


def make_table(doc: Document, headers: list[str], rows: list[list[str]]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    mark_header(table.rows[0])
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value
    return table


def replace_table(old, new) -> None:
    old._tbl.addprevious(new._tbl)
    old._element.getparent().remove(old._element)


def add_s4(doc: Document, anchor) -> None:
    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.add_run("表S4  条件层级概率路径、校准指标与类别级概率门控（固定种子42检查点）").bold = True
    anchor._p.addnext(caption._p)
    outputs = make_table(
        doc,
        ["输出", "OA", "macro mIoU", "macro-F1", "Brier", "ECE（15 bins）"],
        [
            ["Direct Pᵈ", "0.8787", "0.5947", "0.7338", "0.1801", "0.0323"],
            ["Hierarchical Pʰ", "0.8768", "0.5729", "0.7146", "0.1819", "0.0356"],
            ["Final P", "0.8789", "0.5947", "0.7339", "0.1797", "0.0323"],
        ],
    )
    caption._p.addnext(outputs._tbl)
    gate_caption = doc.add_paragraph("类别级融合系数 αc（Final P = (1−αc)Pᵈ + αcPʰ）：")
    outputs._tbl.addnext(gate_caption._p)
    gates = make_table(
        doc,
        ["类别", "αc", "类别", "αc"],
        [
            ["住宅", "0.0014", "工业仓储", "0.0008"],
            ["公共机构与休闲", "0.0345", "交通设施", "0.0035"],
            ["其他建设用地", "0.0262", "农业", "0.0018"],
            ["林地/灌丛/草地/湿地", "0.0421", "水体", "0.0021"],
        ],
    )
    gate_caption._p.addnext(gates._tbl)
    note = doc.add_paragraph(
        "注：均在原始固定128×128空间阻隔测试patch上计算。Final P相对Direct P的校准指标仅有小幅改善，"
        "且各αc较小；因此这些结果支持将层级路径视为受约束的概率组织，而不支持将其表述为最终精度提升的主要来源。"
    )
    gates._tbl.addnext(note._p)


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    shutil.copy2(SRC, DST)
    doc = Document(DST)

    replace(doc.paragraphs[2],
        "高密度城市土地利用/土地覆盖制图需要同时识别自然地表物理属性与建成区使用功能，但单一遥感影像难以区分光谱相似的城市功能类别，"
        "开放地理数据的空间异质性、数据谱系相关性和类别不平衡也会降低模型泛化能力。本文面向香港全域10 m输出网格，构建融合Sentinel-1/2、"
        "高分辨率影像、VIIRS夜间灯光、建筑功能与高度、OSM道路、POI、AOI及文本语义的多模态FCNN。模型以直接分类路径保持逐像元分割，"
        "以城市上下文残差将局地POI、AOI、道路、建筑和社会经济信息再次注入类别logit，并以条件层级概率组织建成区、自然地表和水体之间的类别关系，"
        "最后通过类别级全局概率门控融合两条路径。空间阻隔主测试集上，种子42模型取得0.8789的总体精度、0.5946的macro mIoU和0.7338的macro-F1；"
        "同等47 epoch训练轮数下，FCNN的OA为0.8794、macro mIoU为0.5885、macro-F1为0.7295。因此，本文模型的主证据是类别均衡指标的最高点估计，"
        "而非所有总体指标均优。经结构消融的Holm族内校正，去除城市上下文残差后的macro mIoU下降0.0210仍达到0.05水平；"
        "去除条件层级分类和以共享门控替代类别门控的差异未达校正后显著性。完整128×128输入的Expected Gradients结果、"
        "建筑功能与建筑形态拆分消融及直接48通道预训练主干适配补充实验用于界定数据谱系、概率组织和网络结构的作用边界。"
    )

    # Preserve the original modality-group ablation, while preventing it from
    # being interpreted as the fixed-geometry conditional building effect.
    replace(doc.paragraphs[226],
        "基于表5，本文模型从仅Sentinel-2扩展至全模态后，OA由0.8253提高至0.8789，macro mIoU由0.4392提高至0.5946，"
        "macro-F1由0.5874提高至0.7338。原模态组消融中，联合删除POI、AOI、道路与OSM文本时，OA、macro mIoU和macro-F1分别下降"
        "0.0319、0.1111和0.1034，表明该联合删除会造成显著性能退化。该原模态组实验中，去除全部建筑信息对应的mIoU变化为-0.0151，"
        "但该操作同时改变了输入组合/转码结构，因此包含信息删除与输入结构变化的共同影响。为隔离建筑信息本身的条件边际作用，后续固定输入几何的表S5"
        "应作为建筑贡献的主要依据：全部14个建筑通道置零时，macro mIoU为0.5916，相对完整输入下降0.0030。"
    )
    replace(doc.paragraphs[228],
        "不同模态的效应呈现明显的结构依赖性。去除夜间灯光使FCN-ResNet50的macro mIoU下降0.1261；去除高分辨率影像使CNN、FCNN、Transformer和"
        "FCN-ResNet50分别下降0.0260、0.0198、0.0312和0.0606。原模态组实验中“去全部建筑”的-0.0151同时含输入结构变化，"
        "不用于解释建筑信息的条件边际作用；固定95→96→48转码头的谱系拆分见表S5。相反，VGG-UNet-Transformer在去夜间灯光或去高分辨率影像后"
        "macro mIoU分别上升0.0545和0.0192，去除Sentinel-1也小幅上升0.0028。该非单调结果说明全模态配置并非对所有结构均最优，"
        "并可能受到特征冗余、信息配准与训练稳定性的共同影响；不能据此把单一来源解释为可独立替代的因果贡献。"
    )
    replace(doc.paragraphs[230],
        "完整模型测试OA、mIoU和macro-F1分别为0.8789、0.5946和0.7338（表6）。去除条件层级概率组织后，mIoU下降0.0093；"
        "去除城市上下文残差后mIoU下降0.0210；将类别级概率融合替换为共享门控后mIoU变化为-0.0040。三项结构比较在同一mIoU族内进行Holm校正后，"
        "仅城市上下文残差的退化仍达到0.05显著性（原始p=0.0140，Holm校正p=0.0420）。因此，城市上下文残差是当前数据上得到最强统计支持的结构贡献；"
        "条件层级概率和类别级概率融合的证据应理解为方向一致的结构化组织贡献，而非其单独带来决定性性能提升的证明。"
        "冻结最终检查点的路径分解、类别级门控参数和概率校准指标见表S4。"
    )
    replace(doc.paragraphs[239],
        "固定种子42的三次训练稳定性实验表明，本文模型的OA、mIoU和macro-F1分别为0.8772 ± 0.0019、0.5866 ± 0.0071和0.7264 ± 0.0065（均值 ± SD）。"
        "FCNN的mIoU为0.5776 ± 0.0087，Transformer为0.5672 ± 0.0090。本文模型相对FCNN的mIoU平均差为+0.0091，95% t区间为-0.0125至0.0306，"
        "Holm校正p=0.2538。由于n=3时区间较宽且检验功效有限，这些结果应被视为训练稳定性的探索性证据，而不应用于宣称跨随机种子的显著模型优势。"
    )
    replace(doc.paragraphs[212],
        "为将FCNN-47纳入可比比较，本文以统一协议重新计算所有模型的Boundary F1：在每个128×128测试patch内提取类别内侧形态学边界，"
        "使用1像元（10 m）膨胀容忍度匹配预测和参考边界，并在八类上作macro平均。本文模型的macro Boundary F1为0.5832，"
        "高于FCNN-47（0.5763）、FCNN-20（0.5518）、CNN（0.5347）、Transformer（0.5096）、VGG-UNet-Transformer（0.4407）"
        "和FCN-ResNet50（0.3321）（表S3）。该指标仅反映预测与BLU监督参考标签之间的边界一致性，也可能受参考标签边界概化影响。"
    )
    replace(doc.paragraphs[261],
        "公平性补充实验改变了本文对主结果的表述边界。原20 epoch基线下，种子42的本文模型相对FCNN、CNN和FCN-ResNet50等模型具有更高点估计；"
        "但FCNN延长至47 epoch后，其OA达到0.8794，略高于本文模型0.8789。本文模型的优势应限定为macro mIoU和macro-F1等类别均衡指标上的最高点估计，"
        "而非所有应用目标下的绝对最优。47 epoch控制的是训练轮数而非严格等GPU时间或等优化步；直接48通道预训练主干适配补充实验与本文模型使用相同的"
        "95→96→48特征转码头、数据划分、采样、增强、损失和验证选择规则，用于检验较早的48→3通道适配是否引入信息瓶颈。"
    )
    replace(doc.paragraphs[258],
        "FCNN-47的单次训练GPU墙钟时间为41.80 min；FCN-ResNet50直接48通道与VGG直接48通道的单次训练GPU墙钟时间分别为22.35 min和21.35 min。"
        "本文模型的历史分阶段训练未保存可与上述实验一一对应的完整GPU墙钟日志，因此本文不将47 epoch表述为严格等GPU时间或等优化步预算，"
        "而仅将其表述为相同最大训练轮数下的比较。推理时间还受到数据拼接、OSM文本编码和批量大小影响；本文仅将其作为相同硬件条件下的前向效率参考，"
        "不据此推断端到端制图流水线耗时。"
    )
    replace(doc.paragraphs[268],
        "本研究仍受数据覆盖、验证范围和时相相关的局限。三随机种子仅提供有限的训练稳定性证据，且df=2时置信区间较宽；"
        "leave-region-out结果提示模型在区域分布偏移时泛化仍会降低。BLU、OSM、夜间灯光和建筑数据并非完全同日，高分辨率RGB主要为2021年Google Earth影像，"
        "因此可能引入过时形态线索；其去除实验的mIoU变化为-0.0149，说明该源不是唯一决定因素，但不能消除时相差异风险。"
        "全港制图采用128×128滑窗并写入中心96×96像元，而表3报告的是固定测试patch协议下的参考标签一致性；"
        "相邻滑窗的OSM文本可能复用原始manifest文本，且未对每个滑窗重新构造。因此表3精度不应被严格等同于图4的部署推理精度，"
        "图4应作为滑窗制图展示。Expected Gradients仅覆盖80个类别均衡样本，且加性残差中位数为0.374，应视为模型敏感性而非全港总体因果归因。"
    )
    replace(doc.paragraphs[270],
        "本研究面向香港全域10 m输出网格，构建由多模态FCNN空间特征转码、城市上下文残差、条件层级概率组织和类别级概率融合组成的预测框架。"
        "固定空间阻隔主测试集上，种子42模型取得OA=0.8789、macro mIoU=0.5946和macro-F1=0.7338；同等47 epoch训练轮数下，"
        "FCNN取得略高OA（0.8794），但本文模型的macro mIoU和macro-F1点估计更高。结构消融中，去除城市上下文残差的mIoU退化最大，"
        "原始配对p=0.0140、经三项结构比较的Holm校正后p=0.0420；条件层级概率组织和类别级融合呈方向一致但未达校正后显著的增益。"
        "AOI的单项退化点估计最大，但其Holm校正p约为0.0508；联合OSM删除在Holm校正后仍显著。本文的实证贡献因此限定为："
        "在当前香港空间阻隔协议下，城市上下文残差与多源物理遥感特征的协同使用具有可检验价值，条件层级概率提供受约束的输出组织。"
        "这些结论仍受训练预算、空间泛化、异时相辅助影像、建筑数据谱系和滑窗部署协议差异限制，需要在更多城市和时相下进一步验证。"
    )

    # Table 1: do not mistake publication year for the observation date.
    doc.tables[0].cell(8, 2).text = "Li et al. (2026)产品；单栋观测日期不可统一确定（多源异质时相）"

    # Table 3: Kappa is removed because it is not part of the defined evaluation suite.
    old = doc.tables[2]
    replacement = make_table(doc,
        ["模型", "输入/训练设置", "OA", "macro mIoU", "macro-F1", "最佳epoch"],
        [
            ["本文模型", "全模态层级融合", "0.8789", "0.5946", "0.7338", "联合阶段3"],
            ["FCNN", "原20 epoch", "0.8733", "0.5796", "0.7211", "20"],
            ["FCNN", "同等47 epoch", "0.8794", "0.5885", "0.7295", "37"],
            ["Transformer", "原20 epoch", "0.8714", "0.5775", "0.7185", "18"],
            ["CNN", "原20 epoch", "0.8679", "0.5698", "0.7123", "19"],
            ["FCN-ResNet50", "48→3通道", "0.8189", "0.4239", "0.5724", "15"],
            ["FCN-ResNet50", "直接48通道", "0.8124", "0.4143", "0.5647", "16"],
            ["VGG-UNet-Transformer", "48→3通道", "0.8233", "0.4255", "0.6351", "18"],
            ["VGG-UNet-Transformer", "直接48通道", "0.8640", "0.5467", "0.6940", "18"],
        ])
    replace_table(old, replacement)

    # Structural and OSM tables show the complete family-wise inference.
    old = doc.tables[4]
    replacement = make_table(doc,
        ["结构变体", "OA", "mIoU", "macro-F1", "ΔmIoU", "Raw p", "Holm校正p"],
        [
            ["本文模型", "0.8789", "0.5946", "0.7338", "+0.0000", "—", "—"],
            ["无层级概率", "0.8747", "0.5853", "0.7254", "-0.0093", "0.1129", "0.1480"],
            ["无城市上下文残差", "0.8718", "0.5736", "0.7160", "-0.0210", "0.0140", "0.0420"],
            ["共享门控（替代类别门控）", "0.8780", "0.5906", "0.7301", "-0.0040", "0.0740", "0.1480"],
        ])
    replace_table(old, replacement)

    old = doc.tables[6]
    replacement = make_table(doc,
        ["OSM变体", "OA", "mIoU", "macro-F1", "ΔmIoU", "Raw p", "Holm校正p"],
        [
            ["全模态", "0.8789", "0.5946", "0.7338", "+0.0000", "—", "—"],
            ["去POI", "0.8782", "0.5921", "0.7314", "-0.0025", "0.2598", "0.2652"],
            ["去AOI", "0.8614", "0.5290", "0.6733", "-0.0656", "0.0127", "0.0508"],
            ["去道路", "0.8724", "0.5761", "0.7166", "-0.0185", "0.0924", "0.2652"],
            ["去OSM文本", "0.8731", "0.5745", "0.7149", "-0.0201", "0.0884", "0.2652"],
            ["联合去POI/AOI/道路/文本", "0.8470", "0.4835", "0.6304", "-0.1111", "0.0083", "0.0415"],
        ])
    replace_table(old, replacement)

    # S3 is recomputed for every model under one explicit protocol, including
    # the equal-horizon FCNN, rather than combining historically incomparable values.
    old = doc.tables[13]
    replacement = make_table(doc,
        ["模型", "macro Boundary F1（1像元/10 m容忍）"],
        [
            ["本文模型", "0.5832"],
            ["CNN", "0.5347"],
            ["FCNN（20 epoch）", "0.5518"],
            ["FCNN（47 epoch）", "0.5763"],
            ["Transformer", "0.5096"],
            ["FCN-ResNet50", "0.3321"],
            ["VGG-UNet-Transformer", "0.4407"],
        ])
    replace_table(old, replacement)

    # The current document has no S4. Add it directly after the structural interpretation.
    add_s4(doc, doc.paragraphs[230])
    doc.save(DST)
    print(DST)


if __name__ == "__main__":
    main()
