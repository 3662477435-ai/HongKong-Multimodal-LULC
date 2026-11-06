"""Create a clean, minimally revised ISPRS-JPRS manuscript copy.

The source manuscript is never overwritten. The script only replaces targeted
paragraphs, expands the reference list, and appends two supplementary tables
whose values are taken from completed experiments.
"""

from copy import deepcopy
from pathlib import Path
import shutil

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt


SRC = Path(r"<third_party_sources>\融合多源遥感与城市语义的条件层级概率融合香港土地利用土地覆盖制图.docx")
DST = Path(r"<third_party_sources>\融合多源遥感与城市语义的条件层级概率融合香港土地利用土地覆盖制图_城市上下文残差与条件层级概率修订版.docx")
CHANGELOG = Path(r".\outputs\full\eight_class_fairness_supplement_20260802\manuscript_revision_change_log.md")


def replace_preserving_format(paragraph, text):
    """Replace text while retaining the paragraph properties and first run style."""
    rpr = None
    if paragraph.runs and paragraph.runs[0]._r.rPr is not None:
        rpr = deepcopy(paragraph.runs[0]._r.rPr)
    paragraph.clear()
    run = paragraph.add_run(text)
    if rpr is not None:
        run._r.insert(0, rpr)


def set_cell(cell, text, bold=False, size=8.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_table(doc, rows, widths_cm, template_style):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = template_style
    table.autofit = False
    for ridx, row_data in enumerate(rows):
        for cidx, value in enumerate(row_data):
            cell = table.cell(ridx, cidx)
            cell.width = Cm(widths_cm[cidx])
            set_cell(cell, value, bold=(ridx == 0), size=8.0 if len(rows[0]) > 5 else 8.5)
    mark_header_row(table)
    return table


def mark_header_row(table):
    """Mark the first table row as a repeated/header row for Word and readers."""
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "true")
    tr_pr.append(header)


def add_reference_before(anchor, text):
    p = anchor.insert_paragraph_before(text)
    p.style = anchor.style
    return p


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    shutil.copy2(SRC, DST)
    doc = Document(DST)

    replacements = {
        0: "城市上下文残差与条件层级概率融合的香港土地利用/土地覆盖制图",
        2: (
            "高密度城市土地利用/土地覆盖制图需要同时识别自然地表物理属性与建成区使用功能，"
            "但单一遥感影像难以区分光谱相似的城市功能类别，开放地理数据的空间异质性和类别不平衡也会降低模型泛化能力。"
            "本文面向香港全域10 m输出网格，构建融合Sentinel-1/2、高分辨率影像、VIIRS夜间灯光、建筑功能与高度、"
            "OSM道路、POI、AOI及文本语义的多模态FCNN。模型以直接分类路径保持逐像元分割，"
            "以城市上下文残差将局地POI、AOI、道路、建筑和社会经济信息再次注入类别logit，"
            "并以条件层级概率组织建成区、自然地表和水体之间的类别关系，最后通过类别级全局概率门控融合两条路径。"
            "固定空间分块测试集上，种子42模型取得0.8789的总体精度、0.5946的macro mIoU和0.7338的macro-F1；"
            "同等47 epoch训练预算下，FCNN的OA为0.8794、macro mIoU为0.5885、macro-F1为0.7295。"
            "因此，本文模型的证据主要体现为类别均衡指标的最高点估计，而非所有总体指标均优。"
            "结构消融中，去除城市上下文残差使macro mIoU下降0.0210（p=0.0140），"
            "去除条件层级分类和以共享门控替代类别门控分别下降0.0093和0.0040，当前空间组检验中未达0.05显著性。"
            "在1像元（10 m）容忍距离下，本文模型的macro Boundary F1为0.4915，高于五种原始全模态基线。"
            "完整128×128输入的Expected Gradients结果表明，类别均衡解释样本中Sentinel-2占栅格模态归一化平均绝对归因的49.82%，"
            "OSM道路、Sentinel-1、高分辨率影像和AOI分别占12.36%、11.33%、11.13%和8.82%；"
            "文本潜变量单独报告，不与栅格模态归一化比较。上述结果表明，城市上下文残差与多源城市语义是本模型获得类别均衡增益的主要实证支撑，"
            "而条件层级概率提供了与LULC语义一致的结构化决策组织。"
        ),
        3: "关键词：土地利用/土地覆盖；城市上下文残差；条件层级概率融合；多模态FCNN；OpenStreetMap；香港",
        6: (
            "近年来，深度学习推动了遥感LULC分类由像元光谱判别向空间上下文建模发展。卷积神经网络及其编码器—解码器结构能够联合提取局部纹理与多尺度空间特征，"
            "Transformer则进一步增强了模型对长距离依赖和全局场景关系的刻画能力[1–3, 27–29, 34, 35]。"
            "这些方法显著提升了植被、水体、裸地和建成地表等物理覆盖类型的识别能力；然而，当不同城市功能共享相似光谱和空间形态时，"
            "仅增加影像特征提取深度仍难以消除功能语义歧义。"
        ),
        7: (
            "多源遥感融合为缓解光谱混淆提供了新的技术路径。Sentinel-2能够提供可见光、红边、近红外和短波红外信息，"
            "Sentinel-1合成孔径雷达则对地表结构、粗糙度和含水状态敏感，两者构成互补的物理观测[4, 18–21]。"
            "光学、SAR和高分辨率影像主要回答“地表是什么”，对于“土地被如何使用”的判别能力仍然有限。"
            "住宅、工业和公共服务设施之间的差异往往更多体现在建筑高度与密度、道路组织、夜间活动强度以及周边设施构成，"
            "而非单一像元的光谱响应。"
        ),
        8: (
            "因此，城市功能识别逐渐将遥感影像与POI、建筑轮廓、道路和其他地理大数据结合[5–10, 25, 31]。"
            "这些数据能够补充二维影像难以表达的设施语义、三维形态与交通组织，但也带来覆盖不均、标签噪声和时相不一致等问题。"
            "因此，城市语义数据更适合作为辅助特征和条件上下文，而不是直接替代官方监督标签。"
        ),
        9: (
            "现有研究仍面临三个限制。首先，多数城市功能区研究以街区、建筑或规则网格为分析单元，较少在统一逐像元框架内同时覆盖城市功能与植被、农业和水体等自然地表。"
            "其次，平铺分类未显式利用“物理覆盖—条件功能”的类别关系；硬级联又会在上游误判时切断下游路径[22]。"
            "再次，POI、AOI和OSM道路等开放地理数据具有覆盖不均、时相不一致和标签噪声，随机划分还可能因空间自相关而高估泛化精度[11, 23–25, 32, 33]。"
        ),
        10: (
            "针对上述问题，本文构建以多模态FCNN为空间转码主干的香港全域10 m八类LULC制图框架。"
            "其核心不是将所有特征简单堆叠后依赖单一深度网络完成融合，而是以两步方式协调物理与语义信息："
            "首先，将Sentinel-2、Sentinel-1、高分辨率RGB、VIIRS、建筑、道路及POI/AOI栅格转码为统一的像元级空间特征；"
            "随后，利用城市上下文残差将局地POI/AOI、道路、建筑形态和社会经济信息重新注入基础FCNN的类别logit，"
            "使城市功能语义在深层卷积后仍可直接参与分类修正。"
            "在此基础上，模型以建设地表、自然地表和水体的条件层级概率组织八类叶节点，并用类别级全局概率门控融合直接路径与层级路径。"
            "因此，本文的贡献定位为：在统一逐像元LULC体系中引入可复核的城市上下文残差，以及与物理—功能语义一致的条件层级概率组织；"
            "并通过结构、单项OSM、训练稳定性、空间协议和边界质量实验界定各组成的作用边界。"
        ),
        13: (
            "雷达数据采用 Sentinel-1 GRD产品（COPERNICUS/S1_GRD），筛选 IW 成像模式、VV/VH双极化及10 m像素间距的数据[19]。"
            "以选定的Sentinel-2成像日期为基准，在前后7天内优先选择有效覆盖率不低于95%的Sentinel-1数据；若无合适影像，"
            "则将时间窗口扩展至前后15天，最终选定时间为2025-01-12 10:25:26。同一日期的多景Sentinel-1影像进行拼接，"
            "并保留VV、VH及VV−VH差值特征。Sentinel-1与Sentinel-2影像最终统一投影至WGS 84/UTM Zone 50N（EPSG:32650），"
            "以10 m空间分辨率导出，并完成轨道文件更新、边界与热噪声去除、辐射定标及基于DEM的地形校正，输出为dB单位的后向散射系数。"
            "处理链未另行施加独立斑点滤波或单景几何的叠掩／阴影掩膜；因此，香港山地区域残余的SAR几何畸变与斑点不确定性仍是本研究的输入数据局限。"
        ),
        15: (
            "高分辨率光学影像来源于Google Earth影像服务，研究区影像界面显示的主要获取时间为2021年。"
            "本文使用覆盖香港研究区的RGB三通道影像补充建筑边界、道路纹理、地块形态和建成区内部空间细节，"
            "并经研究区裁剪、几何配准和重采样后对齐至EPSG:32650的10 m输出网格。"
            "该影像与2025年Sentinel-1/2存在时相差异，且可能由不同供应商和局部时相数据拼接形成，"
            "因此仅作为形态与纹理辅助输入，不将其视为同日、统一辐射定标的标准遥感产品。"
        ),
        16: (
            "模型监督标签来源于香港特别行政区政府空间数据共享平台CSDI Geoportal[17]发布的香港土地用途栅格（Raster Grids on Land Utilization，本文简称BLU标签）。"
            "原始标签经类别编码核对、最近邻重投影和空间对齐后转换至EPSG:32650的10 m输出网格。"
            "本文将任务限定为八类：住宅、工业仓储、公共机构与休闲、交通设施、其他建设用地、农业、林地/灌丛/草地/湿地和水体（表S1）。"
            "商业办公（BLU=11）与荒地/岩岸（BLU=81、83）在有标签像元中分别占0.424%和0.558%。"
            "它们被排除是八类研究范围的预先定义，而不是“仅因像元数少而无法训练”的技术断言：商业办公与其他高密度建成区存在显著功能混合，"
            "荒地/岩岸同时包含施工裸地、裸岩与海岸等内部异质子类；二者若进入模型，需要独立设计稀有类采样、标签细化和空间组平衡评价。"
            "为使该范围选择透明，表S2报告两类在训练、验证和测试中的像元与独立空间组支持。"
            "在本八类实验中，它们与BLU=0均重编码为ignore index，不参与监督损失或精度计算。"
            "OSM AOI、POI及建筑功能中的商业办公标签仅作为外部城市语义输入，而非BLU监督标签或预测目标；"
            "商业办公像元本身不产生类别监督损失，也不计入八类精度评价。"
        ),
        17: (
            "开放街图（OpenStreetMap，OSM）数据用于补充遥感影像难以直接反映的城市功能语义与交通结构信息，包括兴趣点（POI）、"
            "功能区面状对象（AOI）和道路。研究范围由香港 Sentinel-2 影像的空间边界确定，并通过 Overpass API 提取范围内的OSM节点、道路和关系对象。"
            "OSM属于志愿地理信息，其完整性与属性质量会随区域和要素类型变化[25, 31]，因此本文将其作为辅助输入而非监督真值。"
            "为减小多源数据的时相差异，OSM历史快照设为最终选定的Sentinel-2成像日期（2025-01-13）。"
        ),
        56: (
            "图1中深度网络的作用是特征转码，而不是以黑箱方式完成全部多模态融合。本文设置空间栅格编码器和OSM文本编码器两个深度特征编码网络："
            "空间栅格编码器以FCNN为主体，将多源连续栅格转化为与输入空间对应的多尺度像元特征[13, 27]；"
            "OSM文本编码器以Transformer将patch内的POI/AOI标签与属性序列转化为上下文向量[14]。"
            "第一次融合发生在95通道输入的卷积转码层，核心的第二次融合则发生在后续城市上下文残差模块："
            "局地POI/AOI、道路以及建筑—社会经济特征被分别重新编码，再与FCNN浅层特征共同生成类别logit残差。"
            "条件层级概率和类别级概率融合在该直接路径上进一步组织输出关系。"
        ),
        65: (
            "由于文本特征对应整个128×128 patch，模型将q广播为16×H×W的空间特征图，使其可与像元级栅格特征共同输入空间分类网络。"
            "该文本表示提供的是patch尺度的上下文语义，而不是每个10 m像元各自独立的语义标签；"
            "像元级位置差异主要由POI计数、AOI one-hot、道路和其他栅格特征表达。"
        ),
        75: (
            "其中，表示卷积、GroupNorm和GELU激活。1×1卷积学习不同数据源之间的通道组合，3×3卷积进一步提取局部跨模态空间关系。"
            "该早期转码层实现第一次特征融合；其后通过城市上下文残差进行第二次、面向类别logit的语义融合。"
        ),
        85: (
            "仅依赖统一输入转码时，光谱和纹理特征可能在深层卷积中占据主导，使POI、AOI、建筑形态、夜间灯光和道路等城市功能线索被弱化。"
            "为此，本文设置城市上下文残差：它不替代基础FCNN，而是在直接分类logit形成前将局地城市语义转化为可学习的加性校正。"
        ),
        97: (
            "因此，POI/AOI、道路、建筑、夜间灯光和高分辨率形态信息既参与第一次全模态特征转码，又通过城市上下文残差对直接分类结果进行第二次类别相关修正。"
            "这一模块是本文物理—城市语义融合的主要显式机制，其条件边际作用由“无语义残差”重训练消融检验。"
        ),
        98: "3.2.5 条件层级概率组织",
        106: (
            "水体没有额外条件分类头。上述概率链式分解不使用硬阈值，因此粗分类的不确定性会以连续概率传递到叶类别。"
            "本文将该输出称为“条件层级概率”或“层级链式概率”，强调其对类别关系的组织作用，而不将其与类别级概率融合混称。"
        ),
        107: "3.2.6 类别级概率融合",
        116: (
            "再通过logsumexp归一化。最终八类门控值由训练过程学习得到。这些门控是类别级、训练集共享的全局标量，"
            "不是输入模态注意力、像元自适应门控或mixture-of-experts式路由；因此本文将最终输出称为“类别级概率融合”。"
            "直接全模态FCNN仍是主要像元预测路径，条件层级分支提供结构化概率信息。"
        ),
        120: (
            "因此，相邻滑动窗口可能复用同一原始patch文本。中心裁剪能够减弱栅格卷积的边缘效应，但不能完全消除patch级文本复用带来的局部依赖。"
            "该文本应解释为窗口上下文而非像元局部证据；未来可采用随推理窗口动态重建文本、局部跨注意力或重叠预测融合来进一步减少窗口依赖。"
        ),
        155: (
            "完整训练预算包括FCNN 20 epoch、条件层级分支8 epoch、城市上下文残差与类别级概率融合12 epoch以及联合微调7 epoch，共47 epoch。"
            "因此，本文模型与仅训练20 epoch的基线之间的差异同时包含结构和优化预算影响，不能表述为“仅由网络结构造成”。"
            "FCNN-47用于控制阶段数预算，但本文模型的历史分阶段运行未保留可按同一口径汇总的完整墙钟日志，"
            "故两者不被表述为严格等GPU时间比较；模型端FLOPs和推理速度另行报告。"
            "随机种子控制Python random、NumPy、PyTorch CPU/CUDA及加权采样器，但未强制所有CUDA算子采用确定性实现。"
        ),
        157: (
            "为区分条件层级概率组织、城市上下文残差和类别级概率融合的贡献，本文采用一次改变一个结构组成的重训练消融。"
            "“无层级分类”令最终输出等于直接分类概率；“无城市上下文残差”令直接logit等于基础FCNN输出；"
            "“共享门控”以一个全局标量替代八个类别标量。各结构变体保持输入、FCNN初始化、数据划分和检查点选择规则一致。"
            "差异表示相应组成在其他模块存在条件下的边际效应，不解释为彼此独立可加的因果贡献。"
            "为回应训练预算公平性问题，本文另将FCNN在相同数据划分和模型选择规则下延长至47 epoch，并将其作为主要公平性补充基线。"
        ),
        164: (
            "本文模型、CNN、FCNN、Transformer、FCN-ResNet50和VGG-UNet-Transformer的可训练参数量分别为4.788、0.829、3.915、1.515、33.474和31.758 M。"
            "不同模型的batch size根据显存调整。本文模型在20 epoch FCNN检查点基础上继续分阶段优化，总预算为47 epoch；"
            "因此主结果同时报告原20 epoch基线和同等47 epoch FCNN补充基线。"
            "该补充控制了阶段数，但不构成严格等墙钟或等优化步比较；跨模型结果仍受到架构、参数量、预训练方式和多模态适配策略影响。"
            "共享FCNN初始化和训练流程的结构消融更适合判断本文新增模块的条件边际作用。"
        ),
        195: (
            "除总体指标外，本文还报告逐类IoU、F1、生产者精度、用户精度和行归一化混淆矩阵。"
            "行归一化混淆矩阵以真实类别为分母，用于识别住宅—工业仓储、农业—林地以及水体—植被等主要误差路径。"
            "为补充面积型指标对边界误差不敏感的问题[26]，本文还在测试patch内计算1像元（10 m）容忍距离下的类别宏平均Boundary F1；"
            "该指标评价预测边界与BLU参考边界的一致性，不等同于独立实测边界精度。"
        ),
        206: (
            "在统一全模态输入、数据划分和评价流程下，本文模型取得OA=0.8789、macro mIoU=0.5946和macro-F1=0.7338（表3）。"
            "相较原20 epoch FCNN，本文模型的OA、macro mIoU和macro-F1分别提高0.0056、0.0150和0.0127。"
            "当FCNN训练预算延长至47 epoch后，其OA达到0.8794，较本文模型高0.0005；"
            "本文模型的macro mIoU和macro-F1仍分别高0.0061和0.0043。"
            "因此，主结果支持本文模型在当前数据与空间分块测试划分下取得最高类别均衡性能点估计，"
            "而不支持“所有总体指标均最高”的表述。"
        ),
        207: (
            "基于14个空间测试组的精确配对置换检验并对五个原始基线实施Holm校正后，本文模型相对于CNN和原20 epoch FCNN的macro mIoU优势达到0.05显著性水平"
            "（校正后p分别为0.0098和0.0421），macro-F1差异亦显著（p分别为0.0103和0.0406）。"
            "相对于FCN-ResNet50，OA、macro mIoU和macro-F1差异均显著。"
            "本文模型与Transformer之间三项指标差异未显著，与VGG-UNet-Transformer的差异在Holm校正后也未达到0.05水平。"
            "同等47 epoch FCNN补充实验进一步说明，训练预算会影响OA排序。"
            "因此，本文将多模型结果定位为类别均衡性能、计算代价和适配方式之间的比较，而不作对所有基线均显著优越的泛化宣称。"
        ),
        212: (
            "行归一化混淆矩阵（图5）显示，本文模型对林地／灌丛／草地／湿地、水体、公共机构与休闲、交通设施和住宅的召回率分别为0.9430、0.8288、0.7689、0.7465和0.7279。"
            "主要误差路径集中在相近地表或功能类别之间：27.97%的农业像元被分为林地／灌丛／草地／湿地，17.58%的工业仓储被分为其他建设用地，"
            "12.29%的水体被分为林地／灌丛／草地／湿地，10.22%的其他建设用地被分为林地／灌丛／草地／湿地。"
            "在1像元（10 m）容忍距离下，本文模型的macro Boundary F1为0.4915，高于CNN（0.4372）、FCNN（0.4537）、"
            "Transformer（0.3954）、FCN-ResNet50（0.2145）和VGG-UNet-Transformer（0.4017）（表S3）。"
            "该补充结果表明本文模型的改进并非只来自大面积类别的区域重叠，也体现在参考标签边界附近的类别一致性。"
        ),
        230: (
            "完整模型测试OA、mIoU和macro-F1分别为0.8789、0.5946和0.7338（表6）。"
            "去除条件层级概率组织后，mIoU下降0.0093（配对置换p=0.1129）；去除城市上下文残差后mIoU下降0.0210（p=0.0140）；"
            "将类别级概率融合替换为共享门控后mIoU变化为-0.0040（p=0.0740）。"
            "因此，三种组成均呈正向点估计，其中城市上下文残差具有当前空间组检验下最强的统计支持。"
            "条件层级概率与类别级概率融合的证据应理解为方向一致的结构化组织贡献，而非其单独带来决定性性能提升的证明。"
        ),
        233: (
            "四种OSM信息分别删除并重新训练后，最大的单项mIoU退化来自去除AOI（-0.0656，p=0.0127）；"
            "去除POI、道路和OSM文本的变化分别为-0.0025、-0.0185和-0.0201。"
            "联合删除POI、AOI、道路与文本时mIoU下降0.1111。"
            "由于这四种信息在空间与语义上相关，联合下降不应被线性分摊为各自独立贡献。"
            "在当前模型与数据条件下，AOI提供了最强的单项边际证据；道路与文本的方向性贡献需要与其空间上下文角色一起解释，而POI单项效应较弱。"
        ),
        245: (
            "为探索最终分类网络对输入信息的局部依赖，本文对种子42的层级多模态FCNN检查点实施基于SHAP GradientExplainer的Expected Gradients。"
            "解释目标为中心像元真实类别的最终类别级概率融合对数概率。补充后的程序使用完整128×128 patch输入模型，"
            "仅在归因汇总时统计中心49×49区域；解释样本从八类测试区中每类抽取10个，共80个，背景分布由8个训练样本构成，nsamples=256。"
            "79个栅格通道在原始输入空间内归一化汇总，16维OSM文本潜变量独立报告。"
        ),
        251: (
            "按80个类别均衡测试样本的mean |Expected Gradients|汇总（表10），Sentinel-2在栅格模态归一化贡献中占49.82%（95% CI：45.85%–53.67%），"
            "仍是最主要的栅格信息来源。OSM道路、Sentinel-1、高分辨率影像和AOI分别占12.36%、11.33%、11.13%和8.82%，"
            "夜间灯光、建筑特征和POI分别占2.91%、2.85%和0.78%。OSM文本潜变量的类别均衡平均绝对归因为0.925，"
            "但该数值位于模型内部连续投影空间，不能解释为百分比，也不能与栅格模态的数值比较。"
            "当前加和性残差绝对值中位数为0.374，且样本量仅为80，因此结果仅描述类别均衡解释样本上的相对敏感性；"
            "更大样本量或不同背景集下的归因稳定性仍有待后续评估。"
        ),
        256: (
            "补充训练的单GPU墙钟时间分别为：FCNN-47为41.80 min，FCN-ResNet50直接48通道为22.35 min，"
            "VGG直接48通道为21.35 min。本文模型的历史分阶段运行未保存可按相同口径汇总的完整墙钟日志，"
            "因此本文不将47 epoch称为严格等GPU时间预算，而将其作为控制训练阶段数的敏感性比较。"
            "推理计时不包含磁盘读取、滑窗拼接和后处理，表11反映的是模型端前向计算效率，而不是完整制图流水线耗时。"
        ),
        258: (
            "本文的核心贡献应从城市物理观测与功能语义的协调来理解，而非将条件层级概率单独视为决定性性能来源。"
            "基础FCNN将多源栅格转码为逐像元空间特征；城市上下文残差再将局地POI/AOI、道路、建筑与社会经济信息直接反馈至类别logit，"
            "从而避免城市功能线索仅在早期通道拼接中被吸收或削弱。"
            "结构消融显示，去除该残差带来0.0210的mIoU下降并达到空间组检验显著性；AOI单项消融和OSM联合消融也支持城市上下文信息的功能价值。"
            "条件层级概率则将建设、自然与水体的语义关系显式编码为连续概率链，避免硬级联对下游类别的截断。"
            "因此，本文将其定位为与城市上下文残差协同的结构化预测组织。"
        ),
        259: (
            "公平性补充实验改变了本文对主结果的表述边界。原20 epoch基线下，种子42的本文模型相对FCNN、CNN和FCN-ResNet50等模型具有更高点估计；"
            "但FCNN延长至47 epoch后，其OA达到0.8794，略高于本文模型0.8789。"
            "本文模型的优势应限定为macro mIoU和macro-F1等类别均衡指标上的最高点估计，而非所有应用目标下的绝对最优。"
            "同时，47 epoch控制的是训练阶段数而不是严格等墙钟时间或等优化步，完整墙钟日志缺失使计算公平性仍有边界。"
            "对于高类别不平衡的香港LULC任务，macro mIoU、macro-F1及补充的Boundary F1共同表明城市上下文残差在小类和边界附近带来的改进；"
            "但轻量FCNN在OA和速度上的表现仍使其成为重要的工程基线。"
        ),
        261: (
            "消融实验与完整128×128 Expected Gradients从互补角度揭示不同输入的作用。全部模态相较仅Sentinel-2使本文模型macro mIoU提高0.1554；"
            "OSM单项重训练中，去除AOI产生最大的单项下降（0.0656），联合去除POI、AOI、道路和文本下降0.1111。"
            "Expected Gradients显示模型仍以遥感物理观测为主要栅格敏感性来源：Sentinel-2占49.82%，"
            "Sentinel-1、高分辨率影像、道路和AOI构成第二梯队。"
            "文本是patch尺度上下文而非像元局部语义，其内部潜变量归因平均绝对值为0.925，未与79个原始栅格通道共同归一化。"
        ),
        264: (
            "模型效率结果说明，类别均衡精度增益伴随计算代价。本文模型参数量为4.79 M，远小于直接48通道FCN-ResNet50和VGG-UNet-Transformer，"
            "但24.88 GFLOPs和8.06 ms单patch推理时间均高于FCNN-47。"
            "其额外成本来自文本转码、城市上下文残差、条件层级分支和类别级概率融合。"
            "对于关注少数类均衡性能、边界一致性和城市功能解释的研究制图任务，这一成本可以接受；"
            "对于高频、大范围快速更新场景，FCNN-47的速度和OA使其成为更轻量的选择。"
        ),
        266: (
            "本研究仍存在若干与数据和评估范围相关的局限。三随机种子只能提供初步训练稳定性证据，基于df=2的置信区间较宽；"
            "leave-region-out仅覆盖香港西部一个地理方向，尚不能代表跨城市或跨时相泛化。"
            "BLU、卫星影像、OSM、建筑功能与高度数据并非完全同日；尤其是主要为2021年的Google Earth RGB可能引入陈旧形态线索。"
            "去除高分辨率影像使mIoU下降0.0149，表明该输入提供补充而非主导信息，但该消融不能消除时相不一致本身的影响。"
            "全香港滑动推理中的OSM文本仍以128×128原始patch manifest为基础，相邻窗口可能复用相同的上下文文本；"
            "因此其贡献不应解释为像元局部语义。最后，Expected Gradients只基于80个类别均衡样本，且近似残差中位数为0.374，"
            "应被视为模型审计而非全域因果归因。"
        ),
        268: (
            "本文面向香港全域10 m输出网格的八类土地利用/土地覆盖制图，构建了以多模态FCNN空间转码、城市上下文残差、"
            "条件层级概率组织和类别级概率融合为组成的预测框架。固定空间分块测试集上，种子42模型取得OA=0.8789、"
            "macro mIoU=0.5946和macro-F1=0.7338；在同等47 epoch训练阶段数下，FCNN取得略高OA（0.8794），"
            "但本文模型保持最高macro mIoU和macro-F1点估计。结构消融中，城市上下文残差的mIoU边际下降最大且达到p=0.0140，"
            "条件层级概率组织与类别级概率融合呈方向一致但未单独显著的增益。AOI单项消融和OSM联合消融支持城市上下文信息对功能判别的价值；"
            "1像元容忍下的macro Boundary F1为0.4915，高于五种原始全模态基线。"
            "因此，本文的实证贡献在于将局地城市上下文作为可检验残差机制与物理遥感特征协同使用，并用条件层级概率组织复杂城市LULC的输出关系。"
            "当前结论受训练预算、空间泛化、异时相辅助影像和解释样本规模约束，未来应在更多城市、时间和本地化文本机制下进一步验证。"
        ),
        293: (
            "注：商业办公与荒地/岩岸合计为109,976像元、10.9976 km²，占有标签像元的0.982%。"
            "它们在八类实验中被设为ignore并非基于“像元数不足以训练”的单一理由，而是为了将本研究限定在类别定义相对一致的八类LULC目标；"
            "两类在最终训练、验证、测试空间组内的实际支持见表S2。"
        ),
        294: (
            "注：OSM AOI、POI及建筑功能中的商业办公标签仅作为外部城市语义输入，而非BLU监督标签或预测目标；"
            "商业办公像元本身不产生类别监督损失，也不计入八类精度评价。"
        ),
    }

    for idx, value in replacements.items():
        replace_preserving_format(doc.paragraphs[idx], value)

    # Expanded citations: inserted immediately before the supplemental material.
    anchor = doc.paragraphs[290]
    refs = [
        "[18] Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., et al. (2012). Sentinel-2: ESA’s optical high-resolution mission for GMES operational services. Remote Sensing of Environment, 120, 25–36. https://doi.org/10.1016/j.rse.2011.11.026",
        "[19] Torres, R., Snoeij, P., Geudtner, D., Bibby, D., Davidson, M., Attema, E., et al. (2012). GMES Sentinel-1 mission. Remote Sensing of Environment, 120, 9–24. https://doi.org/10.1016/j.rse.2011.05.028",
        "[20] Schmitt, M., Hughes, L. H., Qiu, C., & Zhu, X. X. (2019). SEN12MS: A curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion. ISPRS Annals, IV-2/W7, 153–160. https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019",
        "[21] Audebert, N., Le Saux, B., & Lefèvre, S. (2018). Beyond RGB: Very high resolution urban remote sensing with multimodal deep networks. ISPRS Journal of Photogrammetry and Remote Sensing, 140, 20–32. https://doi.org/10.1016/j.isprsjprs.2017.11.011",
        "[22] Silla, C. N., Jr., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22, 31–72. https://doi.org/10.1007/s10618-010-0175-9",
        "[23] Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment, 148, 42–57. https://doi.org/10.1016/j.rse.2014.02.015",
        "[24] Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40, 913–929. https://doi.org/10.1111/ecog.02881",
        "[25] Haklay, M. (2010). How good is volunteered geographical information? A comparative study of OpenStreetMap and Ordnance Survey datasets. Environment and Planning B, 37, 682–703. https://doi.org/10.1068/b35097",
        "[26] Cheng, B., Girshick, R., Dollár, P., Berg, A. C., & Kirillov, A. (2021). Boundary IoU: Improving object-centric image segmentation evaluation. Proceedings of CVPR, 15334–15342. https://doi.org/10.1109/CVPR46437.2021.01508",
        "[27] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28",
        "[28] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of CVPR, 770–778. https://doi.org/10.1109/CVPR.2016.90",
        "[29] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. Proceedings of ICCV, 10012–10022. https://doi.org/10.1109/ICCV48922.2021.00986",
        "[30] Berman, M., Rannen Triki, A., & Blaschko, M. B. (2018). The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. Proceedings of CVPR, 4413–4421.",
        "[31] Goodchild, M. F. (2007). Citizens as sensors: The world of volunteered geography. GeoJournal, 69, 211–221. https://doi.org/10.1007/s10708-006-9111-y",
        "[32] Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nature Communications, 11, 4540. https://doi.org/10.1038/s41467-020-18321-y",
        "[33] Meyer, H., Reudenbach, C., Wöllauer, S., & Nauss, T. (2019). Importance of spatial predictor variable selection in machine learning applications—Moving from data reproduction to spatial prediction. Ecological Modelling, 411, 108815. https://doi.org/10.1016/j.ecolmodel.2019.108815",
        "[34] Zhu, X. X., Tuia, D., Mou, L., Xia, G.-S., Zhang, L., Xu, F., & Fraundorfer, F. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. IEEE Geoscience and Remote Sensing Magazine, 5(4), 8–36. https://doi.org/10.1109/MGRS.2017.2762307",
        "[35] Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., & Johnson, B. A. (2019). Deep learning in remote sensing applications: A meta-analysis and review. ISPRS Journal of Photogrammetry and Remote Sensing, 152, 166–177. https://doi.org/10.1016/j.isprsjprs.2019.04.015",
    ]
    for ref in refs:
        add_reference_before(anchor, ref)

    # Append concise supplementary evidence tables with experimental values.
    doc.add_paragraph()
    p = doc.add_paragraph("表S2  被排除BLU类别在最终空间划分中的像元与空间组支持。")
    p.style = doc.paragraphs[291].style
    add_table(
        doc,
        [
            ["类别", "训练像元", "训练空间组", "验证像元", "验证空间组", "测试像元", "测试空间组"],
            ["商业办公", "34,672", "36", "8,235", "7", "4,534", "7"],
            ["荒地/岩岸", "38,203", "45", "6,337", "9", "10,207", "12"],
        ],
        [2.4, 2.0, 1.8, 2.0, 1.8, 2.0, 1.8],
        doc.tables[10].style,
    )
    p = doc.add_paragraph(
        "注：统计范围为最终训练、验证和测试patch。该表用于透明报告空间支持；八类范围选择基于类别定义与内部异质性，"
        "不把像元计数本身作为无法训练的唯一理由。"
    )
    p.style = doc.paragraphs[292].style

    p = doc.add_paragraph("表S3  六种原始全模态模型在测试patch内的边界一致性。")
    p.style = doc.paragraphs[291].style
    add_table(
        doc,
        [
            ["模型", "macro Boundary F1（1像元/10 m容忍）"],
            ["本文模型", "0.4915"],
            ["CNN", "0.4372"],
            ["FCNN（原20 epoch）", "0.4537"],
            ["Transformer", "0.3954"],
            ["FCN-ResNet50", "0.2145"],
            ["VGG-UNet-Transformer", "0.4017"],
        ],
        [7.2, 7.2],
        doc.tables[10].style,
    )
    p = doc.add_paragraph(
        "注：从预测与BLU参考标签中在各测试patch内部提取类别边界，并在1像元（10 m）容忍距离下计算宏平均Boundary F1。"
        "该指标反映与监督参考标签的边界一致性，不代表独立实测地块边界精度。"
    )
    p.style = doc.paragraphs[292].style

    # Preserve all original layout while adding semantic accessibility metadata.
    for table in doc.tables:
        mark_header_row(table)
    for shape_idx, shape in enumerate(doc.inline_shapes, start=1):
        doc_pr = shape._inline.docPr
        if not (doc_pr.get("descr") or doc_pr.get("title")):
            doc_pr.set("descr", f"图{shape_idx}；具体内容见相邻图题。")

    doc.core_properties.title = "城市上下文残差与条件层级概率融合的香港土地利用/土地覆盖制图"
    doc.core_properties.comments = "Clean revised copy; source manuscript preserved."
    doc.save(DST)

    CHANGELOG.write_text(
        "# Manuscript revision change log\n\n"
        "- Reframed the central contribution as urban contextual residual fusion plus conditional hierarchical probability organisation.\n"
        "- Replaced soft-routing wording with class-wise probability fusion and stated its global-scalar scope.\n"
        "- Added transparent excluded-class spatial-support table (S2) and boundary-consistency table (S3).\n"
        "- Added high-resolution-image temporal-mismatch, patch-level text-context, fairness-budget, and Expected-Gradients scope statements.\n"
        "- Expanded the bibliography from 17 to 35 entries and inserted contextual citations.\n"
        "- Added only results already present in completed output files; no training was run.\n",
        encoding="utf-8",
    )
    print(DST)
    print(CHANGELOG)


if __name__ == "__main__":
    main()
