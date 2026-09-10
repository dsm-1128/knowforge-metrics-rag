"""RAG 问题类别识别 — 决定 Prompt 风险模板和输出要求。
与意图识别互补：意图回答"是什么"，问题类别控制"怎么答"（费用/合规/排障等不同模板）。"""

from __future__ import annotations
import re
from typing import Literal
QuestionCategory = Literal["default", "pricing", "compliance", "troubleshooting", "summary"]

PRICING_HINTS = re.compile(
    r"(费用|价格|学费|报价|优惠|折扣|退费|退款|发票|账单|支付|开票|续费|合同金额|收费|付款|"
    r"赔付|赔款|金额|回款|预算|采购|报销|信用证|保证金|付款条件|收款账户|打款|结算|税费)"
)
COMPLIANCE_HINTS = re.compile(
    r"(合规|隐私|个人信息|客户数据|外部平台|合同|条款|审计|风险|违规|保密|权限审批|数据出境|授权|责任|"
    r"制裁|名单|出口管制|强制性|规范|标准|图纸冲突|除外|免责|不赔|拒赔|HS\s*编码|海关|报关|申报|"
    r"归类|最终用途|最终用户|尽调|受限空间|作业审批|安全确认|安全技术交底|检验批|隐蔽工程|验收|"
    r"既往症|如实告知|保单复效)"
)
TROUBLESHOOTING_HINTS = re.compile(r"(故障|报错|失败|无法|异常|告警|报警|排查|维修|巡检|安装|配置|连接不上|超时|错误码)")
SUMMARY_HINTS = re.compile(r"(总结|概括|归纳|整理|对比|区别|有哪些|都包括|主要内容|学习内容|大纲)")
TABLE_QUERY_HINTS = re.compile(
    r"(表格|清单|台账|明细|字段|列名|行号|sheet|工作表|导出|列表|矩阵|评分表|检查表|对账|"
    r"材料清单|验收项|付款节点|状态|责任人|金额|数量|单价|收款账户|银行卡|发票号码|箱单|装箱单|"
    r"数据字典|计算口径|统计口径|统计粒度|分子|分母|枚举|主键|数据类型)"
)


def infer_question_category(query: str) -> QuestionCategory:
    """判断 RAG 回答风险类别。示例："费用多少"→pricing；"报错了"→troubleshooting。（★★★ 核心）

    风险类别驱动检索策略（召回阈值/上下文大小）和 Prompt 模板（不同风险类别使用不同的回答约束和免责声明）。
    分类优先级：费用 > 合规 > 排障 > 总结：费用和合规是高风险类别，优先匹配。

    调用顺序：检索准备阶段 -> infer_question_category()。
    """
    normalized = query.strip().lower()
    # 费用/价格类（高优先级）：返回金额/价格错误可能导致经济损失或客诉，需最严格的检索控制，召回阈值 ≥ 0.86
    if PRICING_HINTS.search(normalized):
        return "pricing"
    # 合规/法律类（极高优先级）：合规答案错误可能导致监管处罚，检索门槛最高（≥0.86），并启用免责声明模板
    if COMPLIANCE_HINTS.search(normalized):
        return "compliance"
    # 排障类（中优先级）：用户正在遭遇问题，需要丰富且准确的步骤化上下文，Prompt 侧重操作指引
    if TROUBLESHOOTING_HINTS.search(normalized):
        return "troubleshooting"
    # 总结类（低优先级）：信息量需求大但精度要求相对宽松，召回阈值可降至 0.5 以获取更多上下文
    if SUMMARY_HINTS.search(normalized):
        return "summary"
    # 默认类：常规检索策略和通用 Prompt 模板，不做特殊约束
    return "default"


def is_table_query(query: str) -> bool:
    """判断是否为表格/清单查询。示例："清单有哪些字段"→True；"费用多少"→False。（★★ 理解）

    表格查询的业务影响：匹配到表格类时，检索策略会扩大文档候选并禁用模糊 FAQ 直出，
    因为表格内容语义相似但不同列/行会导致严重误导（如"验收项清单"和"付款节点清单"语义相近但内容完全不同）。

    调用顺序：检索准备阶段 -> is_table_query()。
    """
    normalized = query.strip().lower()
    # 命中任意关键词即判定为表格类查询，后续链路据此调整检索策略和后处理逻辑
    return bool(TABLE_QUERY_HINTS.search(normalized))

