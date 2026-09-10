"""意图识别模块：在检索前决定问题应该走哪条处理路径。

它会判断当前检索问题是追问、FAQ 还是文档知识查询，并把结果交给检索计划和
Prompt 模板使用。问候、人工客服和越界请求由路由层的 classify_direct_intent()
提前收口，不进入这里。整体策略是“规则候选 + 意图模型 + 网关仲裁”：高频确定
场景保留规则可解释性，长尾表达由模型参与判定，最终由网关输出可治理结果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import BaseMessage

from qa_core.config.rules import get_rule_config
from qa_core.intent.decision import apply_intent_decision_gateway
from qa_core.scenarios.boundary import SourceMatch, rank_source_matches
from qa_core.scenarios.registry import ScenarioDefinition


Intent = Literal["GREETING", "FOLLOW_UP", "KNOWLEDGE_QUERY", "FAQ_QUERY", "HUMAN_SERVICE", "OUT_OF_SCOPE"]


GREETING_PATTERNS = [
    r"^(你好|您好|hi|hello|哈喽|在吗|在不在|有人吗)[啊呀呢嘛么\s,，!！。.?？]*$",
    r"^(你是谁|您是谁|你叫什么|你的名字|who are you)[\s,，!！。.?？]*$",
]

FOLLOW_UP_HINTS = re.compile(r"^(那|这个|那个|它|他们|她们|这些|上面|刚才|继续|还有|费用呢|审批呢|权限呢|发票呢|告警呢|步骤呢)")
OFF_TOPIC_HINTS = re.compile(r"(彩票|赌博|股票内幕|色情|违法|攻击|破解|黑客入侵)")
HUMAN_SERVICE_HINTS = re.compile(r"(人工客服|转人工|客服|客服电话|电话|联系方式|联系顾问|联系电话)")
FAQ_HINTS = re.compile(r"(费用|价格|安装|环境|失败|报错|地址|时间|退费|优惠|发票|账号|登录|权限|审批|合同|隐私|账单|支付|开票|工单|售后)")
KNOWLEDGE_HINTS = re.compile(r"(知识库|文档|手册|流程|制度|规范|说明|配置|接口|功能|排查|故障|步骤|sop|告警|巡检|设备|合规|条款|入职|审批|合同|隐私|webhook|回调|发票|账单)")
FAQ_QUESTION_SHAPE_HINTS = re.compile(r"(怎么办|如何处理|怎么处理|需要什么|需要哪些|需要准备哪些|有哪些|为什么|什么时候|由谁|能不能|会不会)")
DIRECT_FAQ_SHAPE_HINTS = re.compile(r"(资料呢|材料呢|是什么|如何回收|怎么排查|怎么处理|能不能|可以吗|要看什么)")

@dataclass(frozen=True)
class IntentResult:
    """意图识别的标准输出，供检索计划和下游链路消费。

    这里不是只返回一个标签，而是把“是否可直接回答、是否需要追问改写、检索类建议业务分类、
    判断原因、规则候选分数、模型候选分数和最终仲裁分数”一起返回。rule_score
    表示规则判断强弱；confidence 对外展示网关融合后的最终分数，并会被检索计划
    用来调整 FAQ 直出门槛和召回范围。

    调用顺序：检索准备阶段 -> IntentResult。
    """

    intent: Intent
    direct_answer: str | None = None
    rule_score: float = 0.6
    reason: str = "rule"
    requires_rewrite: bool = False
    suggested_source: str | None = None
    source_score: int = 0
    source_confidence: float = 0.0
    source_candidates: tuple[dict[str, str | int | float], ...] = ()
    final_score: float | None = None
    risk_tags: tuple[str, ...] = ()
    decision_policy: str = "rule_candidate"
    candidate_intents: tuple[dict[str, str | float], ...] = ()
    model_score: float | None = None
    model_version: str | None = None
    policy_version: str = "intent-policy-v1-bert"

    @property
    def confidence(self) -> float:
        """对外展示分数，由 rule_score 派生。

        调用顺序：检索准备阶段 -> IntentResult.confidence()。
        """
        return self.final_score if self.final_score is not None else self.rule_score

    def as_dict(self) -> dict:
        """转换为可 JSON 序列化的字典，供 API 诊断信息返回。

        返回：
            包含 intent、rule_score、对外 confidence、reason、requires_rewrite、检索类 suggested_source 的字典。

        调用顺序：检索准备阶段 -> IntentResult.as_dict()。
        """
        return {
            "intent": self.intent,
            "rule_score": self.rule_score,
            "confidence": self.confidence,
            "reason": self.reason,
            "requires_rewrite": self.requires_rewrite,
            "suggested_source": self.suggested_source,
            "source_score": self.source_score,
            "source_confidence": self.source_confidence,
            "source_candidates": list(self.source_candidates),
            "final_score": self.confidence,
            "risk_tags": list(self.risk_tags),
            "decision_policy": self.decision_policy,
            "candidate_intents": list(self.candidate_intents),
            "model_score": self.model_score,
            "model_version": self.model_version,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class _IntentCandidate:
    """Internal candidate used to resolve overlapping strong intent rules.

    调用顺序：检索准备阶段 -> _IntentCandidate。
    """

    result: IntentResult
    priority: int


@dataclass(frozen=True)
class _SourceInference:
    """Source inference diagnostics attached to retrieval intents.

    调用顺序：检索准备阶段 -> _SourceInference。
    """

    source: str | None
    score: int = 0
    confidence: float = 0.0
    candidates: tuple[dict[str, str | int | float], ...] = ()

    @classmethod
    def from_matches(cls, matches: tuple[SourceMatch, ...]) -> "_SourceInference":
        """Build diagnostics from ranked source matches.

        调用顺序：检索准备阶段 -> _SourceInference.from_matches()。
        """

        if not matches:
            return cls(source=None)
        best = matches[0]
        return cls(
            source=best.source,
            score=best.score,
            confidence=best.confidence,
            candidates=tuple(match.as_dict() for match in matches),
        )

    def intent_kwargs(self) -> dict[str, object]:
        """Return kwargs shared by retrieval IntentResult constructors.

        调用顺序：检索准备阶段 -> _SourceInference.intent_kwargs()。
        """

        return {
            "suggested_source": self.source,
            "source_score": self.score,
            "source_confidence": self.confidence,
            "source_candidates": self.candidates,
        }


def classify_direct_intent(query: str, scenario: ScenarioDefinition) -> IntentResult | None:
    """供查询路由层复用的确定性直答规则：问候、越界、短句转人工。

    这个函数只处理必须优先收口的协议/安全类问题；
    普通 FAQ、知识咨询和追问仍交给检索准备阶段的 ``classify_intent()`` 判断。

    调用顺序：检索准备阶段 -> classify_direct_intent()。
    """
    normalized = query.strip().lower()
    greeting_answers = [
        f"你好！我是{scenario.assistant_name}，可以帮你查询{scenario.business_domain}中的制度、流程、FAQ 和文档资料。",
        f"我是{scenario.assistant_name}，负责解答{scenario.business_domain}相关问题。",
    ]
    # 规则 1 — GREETING：只拦截纯问候/身份问题；问候后带业务问题时继续进入检索链路
    for pattern, answer in zip(GREETING_PATTERNS, greeting_answers):
        if re.match(pattern, normalized, re.IGNORECASE):
            return IntentResult(intent="GREETING", direct_answer=answer, rule_score=1.0, reason="greeting_rule")
    # 规则 2 — OUT_OF_SCOPE：越界话题必须在任何检索之前拦截
    if OFF_TOPIC_HINTS.search(normalized):
        return IntentResult(intent="OUT_OF_SCOPE", direct_answer=f"这个问题超出了{scenario.business_domain}的问答范围，我无法提供帮助。", rule_score=0.95, reason="safety_rule")
    # 规则 3 — HUMAN_SERVICE：转人工请求需在短句中识别，避免长文本误触发
    if HUMAN_SERVICE_HINTS.search(normalized) and len(normalized) <= 18:
        return IntentResult(
            intent="HUMAN_SERVICE",
            direct_answer=f"可以联系人工支持，联系方式：{scenario.support_contact}。",
            rule_score=0.9,
            reason="human_service_rule",
        )
    return None


def classify_intent(query: str, history: list[BaseMessage], scenario: ScenarioDefinition) -> IntentResult:
    """识别检索分支的问题意图，规则无法判断时默认知识查询。（★★★ 核心）

    执行流程（命中即返回）：
      1. 追问识别：有历史且当前问题是“那这个呢”等省略表达，需要先改写。
      2. 规则识别：FAQ/知识库关键词和问法足够明确时直接分类。
      3. 默认知识查询：前面规则都不命中时，交给后续检索链路处理。

    规则优先的核心业务决策：追问、FAQ 和明确知识查询用规则快速分类，避免不必要的模型调用成本和延迟。

    参数：
        query: 已经过 normalize_user_query() 处理的业务有效问题。
        history: 历史对话消息列表。
        scenario: 当前业务场景，用于注入助手名称、业务域、联系方式和 source 白名单。

    返回：
        标准化 IntentResult。
    """
    source_details = infer_source_details(query, scenario)
    # 检索规则 1 — FOLLOW_UP：命中后立即返回，不再进入强领域规则；缺乏历史时不能走此路避免误判
    if history and (FOLLOW_UP_HINTS.search(query.strip()) or len(query.strip()) <= 8):
        rule_result = IntentResult(
            intent="FOLLOW_UP",
            rule_score=0.8,
            reason="follow_up_rule",
            requires_rewrite=True,
            **source_details.intent_kwargs(),
        )
        return apply_intent_decision_gateway(query, history, scenario, rule_result)
    # 检索规则 2 — 领域规则：只有追问规则未命中，才通过 FAQ/knowledge 关键词和问题句式判断业务意图
    strong_rule_intent = _strong_rule_domain_intent(query, source_details)
    if strong_rule_intent is not None:
        return apply_intent_decision_gateway(query, history, scenario, strong_rule_intent)
    # 检索规则 3 — 默认知识查询：规则无法明确细分时，继续进入后续检索链路
    rule_result = IntentResult(
        intent="KNOWLEDGE_QUERY",
        rule_score=0.6,
        reason="default_knowledge",
        **source_details.intent_kwargs(),
    )
    return apply_intent_decision_gateway(query, history, scenario, rule_result)

def infer_source(query: str, scenario: ScenarioDefinition) -> str | None:
    """只根据当前业务场景配置推断问题所属 source。

    source 是数据隔离和检索过滤的重要字段，必须来自当前场景的 `valid_sources`。
    因此这里只读取 `scenario.toml` 中的 `source_patterns`。新增行业或业务分类时，
    改场景配置即可，主链路代码不用变。

    参数：
        query: 用户问题。
        scenario: 当前业务场景。

    返回：
        当前场景 valid_sources 中的 source，或 None。

    调用顺序：检索准备阶段 -> infer_source()。
    """
    return infer_source_details(query, scenario).source


def infer_source_details(query: str, scenario: ScenarioDefinition) -> _SourceInference:
    """Return source inference diagnostics for the current scenario.

    调用顺序：检索准备阶段 -> infer_source_details()。
    """

    return _SourceInference.from_matches(rank_source_matches(query, scenario))


def _strong_rule_domain_intent(query: str, source_details: _SourceInference) -> IntentResult | None:
    """用领域关键词和问法强度识别高频业务问题。（★★ 理解）

    这是业务质量分流层，不是运行必需开关。没有命中时问题仍会进入
    default_knowledge 兜底；命中时可以把高频 FAQ、明确知识查询和模糊兜底
    分开，让第 06 章检索计划选择更合适的 FAQ 直出阈值、文档召回范围和
    Trace 诊断 reason。

    执行流程：
      1. 收集所有命中的 FAQ/knowledge 候选规则。
      2. 按 rule_score 选择最高分候选。
      3. 如果多个候选同分，用显式 priority 作为稳定的业务优先级。

    这里不使用命中即返回，是为了避免“低分规则排在前面，高分规则永远没机会执行”。
    rule_score 既会用于当前规则选择，也会进入第 06 章检索计划。

    1. rule_score=0.82 / 0.84 / 0.85 / 0.86
        rule_score 不是概率，也不是 Milvus 相似度分数。它表达的是规则强弱排序：
        0.82：只命中 FAQ 高频词，比如“费用/发票/报错”，可靠但偏宽泛
        0.84：命中文档、流程、制度等知识类关键词，需要扩大文档证据
        0.85：命中业务 source + 标准问法，比如“需要哪些/怎么办”，更像标准 FAQ
        0.86：命中业务 source + 直接 FAQ 问法，比如“是什么/可以吗”，更确定

        这些值会进入第 06 章检索计划：较高分数的问题可以更积极地尝试 FAQ 优先，
        低分兜底问题会提高 FAQ 直出门槛、扩大文档召回或只允许精确 FAQ 直出。
    2. len(normalized) <= 32 / 36 / 24
        这是“短句保护阈值”，会影响行为。目的不是精确统计，而是降低误判：
        <=32：标准 FAQ 问法通常比较短，比如“报销需要准备哪些材料？”
        <=36：直接 FAQ 问法稍微放宽，因为可能带具体对象，比如“系统权限回收流程是什么？”
        <=24：知识查询如果没有明确 source，只允许短句命中，避免长文本里偶然出现“文档/流程/制度”就被误判

        规则分类里的数字，一开始通常来自业务样本观察和风险取舍，不是凭空神奇数字。短句阈值用来限制规则的适用范围，rule_score 用来表达规则强弱。
        上线后应该用真实问答集评测，再把这些数字调优。
    参数：
        query: 用户问题。
        source_details: infer_source_details() 推断出的业务分类和诊断信息。

    返回：
        命中规则时返回 IntentResult，否则返回 None。
    """
    normalized = query.strip().lower()
    scores = get_rule_config().intent_rule_scores
    candidates: list[_IntentCandidate] = []
    suggested_source = source_details.source
    # FAQ 宽关键词匹配：只要出现"费用/价格/安装/报错"等高频业务词即可判定，覆盖绝大多数业务查询
    if FAQ_HINTS.search(normalized):
        candidates.append(
            _IntentCandidate(
                result=IntentResult(
                    intent="FAQ_QUERY",
                    rule_score=scores.strong_faq,
                    reason="strong_faq_rule",
                    **source_details.intent_kwargs(),
                ),
                priority=10,
            )
        )
    # FAQ 句式匹配：已知业务域 + 问题句式（"怎么办/需要什么"）组合，规则分数高于纯关键词
    if suggested_source and len(normalized) <= 32 and FAQ_QUESTION_SHAPE_HINTS.search(normalized):
        candidates.append(
            _IntentCandidate(
                result=IntentResult(
                    intent="FAQ_QUERY",
                    rule_score=scores.source_question_shape,
                    reason="source_question_shape_rule",
                    **source_details.intent_kwargs(),
                ),
                priority=30,
            )
        )
    # FAQ 句式精确匹配：已知业务域 + 直接 FAQ 句式（"是什么/可以吗"），短句限制降低误判率
    if suggested_source and len(normalized) <= 36 and DIRECT_FAQ_SHAPE_HINTS.search(normalized):
        candidates.append(
            _IntentCandidate(
                result=IntentResult(
                    intent="FAQ_QUERY",
                    rule_score=scores.direct_faq_shape,
                    reason="direct_faq_shape_rule",
                    **source_details.intent_kwargs(),
                ),
                priority=40,
            )
        )
    # 知识查询判定：知识库相关关键词 + 已知业务域或短句，避免长文本中偶然出现"文档"等词导致误判
    if KNOWLEDGE_HINTS.search(normalized) and (suggested_source or len(normalized) <= 24):
        candidates.append(
            _IntentCandidate(
                result=IntentResult(
                    intent="KNOWLEDGE_QUERY",
                    rule_score=scores.knowledge,
                    reason="strong_knowledge_rule",
                    **source_details.intent_kwargs(),
                ),
                priority=20,
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.result.rule_score, candidate.priority)).result
