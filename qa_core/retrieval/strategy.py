"""RAG 链路动态检索计划，将意图结果转为检索参数。

它把意图识别结果转换成具体检索参数，例如 top_k、阈值、是否查询 FAQ/文档集合。
不同检索类问题（FAQ、知识查询、追问、短问题、低决策分数、表格问题）会经过多层决策链得到不同策略。

决策层次：
    1. 意图分支：FAQ、知识查询或追问。
    2. 短问题保护：短句歧义大，限制文档检索并提高直出门槛。
    3. 规则分数保护：规则判断越弱，FAQ 模糊直出越保守，文档召回越充分。
    4. 风险类别：费用、合规、排障、总结类问题使用不同参数。
    5. 表格偏好：表格类查询扩大文档候选池。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qa_core.config.rules import RetrievalStrategyRules, get_rule_config
from qa_core.config.settings import get_settings
from qa_core.intent.classifier import IntentResult
from qa_core.intent.question_category import infer_question_category, is_table_query


@dataclass(frozen=True)
class RetrievalPlan:
    """单个用户问题对应的具体检索参数。QAService 消费此计划而非直接读取 settings，便于策略调整和诊断。

    字段说明：
      - run_faq/run_doc：是否查询 FAQ/文档集合。
      - faq_top_k/doc_top_k：FAQ 和文档初始召回数量。
      - rerank：是否启用 CrossEncoder 重排。
      - faq_direct_threshold：FAQ 直出最低分数阈值。
      - final_context_top_n：最终进入 LLM 上下文的最多片段数。
      - min_context_score：进入上下文的最低相关性分数。
      - max_context_chars/max_context_doc_chars：总上下文和单文档字符上限。
      - use_query_variants：是否生成查询变体。
      - question_category：问题风险类别。
      - prefer_table：是否偏向表格行资料。
      - faq_direct_exact_only：是否只允许精确 FAQ 直出，禁用相似分数直出。
      - intent_rule_score：规则候选分，来自 IntentResult.rule_score。
      - intent_decision_score：网关最终决策分，来自 IntentResult.confidence。
      - reason：检索策略原因标签，用于诊断。

    调用顺序：检索准备或检索执行 -> RetrievalPlan。
    """

    run_faq: bool
    run_doc: bool
    faq_top_k: int
    doc_top_k: int
    rerank: bool
    faq_direct_threshold: float
    final_context_top_n: int
    min_context_score: float
    max_context_chars: int
    max_context_doc_chars: int
    use_query_variants: bool
    question_category: str
    prefer_table: bool
    faq_direct_exact_only: bool
    intent_rule_score: float
    intent_decision_score: float
    reason: str

    def as_dict(self) -> dict:
        """返回可 JSON 序列化的检索计划，供诊断接口使用。

        返回：
            RetrievalPlan 字段字典。

        调用顺序：检索准备或检索执行 -> RetrievalPlan.as_dict()。
        """
        return asdict(self)


# ── Rule table helpers ──────────────────────────────────────────────

PlanParams = dict[str, Any]


@dataclass(frozen=True)
class PlanPatch:
    """检索计划规则补丁，只描述需要调整的字段。

    PlanPatch 是规则层输出，被 _apply_patch 合并到基础参数字典中。
    所有 *Field 为 None 的字段表示该补丁不修改对应参数。

    字段说明：
      - reason: 补丁原因标签，用于检索计划的 reason 字段拼接。
      - replace_reason: True 时补丁原因完全替换原 reason，False 时追加。
      - faq_top_k: 精确覆盖 FAQ 召回数量（覆盖而非取 max）。
      - faq_top_k_min: FAQ 召回的兜底下限（取 max）。
      - doc_top_k: 精确覆盖文档召回数量。
      - doc_top_k_min/doc_top_k_max: 文档召回的兜底下限/上限。
      - final_context_top_n_min: 最终上下文片段数的兜底下限。
      - direct_threshold: 精确覆盖 FAQ 直出阈值。
      - direct_threshold_min: FAQ 直出阈值的兜底下限（取 max）。
      - faq_direct_exact_only: True=仅允许精确匹配直出，禁用模糊直出。

    调用顺序：检索准备或检索执行 -> PlanPatch。
    """

    reason: str
    replace_reason: bool = False
    faq_top_k: int | None = None
    faq_top_k_min: int | None = None
    doc_top_k: int | None = None
    doc_top_k_min: int | None = None
    doc_top_k_max: int | None = None
    final_context_top_n_min: int | None = None
    direct_threshold: float | None = None
    direct_threshold_min: float | None = None
    faq_direct_exact_only: bool | None = None


def _intent_rules(settings) -> dict[str, PlanPatch]:
    """返回意图到检索参数补丁的映射。

    调用顺序：检索准备或检索执行 -> _intent_rules()。
    """
    rules = get_rule_config().retrieval_strategy
    return {
        "FAQ_QUERY": PlanPatch(
            reason="faq_first",
            replace_reason=True,
            # FAQ 意图优先从 FAQ 集合回答，文档检索量减半以降低噪声
            doc_top_k=max(rules.strong_faq_doc_top_k_min, settings.doc_top_k // 2),
            # FAQ 直接回答阈值适度降低（允许相似匹配），但不低于兜底阈值
            direct_threshold=max(
                rules.faq_direct_floor,
                settings.faq_direct_score_threshold - rules.faq_direct_discount,
            ),
        ),
        "KNOWLEDGE_QUERY": PlanPatch(
            reason="knowledge_doc_enriched",
            replace_reason=True,
            # 知识查询需要更多文档证据支撑，doc_top_k 至少取复杂查询的召回量
            doc_top_k_min=max(settings.doc_top_k, settings.doc_complex_query_top_k),
            # 上下文窗口相应扩大，保证足够信息进入 LLM
            final_context_top_n_min=rules.knowledge_context_top_n_min,
        ),
        "FOLLOW_UP": PlanPatch(
            reason="history_aware_follow_up",
            replace_reason=True,
            # 追问依赖历史上下文，FAQ 召回量适度扩大
            faq_top_k=max(settings.faq_top_k, rules.follow_up_faq_top_k_min),
            doc_top_k_min=max(settings.doc_top_k, settings.doc_complex_query_top_k),
            final_context_top_n_min=rules.knowledge_context_top_n_min,
            # 追问歧义最大（"那这个呢"），直接回答阈值取三者的最大值以保守处理
            direct_threshold_min=max(
                settings.faq_direct_score_threshold,
                rules.short_query_guard_threshold,
                rules.medium_rule_score_direct_threshold,
            ),
        ),
    }


def _category_rules(settings) -> dict[str, PlanPatch]:
    """返回问题类别到检索参数补丁的映射。

    调用顺序：检索准备或检索执行 -> _category_rules()。
    """
    rules = get_rule_config().retrieval_strategy
    # expanded_doc: 高风险类别使用更大的文档召回量，确保不遗漏关键证据
    expanded_doc = settings.doc_complex_query_top_k
    return {
        "pricing": PlanPatch(
            reason="pricing_guard",
            faq_top_k_min=settings.faq_top_k,
            doc_top_k_min=expanded_doc,
            final_context_top_n_min=rules.guard_context_top_n_min,
            # 费用类问题对准确性要求最高，直接回答阈值相应提高
            direct_threshold_min=rules.pricing_direct_threshold,
        ),
        "compliance": PlanPatch(
            reason="compliance_guard",
            doc_top_k_min=expanded_doc,
            final_context_top_n_min=rules.guard_context_top_n_min,
            # 合规类问题阈值与费用类独立配置，合规可能允许更宽松的模糊匹配
            direct_threshold_min=rules.compliance_direct_threshold,
        ),
        "troubleshooting": PlanPatch(
            reason="troubleshooting_expanded",
            # 排障场景需要更多上下文（日志、配置、历史案例），扩大文档召回和上下文
            doc_top_k_min=expanded_doc,
            final_context_top_n_min=rules.guard_context_top_n_min,
        ),
        "summary": PlanPatch(
            reason="summary_expanded",
            # 总结类问题需覆盖多份文档，上下文窗口相应扩大
            doc_top_k_min=expanded_doc,
            final_context_top_n_min=rules.guard_context_top_n_min,
        ),
    }


def _rule_score_guard(settings, rules: RetrievalStrategyRules, decision_score: float) -> PlanPatch | None:
    """根据意图决策分数补齐保守检索策略。

    decision_score 来自 IntentResult.confidence，是规则候选和模型候选经过网关仲裁后的
    最终分数。分数越低，说明入口判断越不稳定，后续就应该少做 FAQ 模糊直出，
    多保留文档证据。

    调用顺序：检索准备或检索执行 -> _rule_score_guard()。
    """
    expanded_doc = settings.doc_complex_query_top_k
    if decision_score < rules.low_rule_score_threshold:
        return PlanPatch(
            reason="low_rule_score_guard",
            doc_top_k_min=expanded_doc,
            final_context_top_n_min=rules.guard_context_top_n_min,
            direct_threshold_min=max(settings.faq_direct_score_threshold, rules.low_rule_score_direct_threshold),
            faq_direct_exact_only=True,
        )
    if decision_score < rules.medium_rule_score_threshold:
        return PlanPatch(
            reason="rule_score_guard",
            doc_top_k_min=settings.doc_top_k,
            direct_threshold_min=max(settings.faq_direct_score_threshold, rules.medium_rule_score_direct_threshold),
        )
    return None


def _base_params(settings, is_short: bool) -> PlanParams:
    """构建检索计划的基础参数。

    调用顺序：检索准备或检索执行 -> _base_params()。
    """
    return {
        "run_faq": True,
        "run_doc": True,
        "faq_top_k": settings.faq_short_query_top_k if is_short else settings.faq_top_k,
        "doc_top_k": settings.doc_top_k,
        "final_context_top_n": settings.final_context_top_n,
        "direct_threshold": settings.faq_direct_score_threshold,
        "faq_direct_exact_only": False,
        "reason": "balanced_retrieval",
    }


def _apply_patch(params: PlanParams, patch: PlanPatch) -> PlanParams:
    """把一条规则补丁应用到基础参数字典，支持精确覆盖和兜底最小值两种调整方式。（★★★ 核心）

    补丁调整规则：
    - *Field（精确覆盖）：直接覆盖 params 对应字段。
    - *Field_min（最小值兜底）：取 params[字段] 和 min 值的较大值。
    - *Field_max（最大值兜底）：取 params[字段] 和 max 值的较小值。
    - reason：replace_reason=True 时完全替换原因；False 时追加到原原因后。

    执行流程：逐个检查 patch 中的非 None 字段，按上述规则修改 params 对应值。

    参数：
        params: 基础参数字典（会被原地修改）。
        patch: 规则补丁（只有非 None 字段会生效）。

    返回：
        修改后的 params 字典（原地操作，返回便于链式调用）。

    调用顺序：检索准备或检索执行 -> _apply_plan_rules() -> _apply_patch()。
    """
    if patch.faq_top_k is not None:
        params["faq_top_k"] = patch.faq_top_k
    if patch.faq_top_k_min is not None:
        params["faq_top_k"] = max(params["faq_top_k"], patch.faq_top_k_min)
    if patch.doc_top_k is not None:
        params["doc_top_k"] = patch.doc_top_k
    if patch.doc_top_k_min is not None:
        params["doc_top_k"] = max(params["doc_top_k"], patch.doc_top_k_min)
    if patch.doc_top_k_max is not None:
        params["doc_top_k"] = min(params["doc_top_k"], patch.doc_top_k_max)
    if patch.final_context_top_n_min is not None:
        params["final_context_top_n"] = max(params["final_context_top_n"], patch.final_context_top_n_min)
    if patch.direct_threshold is not None:
        params["direct_threshold"] = patch.direct_threshold
    if patch.direct_threshold_min is not None:
        params["direct_threshold"] = max(params["direct_threshold"], patch.direct_threshold_min)
    if patch.faq_direct_exact_only is not None:
        params["faq_direct_exact_only"] = patch.faq_direct_exact_only
    params["reason"] = patch.reason if patch.replace_reason else f"{params['reason']}_{patch.reason}"
    return params


def _apply_plan_rules(
    params: PlanParams,
    *,
    intent: IntentResult,
    is_short: bool,
    question_category: str,
    prefer_table: bool,
    settings,
) -> PlanParams:
    """按固定顺序应用意图、短问题、决策分数、类别和表格规则。

    调用顺序：检索准备或检索执行 -> _apply_plan_rules()。
    """
    rules = get_rule_config().retrieval_strategy
    intent_patch = _intent_rules(settings).get(intent.intent)
    if intent_patch:
        _apply_patch(params, intent_patch)
    if is_short and intent.intent != "FOLLOW_UP":
        _apply_patch(
            params,
            PlanPatch(
                reason="short_query_guard",
                doc_top_k_max=max(12, settings.final_context_top_n * 2),
                direct_threshold_min=rules.short_query_guard_threshold,
            ),
        )
    score_guard = _rule_score_guard(settings, rules, intent.confidence)
    if score_guard:
        _apply_patch(params, score_guard)
    category_patch = _category_rules(settings).get(question_category)
    if category_patch:
        _apply_patch(params, category_patch)
    if prefer_table and params["run_doc"]:
        _apply_patch(
            params,
            PlanPatch(
                reason="table_row_preferred",
                doc_top_k_min=settings.doc_complex_query_top_k,
                final_context_top_n_min=rules.table_context_top_n_min,
                faq_direct_exact_only=True,
            ),
        )
    return params


def build_retrieval_plan(query: str, intent: IntentResult) -> RetrievalPlan:
    """根据问题形态和意图结果构建检索策略。按多层规则逐层收紧参数。（★★★ 核心）

    执行流程：
      1. 初始化默认参数：问题清洗、类别识别、表格偏好识别、短句识别。
      2. 应用意图分支：FAQ/知识/追问分别调整召回量和阈值。
      3. 应用短问题保护：短句提高 FAQ 直出门槛并收缩文档召回。
      4. 应用规则分数保护：低分兜底问题提高 FAQ 直出门槛并扩大证据召回。
      5. 应用风险类别：费用、合规、排障、总结问题扩大召回或提高阈值。
      6. 应用表格偏好：表格问题扩大文档候选并禁用模糊 FAQ 直出。
      7. 组装不可变 RetrievalPlan；知识查询和追问启用查询变体。

    参数：
        query: 已经过 normalize_user_query() 处理的业务有效问题。
        intent: classify_intent() 输出的检索类 IntentResult.

    返回：
        QAService 执行检索所需的完整 RetrievalPlan。
    """
    settings = get_settings()
    compact_query = query.strip()
    # 推断问题风险类别（pricing/compliance/troubleshooting/summary/other）—— 风险类别驱动检索阈值和回答模板
    question_category = infer_question_category(compact_query)
    # 判断是否为表格类查询 —— 表格问题需扩大候选集并禁用模糊 FAQ 直出，语义相似但不同列的表格内容误导性极强
    prefer_table = is_table_query(compact_query)
    # 短问题歧义大（如"怎么做"、"费用呢"），需要更严格的保护策略
    is_short = len(compact_query) <= settings.short_query_max_chars

    params = _apply_plan_rules(
        _base_params(settings, is_short),
        intent=intent,
        is_short=is_short,
        question_category=question_category,
        prefer_table=prefer_table,
        settings=settings,
    )

    return RetrievalPlan(
        run_faq=params['run_faq'],
        run_doc=params['run_doc'],
        faq_top_k=params['faq_top_k'],
        doc_top_k=params['doc_top_k'],
        rerank=True,
        faq_direct_threshold=params['direct_threshold'],
        final_context_top_n=params['final_context_top_n'],
        min_context_score=settings.rag_min_score_threshold,
        max_context_chars=settings.max_prompt_context_chars,
        max_context_doc_chars=settings.max_context_doc_chars,
        # 知识查询和追问启用查询变体（同义改写），FAQ 查询不启用
        # 原因：FAQ 标准问题固定不变，查询变体只会增加误匹配风险
        use_query_variants=intent.intent in {"KNOWLEDGE_QUERY", "FOLLOW_UP"},
        question_category=question_category,
        prefer_table=prefer_table,
        faq_direct_exact_only=params['faq_direct_exact_only'],
        intent_rule_score=intent.rule_score,
        intent_decision_score=intent.confidence,
        reason=params['reason'],
    )
