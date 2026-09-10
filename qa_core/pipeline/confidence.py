"""答案置信度公共计算模块。

检索命中的 ``score`` 只表示候选内容和查询的相关性排序，不等价于最终答案可信度。
本模块只做两类低成本判断：生成前判断证据是否足够，生成后判断答案是否带有可追溯支撑。
它输出的是可解释的工程信号，不是经过统计校准的正确概率。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


LEVEL_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}

_CITATION_RE = re.compile(r"\[(\d+)\]")
_REFERENCE_SECTION_RE = re.compile(r"(?:^|\n)\s*(?:参考来源|来源参考|references?)\s*[:：]", re.IGNORECASE)
_CLAIM_SPLIT_RE = re.compile(r"[\r\n。！？!?；;]+")
_CJK_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{1,}")

# 这些阈值是工程分档，不代表模型概率。集中定义后，调参和离线评估都更容易。
_RETRIEVAL_MEDIUM_THRESHOLD = 0.50
_RETRIEVAL_HIGH_THRESHOLD = 0.75
_INTENT_STABLE_THRESHOLD = 0.70
_CITATION_COVERAGE_THRESHOLD = 0.50
_CONTEXT_SUPPORT_THRESHOLD = 0.45

__all__ = [
    "AnswerConfidence",
    "calculate_evidence_confidence",
    "calculate_generation_confidence",
    "combine_answer_confidence",
    "confidence_level",
    "evaluate_generated_answer",
    "faq_exact_match",
    "finalize_generated_answer_confidence",
    "mark_answer_confidence_not_applicable",
    "normalize_retrieval_score",
    "record_answer_confidence",
    "record_evidence_confidence",
]


@dataclass(frozen=True)
class AnswerConfidence:
    """生成前证据置信度结果。

    score 是 [0, 1] 区间的工程评分；level 是便于前端展示的粗粒度等级。

    调用顺序：RAG 管线 -> calculate_evidence_confidence() -> AnswerConfidence。
    """

    score: float
    level: str
    reasons: list[str]
    signals: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """转换为 API / Trace 可直接序列化的结构。

        返回：
            dict: 含 score（四舍五入到 0.01）/ level / label / reasons /
            signals 的可序列化字典。

        调用顺序：RAG 管线 -> AnswerConfidence.as_dict()。
        """
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "label": LEVEL_LABELS[self.level],
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
        }


def calculate_evidence_confidence(
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    intent_rule_score: float,
    query: str,
    raw_query: str,
    rewritten_query: str | None,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
    intent_rule_candidate_score: float | None = None,
) -> AnswerConfidence:
    """计算生成前的证据等级。

    这里使用“路径优先、检索分分档、风险降级”的规则，而不是把多个信号
    乘权重后相加。``score`` 仍保留在 [0, 1]，主要用于兼容 API 和 Trace；
    真正有业务含义的是 ``level`` 和 ``reasons``。
    """
    # 检索分只衡量“候选是否相关”，不能直接当作最终答案置信度。
    # retrieval_top_score 是排序后的第一条候选分数：它代表最强证据的相关性，
    # 不是所有候选的平均分，也不是“最终进入 Prompt 的文档数量”。
    # 先统一到 [0, 1]，并保留原始分数供 Trace 诊断。
    normalized_score = normalize_retrieval_score(retrieval_top_score)
    decision_score = _clamp(intent_rule_score)
    rule_candidate_score = _clamp(intent_rule_candidate_score if intent_rule_candidate_score is not None else intent_rule_score)
    history_rewrite_used = _history_rewrite_used(
        query=query,
        raw_query=raw_query,
        rewritten_query=rewritten_query,
    )

    # 评分拆成两段，便于讲义和跟敲代码阅读：
    # 1. 先按回答路径计算基础分；
    # 2. 再按追问改写、低意图分、空上下文等风险统一扣分/封顶。
    score, reasons = _base_evidence_score(
        hit_type=hit_type,
        normalized_score=normalized_score,
        context_count=context_count,
        decision_score=decision_score,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
    )
    score, reasons = _apply_evidence_risk_adjustments(
        score=score,
        reasons=reasons,
        hit_type=hit_type,
        context_count=context_count,
        decision_score=decision_score,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
        history_rewrite_used=history_rewrite_used,
    )

    # 最后统一做边界收口和展示精度处理，确保 API、前端和 Trace 使用同一数值。
    final_score = round(_clamp(score), 2)
    return AnswerConfidence(
        score=final_score,
        level=confidence_level(final_score),
        reasons=reasons,
        signals={
            "hit_type": hit_type,
            "retrieval_top_score": retrieval_top_score,
            "normalized_retrieval_score": round(normalized_score, 2),
            "context_count": context_count,
            "source_count": source_count,
            "intent_rule_score": round(rule_candidate_score, 2),
            "intent_decision_score": round(decision_score, 2),
            "history_rewrite_used": history_rewrite_used,
            "faq_exact_match": faq_exact_match,
            "deterministic_route": deterministic_route,
        },
    )


def _base_evidence_score(
    *,
    hit_type: str,
    normalized_score: float,
    context_count: int,
    decision_score: float,
    deterministic_route: bool,
    faq_exact_match: bool,
) -> tuple[float, list[str]]:
    """按路径和少量硬阈值计算生成前基础分。"""
    if deterministic_route:
        return 0.90, ["deterministic_route"]
    if hit_type == "faq_direct" and faq_exact_match:
        return 0.95, ["faq_exact_match"]
    if hit_type == "insufficient_context":
        return 0.20, ["insufficient_context"]
    if hit_type == "rag" and context_count == 0:
        return 0.20, ["rag_without_context"]
    if hit_type == "faq_direct":
        if normalized_score >= _RETRIEVAL_HIGH_THRESHOLD:
            return 0.85, ["faq_score_direct"]
        if normalized_score >= _RETRIEVAL_MEDIUM_THRESHOLD:
            return 0.65, ["faq_score_direct"]
        return 0.45, ["faq_score_direct"]
    if normalized_score >= _RETRIEVAL_HIGH_THRESHOLD and decision_score >= _INTENT_STABLE_THRESHOLD:
        return 0.85, ["rag_with_context", "strong_retrieval"]
    if normalized_score >= _RETRIEVAL_MEDIUM_THRESHOLD:
        return 0.65, ["rag_with_context"]
    return 0.45, ["rag_with_context", "weak_retrieval"]


def _apply_evidence_risk_adjustments(
    *,
    score: float,
    reasons: list[str],
    hit_type: str,
    context_count: int,
    decision_score: float,
    deterministic_route: bool,
    faq_exact_match: bool,
    history_rewrite_used: bool,
) -> tuple[float, list[str]]:
    """应用少量可解释的风险降级。"""
    adjusted_score = score
    adjusted_reasons = list(reasons)
    if history_rewrite_used:
        adjusted_score = min(adjusted_score, 0.65)
        adjusted_reasons.append("history_rewrite_used")
    if decision_score < _INTENT_STABLE_THRESHOLD and not deterministic_route and not faq_exact_match:
        adjusted_score = min(adjusted_score, 0.54)
        adjusted_reasons.append("low_intent_decision_score")
    if hit_type == "rag" and context_count == 0:
        adjusted_score = min(adjusted_score, 0.20)
        adjusted_reasons.append("no_selected_context")
    return adjusted_score, adjusted_reasons


def record_evidence_confidence(
    context: Any,
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
) -> dict[str, Any]:
    """计算并写入生成前证据置信度，作为后续最终置信度合并的基础。

    参数：
        context: RAG 查询上下文，读取 query/raw_query/rewritten_query 与
            intent_payload（confidence、rule_score），并写入
            answer_confidence 和 retrieval_info。
        hit_type: 回答路径类型。
        retrieval_top_score: top-1 检索相关性分。
        context_count: 最终放入 Prompt 的上下文片段数。
        source_count: 去重来源数量。
        deterministic_route: 是否为确定性路由。
        faq_exact_match: FAQ 是否精确匹配标准问题。

    返回：
        dict: 写入后的置信度字典（含 evidence_confidence 摘要与
        generation_verification 占位）。

    调用顺序：RAG 管线 -> record_evidence_confidence() -> calculate_evidence_confidence()。
    """
    confidence = calculate_evidence_confidence(
        hit_type=hit_type,
        retrieval_top_score=retrieval_top_score,
        context_count=context_count,
        source_count=source_count,
        intent_rule_score=float(context.intent_payload["confidence"]) if context.intent_payload else 0.6,
        intent_rule_candidate_score=float(context.intent_payload["rule_score"]) if context.intent_payload else 0.6,
        query=context.query,
        raw_query=context.raw_query,
        rewritten_query=context.rewritten_query,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
    ).as_dict()
    # 生成前先明确标记 pending：
    # RAG 主链路后续会在 LLM 输出完成后更新为 verified/partial/failed；
    # FAQ 直出、确定性回答和信息不足分支则会标记为 not_applicable。
    confidence["evidence_confidence"] = _confidence_summary(confidence["score"])
    confidence["generation_verification"] = {
        "status": "pending",
        "score": None,
        "reasons": [],
        "signals": {},
    }
    context.answer_confidence = confidence
    context.retrieval_info["answer_confidence"] = confidence
    return confidence


def record_answer_confidence(
    context: Any,
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
) -> dict[str, Any]:
    """兼容旧名称：写入生成前证据置信度。新代码优先用 record_evidence_confidence()。

    参数：
        context: RAG 查询上下文。
        hit_type: 回答路径类型。
        retrieval_top_score: top-1 检索相关性分。
        context_count: 最终放入 Prompt 的上下文片段数。
        source_count: 去重来源数量。
        deterministic_route: 是否为确定性路由。
        faq_exact_match: FAQ 是否精确匹配标准问题。

    返回：
        dict: 同 record_evidence_confidence() 的写入结果。

    调用顺序：RAG 管线 -> record_answer_confidence() -> record_evidence_confidence()。
    """
    return record_evidence_confidence(
        context,
        hit_type=hit_type,
        retrieval_top_score=retrieval_top_score,
        context_count=context_count,
        source_count=source_count,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
    )


def finalize_generated_answer_confidence(
    context: Any,
    *,
    answer: str,
    context_docs: list[Document],
) -> dict[str, Any]:
    """Stage 6：在 LLM 生成完成后核验答案，并更新最终答案置信度。

    生成前的 ``calculate_evidence_confidence()`` 只能判断“证据是否足以支撑生成”，
    无法知道模型是否真的使用了这些证据。因此 RAG 主流程在引用补强之后调用本函数，
    检查最终文本的几个可观测属性：

    1. 答案是否为空；
    2. 事实段落是否带有行内引用；
    3. 引用编号是否落在当前上下文文档范围内；
    4. 答案中的词组和关键数字是否能在上下文中找到词面支撑。

    这里使用的是低延迟、可解释的确定性核验，不声称完成了语义蕴含判断。
    最终分数采用保守合并：``min(evidence_confidence, generation_verification)``。
    原因是“证据很强但答案没有引用/明显偏离上下文”时，不能继续保留高置信度。

    参数：
        context: RAG 查询上下文，读取已写入的 answer_confidence，
            核验后更新 answer_confidence 与 retrieval_info。
        answer: LLM 生成并完成引用补强的最终答案文本。
        context_docs: 进入 Prompt 的上下文文档，用于引用合法性判断
            与词面支撑度计算。

    返回：
        dict: 合并后的最终置信度字典；尚未写入生成前置信度时返回空 dict。

    调用顺序：RAG 管线 -> enforce_answer_citations() ->
    finalize_generated_answer_confidence() -> finish_success()。
    """
    current = dict(context.answer_confidence or {})
    if not current:
        return {}

    verification = calculate_generation_confidence(answer=answer, context_docs=context_docs)
    finalized = combine_answer_confidence(current, verification)
    context.answer_confidence = finalized
    context.retrieval_info["answer_confidence"] = finalized
    return finalized


def combine_answer_confidence(
    evidence_confidence: dict[str, Any],
    generation_confidence: dict[str, Any],
) -> dict[str, Any]:
    """合并生成前证据评分和生成后答案核验，产出最终答案置信度。

    合并策略刻意保守：只要生成后核验给出了有效分数，最终分数就取
    ``min(evidence_score, generation_score)``。这样可以避免“检索很强，但 LLM
    没有引用、引用非法或答案明显偏离上下文”时仍显示高置信。

    参数：
        evidence_confidence: record_evidence_confidence() 写入的生成前
            置信度字典（score、reasons、signals）。
        generation_confidence: calculate_generation_confidence() 产出的
            核验结果；status 为 not_applicable 或 score 为 None 时
            退化为只采用证据分。

    返回：
        dict: 最终答案置信度，含合并分数、等级以及合并后的 signals/reasons，
        并保留 evidence_confidence / generation_verification 两个子结构。

    调用顺序：RAG 管线 -> finalize_generated_answer_confidence() -> combine_answer_confidence()。
    """
    evidence_score = _clamp(float(evidence_confidence.get("score") or 0.0))
    generation_score = generation_confidence.get("score")
    if generation_confidence.get("status") == "not_applicable" or generation_score is None:
        final_score = evidence_score
    else:
        final_score = min(evidence_score, _clamp(float(generation_score)))

    final_level = confidence_level(final_score)
    signals = dict(evidence_confidence.get("signals") or {})
    generation_signals = dict(generation_confidence.get("signals") or {})
    signals.update(
        {
            "evidence_confidence_score": round(evidence_score, 2),
            "generation_verification_score": (
                round(float(generation_score), 2) if generation_score is not None else None
            ),
            "generation_verification_status": generation_confidence.get("status"),
            **generation_signals,
        }
    )
    return {
        "score": round(_clamp(final_score), 2),
        "level": final_level,
        "label": LEVEL_LABELS[final_level],
        "reasons": _merge_reasons(
            evidence_confidence.get("reasons") or [],
            generation_confidence.get("reasons") or [],
        ),
        "signals": signals,
        "evidence_confidence": _confidence_summary(evidence_score),
        "generation_verification": generation_confidence,
    }


def mark_answer_confidence_not_applicable(context: Any, *, reason: str) -> dict[str, Any]:
    """标记没有经过 LLM 生成的答案分支，不虚构生成后核验结果。

    FAQ 直出、确定性路由和 ``insufficient_context`` 都没有经过最终 LLM 生成，
    这些分支应保留证据置信度，但把生成核验标记为 ``not_applicable``。

    参数：
        context: RAG 查询上下文，更新其 answer_confidence 与 retrieval_info。
        reason: 标记原因，写入 generation_verification.reasons 供 Trace 排查。

    返回：
        dict: 更新后的置信度字典；尚无生成前置信度时返回空 dict。

    调用顺序：RAG 管线 -> _finish_with_single_answer() ->
    mark_answer_confidence_not_applicable()。
    """
    current = dict(context.answer_confidence or {})
    if not current:
        return {}
    score = _clamp(float(current.get("score") or 0.0))
    current["evidence_confidence"] = _confidence_summary(score)
    current["generation_verification"] = {
        "status": "not_applicable",
        "score": None,
        "reasons": [reason],
        "signals": {"generation_attempted": False},
    }
    signals = dict(current.get("signals") or {})
    signals.update(
        {
            "evidence_confidence_score": round(score, 2),
            "generation_verification_score": None,
            "generation_verification_status": "not_applicable",
            "generation_attempted": False,
        }
    )
    current["signals"] = signals
    context.answer_confidence = current
    context.retrieval_info["answer_confidence"] = current
    return current


def calculate_generation_confidence(
    *,
    answer: str,
    context_docs: list[Document],
) -> dict[str, Any]:
    """对生成答案做低延迟的引用和词面支撑核验。

    该函数故意不把“出现了来源编号”直接等同于“事实已被证明”：
    - 只出现在“参考来源”尾部的编号不算行内引用；
    - 超出上下文范围的编号会被标记为非法；
    - 每个事实段落会计算与上下文的词面重合度；
    - 词面重合度只是启发式信号，不替代 NLI、LLM Judge 或人工复核。

    参数：
        answer: LLM 生成、完成引用补强后的答案文本。
        context_docs: 进入 Prompt 的上下文文档，文档数量即合法引用编号上限，
            也是词面支撑度计算的对照语料。

    返回：
        dict: 包含 score、status、reasons 和 signals 的可序列化核验结果。

    调用顺序：RAG 管线 -> finalize_generated_answer_confidence() -> calculate_generation_confidence()。
    """
    # 先去掉首尾空白，后面的“答案为空”判断和字符数统计都以真正的可见内容为准。
    # 这里核验的是模型最终交付给用户的文本，而不是 Prompt 或检索结果本身。
    clean_answer = answer.strip()
    if not clean_answer:
        # 空答案没有任何事实可以核验，因此直接失败；不进入后续的引用和重合度计算。
        return _generation_confidence_result(
            score=0.0,
            status="failed",
            reasons=["answer_empty"],
            signals=_generation_signals(answer_char_count=0),
        )

    if not context_docs:
        # 没有进入 Prompt 的上下文时，既无法判断引用编号是否越界，也无法判断答案
        # 是否得到知识库支持。这里返回 not_applicable，表示“没有核验条件”，不是“答案已验证”。
        return _generation_confidence_result(
            score=0.0,
            status="not_applicable",
            reasons=["no_context_for_generation_verification"],
            signals=_generation_signals(answer_char_count=len(clean_answer)),
        )

    # 生成链路可能在答案末尾追加“参考来源”清单。清单只用于展示，不能当作事实正文，
    # 否则清单里的 [1]、[2] 会把行内引用覆盖率虚高。
    answer_body = _answer_body_without_reference_section(clean_answer)

    # 把正文切成若干“事实单元”，后面按事实单元统计“有多少内容带了引用”。
    # 这里采用轻量规则拆分，目的是低延迟诊断，不把它包装成严格的语义断言识别。
    claims = _extract_claim_units(answer_body)
    if not claims and answer_body:
        # 如果规则没有拆出有效片段，但正文确实存在，就保留整段作为一个兜底事实单元，
        # 避免因为拆分规则过严而把真实答案误判成“没有可核验内容”。
        claims = [answer_body]

    # 一次遍历同时收集三类信号：事实单元数量、合法/非法引用编号、上下文词面支撑度。
    # 后续分数只消费这个汇总结果，不直接依赖原始文本，便于 Trace 和接口复用。
    evidence = _inspect_generated_claims(claims=claims, context_docs=context_docs)

    # 先按明确的分段规则计算生成后核验分：非法引用优先降级；引用覆盖率和上下文
    # 支撑度都达标才到 0.85，只满足其中一项到 0.65，两项都不足为 0.35。
    score = _score_generation_grounding(
        citation_coverage=evidence["citation_coverage"],
        context_overlap=evidence["context_overlap"],
        invalid_citation_count=len(evidence["invalid_numbers"]),
    )
    # 统一做边界保护和小数位处理，保证 API、日志和最终置信度合并拿到稳定格式。
    score = round(_clamp(score), 2)

    # 分数用于机器处理，status/reasons 用于人和前端理解“为什么是这个分数”。
    # 两者分开计算，避免为了展示原因而改变核心评分规则。
    status, reasons = _generation_status_and_reasons(
        score=score,
        citation_coverage=evidence["citation_coverage"],
        context_overlap=evidence["context_overlap"],
        invalid_citation_count=len(evidence["invalid_numbers"]),
    )
    # 最后将结果封装成固定结构。signals 保留原始诊断证据，便于排查“哪条回答没有引用”
    # 或“哪些引用编号越界”，而 score/status/reasons 供上层合并和展示。
    return _generation_confidence_result(
        score=score,
        status=status,
        reasons=reasons,
        signals=_generation_signals(
            answer_char_count=len(clean_answer),
            claim_count=evidence["claim_count"],
            cited_claim_count=evidence["cited_claim_count"],
            citation_coverage=evidence["citation_coverage"],
            context_overlap=evidence["context_overlap"],
            valid_citation_numbers=sorted(evidence["valid_numbers"]),
            invalid_citation_numbers=sorted(evidence["invalid_numbers"]),
            inline_citation_count=len(_CITATION_RE.findall(answer_body)),
        ),
    )


def _generation_confidence_result(
    *,
    score: float,
    status: str,
    reasons: list[str],
    signals: dict[str, Any],
) -> dict[str, Any]:
    """构造生成后核验的统一返回结构。

    参数：
        score: 核验分（[0,1]）。
        status: 核验状态，取值 verified / partial / failed / not_applicable。
        reasons: 核验原因列表。
        signals: 供前端和 Trace 展示的信号字典。

    返回：
        dict: 统一四字段核验结构 {score, status, reasons, signals}。

    调用顺序：calculate_generation_confidence() -> _generation_confidence_result()。
    """
    # 保持四个字段始终存在，调用方就不需要根据不同失败分支编写多套解析逻辑。
    # 该函数只负责协议封装，不在这里再次计算或修正 score。
    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "signals": signals,
    }


def evaluate_generated_answer(
    *,
    answer: str,
    context_docs: list[Document],
) -> dict[str, Any]:
    """兼容旧名称：新代码优先调用 calculate_generation_confidence()。

    参数：
        answer: LLM 生成的答案文本。
        context_docs: 进入 Prompt 的上下文文档。

    返回：
        dict: 与 calculate_generation_confidence() 相同的核验结果。

    调用顺序：RAG 管线 -> evaluate_generated_answer() -> calculate_generation_confidence()。
    """
    return calculate_generation_confidence(answer=answer, context_docs=context_docs)


def faq_exact_match(query: str, doc: Document | None) -> bool:
    """判断 FAQ 命中是否为标准问题精确匹配。

    参数：
        query: 归一化后的用户问题。
        doc: FAQ 候选文档，从 metadata 读取 standard_question 作为比对基准。

    返回：
        bool: 查询与标准问题完全一致时为 True；doc 为 None 时为 False。

    调用顺序：RAG 管线 -> faq_exact_match()。
    """
    if doc is None:
        return False
    metadata = doc.metadata
    standard_question = str(metadata.get("standard_question") or metadata.get("question") or doc.page_content).strip()
    return query.strip() == standard_question


def normalize_retrieval_score(score: float) -> float:
    """把检索/重排分数压到 [0, 1]，仅用于置信度派生，不改变排序。

    Milvus/LangChain 返回值在本项目里按”越大越相关”使用；CrossEncoder 有时会返回
    大于 1 的 logits，因此这里用平滑压缩而不是直接截断（直接截断会丢失区分度：
    logit=5 和 logit=50 在截断后都是 1.0，但平滑压缩仍能区分出差异）。

    参数：
        score: 检索/重排后的原始相关性分，可大于 1。

    返回：
        float: [0,1] 区间的归一化分数；score <= 0 时返回 0.0。

    调用顺序：RAG 管线 -> normalize_retrieval_score()。
    """
    if score <= 0:
        return 0.0
    if score <= 1:
        return score
    return 1 - (1 / (1 + score))


def confidence_level(score: float) -> str:
    """将连续分数映射为前端展示等级。

    参数：
        score: [0,1] 区间的置信度分数。

    返回：
        str: "high"（>=0.82）/ "medium"（>=0.55）/ "low"（<0.55）。

    调用顺序：RAG 管线 -> confidence_level()。
    """
    if score >= 0.82:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _history_rewrite_used(*, query: str, raw_query: str, rewritten_query: str | None) -> bool:
    """判断当前回答是否依赖历史追问改写。

    参数：
        query: 当前用于业务处理的归一化问题。
        raw_query: 用户原始输入。
        rewritten_query: 改写后的问题，无改写时传 ``None``。

    返回：
        bool: 改写后问题既不同于当前问题也不同于原始问题时为 True，
        即本次回答确实借助了历史改写补全。

    调用顺序：RAG 管线 -> calculate_evidence_confidence() -> _history_rewrite_used()。
    """
    rewritten = (rewritten_query or "").strip()
    if not rewritten:
        return False
    return rewritten not in {query.strip(), raw_query.strip()}


def _clamp(value: float) -> float:
    """将数值限制在 [0, 1] 区间。

    参数：
        value: 任意数值。

    返回：
        float: 截断到 [0,1] 后的数值。

    调用顺序：RAG 管线 -> _clamp()。
    """
    return max(0.0, min(1.0, value))


def _confidence_summary(score: float) -> dict[str, Any]:
    """构造证据置信度摘要，避免前端重复解释同一个 score。

    参数：
        score: 置信度分数（允许越界，内部先 clamp 再取两位小数）。

    返回：
        dict: {score, level, label} 三元摘要，供 API 与 Trace 复用。

    调用顺序：RAG 管线 -> _confidence_summary()。
    """
    normalized = round(_clamp(score), 2)
    level = confidence_level(normalized)
    return {
        "score": normalized,
        "level": level,
        "label": LEVEL_LABELS[level],
    }


def _merge_reasons(*groups: list[str]) -> list[str]:
    """合并 reasons 并保持首次出现顺序。

    参数：
        *groups: 多组原因列表（如生成前原因 + 生成后原因）。

    返回：
        list[str]: 去重且保持首次出现顺序的原因列表，保证前端展示顺序稳定。

    调用顺序：combine_answer_confidence() -> _merge_reasons()。
    """
    merged: list[str] = []
    for group in groups:
        for reason in group:
            if reason not in merged:
                merged.append(reason)
    return merged


def _answer_body_without_reference_section(answer: str) -> str:
    """移除末尾自动补充的来源列表，只核验答案事实正文。

    原因：答案尾部常带"参考来源"清单，其中的编号不属于行内引用，
    若参与核验会把引用覆盖统计虚高，因此核验前先剔除。

    参数：
        answer: 完整答案文本。

    返回：
        str: 去掉来源清单后的正文；无来源清单时原样返回。

    调用顺序：calculate_generation_confidence() -> _answer_body_without_reference_section()。
    """
    # 只截断自动来源清单的起点，不删除正文中的 [1] 这类行内引用；
    # 正文引用正是后续“引用覆盖率”要检查的对象。
    match = _REFERENCE_SECTION_RE.search(answer)
    return answer[: match.start()].strip() if match else answer.strip()


def _inspect_generated_claims(
    *,
    claims: list[str],
    context_docs: list[Document],
) -> dict[str, Any]:
    """逐条核验事实单元的引用编号和上下文词面重合度。

    参数：
        claims: 从答案正文拆分出的事实单元列表。
        context_docs: 进入 Prompt 的上下文文档，文档数量即合法引用编号上限。

    返回：
        dict: 含 claim_count / cited_claim_count / citation_coverage /
        context_overlap / valid_numbers / invalid_numbers 的核验汇总。

    调用顺序：calculate_generation_confidence() -> _inspect_generated_claims()。
    """
    # 上下文按 [1]...[N] 注入 Prompt，因此当前请求的合法引用编号上限就是 N。
    # 不能使用全局知识库文档数量，否则模型可能引用本次请求根本没有看到的文档。
    max_citation_number = len(context_docs)
    valid_numbers: set[int] = set()
    invalid_numbers: set[int] = set()
    cited_claim_count = 0
    overlap_values: list[float] = []
    for claim in claims:
        # 一个事实单元可能带多个引用，例如“支持 FAQ 和制度文档 [1][2]”。
        # 使用 set 去重，避免同一个编号重复出现影响诊断统计。
        numbers = {int(value) for value in _CITATION_RE.findall(claim)}
        # 原因：引用编号与上下文文档一一对应（1..N），越界编号视为非法引用，
        # 是"模型编造来源编号"的可观测信号，供后续封顶扣分。
        valid = {number for number in numbers if 1 <= number <= max_citation_number}
        invalid = numbers - valid
        valid_numbers.update(valid)
        invalid_numbers.update(invalid)
        if valid:
            # 这里按“有无至少一个合法引用”计数，而不是按引用编号个数计数，
            # 因为我们要衡量的是“多少个事实单元被引用覆盖”。
            cited_claim_count += 1
        # 即使引用非法，也继续计算词面重合度。这样 Trace 能区分“内容有依据但编号写错”
        # 与“内容本身也没有上下文支撑”这两种不同问题。
        overlap_values.append(_context_overlap(claim, context_docs))

    claim_count = len(claims)
    # coverage 的分母是事实单元数量，保证长答案和短答案都按“被引用的事实比例”比较；
    # context_overlap 则是各事实单元重合度的平均值，保留每个单元的支撑情况。
    return {
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "citation_coverage": cited_claim_count / claim_count if claim_count else 0.0,
        "context_overlap": sum(overlap_values) / len(overlap_values) if overlap_values else 0.0,
        "valid_numbers": valid_numbers,
        "invalid_numbers": invalid_numbers,
    }


def _score_generation_grounding(
    *,
    citation_coverage: float,
    context_overlap: float,
    invalid_citation_count: int,
) -> float:
    """按三个可解释门槛计算生成后核验分。

    这里故意使用离散分段，而不是复杂加权公式：面试、Trace 和问题排查时，
    可以直接解释“哪一道门槛没有通过”。
    """
    if invalid_citation_count:
        # 非法编号意味着模型引用了本次上下文之外的来源，属于来源可信度风险，
        # 因此优先返回 0.35，不能被较高的词面重合度抵消。
        return 0.35
    has_citations = citation_coverage >= _CITATION_COVERAGE_THRESHOLD
    has_context_support = context_overlap >= _CONTEXT_SUPPORT_THRESHOLD
    if has_citations and has_context_support:
        # 引用形式正确，同时答案词面能在上下文中找到支撑，进入最高生成核验档位。
        return 0.85
    if has_citations or has_context_support:
        # 只有一项成立：答案可能“引用写得完整但内容不贴合”，也可能“内容贴合但没有
        # 逐条引用”，所以保留为部分可信而不是完全通过。
        return 0.65
    # 没有引用覆盖，也没有上下文词面支撑，说明生成结果缺少可观测依据。
    return 0.35


def _generation_status_and_reasons(
    *,
    score: float,
    citation_coverage: float,
    context_overlap: float,
    invalid_citation_count: int,
) -> tuple[str, list[str]]:
    """把生成后核验分映射为状态和可解释原因。

    参数：
        score: 生成后核验分。
        citation_coverage: 带行内引用的事实单元占比。
        context_overlap: 与上下文的词面支撑比例。
        invalid_citation_count: 非法引用编号数量。

    返回：
        tuple[str, list[str]]: (核验状态, 原因列表)；状态取值
        verified / partial / failed。

    调用顺序：calculate_generation_confidence() -> _generation_status_and_reasons()。
    """
    reasons: list[str] = []
    # 原因：三个独立风险信号各自记录——引用覆盖不足、上下文支撑不足、
    # 非法引用，只要其一存在就不可能判定为 verified。
    if citation_coverage < _CITATION_COVERAGE_THRESHOLD:
        # 该原因说明“有事实单元没有合法行内引用”，不是说所有引用都非法。
        reasons.append("low_inline_citation_coverage")
    if context_overlap < _CONTEXT_SUPPORT_THRESHOLD:
        # 该原因说明答案用词与检索上下文的交集不足，属于轻量级的内容支撑风险。
        reasons.append("low_context_overlap")
    if invalid_citation_count:
        # 单独保留非法引用原因，方便定位是编号越界还是普通的覆盖率不足。
        reasons.append("invalid_citation_reference")
    if (
        score >= 0.82
        and citation_coverage >= _CITATION_COVERAGE_THRESHOLD
        and context_overlap >= _CONTEXT_SUPPORT_THRESHOLD
        and invalid_citation_count == 0
    ):
        # verified 要同时满足高分、覆盖率、上下文支撑和无非法引用，不能只看 score。
        reasons.append("generation_grounded")
        return "verified", reasons
    if score >= 0.55:
        # 0.65 档位代表至少通过一项检查，因此标记为 partial，而不是成功或失败。
        return "partial", reasons
    # 0.35 或更低表示生成结果没有足够可观测依据。
    return "failed", reasons


def _generation_signals(
    *,
    answer_char_count: int,
    claim_count: int = 0,
    cited_claim_count: int = 0,
    citation_coverage: float = 0.0,
    context_overlap: float = 0.0,
    valid_citation_numbers: list[int] | None = None,
    invalid_citation_numbers: list[int] | None = None,
    inline_citation_count: int = 0,
) -> dict[str, Any]:
    """构造前端和 Trace 展示用的生成后核验信号。

    参数：
        answer_char_count: 答案字符数。
        claim_count: 事实单元数量。
        cited_claim_count: 带行内引用的事实单元数量。
        citation_coverage: 引用覆盖（[0,1]）。
        context_overlap: 词面支撑比例（[0,1]）。
        valid_citation_numbers: 合法引用编号列表。
        invalid_citation_numbers: 非法引用编号列表。
        inline_citation_count: 行内引用出现次数。

    返回：
        dict: 序列化信号字典，reference_section_excluded 恒为 True，
        表示核验前已剔除来源清单。

    调用顺序：calculate_generation_confidence() -> _generation_signals()。
    """
    # signals 是诊断数据，不参与重新评分；这里统一 round 只为让日志和 JSON 更易读。
    # 空列表用 [] 而不是 None，便于前端直接遍历，也避免不同分支返回不同类型。
    return {
        "answer_char_count": answer_char_count,
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "citation_coverage": round(citation_coverage, 2),
        "context_overlap": round(context_overlap, 2),
        "valid_citation_numbers": valid_citation_numbers or [],
        "invalid_citation_numbers": invalid_citation_numbers or [],
        "inline_citation_count": inline_citation_count,
        "reference_section_excluded": True,
    }


def _extract_claim_units(answer: str) -> list[str]:
    """按句号、分号和换行拆分事实单元，过滤标题和过短片段。

    参数：
        answer: 去掉来源清单后的答案正文。

    返回：
        list[str]: 事实单元列表；过短片段（<5 字）与孤立引用编号会被过滤。

    调用顺序：calculate_generation_confidence() -> _extract_claim_units()。
    """
    claims: list[str] = []
    for raw_unit in _CLAIM_SPLIT_RE.split(answer):
        # 原因：去掉"1. / - / •"等列表前缀，避免序号被误当作事实单元内容。
        unit = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", raw_unit).strip()
        if claims and _CITATION_RE.fullmatch(unit):
            # 原因：引用编号自成一段时（如换行后单独一行"[3]"），归并到上一个
            # 事实单元末尾，保证引用覆盖按"事实单元"而非"编号行"统计。
            claims[-1] = f"{claims[-1]} {unit}"
            continue
        # 判断片段是否有意义时先去掉引用标记，但最终保存原始 unit，保证后面仍能读取 [n]。
        plain_unit = _CITATION_RE.sub("", unit).strip()
        # 原因：短于 5 字的片段多为标题或残留标点，不构成可核验的事实单元。
        if len(plain_unit) >= 5:
            claims.append(unit)
    return claims


def _context_overlap(claim: str, context_docs: list[Document]) -> float:
    """计算答案事实单元与上下文的词面支撑比例。

    中文没有天然空格分词，因此连续中文片段按相邻双字组拆分；英文、数字和文件名
    按完整词片段处理。该比例只用于低成本风险提示，不代表语义蕴含概率。

    参数：
        claim: 单个事实单元（引用编号已剔除）。
        context_docs: 进入 Prompt 的上下文文档，拼接为对照语料。

    返回：
        float: [0,1] 词面支撑比例；claim 或上下文无 token 时返回 0.0。

    调用顺序：calculate_generation_confidence() -> _inspect_generated_claims() -> _context_overlap()。
    """
    # 引用编号属于格式信息，不是事实内容，必须先移除，否则它会污染词面统计。
    claim_tokens = _text_tokens(_CITATION_RE.sub("", claim))
    if not claim_tokens:
        return 0.0
    # 将本次请求的所有上下文合并成一个对照集合：只要某个 token 出现在任一文档中，
    # 就视为得到了词面层面的支持；这不是严格的“该文档蕴含该结论”。
    context_text = "\n".join(doc.page_content for doc in context_docs)
    context_tokens = _text_tokens(context_text)
    if not context_tokens:
        return 0.0
    return len(claim_tokens & context_tokens) / len(claim_tokens)


def _text_tokens(text: str) -> set[str]:
    """提取可用于答案/上下文词面比较的中英文 token。

    参数：
        text: 待提取 token 的文本。

    返回：
        set[str]: token 集合；中文按相邻双字组切分（单字按原字保留），
        英文/数字/文件名按完整词片段小写保留。

    调用顺序：_context_overlap() -> _text_tokens()。
    """
    tokens: set[str] = set()
    for segment in _CJK_SEGMENT_RE.findall(text):
        if len(segment) == 1:
            # 单个汉字没有相邻字符可组成二字片段，因此保留它本身作为最小 token。
            tokens.add(segment)
            continue
        # 中文通常没有空格分词，这里用相邻双字组近似切分，例如“退款流程”会得到
        # “退款”“款流”“流程”，比按单字比较更能减少偶然重合。
        tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    # 英文、数字和文件名按完整片段比较，并统一小写，避免大小写造成无意义的不匹配。
    tokens.update(
        token.lower()
        for token in _WORD_RE.findall(text)
        if len(token) >= 2
    )
    return tokens
