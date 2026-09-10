"""RAG 主流程中的业务步骤。

这里的函数名称刻意写得直白：准备检索、检索 FAQ、检索文档、准备生成参数。
主流程 `rag.py` 只负责把这些步骤串起来，细节放在这里，阅读时不用在一个超长函数里
来回跳。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from qa_core.config.rules import get_rule_config
from qa_core.config.settings import get_settings
from qa_core.cache.manager import get_cache_manager
from qa_core.intent.classifier import IntentResult, classify_direct_intent, classify_intent, infer_source
from qa_core.llm.client import get_chat_model
from qa_core.memory.history import format_messages
from qa_core.pipeline.confidence import record_evidence_confidence
from qa_core.pipeline.context import (
    build_context,
    direct_faq_answer,
    effective_source_filter as resolve_effective_source_filter,
    select_context_docs,
)
from qa_core.pipeline.query_variants import generate_query_variants
from qa_core.pipeline.rewrite import rewrite_query_if_needed
from qa_core.pipeline.runtime import RAGQueryContext
from qa_core.prompts.profiles import PromptProfile
from qa_core.prompts.selector import build_answer_prompt_profile
from qa_core.retrieval.factory import get_faq_store
from qa_core.retrieval.filters import validate_source_filter
from qa_core.retrieval.results import RetrievalResult
from qa_core.retrieval.strategy import RetrievalPlan, build_retrieval_plan
from qa_core.scenarios.boundary import detect_source_boundary

@dataclass
class RetrievalPreparation:
    """prepare_retrieval() 的输出包，供下游 search_faq / search_doc / prepare_answer 消费。

    调用顺序：QAService/RAG 管线 -> RetrievalPreparation。
    """

    history_messages: list[Any]       # 压缩后的对话上下文，用于拼接 user_prompt
    intent: IntentResult              # 检索类意图识别结果，供改写、检索计划和 Prompt 选择使用
    effective_source_filter: str | None  # 最终生效的 source 过滤项
    rewritten_query: str              # 改写后的独立检索问题（或原问题）
    plan: RetrievalPlan               # ★ 检索计划，控制后续所有检索参数
    query_variants: list[str]         # 同义检索表达列表，原问题排第一位
    prompt_profile: PromptProfile     # 回答 Prompt 模板档位


@dataclass
class AnswerPreparation:
    """调用大模型前需要准备好的 Prompt 和来源信息。

    调用顺序：QAService/RAG 管线 -> AnswerPreparation。
    """

    context_docs: list[Document]
    sources: list[dict[str, Any]]
    hit_type: str
    system_prompt: str
    user_prompt: str


RouteName = Literal["direct_answer", "faq_exact", "retrieval"]


@dataclass
class RouteDecision:
    """在线问答的低成本路由结果，统一承载直答、FAQ 精确命中和继续检索三类分支。

    intent 描述“用户想干什么”，route 描述“系统下一步怎么处理”。FAQ 精确命中是
    route=faq_exact，同时 intent=FAQ_QUERY；它不是新的用户意图。

    调用顺序：QAService/RAG 管线 -> RouteDecision。
    """

    route: RouteName
    answer: str | None = None
    intent: IntentResult | None = None
    reason: str = ""


def should_try_faq_fast_path(query: str, scenario) -> bool:
    """判断短问题是否值得先做 FAQ 精确匹配探测。

    参数：
        query: 已经过 normalize_user_query() 处理的业务有效问题
        scenario: ScenarioDefinition 业务场景定义

    返回：
        bool: True 表示允许进入 FAQ 候选检索探测；它不代表已经可以直出。

    说明：
        faq_fast_path.max_chars 是工程保护阈值，不是官方标准。它用于避免长复合问题
        被拿去做 FAQ 精确匹配；这类问题应进入完整检索和生成链路。
        句式特征和 source 推断满足任意一个即可先探测；真正直出仍由后续标准问题
        精确匹配决定。

    调用顺序：QAService/RAG 管线 -> should_try_faq_fast_path()。
    """
    rules = get_rule_config().faq_fast_path
    compact_query = query.strip()
    if not compact_query or len(compact_query) > rules.max_chars or "\n" in compact_query:
        return False
    return bool(rules.hint_matches(compact_query) or infer_source(compact_query, scenario))


def _exact_faq_answer(query: str, faq_result: RetrievalResult) -> tuple[str | None, RetrievalResult]:
    """从 FAQ 候选中寻找与标准问题完全一致的答案（仅允许精确匹配直出）。

    参数：
        query: 用户原始提问
        faq_result: FAQ 检索结果（含 hits 列表）

    返回：
        tuple: (answer_str or None, 可能重排后的 faq_result)

    调用顺序：QAService/RAG 管线 -> _exact_faq_answer()。
    """
    for index, hit in enumerate(faq_result.hits):
        answer = direct_faq_answer(query, hit.document, hit.score, threshold=float("inf"))
        if not answer:
            continue
        if index:
            # 精确匹配项不在首位时将其提到列表最前，保证来源展示顺序一致
            # 示例：hits = [A, B, C, D]，index=2 命中 C
            #       before = hits[:2] = [A, B]
            #       after  = hits[3:]  = [D]
            #       → reordered = [C, A, B, D]
            before = faq_result.hits[:index]
            after = faq_result.hits[index + 1 :]
            reordered = [hit] + before + after
            faq_result = RetrievalResult(
                hits=reordered,
                query=faq_result.query,
                source_type=faq_result.source_type,
                elapsed_ms=faq_result.elapsed_ms,
            )
        return answer, faq_result
    return None, faq_result


def _fast_faq_plan_payload(plan: RetrievalPlan, *, faq_top_k: int) -> dict[str, Any]:
    """返回 FAQ 精确路由实际执行的诊断计划。

    调用顺序：QAService/RAG 管线 -> _fast_faq_plan_payload()。
    """
    payload = plan.as_dict()
    payload.update(
        {
            "run_faq": True,
            "run_doc": False,
            "faq_top_k": faq_top_k,
            "doc_top_k": 0,
            "rerank": False,
            "final_context_top_n": 0,
            "use_query_variants": False,
            "faq_direct_exact_only": True,
            "match_policy": "standard_question_exact",
            "reason": f"{plan.reason}_faq_exact_fast_path",
        }
    )
    return payload


def try_fast_faq_direct_answer(context: RAGQueryContext) -> tuple[str | None, IntentResult]:
    """查询路由内部的 FAQ 精确路径：精确命中标准 FAQ 时直接返回。★★★ 核心

    执行流程：
    1. 调用方已在 decide_route() 中完成 source 校验、确定性直答和 fast path 探测门控判断
    2. 构建轻量检索计划（仅 FAQ，不查文档，不 rerank）
    3. 执行 FAQ 混合检索（仅单变体）
    4. 从候选中寻找精确匹配的标准问题答案
    5. 命中的直接设置 hit_type 并返回答案；未命中的缓存结果供主链路复用

    参数：
        context: RAGQueryContext 请求级状态

    返回：
        tuple[str | None, IntentResult]: 精确匹配答案及其 FAQ_QUERY 意图；未命中时答案为 None
    """
    # 根据问题关键词推断可能的业务分类过滤项
    suggested_source = infer_source(context.query, context.scenario)
    intent = IntentResult(
        intent="FAQ_QUERY",
        rule_score=0.98,
        reason="faq_fast_exact_match_probe",
        requires_rewrite=False,
        suggested_source=suggested_source,
    )
    context.intent_payload = intent.as_dict()
    context.rewritten_query = context.query

    # 确定 fast path 的 source 过滤项：快路径只用前端显式选择，不自动推断
    effective_source_filter = context.run_stage(
        "fast_resolve_source_filter",
        lambda: resolve_effective_source_filter(context.source_filter, None, context.scenario),
    )

    # 复用通用检索计划取得 FAQ 候选数，再写入 fast path 的实际执行计划。
    plan = context.run_stage("fast_build_retrieval_plan", lambda: build_retrieval_plan(context.query, intent))
    # 这里需要覆盖后续主链路可能使用的 FAQ 候选量：精确命中时直接返回；未命中时，
    # 原问题候选可与查询变体的新增候选合并，避免再把原问题完整检索一遍。
    fast_faq_top_k = max(plan.faq_top_k, get_settings().faq_short_query_top_k)

    # FAQ 精确直出不构造 Prompt，也不选择 Prompt Profile；未命中后才进入完整检索/生成链路。
    # 构造检索诊断信息快照，供后续 trace 和错误排查使用
    context.retrieval_info = {
        "fast_path": "faq_exact_match",
        "plan": _fast_faq_plan_payload(plan, faq_top_k=fast_faq_top_k),
        "fast_faq_top_k": fast_faq_top_k,
        "query_variants": [context.query],
        "scenario_id": context.scenario.scenario_id,
        "scenario_name": context.scenario.display_name,
        "data_scope": context.data_scope.as_dict(),
        "source_filter": effective_source_filter,
        "kb_version": context.active_kb_version,
    }

    # 获取当前场景的 FAQ Milvus 混合检索集合，执行单查询变体的快速检索
    cache = get_cache_manager()
    cache_key = cache.retrieval_key(
        kind="retrieval",
        scenario_id=context.scenario.scenario_id,
        collection_name=context.scenario.faq_collection,
        source_type="faq",
        data_scope=context.data_scope,
        kb_version=context.active_kb_version,
        source_filter=effective_source_filter,
        query_variants=[context.query],
        k=fast_faq_top_k,
        rerank=False,
    )

    def run_fast_faq_search() -> RetrievalResult:
        """执行 FAQ 快路径检索：优先读缓存，未命中则执行 Milvus 混合检索并回写缓存。

        FAQ 快路径只查询 FAQ 集合（不查文档），不启用 rerank（减少延迟），
        top_k 由 fast_faq_top_k 参数控制。结果用于精确匹配标准问题直出。

        参数：
            无（闭包依赖外层变量：cache_key、fast_faq_top_k、
            effective_source_filter 以及 context 的场景/数据范围/版本信息）。

        返回：
            RetrievalResult: 包含 FAQ 候选列表和元数据的检索结果对象。

        调用顺序：try_fast_faq_direct_answer() -> context.run_stage("faq_fast_retrieval")
        -> run_fast_faq_search()。
        """
        cached = cache.get_retrieval_result(cache_key, source_type="faq")
        context.record_cache_event(
            stage="faq_fast_retrieval",
            enabled=cache.enabled,
            hit=cached is not None,
            source_type="faq",
            key=cache_key,
        )
        if cached is not None:
            return cached
        result = get_faq_store(context.scenario.faq_collection).search_many(
            [context.query],
            k=fast_faq_top_k,
            source_filter=effective_source_filter,
            kb_version=context.active_kb_version,
            data_scope=context.data_scope,
            scenario_id=context.scenario.scenario_id,
            source_type="faq",
            rerank=False,
        )
        cache.set_retrieval_result(cache_key, result, source_type="faq")
        return result

    faq_result = context.run_stage("faq_fast_retrieval", run_fast_faq_search)
    context.retrieval_info["faq_elapsed_ms"] = round(faq_result.elapsed_ms, 2)
    context.retrieval_info["faq_top_score"] = faq_result.top_score

    # 从 FAQ 候选中找与用户问题标准问题完全一致的答案，只允许精确匹配直出
    answer, faq_result = _exact_faq_answer(context.query, faq_result)
    if not answer:
        # 缓存 fast path 的 FAQ 召回结果，主链路同参数时复用避免重复检索
        context.fast_faq_result = faq_result
        context.fast_faq_source_filter = effective_source_filter
        context.fast_faq_top_k = fast_faq_top_k
        return None, intent
    context.hit_type = "faq_direct"
    context.sources = faq_result.source_payloads()
    context.retrieval_info["fast_path_hit"] = True
    # FAQ fast path 只允许标准问题精确匹配直出，因此最终答案置信度按精确 FAQ 分支记录
    record_evidence_confidence(
        context,
        hit_type="faq_direct",
        retrieval_top_score=faq_result.top_score,
        context_count=1,
        source_count=len(context.sources),
        faq_exact_match=True,
    )
    return answer, intent


def decide_route(context: RAGQueryContext) -> RouteDecision:
    """Stage 1：统一的低成本查询路由：直答/边界、FAQ 精确命中、或继续完整检索准备。★★★ 核心

    这个阶段只做确定性、低成本决策：
    1. source_filter 校验；
    2. 问候、越界、短句转人工、source 边界直接返回；
    3. 短标准问答先做 FAQ 精确匹配探测，命中则以 FAQ_QUERY 意图直出；
    4. 都不命中时返回 retrieval，由后续 prepare_retrieval() 做检索准备。

    返回：
        RouteDecision: route=direct_answer / faq_exact / retrieval

    调用顺序：QAService/RAG 管线 -> decide_route()。
    """
    context.run_stage(
        "validate_source",
        lambda: validate_source_filter(context.source_filter, context.scenario.valid_sources),
    )

    direct_intent = context.run_stage(
        "route_direct_intent",
        lambda: classify_direct_intent(context.query, context.scenario),
    )
    if direct_intent:
        _apply_direct_route(context, direct_intent, route="direct_answer", reason=direct_intent.reason)
        return RouteDecision(
            route="direct_answer",
            answer=direct_intent.direct_answer,
            intent=direct_intent,
            reason=direct_intent.reason,
        )

    boundary_answer = detect_and_apply_boundary_answer(context)
    if boundary_answer:
        intent = IntentResult(
            intent="OUT_OF_SCOPE",
            direct_answer=boundary_answer,
            rule_score=0.98,
            reason=context.retrieval_info["boundary_reason"],
            requires_rewrite=False,
            suggested_source=None,
        )
        _apply_direct_route(context, intent, route="direct_answer", reason=intent.reason)
        return RouteDecision(route="direct_answer", answer=boundary_answer, intent=intent, reason=intent.reason)

    if should_try_faq_fast_path(context.query, context.scenario):
        answer, intent = try_fast_faq_direct_answer(context)
        if answer:
            context.hit_type = "faq_direct"
            context.retrieval_info["route"] = "faq_exact"
            context.retrieval_info["route_reason"] = "faq_exact_match"
            return RouteDecision(route="faq_exact", answer=answer, intent=intent, reason="faq_exact_match")

    context.retrieval_info["route"] = "retrieval"
    context.retrieval_info["route_reason"] = "no_deterministic_route"
    return RouteDecision(route="retrieval", reason="no_deterministic_route")


def _apply_direct_route(context: RAGQueryContext, intent: IntentResult, *, route: RouteName, reason: str) -> None:
    """把直接路由结果写入上下文，供 end 事件和 trace 使用。

    调用顺序：QAService/RAG 管线 -> _apply_direct_route()。
    """
    context.intent_payload = intent.as_dict()
    if context.hit_type == "unknown":
        context.hit_type = intent.intent.lower()
    context.retrieval_info.update(
        {
            "route": route,
            "route_reason": reason,
            "direct_answer_guard": reason,
            "scenario_id": context.scenario.scenario_id,
            "scenario_name": context.scenario.display_name,
            "data_scope": context.data_scope.as_dict(),
            "source_filter": context.source_filter,
            "kb_version": context.active_kb_version,
        }
    )


def _set_retrieval_snapshot(
    context: RAGQueryContext,
    *,
    plan: RetrievalPlan,
    query_variants: list[str],
    effective_source_filter: str | None,
    prompt_profile: PromptProfile,
) -> None:
    """写入检索诊断快照，保留路由阶段已经产生的 route 信息。

    调用顺序：QAService/RAG 管线 -> _set_retrieval_snapshot()。
    """
    route_snapshot = {
        key: context.retrieval_info[key]
        for key in ("route", "route_reason", "cache")
        if key in context.retrieval_info
    }
    context.retrieval_info = {
        **route_snapshot,
        "plan": plan.as_dict(),
        "query_variants": query_variants,
        "scenario_id": context.scenario.scenario_id,
        "scenario_name": context.scenario.display_name,
        "data_scope": context.data_scope.as_dict(),
        "source_filter": effective_source_filter,
        "kb_version": context.active_kb_version,
        "prompt_profile": prompt_profile.as_dict(),
    }


def prepare_retrieval(context: RAGQueryContext) -> RetrievalPreparation:
    """Stage 2：检索参数准备，加载历史、识别意图、按需改写追问、构建检索策略，为下游检索生成完整参数包。★★★ 核心

    执行流程：
    1. 加载历史摘要 + 最近消息作为对话上下文
    2. 规则优先 + 默认知识查询的意图识别
    3. 确定 source 过滤项（前端 > 意图推断 > 不过滤）
    4. 必要时追问改写（将依赖上下文的问法转为独立检索问题）
    5. 构建检索策略（top_k、阈值、是否重排等）
    6. 生成同义检索表达列表（查询变体）
    7. 选择最终回答提示词模板档位
    8. 汇总检索诊断信息快照

    参数：
        context: RAGQueryContext 请求级状态

    返回：
        RetrievalPreparation: 包含检索所需全部参数的数据包
    """
    # 从 MySQL 加载"历史摘要 + 最近消息"作为压缩后的对话上下文
    history_messages = context.run_stage("load_history", lambda: context.history.get_context_messages(context.session_id))

    # 规则优先 + 默认知识查询的检索类意图识别（追问、FAQ 查询、知识查询等）
    intent = context.run_stage("classify_intent", lambda: classify_intent(context.query, history_messages, context.scenario))
    context.intent_payload = intent.as_dict()

    # 确定 source 过滤项：前端显式选择 > 意图推断 > 不过滤
    effective_source_filter = context.run_stage(
        "resolve_source_filter",
        lambda: resolve_effective_source_filter(context.source_filter, intent.suggested_source, context.scenario),
    )

    # ── 步骤 4：追问改写（仅 FOLLOW_UP 类型启用） ──
    # 仅在 intent.requires_rewrite=True 时调用 rewrite_query_if_needed，将依赖上下文
    # 的追问（如"那审批呢"）改写成独立检索问题（如"入职流程中的审批步骤是什么"）。
    # 原因：非追问场景（FAQ_QUERY / KNOWLEDGE_QUERY）本身已独立完整，改写会引入噪声。
    context.rewritten_query = context.run_stage(
        "rewrite_query",
        lambda: rewrite_query_if_needed(context.query, history_messages, intent.requires_rewrite),
    )

    # ── 步骤 5：构建检索策略 ──
    # 根据意图结果、问题长度、风险类别和表格偏好生成完整的检索计划
    #（FAQ/doc top_k、阈值、是否重排等）
    plan = context.run_stage("build_retrieval_plan", lambda: build_retrieval_plan(context.rewritten_query, intent))

    # ── 步骤 6：生成同义检索表达（查询变体） ──
    # 查询变体同时传给 FAQ 和文档检索，提升召回覆盖率而不改变原问题含义。
    # 追问场景（FOLLOW_UP）允许短结构化问题也生成变体，因为追问已经经过改写补全，
    # 此时同义替换可以弥补改写后可能带来的用词偏移。
    query_variants = context.run_stage(
        "generate_query_variants",
        lambda: generate_query_variants(
            context.rewritten_query,
            enabled=plan.use_query_variants,
            allow_short_structured=intent.intent == "FOLLOW_UP",
        ),
    )

    # ── 步骤 7：选择最终回答提示词模板档位 ──
    # 根据意图结果和问题风险类别选择 System Prompt + User Template 组合，
    # 如 FAQ_QUERY 使用简洁 FAQ 模板，KNOWLEDGE_QUERY 使用结构化回答模板，
    # pricing/compliance 使用保守风险模板
    prompt_profile = context.run_stage(
        "select_prompt_profile",
        lambda: build_answer_prompt_profile(intent.intent, context.scenario, context.rewritten_query),
    )

    _set_retrieval_snapshot(
        context,
        plan=plan,
        query_variants=query_variants,
        effective_source_filter=effective_source_filter,
        prompt_profile=prompt_profile,
    )
    return RetrievalPreparation(
        history_messages=history_messages,
        intent=intent,
        effective_source_filter=effective_source_filter,
        rewritten_query=context.rewritten_query,
        plan=plan,
        query_variants=query_variants,
        prompt_profile=prompt_profile,
    )


def prepare_answer(
    context: RAGQueryContext,
    prepared: RetrievalPreparation,
    faq_result: RetrievalResult,
    doc_result: RetrievalResult,
) -> AnswerPreparation:
    """Stage 5：将 FAQ+文档检索结果整理为 LLM Prompt、引用来源和命中类型，为流式生成做准备。★★★ 核心

    执行流程：
    1. 调用 _build_answer_context 筛选上下文、确定来源和命中类型
    2. 记录上下文统计指标（条数/字符数/来源数/分数）到检索诊断信息
    3. 将历史消息和上下文填充到提示词模板中

    参数：
        context: RAGQueryContext 请求级状态
        prepared: RetrievalPreparation 检索参数包
        faq_result: FAQ 检索结果
        doc_result: 文档检索结果

    返回：
        AnswerPreparation: 包含 system_prompt、user_prompt、context_docs、sources、hit_type
    """
    # 整理上下文、引用来源、命中类型和最高分数，为 LLM 生成准备
    context_docs, sources, hit_type, top_score = context.run_stage(
        "build_answer_context",
        lambda: _build_answer_context(prepared, faq_result, doc_result),
    )
    _record_context_stats(context, context_docs, prepared.plan, top_score)
    # 这里先生成“证据置信度”：它描述检索证据是否足以支撑后续 LLM 生成，
    # 生成完成后的引用覆盖和上下文支撑度会在 rag.py 里再合并为最终 answer_confidence。
    record_evidence_confidence(
        context,
        hit_type=hit_type,
        retrieval_top_score=top_score,
        context_count=len(context_docs),
        source_count=len(sources),
    )

    # 将历史消息转为中文对话文本，填充到提示词模板中
    user_prompt = prepared.prompt_profile.user_template.format(
        history=format_messages(prepared.history_messages),
        question=prepared.rewritten_query,
        context=build_context(context_docs) or "无可用上下文。必须明确回答：信息不足，无法确认。",
    )
    return AnswerPreparation(
        context_docs=context_docs,
        sources=sources,
        hit_type=hit_type,
        system_prompt=prepared.prompt_profile.system_template,
        user_prompt=user_prompt,
    )


def _record_context_stats(
    context: RAGQueryContext,
    context_docs: list[Document],
    plan: RetrievalPlan,
    top_score: float,
) -> None:
    """记录上下文统计指标，供状态页和 trace 诊断使用。

    调用顺序：QAService/RAG 管线 -> _record_context_stats()。
    """
    context.retrieval_info.update(
        {
            "context_count": len(context_docs),
            "context_chars": sum(len(doc.page_content) for doc in context_docs),
            "context_source_count": len({str(doc.metadata.get("source") or "") for doc in context_docs}),
            "context_min_score": plan.min_context_score,
            "context_top_score": top_score,
        }
    )


def _build_answer_context(
    prepared: RetrievalPreparation,
    faq_result: RetrievalResult,
    doc_result: RetrievalResult,
) -> tuple[list[Document], list[dict[str, Any]], str, float]:
    """整理上下文文档、引用来源、命中类型和最高分数，为 prepare_answer 提供核心数据。★★★ 核心

    执行流程：
    1. 按分数阈值和条数限制从 FAQ/文档候选中筛选上下文 Doc 列表
    2. prefer_table 时来源以文档为主、FAQ 为辅（表格行优先展示）
    3. 取 FAQ 和文档两端最高分中的较大值
    4. 无上下文通过过滤时命中类型标记为 insufficient_context

    参数：
        prepared: RetrievalPreparation 检索参数包（含 plan）
        faq_result: FAQ 检索结果
        doc_result: 文档检索结果

    返回：
        tuple: (context_docs, sources, hit_type, top_score) 四元组
    """
    # 按分数阈值和条数限制从 FAQ/文档候选中选择最终进入 prompt 的上下文片段
    context_docs = select_context_docs(faq_result.hits, doc_result.hits, prepared.plan)
    # ── 来源排序：prefer_table 时文档优先，否则 FAQ 优先 ──
    # 原因：表格查询场景下用户期望看到表格行中的具体数据（如金额、状态），
    # 这些数据仅在文档集合的表格 chunk 中，FAQ 集合无此粒度。
    # 非表格查询时 FAQ 来源更优先，因为 FAQ 是标准口径，比文档正文更权威。
    if prepared.plan.prefer_table:
        sources = doc_result.source_payloads(limit=5) + faq_result.source_payloads(limit=2)
    else:
        sources = faq_result.source_payloads(limit=2) + doc_result.source_payloads(limit=5)
    top_score = max(faq_result.top_score, doc_result.top_score)
    # 无上下文通过分数过滤时返回 insufficient_context，上游据此返回确定性信息不足回答
    return context_docs, sources, "rag" if context_docs else "insufficient_context", top_score


def build_insufficient_context_answer(context: RAGQueryContext) -> str:
    """无可用上下文时返回确定性"信息不足"回答，避免 LLM 幻觉。

    参数：
        context: RAGQueryContext 请求级状态（含 scenario.support_contact）

    返回：
        str: 确定性"信息不足，无法确认"提示文案

    调用顺序：QAService/RAG 管线 -> build_insufficient_context_answer()。
    """
    context.retrieval_info["insufficient_context_reason"] = "no_context_after_score_filter"
    return f"信息不足，无法确认。当前知识库没有召回到足够可靠的依据，请联系{context.scenario.support_contact}。"


def stream_llm_answer(system_prompt: str, user_prompt: str):
    """Stage 6：调用 LangChain ChatModel 流式生成答案片段。

    参数：
        system_prompt: SystemMessage 系统提示词
        user_prompt: HumanMessage 用户提示词（含上下文和问题）

    返回：
        BaseMessage stream iterable，yield 每个 AIMessageChunk 直到流结束

    调用顺序：QAService/RAG 管线 -> stream_llm_answer()。
    """
    # 获取已缓存的流式 ChatOpenAI 客户端，按 token 逐步推送生成结果
    llm = get_chat_model(streaming=True)
    # 以 SystemMessage + HumanMessage 调用 LLM 流式生成
    return llm.stream([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])


def detect_and_apply_boundary_answer(context: RAGQueryContext) -> str | None:
    """识别 source 边界问题，并返回确定性提示。

    执行流程：
    1. 调用 detect_source_boundary 判断前端 source 是否和问题明显不匹配
    2. 若不匹配，返回"请切换分类后查询"引导
    3. 无边界问题时返回 None，继续正常 RAG 流程

    参数：
        context: RAGQueryContext 请求级状态（含 query、scenario、source_filter）

    返回：
        str | None: 边界问题引导文案；无边界问题时返回 None
    """
    # 判断用户显式选择的 source 是否和问题明显不匹配（如选了 hr 但问题属于 finance）
    source_boundary = detect_source_boundary(context.query, context.scenario, context.source_filter)
    context.retrieval_info["source_boundary"] = source_boundary.as_dict()
    if source_boundary.mismatched:
        context.hit_type = "source_boundary"
        context.retrieval_info["boundary_reason"] = "source_boundary"
        return (
            f"当前选择的是「{source_boundary.selected_source_label}」，但问题更像当前场景下的"
            f"「{source_boundary.matched_source_label}」分类。为避免按错误资料回答，请切换分类后再查询。"
        )
    return None
