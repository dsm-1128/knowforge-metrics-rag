"""RAG 主流程编排：`stream_query`（流式问答）和 `debug_retrieval`（检索调试）。
"""
from __future__ import annotations
from collections.abc import Generator
from typing import Any
from qa_core.config.logging_config import get_logger
from qa_core.pipeline.events import (
    status_event as build_status_event,
    token_event as build_token_event,
)
from qa_core.pipeline.citations import enforce_answer_citations
from qa_core.pipeline.confidence import (
    faq_exact_match,
    finalize_generated_answer_confidence,
    mark_answer_confidence_not_applicable,
    record_evidence_confidence,
)
from qa_core.pipeline.runtime import (
    create_query_context,
    finish_error,
    finish_success,
    start_event as build_query_start_event,
)
from qa_core.pipeline.steps import (
    build_insufficient_context_answer,
    decide_route,
    prepare_answer,
    prepare_retrieval,
    stream_llm_answer,
)
from qa_core.pipeline.retrieval_steps import get_faq_direct_answer, search_doc, search_faq
logger = get_logger(__name__)

def stream_query(
    history,
    query: str,
    source_filter: str | None,
    session_id: str | None,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    user_role: str | None = None,
    user_roles: list[str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """一次完整问答请求的编排主入口，协调 Stage 0-7 管线并持续产出 WebSocket 事件流。★★★ 核心

    执行流程：
    Stage 0: 创建运行时上下文（场景/数据域/会话/trace/知识库版本）
    Stage 1: 低成本查询路由（直答/边界、FAQ 精确命中、继续检索）
    Stage 2: 检索准备（历史、意图、source、按需改写、检索计划、查询变体、Prompt Profile）
    Stage 3: FAQ 检索，判断是否直出
    Stage 4: 文档检索
    Stage 5: 上下文构建
    Stage 6: LLM 流式生成
    Stage 7: 写历史记录、写 trace、发送结束事件

    参数：
        history: 对话历史管理器
        query: 用户原始提问文本
        source_filter: 前端选择的业务分类过滤项
        session_id: 会话 ID
        kb_version: 请求指定的知识库版本号
        scenario_id: 业务场景 ID
        tenant_id: 租户 ID
        dataset_id: 数据集 ID
        visibility: 数据可见级别
        user_role: 用户主角色
        user_roles: 用户的全部角色列表

    返回：
        Generator yielding WebSocket 事件 dict（包含事件类型和数据）
    """
    # ── Stage 0: Create runtime context (scenario, data scope, session, trace_id, kb version) ──
    context = create_query_context(
        history=history,
        query=query,
        source_filter=source_filter,
        session_id=session_id,
        requested_kb_version=kb_version,
        scenario_id=scenario_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_role=user_role,
        user_roles=user_roles,
    )
    # 向前端发送"请求已接收"事件
    yield build_query_start_event(context)

    try:
        # ── Stage 1: Low-cost route decision ──
        # 统一处理确定性直答、边界拦截和 FAQ 精确命中；未命中才进入检索准备。
        yield build_status_event("正在进行查询路由...", context.session_id)
        route = decide_route(context)
        if route.answer:
            yield from _finish_with_single_answer(context, history, query, route.answer)
            return

        # ── Stage 2: Intent recognition + retrieval parameter preparation ──
        # 向前端发送"正在识别问题意图"状态事件
        yield build_status_event("正在识别问题意图...", context.session_id)
        # 生成下游检索参数包：历史、意图、source、按需改写、检索计划、查询变体和 Prompt Profile。
        prepared = prepare_retrieval(context)

        # ── Stage 3-6: FAQ retrieval → doc retrieval → context building → LLM streaming ──
        # 执行检索+生成并产生状态/token/结束事件，返回 None 表示已内部收尾
        helper_result = yield from _search_and_generate(context, prepared, query, history)
        # _search_and_generate 内已 yield 收尾事件，无需继续走引用补强
        if helper_result is None:
            return

        answer_prepared = helper_result
        raw_answer = context.answer
        # ── Stage 6 continuation: 引用补强与生成后核验 ──
        # stream_llm_answer() 完成 token 生成后，仍属于 Stage 6 的答案后处理。
        # 先补齐可见引用，再核验引用覆盖、编号合法性和上下文支撑度。
        # 确保 RAG 答案带有可见来源编号，模型漏写时在末尾补充"参考来源"
        answer = enforce_answer_citations(raw_answer, answer_prepared.context_docs)
        # LLM 已经完成生成，此处把生成结果纳入答案置信度；引用补强只修补展示文本，
        # 不等于事实核验，因此必须在最终文本收口前单独执行生成后核验。
        finalize_generated_answer_confidence(
            context,
            answer=answer,
            context_docs=answer_prepared.context_docs,
        )
        # 生成结果为空时返回确定性信息不足提示；同时保留 answer_empty 的低置信核验结果，
        # 避免用后续兜底文案覆盖“LLM 没有产出答案”这一真实诊断。
        if not answer:
            answer = f"信息不足，无法确认，请联系人工支持：{context.scenario.support_contact}。"
            yield from _finish_with_single_answer(context, history, query, answer, record_save_stage=True)
            return
        # 引用补强追加了新文本（如"参考来源：..."），将增量部分推送给前端
        elif answer != raw_answer:
            extra_token = answer[len(raw_answer):] if answer.startswith(raw_answer) else answer
            # 将补充的引用来源片段推送给前端
            yield build_token_event(extra_token, context.session_id)
            context.answer_parts = [answer]

        # ── Stage 7: Save history, write trace, send end event ──
        with context.stage("save_history"):
            # 将本轮问答写入 MySQL 历史表
            history.add_turn(context.session_id, query, answer)
        # 向前端发送 success 结束事件（含来源、耗时、意图、检索诊断信息）并写入 trace
        yield finish_success(context, answer=answer)
    except Exception as exc:
        logger.exception("QA stream failed")
        # 向前端发送 error 结束事件（WebSocket 不断连，前端显示可恢复的失败提示）
        yield finish_error(context, exc)


def _search_and_generate(context, prepared, query, history) -> Generator:
    """检索+生成子管线：FAQ 直出判断 → 文档检索 → 上下文构建 → LLM 流式生成。★★★ 核心

    执行流程：
    1. FAQ 检索：按查询变体召回 FAQ，判断是否分数直出（无需 LLM）
    2. 文档检索：按查询变体召回文档
    3. 上下文构建：合并 FAQ+文档候选中筛选出最终 prompt 上下文
    4. 上下文为空时返回"信息不足"兜底，不走 LLM
    5. LLM 流式生成：逐 chunk yield token 事件

    参数：
        context: RAGQueryContext 请求级状态
        prepared: RetrievalPreparation 检索参数包
        query: 用户原始提问
        history: 对话历史管理器

    返回：
        Generator yielding token/status/end 事件；
        function return value 为 None（已收尾）或 AnswerPreparation（需上游继续引用补强）
    """
    # ── Stage 3: FAQ retrieval with direct-answer bypass ──
    # 向前端发送"正在检索 FAQ 知识库"状态事件
    yield build_status_event("正在检索业务 FAQ 知识库...", context.session_id)
    # 按检索计划查询 FAQ 集合
    faq_result = search_faq(context, prepared)
    # 判断 FAQ 是否达到直出条件（精确匹配或分数超阈值）
    direct_answer = get_faq_direct_answer(context, prepared, faq_result)
    # FAQ 检索分数超阈值时可直接返回标准答案，无需 LLM 生成
    if direct_answer:
        context.hit_type = "faq_direct"
        context.sources = faq_result.source_payloads()
        # FAQ 分数直出不是标准问题精确匹配时，置信度需要同时参考检索分和规则分
        record_evidence_confidence(
            context,
            hit_type="faq_direct",
            retrieval_top_score=faq_result.top_score,
            context_count=1 if faq_result.top_document else 0,
            source_count=len(context.sources),
            faq_exact_match=faq_exact_match(context.query, faq_result.top_document),
        )
        yield from _finish_with_single_answer(context, history, query, direct_answer)
        return None

    # ── Stage 4: Document retrieval ──
    # 向前端发送"正在匹配业务资料"状态事件
    yield build_status_event("正在匹配相关业务资料...", context.session_id)
    # 按检索计划查询文档集合
    doc_result = search_doc(context, prepared)
    # ── Stage 5: Answer context preparation ──
    # 将检索结果整理成最终 Prompt、引用来源和命中类型
    answer_prepared = prepare_answer(context, prepared, faq_result, doc_result)
    context.sources = answer_prepared.sources
    context.hit_type = answer_prepared.hit_type

    # 合并后上下文仍为空（所有候选低于分数阈值），直接返回确定性"信息不足"避免 LLM 幻觉
    if context.hit_type == "insufficient_context":
        answer = build_insufficient_context_answer(context)
        yield from _finish_with_single_answer(context, history, query, answer, record_save_stage=True)
        return None

    # ── Stage 6: LLM streaming generation ──
    # 向前端发送"正在生成回答"状态事件
    yield build_status_event("正在生成回答...", context.session_id)
    context.retrieval_info["generation_attempted"] = True
    with context.stage("llm_generation"):
        # 调用 LangChain ChatOpenAI 流式接口，逐 chunk 产生 token
        for chunk in stream_llm_answer(answer_prepared.system_prompt, answer_prepared.user_prompt):
            token = str(getattr(chunk, "content", "") or "")
            # LangChain 可能产出空 content 的 chunk（如 finish_reason 块），跳过
            if not token:
                continue
            context.answer_parts.append(token)
            context.mark_first_token()
            # 向 WebSocket 推送当前 token 片段，前端逐字展示
            yield build_token_event(token, context.session_id)

    return answer_prepared


def _finish_with_single_answer(
    context,
    history,
    query: str,
    answer: str,
    *,
    record_save_stage: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """无需 LLM 流式的答案收口：发 token → 写历史 → 写 trace → 发 end，四类直出分支共用。★★★ 核心

    处理步骤：发 token → 写历史 → 写 trace → 发 end。
    FAQ 直出、直接意图、信息不足兜底四类已有完整答案的分支统一走此收尾。

    参数：
        context: RAGQueryContext 请求级状态
        history: 对话历史管理器
        query: 用户原始提问
        answer: 完整答案文本（非流式，一次性推送）
        record_save_stage: 是否包裹 stage 计时上下文（信息不足等兜底分支需要）

    返回：
        Generator yielding token 事件 + success 结束事件
    """
    if context.scenario.scenario_id == "metrics_dictionary":
        from qa_core.prompts.metrics_dictionary import format_direct_answer
        answer = format_direct_answer(answer, has_sources=bool(context.sources))
    context.answer_parts = [answer]
    context.mark_first_token()
    # 将完整答案作为单次 token 推送（非流式场景直接透传）
    yield build_token_event(answer, context.session_id)
    # record_save_stage=True 时包裹 stage 计时（信息不足等兜底分支），
    # 其他简单直出路径不加额外包装直接写入历史
    if record_save_stage:
        with context.stage("save_history"):
            history.add_turn(context.session_id, query, answer)
    else:
        history.add_turn(context.session_id, query, answer)
    # 向前端发送 success 结束事件并写入 trace
    yield finish_success(context, answer=answer)

def debug_retrieval(
    history,
    query: str,
    source_filter: str | None,
    session_id: str | None = None,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    user_role: str | None = None,
    user_roles: list[str] | None = None,
) -> dict[str, Any]:
    """路由 + 检索半链路调试入口：先判断查询路由；检索类问题继续执行 FAQ/文档检索，不调用 LLM。★★★ 核心

    执行流程：
    1. 创建运行时上下文（场景/数据域/会话/trace/知识库版本）
    2. 查询路由：直答/边界问题直接返回 route 诊断
    3. 检索准备：检索类问题生成历史、意图、source、按需改写、检索计划、查询变体和 Prompt Profile
    4. FAQ 检索 + 文档检索（检索计划禁用某一路时返回空结果）
    5. 汇总阶段耗时和检索诊断信息

    参数：
        history: 对话历史管理器
        query: 用户原始提问
        source_filter: 前端选择的业务分类过滤项
        session_id: 会话 ID
        kb_version: 请求指定的知识库版本号
        scenario_id: 业务场景 ID
        tenant_id: 租户 ID
        dataset_id: 数据集 ID
        visibility: 数据可见级别
        user_role: 用户主角色
        user_roles: 用户的全部角色列表

    返回：
        dict: 路由和检索调试诊断数据包，包含链路中的参数和结果
    """
    # ── Stage 0: Create runtime context ──
    context = create_query_context(
        history=history,
        query=query,
        source_filter=source_filter,
        session_id=session_id,
        requested_kb_version=kb_version,
        scenario_id=scenario_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_role=user_role,
        user_roles=user_roles,
    )
    # ── Stage 1: Low-cost route decision ──
    # 调试入口也先走路由；直答/边界问题不生成检索计划。
    route = decide_route(context)
    if route.answer:
        if "answer_confidence" not in context.retrieval_info:
            record_evidence_confidence(
                context,
                hit_type=context.hit_type,
                retrieval_top_score=float(context.retrieval_info.get("faq_top_score") or 0.0),
                context_count=1 if route.route == "faq_exact" else 0,
                source_count=len(context.sources),
                deterministic_route=route.route == "direct_answer",
                faq_exact_match=route.route == "faq_exact",
            )
        mark_answer_confidence_not_applicable(context, reason="generation_not_called")
        context.finalize_timings()
        faq_sources = context.sources if route.route == "faq_exact" else []
        return {
            "query": query,
            "raw_query": context.raw_query,
            "effective_query": context.query,
            "query_normalized": context.raw_query != context.query,
            "rewritten_query": context.query,
            "scenario_id": context.scenario.scenario_id,
            "scenario_name": context.scenario.display_name,
            "data_scope": context.data_scope.as_dict(),
            "tenant_id": context.data_scope.tenant_id,
            "dataset_id": context.data_scope.dataset_id,
            "visibility": context.data_scope.visibility,
            "source_filter": context.source_filter,
            "kb_version": context.active_kb_version,
            "route": route.route,
            "route_reason": route.reason,
            "answer": route.answer,
            "answer_confidence": context.answer_confidence,
            "intent": route.intent.as_dict(),
            "retrieval_plan": None,
            "stage_timings_ms": context.retrieval_info["stage_timings_ms"],
            "slowest_stage": context.retrieval_info["slowest_stage"],
            "faq_sources": faq_sources,
            "doc_sources": [],
        }

    # ── Stage 2: Intent, rewrite, retrieval plan, query variants ──
    # 完成检索类意图识别、改写、检索计划和查询变体生成
    prepared = prepare_retrieval(context)

    # ── Stage 3-5: FAQ retrieval → document retrieval → answer context preparation ──
    # 调试入口按业务流程图顺序执行 Stage 3、4、5，不进入 Stage 6 LLM 生成。
    faq_result = search_faq(context, prepared)
    doc_result = search_doc(context, prepared)
    answer_prepared = prepare_answer(context, prepared, faq_result, doc_result)
    # 将各阶段耗时写回 retrieval_info
    context.finalize_timings()

    # 构造调试诊断数据包，包含完整检索链路中的参数和结果
    return {
        "query": query,
        "raw_query": context.raw_query,
        "effective_query": context.query,
        "query_normalized": context.raw_query != context.query,
        "rewritten_query": prepared.rewritten_query,
        "scenario_id": context.scenario.scenario_id,
        "scenario_name": context.scenario.display_name,
        "data_scope": context.data_scope.as_dict(),
        "tenant_id": context.data_scope.tenant_id,
        "dataset_id": context.data_scope.dataset_id,
        "visibility": context.data_scope.visibility,
        "source_filter": prepared.effective_source_filter,
        "kb_version": context.active_kb_version,
        "route": "retrieval",
        "route_reason": context.retrieval_info["route_reason"],
        "intent": prepared.intent.as_dict(),
        "answer_confidence": context.answer_confidence,
        "retrieval_plan": {
            **prepared.plan.as_dict(),
            "query_variants": prepared.query_variants,
            "prompt_profile": prepared.prompt_profile.as_dict(),
        },
        "stage_timings_ms": context.retrieval_info["stage_timings_ms"],
        "slowest_stage": context.retrieval_info["slowest_stage"],
        "faq_sources": faq_result.source_payloads(limit=10),
        "doc_sources": doc_result.source_payloads(limit=10),
        "context_count": len(answer_prepared.context_docs),
        "hit_type": answer_prepared.hit_type,
    }
