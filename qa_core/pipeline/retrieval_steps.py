"""RAG 主流程中的检索执行步骤。

`steps.py` 负责意图、改写、Prompt 等准备工作；本文件只负责把准备好的
`RetrievalPreparation` 落到 FAQ 和文档检索上。这样阅读主链路时可以清楚区分：
先决定“怎么查”，再执行“真正去哪里查”。
"""

from __future__ import annotations

from qa_core.cache.manager import get_cache_manager
from qa_core.config.settings import get_settings
from qa_core.pipeline.context import direct_faq_answer
from qa_core.pipeline.runtime import RAGQueryContext
from qa_core.pipeline.steps import RetrievalPreparation
from qa_core.retrieval.factory import get_doc_store, get_faq_store
from qa_core.retrieval.ranking import merge_hits_by_document, normalize_queries, rerank_hits, sort_hits_by_score
from qa_core.retrieval.results import RetrievalResult


def search_faq(context: RAGQueryContext, prepared: RetrievalPreparation) -> RetrievalResult:
    """Stage 3：按检索计划执行 FAQ 混合检索，并把耗时、最高分写入检索诊断信息。★★★ 核心

    使用场景：
    - FAQ 标准问答优先召回，用于判断是否可以直接返回标准答案；
    - FAQ 未直出时，FAQ 片段仍可和文档片段一起进入 Prompt，补充标准口径。

    为什么单独封装：FAQ 检索涉及 fast path 结果复用、source 过滤、版本过滤和数据隔离。
    这些属于“执行检索”的细节，不应该挤在 `rag.py` 的流程编排里。

    调用顺序：QAService/RAG 管线 -> search_faq()。
    """

    def do_search() -> RetrievalResult:
        """执行 FAQ 混合检索：优先复用 fast path 缓存结果，缓存未命中时查询 Milvus FAQ 集合。

        调用顺序：QAService/RAG 管线 -> do_search()。
        """
        if not prepared.plan.run_faq:
            return RetrievalResult(query=prepared.rewritten_query, source_type="faq")
        reused = _reuse_fast_faq_result(context, prepared)
        if reused is not None:
            return reused
        context.retrieval_info["faq_reused_from_fast_path"] = False
        context.retrieval_info["faq_reuse_reason"] = "full_faq_search_required"
        return _search_with_cache(
            context,
            source_type="faq",
            collection_name=context.scenario.faq_collection,
            query_variants=prepared.query_variants,
            k=prepared.plan.faq_top_k,
            source_filter=prepared.effective_source_filter,
            rerank=prepared.plan.rerank,
            stage_name="faq_retrieval",
            search=lambda: get_faq_store(context.scenario.faq_collection).search_many(
                prepared.query_variants,
                k=prepared.plan.faq_top_k,
                source_filter=prepared.effective_source_filter,
                kb_version=context.active_kb_version,
                data_scope=context.data_scope,
                scenario_id=context.scenario.scenario_id,
                source_type="faq",
                rerank=prepared.plan.rerank,
            ),
        )

    faq_result = context.run_stage("faq_retrieval", do_search)
    context.retrieval_info["faq_elapsed_ms"] = round(faq_result.elapsed_ms, 2)
    context.retrieval_info["faq_top_score"] = faq_result.top_score
    return faq_result


def _reuse_fast_faq_result(context: RAGQueryContext, prepared: RetrievalPreparation) -> RetrievalResult | None:
    """复用 FAQ 精确探测的原问题候选，只检索额外查询变体。

    快路径先以原问题召回足够容量的未重排候选。若精确 FAQ 未命中，主链路生成
    ``[原问题, 变体1, ...]`` 后，不应再次查询原问题：只查询新增变体，再把两批候选
    去重、统一重排，结果与完整多变体检索保持相同的排序语义。
    """
    variants = normalize_queries(prepared.query_variants)
    if context.fast_faq_result is None:
        return None
    if prepared.rewritten_query != context.query or not variants or variants[0] != context.query:
        return None
    if prepared.effective_source_filter != context.fast_faq_source_filter:
        return None
    if context.fast_faq_top_k < prepared.plan.faq_top_k:
        return None

    extra_variants = variants[1:]
    extra_result = RetrievalResult(query="", source_type="faq")
    if extra_variants:
        extra_result = _search_with_cache(
            context,
            source_type="faq",
            collection_name=context.scenario.faq_collection,
            query_variants=extra_variants,
            k=prepared.plan.faq_top_k,
            source_filter=prepared.effective_source_filter,
            rerank=False,
            stage_name="faq_variant_retrieval",
            search=lambda: get_faq_store(context.scenario.faq_collection).search_many(
                extra_variants,
                k=prepared.plan.faq_top_k,
                source_filter=prepared.effective_source_filter,
                kb_version=context.active_kb_version,
                data_scope=context.data_scope,
                scenario_id=context.scenario.scenario_id,
                source_type="faq",
                rerank=False,
            ),
        )

    merged = {}
    merge_hits_by_document(merged, context.fast_faq_result.hits)
    merge_hits_by_document(merged, extra_result.hits)
    hits = sort_hits_by_score(merged.values())
    if prepared.plan.rerank and hits:
        hits = _rerank_merged_faq_hits(context.query, variants, hits)

    context.retrieval_info.update(
        {
            "faq_reused_from_fast_path": True,
            "faq_reuse_reason": "reuse_original_and_search_variants",
            "faq_fast_reused_hit_count": len(context.fast_faq_result.hits),
            "faq_incremental_variant_queries": extra_variants,
        }
    )
    return RetrievalResult(
        hits=hits[:prepared.plan.faq_top_k],
        query=" | ".join(variants),
        source_type="faq",
        elapsed_ms=context.fast_faq_result.elapsed_ms + extra_result.elapsed_ms,
    )


def _rerank_merged_faq_hits(query: str, variants: list[str], hits):
    """对复用后的 FAQ 候选执行一次统一 CrossEncoder 重排。"""
    # 只有当前检索计划需要重排时才加载大模型，避免 FAQ 直出和纯合并路径触发额外依赖。
    from qa_core.retrieval.models import get_reranker

    settings = get_settings()
    candidate_limit = max(settings.rerank_top_n * len(variants), settings.rerank_top_n)
    return rerank_hits(
        query,
        hits[:candidate_limit],
        reranker=get_reranker(),
        top_n=settings.rerank_top_n,
    )


def get_faq_direct_answer(
    context: RAGQueryContext,
    prepared: RetrievalPreparation,
    faq_result: RetrievalResult,
) -> str | None:
    """判断 FAQ top 命中是否足够可靠，可靠时直接返回标准答案。★★★ 核心

    使用场景：用户问的是制度型、流程型、标准口径型问题，例如“报销需要哪些材料”。
    如果 FAQ 中已经有高置信标准答案，就不必再让 LLM 重新生成，减少幻觉和延迟。

    调用顺序：QAService/RAG 管线 -> get_faq_direct_answer()。
    """

    # 指标问法可能只差日期或版本，分数再高也不能用当前 FAQ 替代历史口径。
    exact_only = prepared.plan.faq_direct_exact_only or context.scenario.scenario_id == "metrics_dictionary"
    threshold = float("inf") if exact_only else prepared.plan.faq_direct_threshold
    return direct_faq_answer(
        context.query,
        faq_result.top_document,
        faq_result.top_score,
        threshold,
    )


def search_doc(context: RAGQueryContext, prepared: RetrievalPreparation) -> RetrievalResult:
    """Stage 4：按检索计划执行文档混合检索，并把耗时、最高分写入检索诊断信息。★★★ 核心

    使用场景：
    - FAQ 没有直接命中，需要从制度、合同、规范、表格等正文资料里召回证据；
    - 表格问题、复杂资料问题通常依赖文档检索而不是 FAQ 直出。

    调用顺序：QAService/RAG 管线 -> search_doc()。
    """

    def do_search() -> RetrievalResult:
        """执行文档混合检索：按检索计划的 top_k 和过滤条件查询 Milvus 文档集合。

        调用顺序：QAService/RAG 管线 -> do_search()。
        """
        if not prepared.plan.run_doc:
            return RetrievalResult(query=prepared.rewritten_query, source_type="doc")
        return _search_with_cache(
            context,
            source_type="doc",
            collection_name=context.scenario.doc_collection,
            query_variants=prepared.query_variants,
            k=prepared.plan.doc_top_k,
            source_filter=prepared.effective_source_filter,
            rerank=prepared.plan.rerank,
            stage_name="doc_retrieval",
            search=lambda: get_doc_store(context.scenario.doc_collection).search_many(
                prepared.query_variants,
                k=prepared.plan.doc_top_k,
                source_filter=prepared.effective_source_filter,
                kb_version=context.active_kb_version,
                data_scope=context.data_scope,
                scenario_id=context.scenario.scenario_id,
                source_type="doc",
                rerank=prepared.plan.rerank,
            ),
        )

    doc_result = context.run_stage("doc_retrieval", do_search)
    context.retrieval_info["doc_elapsed_ms"] = round(doc_result.elapsed_ms, 2)
    context.retrieval_info["doc_top_score"] = doc_result.top_score
    return doc_result


def _search_with_cache(
    context: RAGQueryContext,
    *,
    source_type: str,
    collection_name: str,
    query_variants: list[str],
    k: int,
    source_filter: str | None,
    rerank: bool,
    stage_name: str,
    search,
) -> RetrievalResult:
    """统一执行检索缓存读写：先查 Redis 缓存，命中直接返回；未命中调底层检索后写入缓存。（★★★ 核心）

    执行流程：
      1. 通过 CacheManager 生成缓存 key（绑定场景、集合、查询变体、过滤条件等所有参数）。
      2. 先查 L2 Redis 缓存，命中直接返回结果（含诊断记录）。
      3. 未命中时调用 search 回调执行真正的 Milvus 混合检索。
      4. 将检索结果写入缓存（TTL 由 CacheManager 统一控制）。

    参数：
        context: RAGQueryContext（含场景、数据域等信息）。
        source_type: "faq" 或 "doc"，用于缓存 key 和诊断记录。
        collection_name: Milvus 集合名。
        query_variants: 查询变体列表。
        k: 检索返回 top_k 数量。
        source_filter: source 过滤项。
        rerank: 是否启用 CrossEncoder 重排。
        stage_name: 阶段名称（"faq_retrieval" 或 "doc_retrieval"），用于缓存诊断记录。
        search: 执行实际检索的回调函数（无参，返回 RetrievalResult）。

    返回：
        检索结果（缓存命中或新查询结果）。

    调用顺序：检索执行阶段 -> search_faq() / search_doc() -> _search_with_cache()。
    """
    cache = get_cache_manager()
    cache_key = cache.retrieval_key(
        kind="retrieval",
        scenario_id=context.scenario.scenario_id,
        collection_name=collection_name,
        source_type=source_type,
        data_scope=context.data_scope,
        kb_version=context.active_kb_version,
        source_filter=source_filter,
        query_variants=query_variants,
        k=k,
        rerank=rerank,
    )
    # ── 步骤 1：查缓存 ──
    # 缓存 key 绑定场景、集合、查询变体和过滤条件等所有参数，
    # 任何参数变化都会产生不同的 key，避免不同查询命中的数据污染
    cached = cache.get_retrieval_result(cache_key, source_type=source_type)
    # ── 步骤 2：记录缓存诊断事件 ──
    context.record_cache_event(
        stage=stage_name,
        enabled=cache.enabled,
        hit=cached is not None,
        source_type=source_type,
        key=cache_key,
    )
    # ── 步骤 3：命中直接返回 ──
    if cached is not None:
        return cached
    # ── 步骤 4：未命中时执行实际检索并写入缓存 ──
    result = search()
    cache.set_retrieval_result(cache_key, result, source_type=source_type)
    return result
