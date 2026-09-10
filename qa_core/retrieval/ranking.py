"""检索候选合并与重排，不依赖 Milvus 连接的纯逻辑。

这里放的是去重、查询清洗、排序和 CrossEncoder 重排等纯逻辑。它和 `store.py` 分开，
是为了让 Milvus 交互只留在 store 层，而这些函数可以在没有 Milvus 的情况下单独测试。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document

from qa_core.retrieval.results import RetrievalHit


def document_key(document: Document) -> str:
    """返回用于合并重复命中文档的稳定标识（chunk_id > faq_id > 内容前 120 字符）。

    参数：
        document: LangChain Document。

    返回：
        稳定去重 key。

    调用顺序：检索准备或检索执行 -> document_key()。
    """
    metadata = document.metadata
    # 优先使用 chunk_id（文档分块）作为唯一标识；faq_id 是 FAQ 命中的备用方案
    # 如果两者都不存在（极端兜底），取内容前 120 字符作为近似标识
    # 原因：同一文档的多个检索结果应合并为一条，避免召回列表膨胀和上下文窗口浪费
    return str(metadata.get("chunk_id") or metadata.get("faq_id") or document.page_content[:120])


def normalize_queries(queries: Iterable[str]) -> list[str]:
    """清洗查询变体列表：去空白、去空串、按顺序去重，保持第一个查询（原问题）用于后续 rerank。

    参数：
        queries: 查询变体文本列表。

    返回：
        清洗后且按原顺序去重的查询列表。

    调用顺序：检索准备或检索执行 -> normalize_queries()。
    """
    result: list[str] = []
    for query in queries:
        cleaned = query.strip()
        # 跳过空字符串查询（改写阶段可能产生空查询变体）
        # 如果 cleaned 已经在 result 中则跳过（保持每个查询变体唯一）
        # 原因：重复的查询变体在检索阶段会产生完全相同的检索结果列表，徒增合并开销
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def merge_hits_by_document(merged: dict[str, RetrievalHit], hits: list[RetrievalHit]) -> None:
    """把一批候选命中合并到已有结果中，同一 chunk 被多次命中时只保留最高分。

    参数：
        merged: 已累计的命中字典，会被原地更新。
        hits: 本批新增候选命中。

    调用顺序：检索准备或检索执行 -> merge_hits_by_document()。
    """
    for hit in hits:
        key = document_key(hit.document)
        previous = merged.get(key)
        # 同一文档被多个查询变体命中时只保留最高分，避免召回阶段排序膨胀
        # 原因：merged 字典的 value 可能来自不同查询变体召回的结果，
        # 每个查询变体对同一文档的打分可能不同（不同改写对文档的语义匹配度有差异）
        # 保留最高分意味着只取该文档在所有查询变体中的最佳匹配表现
        if previous is None or hit.score > previous.score:
            merged[key] = hit


def sort_hits_by_score(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    """按分数从高到低排序候选结果。

    参数：
        hits: 待排序的候选命中。

    返回：
        按分数降序排列的候选列表。

    调用顺序：检索准备或检索执行 -> sort_hits_by_score()。
    """
    # 降序排列：分数越高表示与 query 的语义相关性越强
    # 此排序结果用于后续重排的输入或直接作为最终排序（当未启用 reranker 时）
    return sorted(hits, key=lambda item: item.score, reverse=True)


def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    *,
    reranker: Any,
    top_n: int,
) -> list[RetrievalHit]:
    """使用 CrossEncoder 对 query-passage 对打分排序，比向量相似度更准确。

    执行流程：
      1. hits 为空时直接返回空列表。
      2. 检索计划要求重排但 reranker 为空时抛错，避免静默降级。
      3. 为每个候选构造 (query, passage_text)。
      4. 调用 CrossEncoder predict() 得到相关性分数。
      5. 按新分数降序排序。
      6. 截断为 top_n 条。

    参数：
        query: 原始用户问题。
        hits: 待重排候选。
        reranker: CrossEncoder 模型实例，需实现 predict()。
        top_n: 重排后最多返回条数。

    返回：
        重排并截断后的候选列表。

    异常：
        RuntimeError: 检索计划要求重排但 reranker 未初始化。
    """
    # ── 步骤 1：空候选防御 ──
    # 没有候选命中时直接返回空列表，避免后续构造 pairs 或调用模型报错
    if not hits:
        return []
    # ── 步骤 2：reranker 未初始化时硬失败 ──
    # 检索计划要求重排但 reranker 未配置，此时不应静默降级（会显著降低排序质量）
    # 原因：如果跳过 reranker 而直接用向量相似度排序返回，检索质量将严重下降
    # 抛出 RuntimeError 而非静默兜底，是为了让调用方感知配置缺失
    if reranker is None:
        raise RuntimeError("Reranker 未初始化，但当前检索计划要求重排。")
    # 构造 (query, passage_text) 对作为 CrossEncoder 的输入
    # 原因：CrossEncoder 同时对 query 和 passage 做注意力计算，比双编码器的余弦相似度更精确
    pairs = [(query, hit.document.page_content) for hit in hits]
    # 用 CrossEncoder 对 query-passage 对重新打分，比向量余弦距离更准确，用于最终排序决策
    # predict() 返回每个 pair 的相关性分数列表，值范围通常为 [-1, 1] 或 [0, 1]
    scores = reranker.predict(pairs)
    # 将分数和候选重新配对，按分数降序排列，保留每个命中的原始 Document
    reranked = [
        RetrievalHit(document=hit.document, score=float(score))
        for hit, score in sorted(zip(hits, scores), key=lambda item: float(item[1]), reverse=True)
    ]
    # 截断为 top_n 条，控制送入 LLM 的上下文窗口大小和调用开销
    return reranked[:top_n]
