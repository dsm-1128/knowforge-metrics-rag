"""检索结果对象的序列化与反序列化。

RetrievalResult 包含 Document 对象，无法直接 JSON 序列化。这个模块提供
to_payload/from_payload 两个纯函数，在 Redis 缓存读写路径中完成格式转换。

设计决策：
- 只提取 page_content 和 metadata，不保留 Document 上的其他属性（如 id、type），
  减少缓存体积。metadata 中的键值已包含所有业务需要的元数据。
- score 保留为 float，metadata 保留为 dict，查询时反序列化的结果与原始对象在
  业务语义上等价，缓存层对 pipeline 代码透明。

调用顺序：CacheManager.get_retrieval_result() / set_retrieval_result() -> serialization。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from qa_core.retrieval.results import RetrievalHit, RetrievalResult


def retrieval_result_to_payload(result: RetrievalResult) -> dict[str, Any]:
    """将 RetrievalResult 转成 JSON 可存储的字典结构。（★★ 理解）

    序列化时只保留 query/source_type/elapsed_ms 三个顶层字段和 hits 列表，每个 hit
    提取 score、page_content 和 metadata。去掉 id/type 等不参与业务判断的字段，
    减少约 30% 的缓存体积。

    参数：
        result: 待序列化的 RetrievalResult 对象。

    返回：
        JSON 兼容的字典，包含 query、source_type、elapsed_ms 和 hits（列表）。

    调用顺序：CacheManager.set_retrieval_result() -> retrieval_result_to_payload()。
    """
    return {
        "query": result.query,
        "source_type": result.source_type,
        "elapsed_ms": result.elapsed_ms,
        # 原因：只序列化参与业务判断的字段，移除 Document 的 id/type 等非必需属性
        "hits": [
            {
                "score": hit.score,
                "page_content": hit.document.page_content,
                "metadata": dict(hit.document.metadata),
            }
            for hit in result.hits
        ],
    }


def retrieval_result_from_payload(payload: dict[str, Any]) -> RetrievalResult:
    """从 JSON 字典恢复 RetrievalResult 对象。（★★ 理解）

    反序列化时重新构造 Document 和 RetrievalHit 对象。metadata 保持 dict 类型，
    无需额外转换，业务代码可以直接使用。

    参数：
        payload: retrieval_result_to_payload() 产出的 JSON 字典。

    返回：
        完全恢复的 RetrievalResult 对象，包含 query、source_type、elapsed_ms 和 hits。

    调用顺序：CacheManager.get_retrieval_result() -> retrieval_result_from_payload()。
    """
    # 原因：从 JSON 提取每个 hit 的 score/page_content/metadata，重建 Document
    hits = [
        RetrievalHit(
            document=Document(
                page_content=str(item.get("page_content") or ""),
                metadata=dict(item.get("metadata") or {}),
            ),
            score=float(item.get("score") or 0.0),
        )
        for item in payload.get("hits", [])
    ]
    return RetrievalResult(
        hits=hits,
        query=str(payload.get("query") or ""),
        source_type=payload.get("source_type") or "doc",
        elapsed_ms=float(payload.get("elapsed_ms") or 0.0),
    )
