"""V2 GraphRAG（图检索增强生成）索引构建与路径检索服务。

本包提供了 KnowForge V2.1 的轻量 GraphRAG 实现，包含：
  - schema.py：核心数据结构（实体、关系、路径、索引）。
  - extractor.py：基于领域术语和正则规则的确定性抽取器。
  - graph_store.py：JSON 持久化的图存储与查询服务。
  - evaluation.py：用于验收的质量评估工具。

使用方式:
    from qa_core.graphrag import rebuild_graphrag_index, query_graphrag

    rebuild_graphrag_index("it_helpdesk")
    result = query_graphrag("VPN 故障如何处理？", scenario_id="it_helpdesk")
"""

from qa_core.graphrag.graph_store import (
    GraphRAGStore,
    get_graphrag_store,
    list_graphrag_paths,
    query_graphrag,
    rebuild_graphrag_index,
)

__all__ = [
    "GraphRAGStore",
    "get_graphrag_store",
    "list_graphrag_paths",
    "query_graphrag",
    "rebuild_graphrag_index",
]
