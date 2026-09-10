"""缓存 key 生成规则，绑定知识库版本、权限域和配置版本。

设计决策：
- key 必须绑定所有可能影响检索结果的维度，包括：数据域（tenant_id / dataset_id /
  visibility / user_roles）、知识库版本、缓存 epoch、检索参数和模型版本。
- 任一维度变化都会产生不同的 key，确保跨版本、跨租户或跨角色的缓存结果不混用。
- key 前缀按可读维度分层，冒号分隔；digest 放在末尾作为唯一性保证。

key 格式：{prefix}:{kind}:{scenario_id}:{tenant_id}:{dataset_id}:e{epoch}:{digest}

调用顺序：CacheManager.retrieval_key() -> keys.retrieval_cache_key() / key_digest()。
"""

from __future__ import annotations

import json
from typing import Any

from qa_core.config.settings import get_settings
from qa_core.governance.data_scope import DataScope
from qa_core.utils import stable_hash


def key_digest(payload: dict[str, Any]) -> str:
    """将结构化 payload 转成稳定的短摘要（SHA-256 前 24 位）。（★★ 理解）

    sort_keys=True 保证相同内容生成相同摘要；separators=(",", ":") 去掉空格减少
    digest 的输入长度。结果取 SHA-256 前 24 位，在唯一性和 key 可读性之间取得平衡。

    参数：
        payload: 包含所有缓存维度的字典，key 按字母序排序后序列化。

    返回：
        24 字符的十六进制摘要字符串。

    调用顺序：CacheManager.embedding_key() / retrieval_cache_key() -> key_digest()。
    """
    # 原因：sort_keys=True 确保相同内容生成相同摘要，separators 去掉空格减少长度
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return stable_hash(raw)[:24]


def retrieval_cache_key(
    *,
    kind: str,
    scenario_id: str,
    collection_name: str,
    source_type: str,
    data_scope: DataScope,
    kb_version: str,
    cache_epoch: int,
    source_filter: str | None,
    query_variants: list[str],
    k: int,
    rerank: bool,
) -> str:
    """生成 FAQ/Doc 检索结果缓存 key。（★★★ 核心）

    缓存 key 绑定以下维度，任一变化都会导致缓存 miss：
    - 数据域隔离：tenant_id / dataset_id / visibility / user_roles
    - 知识库版本：kb_version 变化意味着底层数据已更新
    - 缓存 epoch：版本发布/回滚时推进，旧 key 自然失效无需扫描删除
    - 模型版本：embedding / reranker / chunk_schema 任一升级都会使旧向量/排序失效
    - 检索参数：source_filter / query_variants / k / rerank

    参数：
        kind: 检索类型（faq / doc / hybrid）。
        scenario_id: 场景 ID。
        collection_name: Milvus collection 名称。
        source_type: 来源类型。
        data_scope: 数据域对象。
        kb_version: 知识库版本号。
        cache_epoch: 缓存 namespace epoch。
        source_filter: 来源过滤条件。
        query_variants: 查询变体列表。
        k: 召回数量。
        rerank: 是否启用重排序。

    返回：
        完整的 Redis key 字符串，格式为 {prefix}:{kind}:{scenario_id}:{tenant_id}:{dataset_id}:e{epoch}:{digest}。

    调用顺序：CacheManager.retrieval_key() -> retrieval_cache_key()。
    """
    settings = get_settings()
    scope = data_scope.as_dict()
    # 将所有影响缓存结果的维度序列化为 JSON -> SHA256 截取前 24 位作为短摘要
    # sort_keys=True 保证相同内容生成相同摘要；separators=(",", ":") 去掉空格减少长度
    digest = key_digest(
        {
            "kind": kind,
            "scenario_id": scenario_id,
            "collection_name": collection_name,
            "source_type": source_type,
            # 数据域隔离字段：跨租户/数据集/可见性/角色的结果不得复用
            "tenant_id": scope["tenant_id"],
            "dataset_id": scope["dataset_id"],
            "visibility": scope["visibility"],
            "user_roles": sorted(scope["user_roles"]),
            # 知识库版本和 epoch：版本更新时自然失效
            "kb_version": kb_version,
            "cache_epoch": int(cache_epoch),
            # 检索参数：任一变化都会产生不同的检索结果
            "source_filter": source_filter or "",
            "query_variants": query_variants,
            "k": int(k),
            "rerank": bool(rerank),
            # 模型版本：向量空间或排序模型变化时旧缓存不可用
            "embedding_model_version": settings.embedding_model_version,
            "reranker_model_version": settings.reranker_model_version,
            "chunk_schema_version": settings.chunk_schema_version,
        }
    )
    # key 前缀按可读维度分层，冒号分隔；digest 放在末尾作为唯一性保证
    return (
        f"{settings.cache_key_prefix}:{kind}:{scenario_id}:{scope['tenant_id']}:"
        f"{scope['dataset_id']}:e{int(cache_epoch)}:{digest}"
    )
