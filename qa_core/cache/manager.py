"""统一缓存入口：L1 进程缓存 + L2 Redis 缓存 + namespace epoch 管理。

模块职责：
1. 为检索结果和 embedding 向量提供两级缓存（L1 进程内存 / L2 Redis）。
2. 通过 namespace epoch 机制实现缓存失效：版本发布时推进 epoch 值，旧的 Redis key
   自然过期，无需扫描删除全量缓存。
3. 提供缓存运行状态查询和统计接口，供 /api/admin/cache 等管理端使用。

设计决策：
- L1 缓存只存 namespace epoch 等低频元数据（TTL 3s），检索结果和 embedding 向量
  直接从 Redis 读写，避免进程内存膨胀导致 OOM。
- 缓存 key 绑定知识库版本、模型版本、数据域和配置版本，跨版本或跨租户的结果不混用。
- 用 LRU 缓存的 get_cache_manager() 确保整个进程生命周期只有一个 CacheManager 实例，
  避免重复创建 Redis 连接池。

依赖分层：
- redis()：延迟初始化 Redis 客户端，首次调用时创建连接。
- namespace_store()：延迟初始化 MySQL namespace 存储，首次调用时创建连接池。
- l1_cache：进程内 TTL 缓存，存储 namespace epoch 等低频元数据。
"""

from __future__ import annotations

import time
from functools import lru_cache

from qa_core.cache.keys import key_digest, retrieval_cache_key
from qa_core.cache.namespaces import CacheNamespaceStore
from qa_core.cache.serialization import retrieval_result_from_payload, retrieval_result_to_payload
from qa_core.cache.stores import RedisJsonCache, TTLMemoryCache
from qa_core.config.settings import get_settings
from qa_core.governance.data_scope import DataScope
from qa_core.retrieval.results import RetrievalResult


class CacheManager:
    """两级缓存入口，封装 L1 进程缓存、L2 Redis 缓存和 MySQL namespace epoch。

    这是缓存系统的统一入口，pipeline 通过 get_cache_manager() 获取单例后调用
    各个方法实现检索结果缓存、embedding 缓存和缓存失效。不直接操作 stores 或
    namespaces 子模块。

    cache_enabled 由 settings 统一控制，业务代码通过 get_cache_manager()
    获取同一实例。

    调用顺序：检索阶段 -> CacheManager。
    """

    def __init__(
        self,
        *,
        redis_cache: RedisJsonCache | None = None,
        namespace_store: CacheNamespaceStore | None = None,
        l1_cache: TTLMemoryCache | None = None,
        enabled: bool | None = None,
    ) -> None:
        """初始化缓存管理器。（★★ 理解）

        构造阶段只保存配置和依赖引用，不建立 Redis/MySQL 连接。真正的连接在
        _redis() / _namespace_store() 首次调用时延迟创建。

        参数：
            redis_cache: Redis 缓存适配器，None 时在首次调用时自动创建。
            namespace_store: MySQL namespace 存储，None 时在首次调用时自动创建。
            l1_cache: 进程内 TTL 缓存，默认创建 TTLMemoryCache 实例。
            enabled: 是否启用缓存，None 时从 settings.cache_enabled 读取。

        调用顺序：进程启动时 -> CacheManager.__init__()。
        """
        # 加载全局运行时配置，缓存开关、TTL 等参数由此决定
        self.settings = get_settings()
        # 原因：允许调用方通过参数显式覆盖配置，便于测试时关闭缓存
        self.enabled = self.settings.cache_enabled if enabled is None else enabled
        # 原因：延迟初始化，避免构造时就建立 Redis/MySQL 连接
        self.redis_cache = redis_cache
        self.namespace_store = namespace_store
        # L1 缓存默认为进程内 TTLMemoryCache，不依赖外部组件
        self.l1_cache = l1_cache or TTLMemoryCache()
        # ── 内部统计状态 ──
        # 这些统计用于 status() 接口和管理端展示，不做精确计数，允许竞态条件下的
        # 微小偏差（约 ±1）。关键业务决策不应依赖这些数字。
        self._stats: dict[str, int] = {
            "retrieval_hits": 0,
            "retrieval_misses": 0,
            "retrieval_writes": 0,
            "embedding_hits": 0,
            "embedding_misses": 0,
            "embedding_writes": 0,
            "invalidations": 0,
        }
        # 最近 20 条缓存事件记录，用于 debug 和管理端近实时查看缓存行为
        self._recent_events: list[dict[str, object]] = []

    def _redis(self) -> RedisJsonCache:
        """延迟初始化 Redis 客户端，首次调用时才建立连接。（★★ 理解）

        原因：Redis 连接在构造时建立会增加启动耗时，且测试场景常需要 mock client。
        延迟初始化让 CacheManager 构造轻量，同时在首次使用时才加载 redis 依赖。

        返回：
            RedisJsonCache 适配器实例。

        调用顺序：CacheManager.get_retrieval_result() / set_retrieval_result() -> _redis()。
        """
        if self.redis_cache is None:
            # 首次调用时创建 Redis 客户端，根据 settings.redis_url 连接
            self.redis_cache = RedisJsonCache()
        return self.redis_cache

    def _namespace_store(self) -> CacheNamespaceStore:
        """延迟初始化 MySQL namespace 存储，首次调用时才建立连接池。

        返回：
            CacheNamespaceStore 实例。

        调用顺序：CacheManager.namespace_epoch() / invalidate_scenario() -> _namespace_store()。
        """
        if self.namespace_store is None:
            self.namespace_store = CacheNamespaceStore()
        return self.namespace_store

    def _record_event(self, *, kind: str, hit: bool | None = None, source_type: str = "", key: str | None = None) -> None:
        """记录进程级缓存观测事件，用于 status() 接口和近实时 debug。

        每次缓存的命中/未命中/写入/失效都会调用此方法。_stats 记录累计计数，
        _recent_events 保留最近 20 条事件记录（按时间逆序展示）。

        参数：
            kind: 事件类型（retrieval_hits / retrieval_misses / retrieval_writes 等）。
            hit: 是否命中缓存。
            source_type: 来源类型（doc / faq / embedding）。
            key: 缓存 key，记录时只取最后一段作为摘要，避免在内存中积累长字符串。

        调用顺序：CacheManager.get_retrieval_result() / set_retrieval_result() -> _record_event()。
        """
        if kind in self._stats:
            self._stats[kind] += 1
        # 原因：只保留 key 的最后一段（digest）用于事件记录，避免长 key 占用内存
        event = {
            "kind": kind,
            "hit": hit,
            "source_type": source_type,
            "key_digest": key.rsplit(":", 1)[-1] if key else "",
            "created_at": round(time.time(), 3),
        }
        self._recent_events.append(event)
        # 只保留最近 20 条事件，避免内存无限增长
        if len(self._recent_events) > 20:
            self._recent_events = self._recent_events[-20:]

    def namespace_epoch(self, *, scenario_id: str, data_scope: DataScope) -> int:
        """读取当前请求所属 namespace 的缓存 epoch。（★★★ 核心）

        执行流程：
          1. 从 data_scope 提取 tenant_id 和 dataset_id，构造 L1 缓存 key。
          2. 先查 L1 进程缓存，命中直接返回，避免每次请求都查询 MySQL。
          3. L1 未命中时查询 MySQL 获取最新 epoch 值。
          4. 将查询结果写入 L1 缓存（TTL 3s），后续相同 namespace 的请求直接从 L1 获取。

        L1 缓存 TTL 设置为 3 秒的原因：版本发布时推进 epoch，3 秒后所有进程的 L1
        缓存自然过期，重新从 MySQL 读取最新 epoch。这比 Redis 发布订阅方案更简单，
        不需要额外的心跳和广播基础设施。

        参数：
            scenario_id: 场景 ID。
            data_scope: 数据域（tenant_id / dataset_id / visibility / user_roles）。

        返回：
            namespace epoch 整数值。

        调用顺序：CacheManager.retrieval_key() -> namespace_epoch()。
        """
        # ── 步骤 1：构造 L1 缓存 key ──
        # 按 scenario_id + tenant_id + dataset_id 三个维度隔离，不同租户/数据集的 epoch 分开缓存
        scope = data_scope.as_dict()
        l1_key = f"epoch:{scenario_id}:{scope['tenant_id']}:{scope['dataset_id']}"
        # ── 步骤 2：查 L1 进程缓存 ──
        # L1 TTL 为 3 秒：版本发布后最多 3 秒所有进程感知到 epoch 变更
        cached = self.l1_cache.get(l1_key)
        if cached is not None:
            return int(cached)
        # ── 步骤 3：L1 未命中，查询 MySQL ──
        # 使用 INSERT ... ON DUPLICATE KEY UPDATE 模式实现"不存在时自动创建"
        epoch = self._namespace_store().get_epoch(
            scenario_id=scenario_id,
            tenant_id=scope["tenant_id"],
            dataset_id=scope["dataset_id"],
        )
        # ── 步骤 4：写入 L1 缓存 ──
        self.l1_cache.set(l1_key, epoch, self.settings.cache_namespace_l1_ttl_seconds)
        return epoch

    def retrieval_key(
        self,
        *,
        kind: str,
        scenario_id: str,
        collection_name: str,
        source_type: str,
        data_scope: DataScope,
        kb_version: str,
        source_filter: str | None,
        query_variants: list[str],
        k: int,
        rerank: bool,
    ) -> str | None:
        """生成检索缓存 key；缓存关闭时返回 None。

        key 的生成委托给 retrieval_cache_key()，绑定知识库版本、数据域、模型版本和
        检索参数等所有可能影响检索结果的维度。任一维度变化都会产生不同的 key。

        参数：
            kind: 检索类型（faq / doc / hybrid）。
            scenario_id: 场景 ID。
            collection_name: Milvus collection 名称。
            source_type: 来源类型。
            data_scope: 数据域（tenant_id / dataset_id / visibility / user_roles）。
            kb_version: 知识库版本号。
            source_filter: 来源过滤条件。
            query_variants: 查询变体列表。
            k: 召回数量。
            rerank: 是否启用重排序。

        返回：
            完整的 Redis key 字符串；缓存关闭时返回 None。

        调用顺序：CacheManager.get_retrieval_result() / set_retrieval_result() -> retrieval_key()。
        """
        if not self.enabled:
            return None
        # 先获取当前 namespace epoch，作为缓存 key 的一部分
        epoch = self.namespace_epoch(scenario_id=scenario_id, data_scope=data_scope)
        return retrieval_cache_key(
            kind=kind,
            scenario_id=scenario_id,
            collection_name=collection_name,
            source_type=source_type,
            data_scope=data_scope,
            kb_version=kb_version,
            cache_epoch=epoch,
            source_filter=source_filter,
            query_variants=query_variants,
            k=k,
            rerank=rerank,
        )

    def get_retrieval_result(self, key: str | None, *, source_type: str = "") -> RetrievalResult | None:
        """读取检索结果缓存。（★★ 理解）

        key 为 None 表示缓存关闭或 key 生成失败，直接返回 None。

        参数：
            key: retrieval_key() 生成的缓存 key，None 时跳过读取。
            source_type: 来源类型（doc / faq），仅用于事件记录。

        返回：
            反序列化的 RetrievalResult；未命中或缓存关闭时返回 None。

        调用顺序：CacheManager.get_retrieval_result() -> retrieval_result_from_payload()。
        """
        if not key or not self.enabled:
            return None
        payload = self._redis().get_json(key)
        hit = bool(payload)
        self._record_event(
            kind="retrieval_hits" if hit else "retrieval_misses",
            hit=hit,
            source_type=source_type,
            key=key,
        )
        return retrieval_result_from_payload(payload) if payload else None

    def set_retrieval_result(self, key: str | None, result: RetrievalResult, *, source_type: str) -> None:
        """写入检索结果缓存，FAQ 和文档使用不同的 TTL。

        FAQ 类型的缓存 TTL 较长（默认 30 分钟），因为 FAQ 的稳定性高、变更频率低。
        文档类型的缓存 TTL 较短（默认 15 分钟），因为文档内容可能频繁更新。

        参数：
            key: retrieval_key() 生成的缓存 key，None 时跳过写入。
            result: 待缓存的检索结果。
            source_type: 来源类型（doc / faq），决定使用哪种 TTL。

        调用顺序：CacheManager.set_retrieval_result() -> retrieval_result_to_payload() -> _redis().set_json()。
        """
        if not key or not self.enabled:
            return
        # 原因：FAQ 的稳定性更高，使用更长的缓存 TTL；文档变更频率更高，TTL 更短
        ttl = self.settings.cache_faq_ttl_seconds if source_type == "faq" else self.settings.cache_doc_ttl_seconds
        self._redis().set_json(key, retrieval_result_to_payload(result), ttl)
        self._record_event(kind="retrieval_writes", source_type=source_type, key=key)

    def embedding_key(self, text: str) -> str | None:
        """生成 query embedding 缓存 key，绑定 embedding 模型版本和路径。

        key 格式：{prefix}:embedding:{model_version}:{digest}
        digest 根据 text、model_version 和 model_path 生成，模型升级后旧 key 自然失效。

        参数：
            text: 用户查询文本。

        返回：
            Redis key 字符串；缓存关闭时返回 None。

        调用顺序：CacheManager.get_embedding_vector() / set_embedding_vector() -> embedding_key()。
        """
        if not self.enabled:
            return None
        # 原因：digest 包含 model_version 和 model_path，模型升级后自动生成不同的 digest
        digest = key_digest(
            {
                "kind": "embedding",
                "text": text,
                "embedding_model_version": self.settings.embedding_model_version,
                "embedding_model_path": self.settings.embedding_model_path,
            }
        )
        return f"{self.settings.cache_key_prefix}:embedding:{self.settings.embedding_model_version}:{digest}"

    def get_embedding_vector(self, key: str | None) -> list[float] | None:
        """读取 query embedding 缓存。（★★ 理解）

        key 为 None 时直接返回 None。Redis 中存储的格式为 {"vector": [float, ...]}，
        反序列化时校验 vector 类型是否为 list，避免脏数据污染上层逻辑。

        参数：
            key: embedding_key() 生成的缓存 key，None 时跳过读取。

        返回：
            embedding 向量（float 列表）；未命中、缓存关闭或数据格式异常时返回 None。

        调用顺序：CachedEmbeddings.embed_query() -> CacheManager.get_embedding_vector()。
        """
        if not key or not self.enabled:
            return None
        payload = self._redis().get_json(key)
        # 原因：校验 vector 字段类型，防止 Redis 中的脏数据导致上层代码报错
        if not payload or not isinstance(payload.get("vector"), list):
            self._record_event(kind="embedding_misses", hit=False, source_type="embedding", key=key)
            return None
        self._record_event(kind="embedding_hits", hit=True, source_type="embedding", key=key)
        return [float(item) for item in payload["vector"]]

    def set_embedding_vector(self, key: str | None, vector: list[float]) -> None:
        """写入 query embedding 缓存。

        参数：
            key: embedding_key() 生成的缓存 key，None 时跳过写入。
            vector: embedding 向量（float 列表）。

        调用顺序：CachedEmbeddings.embed_query() -> CacheManager.set_embedding_vector()。
        """
        if not key or not self.enabled:
            return
        self._redis().set_json(key, {"vector": vector}, self.settings.cache_embedding_ttl_seconds)
        self._record_event(kind="embedding_writes", source_type="embedding", key=key)

    def invalidate_scenario(self, scenario_id: str) -> int:
        """推进场景下所有 namespace 的 cache epoch，使旧缓存 key 自然失效。（★★★ 核心）

        执行流程：
          1. 清空 L1 进程缓存，确保后续请求从 MySQL 读取最新 epoch。
          2. 在 MySQL 中推进该场景下所有 namespace 的 epoch 值（+1）。
          3. 记录失效事件。

        失效后，旧的 Redis key 不会立即被删除——它们会在 TTL 过期后自然淘汰。
        新的请求生成的缓存 key 包含新的 epoch 值，不会命中旧缓存。

        参数：
            scenario_id: 场景 ID。

        返回：
            受影响的 namespace 数量。

        调用顺序：版本发布或回滚时 -> CacheManager.invalidate_scenario()。
        """
        if not self.enabled:
            return 0
        # 清空 L1 缓存：所有进程的 epoch 缓存都会在下次请求时重新从 MySQL 读取
        self.l1_cache.clear()
        affected = self._namespace_store().bump_scenario_epoch(scenario_id)
        self._record_event(kind="invalidations", source_type="namespace", key=scenario_id)
        return affected

    def status(self, *, scenario_id: str | None = None) -> dict[str, object]:
        """返回缓存运行状态，供 /api/admin/cache 等管理端使用。

        返回的信息包括：缓存开关、Redis 连接状态、配置参数、统计计数、最近事件
        和 namespace 列表。

        参数：
            scenario_id: 可选的场景 ID，用于过滤 namespace 列表。

        返回：
            包含 enabled、redis、config、stats、recent_events 和 namespaces 的字典。

        调用顺序：管理 API -> CacheManager.status()。
        """
        namespaces = self._namespace_store().list_namespaces(scenario_id=scenario_id) if self.enabled else []
        redis_ok = self._redis().ping() if self.enabled else False
        return {
            "enabled": self.enabled,
            "redis": {
                "ok": redis_ok,
                "host": self.settings.redis_host,
                "port": self.settings.redis_port,
                "db": self.settings.redis_db,
            },
            "config": {
                "key_prefix": self.settings.cache_key_prefix,
                "faq_ttl_seconds": self.settings.cache_faq_ttl_seconds,
                "doc_ttl_seconds": self.settings.cache_doc_ttl_seconds,
                "embedding_ttl_seconds": self.settings.cache_embedding_ttl_seconds,
                "namespace_l1_ttl_seconds": self.settings.cache_namespace_l1_ttl_seconds,
                "embedding_model_version": self.settings.embedding_model_version,
                "reranker_model_version": self.settings.reranker_model_version,
                "chunk_schema_version": self.settings.chunk_schema_version,
            },
            "stats": dict(self._stats),
            "recent_events": list(reversed(self._recent_events)),
            "namespaces": namespaces,
        }


@lru_cache(maxsize=1)
def get_cache_manager() -> CacheManager:
    """返回进程级 CacheManager 单例。

    使用 lru_cache(maxsize=1) 确保整个进程生命周期只有一个 CacheManager 实例和
    对应的 Redis/MySQL 连接。测试时可以通过 get_cache_manager.cache_clear() 重置。

    调用顺序：检索阶段 -> get_cache_manager()。
    """
    return CacheManager()
