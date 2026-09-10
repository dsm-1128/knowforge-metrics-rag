"""企业级缓存能力。

当前项目使用单层 Redis 缓存配置，并通过缓存 key 生成和 namespace epoch
失效机制控制不同知识库版本之间的缓存边界。

设计决策：
- L1（进程内存）缓存 namespace epoch 等低频元数据，避免每次请求都查询 MySQL。
- L2（Redis）缓存检索结果和 embedding 向量，降低 LLM 和向量模型的重复计算成本。
- 通过 namespace epoch 机制实现缓存失效：版本发布时推进 epoch 值，旧的 Redis key
  自然过期，无需扫描删除全量缓存。

依赖分层：
- manager：统一 L1/L2 缓存入口，负责 key 生成、读写和 epoch 管理。
- keys：缓存 key 生成规则，绑定知识库版本、权限域和配置版本。
- namespaces：缓存 namespace 与 epoch 管理，版本变更时推进 epoch。
- serialization：缓存对象的 JSON 序列化与反序列化。
- stores：底层缓存存储适配器（TTLMemoryCache / RedisJsonCache）。
- embedding：query embedding 缓存包装器，只缓存用户查询不缓存文档向量。
"""

from qa_core.cache.manager import CacheManager, get_cache_manager

__all__ = ["CacheManager", "get_cache_manager"]
