"""Embedding 查询缓存包装器。

对用户 query 的 embedding 结果做缓存，降低热点问题的向量化成本。
文档入库时的 embed_documents 仍然走原始模型，避免大批 chunk 污染 Redis 缓存。

设计决策：
- 只缓存 query embedding，不缓存 document embedding。原因是文档入库时 chunk 量级在
  数百到数千条，全部写入 Redis 会显著增加内存压力和网络开销，且文档 embedding 的
  复用率远低于热点 query。
- 缓存 key 绑定 embedding_model_version 和 embedding_model_path，模型升级或切换
  时旧缓存自动失效，避免不同模型产出的向量混用导致检索质量下降。
- 通过 CacheManager 统一管理 L1（进程内存）+ L2（Redis）两级缓存，业务代码只需
  调用 embed_query/embed_documents，不感知缓存层级。

依赖分层：
- qa_core.cache.manager：统一的 L1/L2 缓存入口，负责 key 生成、读写和 epoch 管理。
"""

from __future__ import annotations
from typing import Any
from qa_core.cache.manager import get_cache_manager

class CachedEmbeddings:
    """只缓存用户 query 的 embedding 向量，不缓存入库文档向量。

    这是 embedding 模型的装饰器/代理，对上层 Embedding 接口保持完全兼容：
    - embed_query：先查 Redis 缓存，命中直接返回；未命中调底层模型计算后写入缓存。
    - embed_documents：直接透传底层模型，不经过缓存层。
    - 其他属性/方法通过 __getattr__ 委托给底层模型。

    缓存策略的业务原因：
    - query embedding 复用率高：热门问题（如"报销流程"）会被频繁查询，缓存命中率
      可达 60-80%，显著降低 BGE 模型 GPU/CPU 推理成本。
    - document embedding 不应缓存：每次入库的文档 chunk 几乎不重复，缓存命中率
      接近零，写入反而浪费 Redis 内存并增加序列化开销。

    调用顺序：检索阶段 / 入库阶段 -> CachedEmbeddings。
    """

    def __init__(self, base_embeddings: Any) -> None:
        """包装底层 embedding 模型，叠加查询缓存层。（★★ 理解）

        构造阶段只保存底层模型引用，不建立 Redis 连接。真正的缓存读写发生在
        embed_query() 首次调用时，由 CacheManager 懒初始化 Redis 客户端。

        参数：
            base_embeddings: 原始 embedding 模型实例（通常是 BGE 模型的
                LangChain Embeddings 包装），必须实现 embed_query() 和
                embed_documents() 方法。

        调用顺序：检索阶段 -> CachedEmbeddings.__init__()。
        """
        # 保存底层 embedding 模型引用，后续 embed_query/embed_documents 都会用到
        self.base_embeddings = base_embeddings

    def embed_query(self, text: str) -> list[float]:
        """缓存用户 query embedding，降低热点问题向量化成本。（★★★ 核心）

        执行流程：
          1. 通过 CacheManager 生成缓存 key（绑定模型版本、文本内容）。
          2. 先查 L2 Redis 缓存，命中直接返回向量，跳过模型推理。
          3. 未命中时调用底层模型 embed_query() 计算稠密向量。
          4. 将计算结果写入 Redis 缓存，供后续相同/相似查询复用。

        缓存 key 由 CacheManager.embedding_key() 统一生成，包含：
          - embedding_model_version：模型版本号，升级后旧 key 自动失效。
          - text 的 SHA-256 摘要：相同文本产生相同 key，不同文本不冲突。

        参数：
            text: 用户查询文本（单个字符串，非列表）。

        返回：
            浮点数列表，即 BGE 模型产出的稠密向量，维度由模型决定（默认 1024）。

        调用顺序：检索阶段 -> CachedEmbeddings.embed_query()。
        """
        # ── 步骤 1：生成缓存 key ──
        # CacheManager 根据 embedding_model_version + text 内容生成唯一 key，
        # 模型版本号绑定确保模型升级后旧向量不会与新向量混用
        manager = get_cache_manager()
        key = manager.embedding_key(text)
        # ── 步骤 2：先查 Redis 缓存 ──
        # 热点问题（如"报销流程是什么"）会被大量用户重复查询，缓存命中时
        # 直接返回向量，跳过 BGE 模型推理，节省数十到数百毫秒
        cached = manager.get_embedding_vector(key)
        if cached is not None:
            return cached
        # ── 步骤 3：缓存未命中，调用底层模型计算 ──
        # BGE 模型推理是 CPU/GPU 密集型操作，单次耗时 20-100ms
        vector = self.base_embeddings.embed_query(text)
        # ── 步骤 4：写入 Redis 缓存 ──
        # TTL 由 CacheManager 统一控制，过期后自动重新计算
        manager.set_embedding_vector(key, vector)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档入库向量不写 Redis，避免大批 chunk 污染业务缓存。（★★ 理解）

        设计原因：文档入库时一次可能产生数百个 chunk，每个 chunk 都需要 embedding。
        如果全部写入 Redis：
          - 缓存命中率极低（每个文档 chunk 几乎只被查询一次）。
          - 大量写入会挤占 Redis 内存，影响 query 缓存的命中率。
          - 序列化/反序列化开销抵消甚至超过缓存节省的时间。

        因此入库路径直接透传底层模型，不做缓存拦截。

        参数：
            texts: 待嵌入的文档文本列表（chunk 切分后的文本片段）。

        返回：
            二维浮点数列表，每行对应一个输入文本的稠密向量。

        调用顺序：入库阶段 -> CachedEmbeddings.embed_documents()。
        """
        # 原因： 文档 chunk 几乎不重复查询，缓存命中率趋近于零，写入反而浪费
        # Redis 内存并增加网络开销。直接透传底层模型，不经过缓存层。
        return self.base_embeddings.embed_documents(texts)

    def __getattr__(self, name: str):
        """未显式定义的方法委托给底层 embedding 模型。

        这是 Python 代理模式的实现：当访问 CachedEmbeddings 上不存在的属性时，
        Python 自动调用 __getattr__，将请求转发给被包装的 base_embeddings 对象。
        这样可以保证 CachedEmbeddings 与原始 Embeddings 接口完全兼容，调用方无需
        感知缓存层的存在。

        参数：
            name: 被访问的属性名（如 model_name、client 等）。

        返回：
            底层模型对应属性的值。

        调用顺序：任意访问 CachedEmbeddings 未定义属性时自动触发。
        """
        return getattr(self.base_embeddings, name)
