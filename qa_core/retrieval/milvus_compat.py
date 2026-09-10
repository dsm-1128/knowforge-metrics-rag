"""Milvus Hybrid Search 兼容工具：集中封装 BM25 内置函数和连接参数，使 store.py 可专注混合检索流程。

设计决策：
- BM25 analyzer_params={"type": "chinese"} 是中文场景的必选项而非可选性能调优：
  中文文本词之间没有空格分隔，必须经过中文分词器（jieba/jieba_simple）才能产生有
  意义的 token。如果使用默认英文分词器（按空白符切分），BM25 稀疏检索对中文 query
  几乎完全失效（每个汉字独立成 token，无语义分组）。
- langchain-milvus 连接参数集中生成：FAQ 和文档集合共享同一 Milvus 实例和 database，
  区别在于 collection_name。集中生成连接参数避免 store 层重复指定 URI。
- ensure_milvus_database 自动创建 database：首次部署时 database 可能尚未创建，
  自动创建可简化部署流程，避免手动创建步骤。

依赖分层：
- 被 retrieval.store.MilvusHybridStore 在构造时调用。
- 被 retrieval.factory.warmup_retrieval_stack() 在启动预热时调用。
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from langchain_milvus import BM25BuiltInFunction
from pymilvus import MilvusClient

from qa_core.config.settings import get_settings


HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF = 64


def langchain_connection_args() -> dict[str, str]:
    """构建传给 langchain-milvus 的连接参数。

    业务检索统一走 langchain-milvus；这里集中生成 Milvus URI 和可选 database 参数。
    FAQ/Doc 查询目标由 collection_name 决定，不再额外维护 collection 级 alias。

    离线重建脚本执行 `--reset-collections` 时也复用这组连接参数，保证“删集合”
    和“后续写入/在线检索”指向同一个 Milvus 实例与 database。

    调用顺序：检索准备或检索执行 -> langchain_connection_args()。
    """
    # 读取与前置校验相同的 Milvus 配置，避免预热和真实检索连到不同实例。
    settings = get_settings()
    args = {"uri": settings.milvus_uri}
    # 如果配置指定了 Milvus database（非默认数据库 "default"），则附加 db_name 参数
    # 原因：多租户或多环境共用一个 Milvus 实例时，用 database 做逻辑隔离
    # FAQ/Doc 查询目标由后续 collection_name 决定，不再额外维护 collection 级 alias
    if settings.milvus_database:
        args["db_name"] = settings.milvus_database
    return args


def ensure_milvus_database() -> None:
    """确保配置中的 Milvus database 可用。

    该函数只保证 database 容器存在，不创建 FAQ/文档 collection，也不写入业务
    向量；collection 的 schema 和索引由 `MilvusHybridStore.store` 首次访问时
    校验或创建。

    调用顺序：启动预热或离线 reset -> ensure_milvus_database()。
    """
    # LangChain wrapper 初始化前先确认目标 database 存在，首次部署时可自动补齐。
    settings = get_settings()
    client = MilvusClient(uri=settings.milvus_uri)
    # list_databases() 读取当前实例已有 database，后面只在缺失时执行创建。
    databases = client.list_databases()
    # 检查配置的 database 是否存在，不存在则自动创建
    # 原因：首次部署时 database 可能尚未创建，自动创建可简化部署流程
    # 注意：此操作仅在进程启动时执行一次，不影响运行时性能
    if settings.milvus_database and settings.milvus_database not in databases:
        client.create_database(settings.milvus_database)


def bm25_function():
    """构建 Milvus 2.5+ 内置 BM25 稀疏向量函数，替换旧版本地 BM25 方案。

    analyzer_params={"type": "chinese"} 是中文场景的必选项，不是可选的性能调优：
    中文文本词之间没有空格分隔，必须经过分词器才能产生有意义的 token。如果使用默认
    的英文分词器（按空白符切分），BM25 稀疏检索对中文 query 几乎失效。

    调用顺序：检索准备或检索执行 -> bm25_function()。
    """
    return BM25BuiltInFunction(
        input_field_names="text",
        output_field_names="sparse",
        analyzer_params={"type": "chinese"},
        enable_match=True,
    )


def hybrid_index_params() -> list[dict[str, object]]:
    """返回 Dense + BM25 Sparse 混合检索的显式索引配置。

    顺序必须与 store.py 中 vector_field=["dense", "sparse"] 保持一致：
    - dense：BGE 向量已做 L2 归一化，使用 HNSW + L2；HNSW 负责近似最近邻加速。
    - sparse：Milvus BM25 Function 输出字段使用 BM25 metric，由 Milvus 服务端管理稀疏索引。

    调用顺序：检索准备或检索执行 -> MilvusHybridStore.store -> hybrid_index_params()。
    """
    return [
        {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
        },
        {"index_type": "AUTOINDEX", "metric_type": "BM25", "params": {}},
    ]


def hybrid_search_params() -> list[dict[str, object]]:
    """返回与 hybrid_index_params() 对应的搜索参数。

    调用顺序：检索准备或检索执行 -> MilvusHybridStore.store -> hybrid_search_params()。
    """
    return [
        {"metric_type": "L2", "params": {"ef": HNSW_EF}},
        {"metric_type": "BM25", "params": {}},
    ]


def milvus_endpoint_available(timeout: float = 1.5) -> bool:
    """快速判断 Milvus TCP 端口是否可达，用于启动前置校验。

    调用顺序：检索准备或检索执行 -> milvus_endpoint_available()。
    """
    # 这里只做轻量 TCP 探测，不创建 MilvusClient，也不触发 collection 初始化。
    settings = get_settings()
    # 从 Milvus URI 中解析主机和端口
    # 支持的 URI 格式示例：http://localhost:19530 或 tcp://10.0.0.1:19530
    parsed = urlparse(settings.milvus_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 19530
    try:
        # 尝试 TCP 连接 Milvus 服务端口
        # 超时时间设为 1.5 秒，避免进程启动阻塞过久
        with socket.create_connection((host, port), timeout=timeout):
            # 连接建立后立即关闭探测 socket，返回 True 交给上层继续完整预热。
            return True
    except OSError:
        # OSError 涵盖连接超时、连接拒绝、DNS 解析失败等所有网络异常
        return False
