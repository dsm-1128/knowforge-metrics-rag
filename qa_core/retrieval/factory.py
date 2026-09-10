"""检索集合工厂与启动预热。

这个模块只负责“拿到检索对象”和“启动时预热检索依赖”。真正执行 Milvus 查询的逻辑在
`store.py`。这里通过 collection_name 缓存 MilvusHybridStore，避免每次请求重复创建
连接对象。
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from qa_core.config.logging_config import get_logger
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.scenarios.registry import get_scenario_registry, resolve_scenario

logger = get_logger(__name__)

_known_active_versions: dict[str, str] = {}
_retrieval_cache_lock = threading.RLock()
_warmup_lock = threading.Lock()
_warmup_thread: threading.Thread | None = None
_warmup_state: dict[str, object] = {
    "status": "not_started",
    "started_at": None,
    "finished_at": None,
    "elapsed_seconds": None,
    "summary": None,
    "error": None,
}


@lru_cache(maxsize=32)
def get_hybrid_store(collection_name: str):
    """按 collection_name 返回已缓存的 Milvus 混合检索封装，多业务场景间隔离集合连接。

    LRU 缓存最多保留 32 个集合实例，足够覆盖当前冻结场景包，同时避免无界增长。

    把 lru_cache 放在 get_hybrid_store 而非顶层，是为了让 warmup_retrieval_stack 可以
    显式遍历每个 collection 触发缓存填充，而顶层调用方的无参 get_faq_store/get_doc_store
    仍享受缓存复用——预热与按需获取共享同一缓存空间。

    参数：
        collection_name: Milvus collection 名称。

    返回：
        已缓存或新创建的 MilvusHybridStore。

    调用顺序：检索准备或检索执行 -> get_hybrid_store()。
    """
    from qa_core.retrieval.store import MilvusHybridStore

    return MilvusHybridStore(collection_name)

def _active_scenario_collection(kind: Literal["faq", "doc"]) -> str:
    """返回当前默认场景的 FAQ 或文档 collection 名。

    参数：
        kind: faq 表示 FAQ 集合，doc 表示文档集合。

    返回：
        当前默认场景配置的 collection 名。

    调用顺序：检索准备或检索执行 -> _active_scenario_collection()。
    """
    # 解析当前业务场景，获取其对应的集合配置
    scenario = resolve_scenario()
    return scenario.faq_collection if kind == "faq" else scenario.doc_collection

def get_faq_store(collection_name: str | None = None):
    """返回已缓存的 FAQ 混合集合封装。

    离线入库总流程会显式传入 `scenario.faq_collection`，保证写入目标和当前
    场景一致；在线调用不传时才使用默认场景的 FAQ collection。最终实例由
    `get_hybrid_store()` 按 collection 名缓存，FAQ 和文档不会共享错误的集合对象。

    参数：
        collection_name: 可选集合名；为空时使用当前默认场景的 faq_collection。

    返回：
        FAQ MilvusHybridStore。

    调用顺序：检索准备或检索执行 -> get_faq_store()。
    """
    return get_hybrid_store(collection_name or _active_scenario_collection("faq"))


def get_doc_store(collection_name: str | None = None):
    """返回已缓存的文档混合集合封装。

    目录入库会显式传入当前场景的 `doc_collection`。保留可选参数是为了兼容
    在线检索和旧调用方；不传时通过场景 registry 推导默认 collection。

    参数：
        collection_name: 可选集合名；为空时使用当前默认场景的 doc_collection。

    返回：
        文档 MilvusHybridStore。

    调用顺序：检索准备或检索执行 -> get_doc_store()。
    """
    return get_hybrid_store(collection_name or _active_scenario_collection("doc"))


def clear_retrieval_store_cache() -> None:
    """清空进程内检索 store 缓存和 active 版本快照。

    用于测试或管理动作。模型缓存不在这里清理，避免每次版本切换都重新加载 BGE / Reranker。

    调用顺序：检索准备或检索执行 -> clear_retrieval_store_cache()。
    """
    with _retrieval_cache_lock:
        get_hybrid_store.cache_clear()
        _known_active_versions.clear()


def sync_retrieval_cache_for_active_version(scenario_id: str, active_kb_version: str) -> None:
    """active kb_version 变化时清空已缓存的 Milvus store wrapper。（★★★ 核心）

    知识库重建可能会 drop/recreate Milvus collection。若 API 进程已经预热过旧 collection，
    LangChain Milvus wrapper 会继续持有旧连接状态，导致 MySQL active 已切换但检索召回为空。
    因此每次请求解析到 active 版本后做一次轻量比对，发现版本变化就清空 store 缓存。

    执行流程：
      1. 查询 _known_active_versions 中记录的上一次版本号。
      2. 无记录（首次请求）或版本相同时不做操作。
      3. 版本变化时：清空 lru_cache 的 Milvus wrapper、清除全部版本快照、更新为当前版本。

    参数：
        scenario_id: 业务场景 ID。
        active_kb_version: 当前请求解析出的 active 知识库版本号。

    调用顺序：检索准备或检索执行 -> sync_retrieval_cache_for_active_version()。
    """
    with _retrieval_cache_lock:
        previous = _known_active_versions.get(scenario_id)
        if previous is None:
            _known_active_versions[scenario_id] = active_kb_version
            return
        if previous == active_kb_version:
            return

        logger.warning(
            "检测到场景 %s 的 active kb_version 从 %s 切换到 %s，清空 Milvus store 缓存。",
            scenario_id,
            previous,
            active_kb_version,
        )
        get_hybrid_store.cache_clear()
        _known_active_versions.clear()
        _known_active_versions[scenario_id] = active_kb_version


def retrieval_warmup_state() -> dict[str, object]:
    """返回当前检索预热状态，供健康检查和诊断接口读取。

    这里只返回状态快照，不等待线程，也不触发新的预热；这样健康检查不会因为
    模型加载耗时而被拖住。复制字典后再返回，避免调用方意外修改模块内状态。

    调用顺序：健康检查/管理接口 -> retrieval_warmup_state()。
    """

    # 使用同一把锁读取状态，避免后台线程更新到一半时被健康检查读到半截数据。
    with _warmup_lock:
        return dict(_warmup_state)


def start_retrieval_warmup_background() -> dict[str, object]:
    """启动检索栈后台预热线程，并立即返回当前状态。

    预热包含模型加载和 Milvus collection 初始化，属于同步且可能较慢的工作。
    这里不在调用线程直接执行，而是创建 daemon 线程；上层可以继续完成启动编排，
    再通过 ``wait_for_retrieval_warmup()`` 等待明确的 ready/failed 结果。

    如果预热已经处于 ``running`` 或 ``ready``，函数不会重复创建线程，保证同一个
    API 进程内不会同时初始化多套相同的检索资源。

    调用顺序：warmup_runtime() -> start_retrieval_warmup_background()。
    """

    global _warmup_thread
    with _warmup_lock:
        # lifespan 可能重复触发检查；运行中或已完成时直接返回现有快照。
        if _warmup_state["status"] == "running":
            return dict(_warmup_state)
        if _warmup_state["status"] == "ready":
            return dict(_warmup_state)

        # 先将状态切换为 running，再创建线程，避免其他观察者误以为预热尚未开始。
        _warmup_state.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "finished_at": None,
                "elapsed_seconds": None,
                "summary": None,
                "error": None,
            }
        )
        # daemon=True 表示解释器退出时不因预热线程残留而阻止进程关闭。
        _warmup_thread = threading.Thread(
            target=_run_retrieval_warmup_in_thread,
            name="retrieval-warmup",
            daemon=True,
        )
        # start() 只负责安排后台执行；真正的异常会由预热线程捕获并写入 _warmup_state。
        _warmup_thread.start()
        return dict(_warmup_state)


def wait_for_retrieval_warmup(timeout: float | None = None) -> dict[str, object]:
    """等待检索预热完成；失败或超时会抛错，调用方不得接收用户检索请求。

    预热线程仍独立运行，以满足 langchain-milvus 的事件循环要求；但 API lifespan 启动阶段会调用
    本函数等待其完成。这样既保留预热状态诊断，也保证 BGE、Reranker 和 Milvus collection
    已经就绪，不会把冷启动延迟转嫁给第一个用户。

    参数：
        timeout: 最大等待秒数；None 表示等待预热自然完成。

    返回：
        预热成功后的摘要信息。

    异常：
        TimeoutError: 预热超过 timeout 仍未完成。
        RuntimeError: 预热失败，服务不应继续启动。
    """
    global _warmup_thread
    # 先读取线程引用和状态，再离开锁等待，避免 join 阻塞其他线程读取健康状态。
    with _warmup_lock:
        status = str(_warmup_state["status"])
        thread = _warmup_thread

    # 如果调用方直接等待而没有先启动预热，这里补一次启动，保证函数具备自洽性。
    if status == "not_started" or thread is None:
        start_retrieval_warmup_background()
        with _warmup_lock:
            thread = _warmup_thread

    # join 只等待后台线程，不会把线程中的异常直接抛回；异常结果由下面的状态判断转换。
    if thread is not None:
        thread.join(timeout=timeout)

    # 重新复制状态，判断线程是否真正结束以及结束原因。
    with _warmup_lock:
        state = dict(_warmup_state)

    # 线程仍存活说明超过等待上限，不能让上层误判为 ready。
    if thread is not None and thread.is_alive():
        raise TimeoutError(f"检索栈预热超过等待上限：{timeout}s")
    # 线程结束但状态不是 ready，说明模型、Milvus 或 active 版本准备失败。
    if state["status"] != "ready":
        raise RuntimeError(f"检索栈预热失败：{state.get('error') or state['status']}")
    # 只返回成功摘要，隐藏内部可变状态结构。
    return dict(state.get("summary") or {})


def _run_retrieval_warmup_in_thread() -> None:
    """在线程中执行预热，并为 langchain-milvus 建立独立事件循环。

    langchain-milvus 的部分初始化路径可能依赖当前线程的事件循环，因此不能只把
    普通函数扔进默认线程池后就结束。这里显式创建、绑定、关闭事件循环，同时把成功
    摘要或失败原因写入共享状态，供 ``wait_for_retrieval_warmup()`` 使用。

    调用顺序：start_retrieval_warmup_background() ->
    _run_retrieval_warmup_in_thread() -> warmup_retrieval_stack()。
    """

    # 预热线程拥有自己的事件循环，避免复用 FastAPI 主事件循环的上下文。
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # 使用单调时钟计算耗时，不受系统时间回拨影响。
    started = time.perf_counter()
    try:
        # 这里是真正加载 embedding、reranker 并触发 Milvus collection 初始化的函数。
        summary = warmup_retrieval_stack()
        with _warmup_lock:
            # 只有完整流程无异常返回，才把状态标记为 ready。
            _warmup_state.update(
                {
                    "status": "ready",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "summary": summary,
                    "error": None,
                }
            )
    except Exception as exc:
        # 捕获异常并记录到状态，而不是让 daemon 线程静默退出；这样主线程能得到可解释错误。
        logger.exception("检索栈后台预热失败")
        with _warmup_lock:
            _warmup_state.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "summary": None,
                    "error": str(exc),
                }
            )
    finally:
        # 清除线程局部事件循环并释放资源，避免后续代码误用已关闭的 loop。
        asyncio.set_event_loop(None)
        loop.close()


def _utc_now() -> str:
    """返回 UTC 时间字符串（ISO 8601 格式），用于检索预热状态记录。

    独立函数而非内联 datetime 调用，是为了方便测试时替换时钟。

    返回：
        形如 "2026-08-02T10:30:00.123456+00:00" 的 ISO 格式字符串。

    调用顺序：warmup_retrieval_stack() -> _utc_now()。
    """
    return datetime.now(timezone.utc).isoformat()


def warmup_retrieval_stack() -> dict[str, object]:
    """服务启动时加载检索模型、全部冻结场景集合和当前 active 版本。任一预热失败直接阻断启动。

    执行流程：
      1. 加载 BGE embedding 模型，并对样例问题做一次向量化。
      2. 检查 Milvus 端点是否可达。
      3. 遍历所有场景，解析 active 知识库版本，并触发 FAQ/文档集合懒加载。
      4. 加载 CrossEncoder reranker，并对样例 query-passage 做一次预测。
      5. 记录预热耗时、场景数量、集合数量和 active 版本。

    异常：
        RuntimeError: Milvus 不可达或关键依赖不可用。
    """
    # 使用固定样例贯穿 embedding 和 reranker 预热，便于日志对比且不引入用户数据。
    sample_query = "当前业务资料有哪些处理流程"
    # 记录整个检索栈初始化耗时，后续可用于定位模型加载或 collection 初始化瓶颈。
    started = time.perf_counter()
    # 延迟导入重量级模块，普通 API 导入和单元测试不必立刻加载 Milvus/模型依赖。
    from qa_core.retrieval.milvus_compat import milvus_endpoint_available
    from qa_core.retrieval.models import get_embeddings, get_reranker

    # 获取场景注册表，后面会遍历所有业务场景，避免只预热默认场景。
    registry = get_scenario_registry()

    # 第一步：加载 BGE，并绕过 CachedEmbeddings 的 query 缓存做一次真实向量化。
    # 加载 BGE 向量模型并绕过 query 缓存做一次真实推理，确保模型权重和推理设备都已热身。
    # 仅调用 CachedEmbeddings.embed_query() 可能命中 Redis，导致第一个真实用户查询仍触发冷推理。
    embeddings = get_embeddings()
    getattr(embeddings, "base_embeddings", embeddings).embed_query(sample_query)

    # 第二步：确认 Milvus 服务端点可达；端口不通时无需继续创建 collection 对象。
    if not milvus_endpoint_available(timeout=3.0):
        raise RuntimeError("Milvus 服务不可达：请先启动 Milvus 2.5+ 服务。")

    warmed_collections: list[str] = []
    active_versions: dict[str, str] = {}
    # 第三步：逐个场景确认 active 版本并触发 FAQ/文档 collection 的懒加载。
    for scenario in registry.list_scenarios():
        # active 版本是线上检索的数据边界；任意场景缺失都会让全局预热失败。
        active_versions[scenario.scenario_id] = resolve_active_kb_version(None, scenario.scenario_id)
        for collection_name in (scenario.faq_collection, scenario.doc_collection):
            # get_hybrid_store() 负责进程内复用 wrapper，.store 会触发底层 collection 初始化。
            _ = get_hybrid_store(collection_name).store
            warmed_collections.append(collection_name)

    # 第四步：加载 CrossEncoder，并对 query-passage 样例打分，确认精排模型可用。
    get_reranker().predict([(sample_query, "业务资料包含处理流程、常见问题和操作规范。")])

    # 第五步：形成可观测摘要，既用于日志，也用于健康检查和故障排查。
    summary = {
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scenario_count": len(active_versions),
        "collection_count": len(warmed_collections),
        "active_versions": active_versions,
    }
    logger.info(
        "检索栈预热完成：耗时 %.2fs，场景数=%s，集合数=%s，active_versions=%s",
        summary["elapsed_seconds"],
        summary["scenario_count"],
        summary["collection_count"],
        active_versions,
    )
    # 清除旧快照后写入当前 active 版本快照，供后续请求检测 active 版本切换。
    with _retrieval_cache_lock:
        _known_active_versions.clear()
        _known_active_versions.update(active_versions)
    return summary
