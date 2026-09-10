"""LLM 客户端工厂：通过 OpenAI 兼容接口创建 ChatOpenAI 实例。

切换模型时只改配置，不改业务代码。当前项目优先使用 LangChain 的 ChatOpenAI
兼容层接入 DashScope 等模型服务，避免在主链路里绑定某个厂商 SDK。

示例：
    get_chat_model(streaming=False) -> 非流式 ChatOpenAI 客户端
    get_chat_model(streaming=True)  -> 流式 ChatOpenAI 客户端
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage

from qa_core.common import utc_now
from qa_core.config.settings import get_settings


_LLM_RUNTIME_STATUS: dict[str, Any] = {
    "status": "unknown",
    "ok": None,
    "checked_at": "",
    "error": "",
}

@lru_cache(maxsize=2)  # 缓存流式/非流式两个客户端实例，避免每次重建 TCP 连接
def get_chat_model(streaming: bool = False) -> Any:
    """返回带缓存的 ChatOpenAI 客户端。

    `streaming` 是缓存 key 的一部分，所以流式和非流式客户端会分开缓存。
    `maxsize=2` 正好覆盖这两种模式，避免重复创建连接，也避免缓存无限增长。

    参数：
        streaming: 是否返回支持流式 token 输出的客户端。

    返回：
        已按全局配置初始化好的 ChatOpenAI 实例。

    调用顺序：LLM 调用阶段 -> get_chat_model()。
    """
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 langchain_openai，请安装 requirements.txt 后重试。") from exc

    # 加载全局设置
    settings = get_settings()
    # 构造 ChatOpenAI 客户端
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        streaming=streaming,
    )

def validate_llm_connectivity() -> None:
    """启动时真实调用一次模型，验证 LLM 服务可用。

    执行流程：
      1. 通过 get_chat_model() 获取非流式 ChatOpenAI 客户端。
      2. 发送最小化的“回复 OK”测试消息。
      3. 检查模型返回内容是否非空。
      4. 如果调用失败或返回空内容，抛出 RuntimeError，让启动阶段暴露配置问题。

    异常：
        RuntimeError: LLM 服务不可达、配置错误或返回空内容。
    """
    try:
        # 启动时发送最小测试请求验证 LLM 服务可用，防止上线后才发现配置错误
        response = get_chat_model(streaming=False).invoke([HumanMessage(content="回复 OK")])
    except Exception as exc:
        raise RuntimeError("LLM 服务不可用：请检查 DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL 和 LLM_MODEL。") from exc
    # HTTP 调用成功但模型返回空内容，也不能视为生成服务可用。
    if not str(getattr(response, "content", "") or "").strip():
        raise RuntimeError("LLM 服务返回空内容：请检查模型服务状态。")


def llm_runtime_status() -> dict[str, Any]:
    """返回最近一次 LLM 连通性探测状态，用于健康检查和管理后台展示。

    调用顺序：FastAPI 状态接口 -> llm_runtime_status()。
    """
    settings = get_settings()
    return {
        **_LLM_RUNTIME_STATUS,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "timeout": settings.llm_timeout,
        "has_api_key": bool(settings.llm_api_key),
    }


def refresh_llm_runtime_status() -> dict[str, Any]:
    """主动探测 LLM 连通性并缓存结果，但不抛出异常阻断应用启动。

    LLM 供应商的临时网络抖动、额度不足或服务异常属于运行状态问题，
    这里将异常转换成统一状态，避免后台探测任务反过来终止整个 API 进程。

    调用顺序：FastAPI lifespan或管理接口 -> refresh_llm_runtime_status()。
    """
    try:
        # 先发起一次最小真实请求，成功后才把状态标记为 available。
        validate_llm_connectivity()
        _LLM_RUNTIME_STATUS.update(
            {
                "status": "available",
                "ok": True,
                "checked_at": utc_now(),
                "error": "",
            }
        )
    except RuntimeError as exc:
        # 保留包装异常和底层原因，方便通过健康检查或管理接口定位故障来源。
        cause = exc.__cause__
        cause_text = f" | {cause.__class__.__name__}: {cause}" if cause else ""
        _LLM_RUNTIME_STATUS.update(
            {
                "status": "unavailable",
                "ok": False,
                "checked_at": utc_now(),
                "error": f"{exc}{cause_text}",
            }
        )
    # 成功和失败都返回相同结构，调用方无需再围绕异常写分支逻辑。
    return llm_runtime_status()

