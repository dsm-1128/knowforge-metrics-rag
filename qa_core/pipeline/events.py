"""WebSocket 流式问答事件构造器：start / status / token / end / error 五种事件格式集中管理。

设计决策：
- 所有事件统一为 dict 格式，通过 type 字段做路由（"start"/"status"/"token"/"end"/"error"），
  前端根据 type 驱动 UI 状态机（加载态 → 阶段提示 → 逐字展示 → 结果展示 → 错误提示）。
- 异常错误文本不透露内部细节：供应商 SDK 的错误文本可能包含账户状态、请求参数和内部标识，
  直接通过 WebSocket 返回会向用户暴露敏感信息。完整异常由 rag.py 写入日志和 LangSmith Trace。
- 所有事件均携带 session_id：WebSocket 是长连接多路复用，前端可能需要根据 session_id
  将事件路由到对应对话气泡，避免多个问题并行时 UI 错乱。

依赖分层：
- 被 pipeline.runtime.finish_success() / finish_error() / start_event() 调用。
- 被 pipeline.rag.stream_query() 直接 yield 给 WebSocket 传输层。
"""

from __future__ import annotations
import time
from typing import Any


def user_facing_error_message(error: BaseException | str) -> str:
    """将内部异常转换为不暴露敏感信息的用户友好提示。（★★ 理解）

    供应商 SDK 的错误文本可能包含账户状态、请求参数和内部标识，直接通过
    WebSocket 返回会向用户暴露内部实现细节。这里按异常文本特征归类映射为
    3 类可展示提示（账户不可用 / 超时 / 连接失败），兜底返回通用失败文案。

    参数：
        error: 异常对象或异常文本。

    返回：
        str: 向用户展示的友好错误提示。

    调用顺序：QAService/RAG 管线异常分支 -> finish_error() -> user_facing_error_message()。
    """
    # ── 步骤 1：将异常统一转为小写文本做关键词匹配 ──
    # SDK 异常格式不统一，有的用大写（"TIMEOUT"）、有的用小写（"timeout"），统一小写减少匹配分支
    text = str(error).lower()
    # ── 步骤 2：按业务语义归类 ──
    # 账户相关错误（欠费/鉴权失败）提示联系管理员，不暴露 API Key 或账户 ID
    if any(
        marker in text
        for marker in (
            "arrearage",
            "overdue-payment",
            "access denied",
            "unauthorized",
            "authentication",
            "api key",
            "invalid_api_key",
            "invalid api key",
            "401",
            "403",
        )
    ):
        return "当前生成服务暂不可用，请检查模型服务账户或 API Key，并联系管理员。"
    # 超时类错误：供应商服务响应慢或客户端连接超时，属于可恢复问题，提示重试
    if any(marker in text for marker in ("timeout", "timed out", "time out")):
        return "模型服务响应超时，请稍后重试。"
    # 连接类错误：网络不通或服务进程宕机，属于可恢复问题，提示稍后重试
    if any(marker in text for marker in ("connection", "connecterror", "connection refused")):
        return "模型服务暂时无法连接，请稍后重试。"
    # ── 兜底：无法归类的异常使用通用文案，避免暴露未预见的 SDK 内部细节 ──
    return "抱歉，处理失败，请稍后重试。"

def start_event(
    *,
    session_id: str,
    trace_id: str,
    scenario_id: str,
    scenario_name: str,
    data_scope: dict[str, Any],
    kb_version: str | None,
) -> dict[str, Any]:
    """通知前端 WebSocket 连接已建立、请求已接收，前端据此展示加载状态。

    调用顺序：QAService/RAG 管线 -> start_event()。
    """
    return {
        "type": "start",
        "session_id": session_id,
        "trace_id": trace_id,
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "data_scope": data_scope,
        "kb_version": kb_version,
    }


def status_event(message: str, session_id: str) -> dict[str, Any]:
    """构造阶段状态事件：让前端在当前阶段名称旁展示进度，缓解用户等待焦虑。（★★ 理解）

    用户等待 LLM 生成时，如果没有阶段提示，可能在 5 秒后认为是系统卡死或崩溃。
    通过每 1-2 秒发送一次状态消息（"正在识别问题意图…" / "正在检索 FAQ…"），
    前端可以展示阶段名称和旋转加载动画，显著降低用户感知等待时间。

    参数：
        message: 阶段状态描述文本（如"正在检索相关业务资料…"）。
        session_id: 会话 ID（用于前端多会话路由）。

    返回：
        WebSocket status 事件 dict。

    调用顺序：QAService/RAG 管线各阶段 -> status_event()。
    """
    return {"type": "status", "message": message, "session_id": session_id}


def token_event(token: str, session_id: str) -> dict[str, Any]:
    """构造流式 token 事件：逐字推送生成结果，让用户逐步看到内容而非等待完整响应。（★★★ 核心）

    执行流程：stream_llm_answer() 每次 yield 一个 AIMessageChunk，其中 content 为
    本次增量 token。rag.py 遍历此 stream，对每个非空 content 调用 token_event()
    构造事件 dict，yield 给前端。

    为什么逐字推送而非一次性推送完整答案：实验数据显示，逐字展示首 token 延迟约 0.2-0.5s，
    用户感知的"等待时间"远短于完整答案全部到达后再展示的 3-8s。

    参数：
        token: LLM 流式输出的增量文本片段（可能为单个字符或多个词）。
        session_id: 会话 ID。

    返回：
        WebSocket token 事件 dict。

    调用顺序：QAService/RAG 管线 Stage 6 -> stream_llm_answer() -> token_event()。
    """
    return {"type": "token", "token": token, "session_id": session_id}


def end_event(
    *,
    session_id: str,
    hit_type: str,
    sources: list[dict[str, Any]],
    started: float,
    rewritten_query: str | None,
    trace_id: str,
    intent: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    """构造请求结束事件：通知前端关闭加载状态，并附带诊断数据供 Trace 和性能面板使用。（★★★ 核心）

    end_event 承载了本轮请求的完整诊断信息：命中类型、来源列表、答案置信度、各阶段耗时
    和首 token 延迟。前端收到后停止旋转加载并将 sources 展示为答案脚注。

    执行流程：
      1. 组装必要字段。
      2. computing processing_time（time.perf_counter() 归零）。

    参数：
        session_id: 会话 ID。
        hit_type: 命中类型（rag / faq_direct / insufficient_context 等）。
        sources: 引用来源列表（每项含 score、source_type、content 等）。
        started: time.perf_counter() 请求开始时间戳。
        rewritten_query: 改写后的查询文本（无改写时为 None）。
        trace_id: LangSmith trace ID。
        intent: 意图识别结果快照。
        retrieval: 检索诊断信息（含 stage_timings_ms、first_token_ms 等）。

    返回：
        WebSocket end 事件 dict。

    调用顺序：QAService/RAG 管线 Stage 7 -> finish_success() -> end_event()。
    """
    return {
        "type": "end",
        "session_id": session_id,
        "trace_id": trace_id,
        "is_complete": True,
        "hit_type": hit_type,
        "sources": sources,
        "answer_confidence": retrieval["answer_confidence"],
        "rewritten_query": rewritten_query,
        "intent": intent,
        "retrieval": retrieval,
        "stage_timings_ms": retrieval["stage_timings_ms"],
        "first_token_ms": retrieval["first_token_ms"],
        "slowest_stage": retrieval["slowest_stage"],
        # 从 started 到现在的 wall-clock 耗时，与各阶段 stage_timings_ms 之和接近但不严格相等
        # 原因：perf_counter 的粒度高于 stage_timings_ms 的累加（stage 间有微小 gap），
        # 且 mark_first_token() 等操作无 stage 包裹
        "processing_time": time.perf_counter() - started,
    }


def error_event(*, error: str, session_id: str, trace_id: str) -> dict[str, Any]:
    """构造异常结束事件：以事件形式将错误推送给前端而非断开 WebSocket，保持对话状态。（★★ 理解）

    设计原因：WebSocket 断开后前端需要用 HTTP 重新建立连接，不仅浪费一次往返，更会
    丢失当前对话上下文（如正在输入的草稿、对话历史状态等）。通过 error 事件保持
    连接存活，前端可在原对话气泡中展示红色错误提示和重试入口，用户点击重试按钮
    时通过已有 session_id 重新发起请求。

    参数：
        error: 已经过 user_facing_error_message() 脱敏的用户友好错误提示。
        session_id: 会话 ID。
        trace_id: LangSmith trace ID。

    返回：
        WebSocket error 事件 dict。

    调用顺序：QAService/RAG 管线异常分支 -> finish_error() -> error_event()。
    """
    return {"type": "error", "error": error, "session_id": session_id, "trace_id": trace_id}
