"""可选的 LangSmith 适配器：为 RAG 运行时附加企业级 Trace 元数据。

项目保持核心 RAG 质量循环在本地 reports、eval_sets 和 gate 脚本中。
此模块仅在 LangSmith Tracing 启用时被调用，用于向 LangSmith 发送领域元数据。

设计决策：
- Trace 是旁路能力：未启用或网络失败时只记录日志，不影响用户请求。
- setdefault 仅在环境变量未设置时写入，优先保留外部配置（如 Docker env_file）。
- 答案文本截断到 800 字符，避免 Trace 输出中写入过长内容。

调用顺序：pipeline 收尾 -> langsmith_adapter。
"""

from __future__ import annotations

import os
from typing import Any

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.scenarios.registry import ScenarioDefinition


logger = get_logger(__name__)


def langsmith_enabled() -> bool:
    """判断当前进程是否启用 LangSmith Trace。

    两个条件同时满足才启用：功能开关为 true 且 API Key 已配置，避免空 Key 导致
    LangSmith SDK 报错。

    返回：
        True 表示 Tracing 已启用。

    调用顺序：record_query_trace() / langsmith_status() -> langsmith_enabled()。
    """
    settings = get_settings()
    # 原因：两个条件同时满足才启用，避免空 Key 导致 LangSmith SDK 报错
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)


def configure_langsmith_environment() -> None:
    """将项目配置同步到 LangSmith/LangChain 读取的环境变量。

    setdefault 仅在环境变量未设置时写入，优先保留外部配置（如 Docker env_file
    注入的值）。这确保容器化部署时环境变量 > settings 配置。

    调用顺序：record_query_trace() -> configure_langsmith_environment()。
    """
    settings = get_settings()
    # setdefault 仅在环境变量未设置时写入，优先保留外部配置（如 Docker env_file 注入的值）
    os.environ.setdefault("LANGSMITH_TRACING", "true" if settings.langsmith_tracing else "false")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)


def langsmith_status() -> dict[str, Any]:
    """返回本地管理页展示用的 LangSmith 状态。

    返回：
        包含 provider、enabled、project、endpoint、has_api_key 和 message 的字典。

    调用顺序：管理 API -> langsmith_status()。
    """
    settings = get_settings()
    return {
        "provider": "langsmith",
        "enabled": langsmith_enabled(),
        "project": settings.langsmith_project,
        "endpoint": settings.langsmith_endpoint,
        "has_api_key": bool(settings.langsmith_api_key),
        "project_url": "https://smith.langchain.com/",
        "message": (
            "LangSmith tracing is enabled."
            if langsmith_enabled()
            else "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable enterprise tracing."
        ),
    }


def _safe_preview(text: str, max_chars: int = 800) -> str:
    """截断长文本，避免 Trace 输出中写入过长答案。

    默认截断到 800 字符：足够在 Trace 列表中预览答案质量，又不会让 Trace payload
    过于臃肿（特别是包含大量表格或长文档时）。

    参数：
        text: 原始文本。
        max_chars: 最大字符数，默认 800。

    返回：
        截断后的文本。

    调用顺序：record_query_trace() -> _safe_preview()。
    """
    return text[:max_chars]


def record_query_trace(
    *,
    trace_id: str,
    session_id: str,
    question: str,
    answer: str,
    hit_type: str,
    scenario: ScenarioDefinition,
    data_scope: dict[str, Any],
    source_filter: str | None,
    kb_version: str,
    rewritten_query: str | None,
    intent: dict[str, Any],
    retrieval: dict[str, Any],
    sources: list[dict[str, Any]],
    elapsed_ms: float,
    error: str | None = None,
) -> None:
    """将一次问答请求记录到 LangSmith。（★★★ 核心）

    执行流程：
      1. configure_langsmith_environment()：将配置同步到环境变量。
      2. langsmith_enabled() 检查：未启用时直接返回，不对业务请求产生副作用。
      3. 构造 metadata 字典（包含场景、数据域、意图、检索结果、耗时等完整诊断信息）。
      4. 构造 inputs/outputs 字典（answer 截断到 800 字符）。
      5. 调用 langsmith.run_helpers.trace() 创建 Trace run tree。
      6. 异常捕获：网络超时或 LangSmith 服务不可用时只打日志。

    Trace 是旁路能力：未启用或网络失败时只记录日志，不影响用户请求。这是核心设计
    决策——确保 Tracing 故障不会导致主链路问题。

    参数：
        trace_id: Trace 唯一 ID。
        session_id: 会话 ID。
        question: 用户原始问题。
        answer: 助手回答。
        hit_type: 命中类型（faq / doc / hybrid）。
        scenario: 当前业务场景定义。
        data_scope: 数据域字典。
        source_filter: 来源过滤条件。
        kb_version: 知识库版本号。
        rewritten_query: 改写后的查询。
        intent: 意图识别结果字典。
        retrieval: 检索结果字典（包含 plan、prompt_profile、answer_confidence 等）。
        sources: 答案来源列表。
        elapsed_ms: 总耗时（毫秒）。
        error: 错误描述（正常时为 None）。

    调用顺序：pipeline 收尾 -> record_query_trace()。
    """
    configure_langsmith_environment()
    if not langsmith_enabled():
        # Tracing 禁用时直接返回，不对业务请求产生任何副作用，确保 Tracing 故障不影响主链路
        return

    try:
        from langsmith.run_helpers import trace
    except Exception as exc:  # pragma: no cover - tracing dependency is optional
        logger.warning("LangSmith trace helper unavailable: %s", exc)
        return

    plan = retrieval.get("plan") or {}
    prompt_profile = retrieval.get("prompt_profile") or {}
    answer_confidence = retrieval.get("answer_confidence") or {}
    evidence_confidence = answer_confidence.get("evidence_confidence") or {}
    generation_verification = answer_confidence.get("generation_verification") or {}
    # 构建完整的 metadata 字典，包含所有业务维度的诊断信息
    metadata = {
        "trace_id": trace_id,
        "session_id": session_id,
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "source_filter": source_filter,
        "effective_source": source_filter,
        "kb_version": kb_version,
        "tenant_id": data_scope.get("tenant_id"),
        "dataset_id": data_scope.get("dataset_id"),
        "visibility": data_scope.get("visibility"),
        "user_role": data_scope.get("user_role"),
        "allowed_roles": data_scope.get("allowed_roles"),
        "intent": intent.get("intent"),
        "intent_reason": intent.get("reason"),
        "hit_type": hit_type,
        "prompt_profile": prompt_profile.get("name"),
        "question_category": plan.get("question_category"),
        "rewritten_query": rewritten_query,
        "sources_count": len(sources),
        "top_source_score": sources[0].get("score") if sources else None,
        "answer_confidence": answer_confidence,
        "answer_confidence_score": answer_confidence.get("score"),
        "answer_confidence_level": answer_confidence.get("level"),
        "answer_confidence_reasons": answer_confidence.get("reasons"),
        "evidence_confidence_score": evidence_confidence.get("score"),
        "evidence_confidence_level": evidence_confidence.get("level"),
        "generation_verification_status": generation_verification.get("status"),
        "generation_verification_score": generation_verification.get("score"),
        "generation_verification_reasons": generation_verification.get("reasons"),
        "first_token_ms": retrieval["first_token_ms"],
        "stage_timings_ms": retrieval["stage_timings_ms"],
        "slowest_stage": retrieval["slowest_stage"],
        "elapsed_ms": round(elapsed_ms, 2),
        "error": error,
    }
    inputs = {
        "question": question,
        "scenario_id": metadata["scenario_id"],
        "source_filter": source_filter,
        "kb_version": kb_version,
    }
    outputs = {
        "answer_preview": _safe_preview(answer),
        "hit_type": hit_type,
        "answer_confidence": answer_confidence,
        "sources": sources[:8],  # 限制来源数量，避免 Trace payload 过大
        "error": error,
    }

    try:
        # 原因：trace() 上下文管理器自动处理 run 的开始和结束
        with trace(
            "qa_stream_query",
            run_type="chain",
            inputs=inputs,
            metadata=metadata,
            project_name=get_settings().langsmith_project,
            run_id=trace_id,
            tags=[str(metadata["scenario_id"]), hit_type],
        ) as run_tree:
            if error:
                run_tree.end(outputs=outputs, error=error)
            else:
                run_tree.end(outputs=outputs)
    except Exception as exc:  # pragma: no cover - tracing must be best effort
        # Tracing 是辅助能力，网络超时或 LangSmith 服务不可用时只打日志，不影响用户答案返回
        logger.warning("LangSmith trace failed: %s", exc)
