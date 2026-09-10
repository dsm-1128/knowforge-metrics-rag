"""FastAPI 对外请求/响应模型。

定义所有 HTTP API 的请求体和响应体的 Pydantic 模型，包含：
- V1 接口：RetrievalDebugRequest/Response、FeedbackRequest。

设计原则：
- 使用 Pydantic Field 的 pattern 参数做字符串格式校验（如 rating）。
- 可选字段使用 Optional/None 默认值，不影响接口兼容性。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalDebugRequest(BaseModel):
    """HTTP 检索诊断接口的请求体。在线问答不复用该模型。

    包含检索诊断所需的全部查询参数：问题文本、场景上下文、数据域隔离参数和知识库版本。

    参数：
        query: 用户问题文本，至少 1 个字符。
        source_filter: 前端选择的业务分类过滤项（可为 None）。
        session_id: 会话 ID（可选，自动生成）。
        scenario_id: 业务场景标识（可选）。
        tenant_id: 租户 ID（可选，默认 "default"）。
        dataset_id: 数据集 ID（可选，默认 "default"）。
        visibility: 可见级别（可选，默认 "public"）。
        user_role: 用户角色（可选，单值）。
        user_roles: 用户角色列表（可选，多值）。
        kb_version: 知识库版本号（可选）。

    调用顺序：上游业务入口 -> RetrievalDebugRequest。
    """

    query: str = Field(..., min_length=1)
    source_filter: str | None = None
    session_id: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    visibility: str | None = None
    user_role: str | None = None
    user_roles: list[str] = Field(default_factory=list)
    kb_version: str | None = None


class FeedbackRequest(BaseModel):
    """用户反馈载荷，rating 约束为 useful/not_useful。

    参数：
        session_id: 会话 ID（可选）。
        scenario_id: 业务场景标识（可选）。
        tenant_id: 租户 ID（可选），用于多租户数据隔离。
        dataset_id: 数据集 ID（可选），用于数据集级别数据隔离。
        question: 用户问题（必填，至少 1 个字符）。
        answer: 系统回答（必填，至少 1 个字符）。
        rating: 评分，"useful" 或 "not_useful"（必填，正则校验）。
        comment: 用户评论文本（可选）。
        sources: 反馈涉及的来源引用列表（可选，默认为空列表）。

    调用顺序：上游业务入口 -> FeedbackRequest。
    """

    session_id: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(useful|not_useful)$")
    comment: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalDebugResponse(BaseModel):
    """检索调试响应，不包含最终答案，faq 和 doc 来源分开返回。

    参数：
        query: 清洗后的查询文本。
        raw_query: 原始查询文本（未归一化前，可选）。
        effective_query: 实际生效的查询文本（可选）。
        query_normalized: 查询是否经过归一化处理。
        rewritten_query: 改写后的查询文本。
        source_filter: 前端选择的业务分类过滤项（可选）。
        scenario_id: 业务场景标识（可选）。
        tenant_id: 租户 ID（可选）。
        dataset_id: 数据集 ID（可选）。
        visibility: 可见级别（可选）。
        data_scope: 数据域隔离元数据字典（可选，包含 tenant/dataset/visibility/roles）。
        kb_version: 知识库版本号（可选）。
        route: 检索路由决策结果（可选，如 "faq_first" / "knowledge_enriched"）。
        route_reason: 路由决策原因（可选）。
        answer: 最终回答文本（可选，debug 模式下通常为空）。
        answer_confidence: 答案置信度字典（默认为空字典）。
        intent: 意图分类结果字典（必填，含 intent/confidence/reason/rule_score 等）。
        retrieval_plan: 检索计划字典（可选，含 top_k/threshold/use_variants 等）。
        hit_type: 命中类型（可选，如 "faq_direct" / "doc_retrieval" / "mixed"）。
        context_count: 最终进入 LLM 的上下文段落数（可选）。
        faq_sources: FAQ 命中列表（默认为空列表）。
        doc_sources: 文档命中列表（默认为空列表）。

    调用顺序：上游业务入口 -> RetrievalDebugResponse。
    """

    query: str
    raw_query: str | None = None
    effective_query: str | None = None
    query_normalized: bool = False
    rewritten_query: str
    source_filter: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    visibility: str | None = None
    data_scope: dict[str, Any] | None = None
    kb_version: str | None = None
    route: str | None = None
    route_reason: str | None = None
    answer: str | None = None
    answer_confidence: dict[str, Any] = Field(default_factory=dict)
    intent: dict[str, Any]
    retrieval_plan: dict[str, Any] | None = None
    hit_type: str | None = None
    context_count: int | None = None
    faq_sources: list[dict[str, Any]] = Field(default_factory=list)
    doc_sources: list[dict[str, Any]] = Field(default_factory=list)
