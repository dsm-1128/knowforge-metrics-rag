"""API 层传给 QAService 的公共调用上下文。

WebSocket stream 和 retrieval debug 都需要同一组字段：场景、租户、数据集、
可见性、角色和知识库版本。集中到这个小对象后，路由层只负责解析请求，业务规则仍在
QAService 和 pipeline 中。

设计决策：
- 使用 frozen dataclass 确保上下文在构造后不可变，避免请求处理过程中意外修改。
- from_debug_request / from_ws_payload 两个工厂方法适配不同的输入源（HTTP body / WebSocket），
  统一输出 QueryServiceContext，下游不需要关心数据来源。
- service_args() 方法按 QAService.stream_query() 的顺序参数解包，保持与 pipeline 接口契约。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qa_core.schemas import RetrievalDebugRequest


@dataclass(frozen=True)
class QueryServiceContext:
    """一次 API 请求调用 QAService 所需的稳定参数包。（★★★ 核心）

    作为路由层与业务层之间的参数传输对象，包含以下维度的隔离信息：
    - 查询维度：query（问题文本）、source_filter（业务分类过滤）、session_id（会话标识）
    - 知识库维度：kb_version（版本号）、scenario_id（业务场景）
    - 数据域维度：tenant_id（租户）、dataset_id（数据集）
    - 权限维度：visibility（可见级别）、user_role / user_roles（用户角色）

    调用顺序：FastAPI 路由层 -> QueryServiceContext -> QAService.stream_query()/debug_retrieval()。
    """

    query: str
    source_filter: str | None
    session_id: str
    kb_version: str | None
    scenario_id: str | None
    tenant_id: str | None
    dataset_id: str | None
    visibility: str | None
    user_role: str | None
    user_roles: list[str] | None

    @classmethod
    def from_debug_request(
        cls,
        request: RetrievalDebugRequest,
        *,
        session_id: str | None = None,
    ) -> "QueryServiceContext":
        """从 HTTP 检索诊断请求模型构造 QAService 调用上下文。

        参数：
            request: RetrievalDebugRequest，包含检索诊断所需的全部查询参数。
            session_id: 可选的会话 ID 覆盖值，未传时使用 request.session_id 或自动生成。

        返回：
            QueryServiceContext 实例，包含清洗后的查询参数。

        调用顺序：FastAPI 路由 /api/retrieval/debug -> QueryServiceContext.from_debug_request()。
        """
        return cls(
            query=request.query.strip(),
            source_filter=request.source_filter,
            session_id=session_id or request.session_id or str(uuid.uuid4()),
            kb_version=request.kb_version,
            scenario_id=request.scenario_id,
            tenant_id=request.tenant_id,
            dataset_id=request.dataset_id,
            visibility=request.visibility,
            user_role=request.user_role,
            user_roles=request.user_roles,
        )

    @classmethod
    def from_ws_payload(cls, payload: dict[str, Any]) -> "QueryServiceContext":
        """从 WebSocket JSON 载荷构造 QAService 调用上下文。

        参数：
            payload: WebSocket 接收到的 JSON 字典，包含 query/session_id/scenario_id 等字段。

        返回：
            QueryServiceContext 实例，字段缺失时使用默认值（session_id 未传时自动生成 UUID）。

        调用顺序：WebSocket /api/stream -> QueryServiceContext.from_ws_payload()。
        """
        return cls(
            query=str(payload.get("query") or "").strip(),
            source_filter=payload.get("source_filter"),
            session_id=payload.get("session_id") or str(uuid.uuid4()),
            kb_version=payload.get("kb_version"),
            scenario_id=payload.get("scenario_id"),
            tenant_id=payload.get("tenant_id"),
            dataset_id=payload.get("dataset_id"),
            visibility=payload.get("visibility"),
            user_role=payload.get("user_role"),
            user_roles=payload.get("user_roles") or [],
        )

    def service_args(self) -> tuple[Any, ...]:
        """返回 QAService 方法当前使用的顺序参数。

        与 QAService.stream_query() 和 debug_retrieval() 的参数列表顺序保持一致，
        可直接解包传递给这些方法。如字段变化需同步更新此处。

        返回：
            按 QAService 方法签名的顺序元组。

        调用顺序：QueryServiceContext -> QAService.stream_query()/debug_retrieval()。
        """
        return (
            self.query,
            self.source_filter,
            self.session_id,
            self.kb_version,
            self.scenario_id,
            self.tenant_id,
            self.dataset_id,
            self.visibility,
            self.user_role,
            self.user_roles,
        )
