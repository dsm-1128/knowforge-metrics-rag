"""应用层 RAG 编排服务。（★★★ 核心）

接收 API 层的 WebSocket 或检索调试请求，依次完成场景解析、数据域隔离、查询路由、
检索准备、FAQ 检索、置信度判断、文档检索、生成和保存，并将最终结果以 Generator 事件流
或检索诊断字典的形式返回给 API 层。

设计决策：
- QAService 不直接处理检索或生成逻辑，将所有参数透传给 pipeline 层。
  QAService 只做请求接入层的工作，pipeline 层专注业务编排。
- 使用 Generator 事件流：stream_query() 以 yield from 方式产出 status/token/end/error 事件，
  API 层 WebSocket 端点逐事件推送到浏览器，支持流式展示。
- debug_retrieval() 与 stream_query() 的区别：不调用 LLM 回答生成，只返回检索中间结果
  供开发者调试，适用于检索质量评估、知识库覆盖度检查等非交互式场景。
- QAService 实例由 factory.get_qa_service() 以进程级单例方式管理，构造函数不保存任何
  请求级状态，无状态设计支持多请求并发。
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from qa_core.config.logging_config import get_logger
from qa_core.memory.history import get_history_store
from qa_core.pipeline.rag import debug_retrieval as rag_debug_retrieval
from qa_core.pipeline.rag import stream_query as rag_stream_query

logger = get_logger(__name__)

class QAService:
    """核心 RAG 问答工作流，统一通过流式主链路编排。（★★★ 核心）

    作为 API 层与 pipeline 层之间的桥梁，负责：
    - 参数透传：将 API 层传入的查询参数完整传递给 pipeline 层。
    - 事件流管理：stream_query() 以 Generator 事件流形式逐块产出问答结果。
    - 检索调试：debug_retrieval() 不调用 LLM 回答生成，只返回检索中间结果。

    典型调用顺序：
      1. API 层通过 WebSocket 接收问题并构造 QueryServiceContext。
      2. stream_query() 委托 pipeline 执行上下文创建、查询路由、检索准备、
         FAQ/文档检索、LLM 生成、历史保存和 Trace。
      3. token 与状态事件逐块通过 Generator[dict] 推送给 API 层 WebSocket 转发。
      4. debug_retrieval() 先执行查询路由；检索类问题复用检索准备和召回逻辑，但不调用 LLM。
    """

    def __init__(self) -> None:
        """初始化共享服务层依赖。

        执行流程：
          1. get_history_store() 获取 MySQL 历史记录适配器单例。

        说明：构造函数不保存任何请求级状态，QAService 实例由
        factory.get_qa_service() 以进程级单例方式管理。

        调用顺序：factory.get_qa_service() -> QAService.__init__()。
        """
        self.history = get_history_store()

    def stream_query(
        self,
        query: str,
        source_filter: str | None,
        session_id: str | None,
        kb_version: str | None = None,
        scenario_id: str | None = None,
        tenant_id: str | None = None,
        dataset_id: str | None = None,
        visibility: str | None = None,
        user_role: str | None = None,
        user_roles: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """完整流式问答入口。（★★★ 核心）

        委托 RAGPipeline 执行查询路由 -> 检索准备 -> 检索 -> 生成全链路，
        以事件生成器（status / token / end / error）形式逐块产出，由 API 层 WebSocket 转发。

        执行流程（由 rag_stream_query 内部完成）：
          1. decide_route()             — 统一处理直答、FAQ 精确路由或继续检索。
          2. prepare_retrieval()        — 历史、意图、source、按需改写、检索计划、查询变体和 Prompt Profile。
          3. search_faq()               — 从 Milvus FAQ 集合检索并判断是否标准直出。
          4. search_doc()               — 低置信度时执行文档 RAG 检索。
          5. stream_llm_answer()        — 调用大模型生成回答，token 逐块 yield。
          6. history_add_turn()         — 将本轮问答写入 MySQL 历史表。
          7. record_trace()             — 将完整 trace 写入持久化存储。

        参数：
            query: 用户输入的问题文本。
            source_filter: 前端选择的业务分类过滤项（可为 None）。
            session_id: 会话 ID，未传则自动生成 UUID。
            kb_version: 知识库版本号（可选）。
            scenario_id: 业务场景标识（可选）。
            tenant_id: 租户 ID（可选）。
            dataset_id: 数据集 ID（可选）。
            visibility: 可见级别（可选）。
            user_role: 用户角色（可选）。
            user_roles: 用户角色列表（可选）。

        Yields:
            dict[str, Any]: 流式事件，包含 type 字段，可能取值：
                - "status": 阶段状态变更事件。
                - "token": LLM 生成的文本片段。
                - "end": 问答结束事件（包含最终回答和来源引用）。
                - "error": 错误事件（包含错误描述）。

        调用顺序：QAService.stream_query() -> pipeine.rag.stream_query()。
        """
        # 委托至 pipeline 层 rag_stream_query 执行完整 RAG 全链路，逐事件 yield
        # QAService 不直接处理检索或生成逻辑，将所有参数透传给 pipeline 层，
        # 这样 QAService 只做请求接入层的工作，pipeline 层专注业务编排
        yield from rag_stream_query(
            self.history,
            query,
            source_filter,
            session_id,
            kb_version=kb_version,
            scenario_id=scenario_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            visibility=visibility,
            user_role=user_role,
            user_roles=user_roles,
        )

    def debug_retrieval(
        self,
        query: str,
        source_filter: str | None,
        session_id: str | None = None,
        kb_version: str | None = None,
        scenario_id: str | None = None,
        tenant_id: str | None = None,
        dataset_id: str | None = None,
        visibility: str | None = None,
        user_role: str | None = None,
        user_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        """路由 + 检索调试入口。（★★ 理解）

        委托 RAGPipeline 先判断查询路由；检索类问题继续执行检索准备 + FAQ 检索 + 文档检索，
        但不调用最终回答 LLM，用于开发者诊断检索质量。

        与 stream_query() 的区别：
        - 不调用 LLM 回答生成，只返回检索中间结果供开发者调试。
        - 适用于检索质量评估、知识库覆盖度检查等非交互式场景。

        执行流程（由 rag_debug_retrieval 内部完成）：
          1. decide_route()             — 统一处理直答、FAQ 精确路由或继续检索。
          2. prepare_retrieval()        — 检索类问题生成意图、source、改写、检索计划和查询变体。
          3. search_faq()               — 从 Milvus FAQ 集合检索。
          4. search_doc()               — 从 Milvus 文档集合检索。
          5. 组装调试信息并返回（含路由、意图、改写、检索计划、FAQ/文档命中）。
          6. record_trace()             — 记录调试 trace（如启用）。

        参数：
            query: 用户输入的问题文本。
            source_filter: 前端选择的业务分类过滤项（可为 None）。
            session_id: 会话 ID（可选）。
            kb_version: 知识库版本号（可选）。
            scenario_id: 业务场景标识（可选）。
            tenant_id: 租户 ID（可选）。
            dataset_id: 数据集 ID（可选）。
            visibility: 可见级别（可选）。
            user_role: 用户角色（可选）。
            user_roles: 用户角色列表（可选）。

        返回：
            dict[str, Any]: 检索诊断结果，包含路由、意图分类、改写结果、FAQ 命中列表、
            文档命中列表、检索计划等调试信息。

        调用顺序：QAService.debug_retrieval() -> pipeline.rag.debug_retrieval()。
        """
        # 委托至 pipeline 层 rag_debug_retrieval 执行路由 + 检索诊断链路（不含 LLM 生成）
        # 与 stream_query 的区别：不调用 LLM 回答生成，只返回检索中间结果供开发者调试
        # 适用于检索质量评估、知识库覆盖度检查等非交互式场景
        return rag_debug_retrieval(
            self.history,
            query,
            source_filter,
            session_id=session_id,
            kb_version=kb_version,
            scenario_id=scenario_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            visibility=visibility,
            user_role=user_role,
            user_roles=user_roles,
        )


