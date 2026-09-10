"""聊天历史持久化和上下文压缩工具。

MySQL 负责保存完整会话，便于持久化、审计和回放；模型侧只接收“历史摘要 + 最近消息”，
避免把全部历史都塞进 Prompt 导致上下文膨胀。
"""

from __future__ import annotations
from functools import lru_cache
from typing import Any, Iterable
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import text
from qa_core.config.logging_config import get_logger
from qa_core.llm.client import get_chat_model
from qa_core.memory.base import _MySqlStore
from qa_core.prompts.constants import HISTORY_SUMMARY_SYSTEM_PROMPT
logger = get_logger(__name__)

SQLChatMessageHistory: Any = None


def _sql_chat_message_history_class() -> Any:
    """按需返回 LangChain SQLChatMessageHistory 类。

    调用顺序：ChatHistoryStore.for_session() -> _sql_chat_message_history_class()。
    """
    global SQLChatMessageHistory
    if SQLChatMessageHistory is not None:
        return SQLChatMessageHistory
    try:
        from langchain_community.chat_message_histories import SQLChatMessageHistory as history_cls
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 langchain_community，请安装 requirements.txt 后重试。") from exc
    SQLChatMessageHistory = history_cls
    return history_cls


class ChatHistoryStore(_MySqlStore):
    """基于 LangChain SQLChatMessageHistory 的聊天历史存储。

    这个类只负责消息读写、清空和摘要维护，不负责 FAQ 检索、文档检索或回答生成。
    这样历史链路和 RAG 检索链路保持解耦。

    调用顺序：问答历史或反馈存储 -> ChatHistoryStore。
    """

    def __init__(self, table_name: str = "chat_messages") -> None:
        """初始化聊天历史存储。

        SQLAlchemy 引擎由父类延迟创建，构造对象时不会立即连接 MySQL。

        参数：
            table_name: 保存聊天消息的 MySQL 表名。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.__init__()。
        """
        super().__init__()
        self.table_name = table_name

    def for_session(self, session_id: str) -> Any:
        """为指定会话创建 LangChain SQLChatMessageHistory 适配器。

        参数：
            session_id: 会话唯一标识。

        返回：
            绑定当前 MySQL 表和 session_id 的 SQLChatMessageHistory。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.for_session()。
        """
        history_cls = _sql_chat_message_history_class()
        # 构造 LangChain SQL 消息历史适配器
        return history_cls(
            session_id=session_id,
            connection=self.engine,
            table_name=self.table_name,
            session_id_field_name="session_id",
        )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[BaseMessage]:
        """读取某个会话的历史消息，可限制只取最近 N 条。

        参数：
            session_id: 会话唯一标识。
            limit: 最大返回消息数；None 表示返回全部。

        返回：
            按时间顺序排列的 LangChain BaseMessage 列表。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.get_messages()。
        """
        history = self.for_session(session_id)
        messages = list(history.messages)
        if limit and len(messages) > limit:
            return messages[-limit:]
        return messages

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """保存一轮完整对话，即用户问题 + 助手回答。

        参数：
            session_id: 会话唯一标识。
            question: 用户问题。
            answer: 助手回答。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.add_turn()。
        """
        history = self.for_session(session_id)
        history.add_messages([HumanMessage(content=question), AIMessage(content=answer)])

    def clear(self, session_id: str) -> None:
        """清空某个会话的全部消息和派生摘要。

        参数：
            session_id: 会话唯一标识。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.clear()。
        """
        self.for_session(session_id).clear()
        self.delete_summary(session_id)

    def get_context_messages(self, session_id: str) -> list[BaseMessage]:
        """构建传给 RAG 链路的压缩上下文：历史摘要 + 最近对话。

        执行流程：
          1. 读取最近 N 条消息，N 来自 history_recent_messages 配置。
          2. 如果启用摘要，读取该会话的历史摘要。
          3. 摘要存在时，把它作为 SystemMessage 放在最近消息前面。
          4. 返回摘要消息和最近 Human/AI 消息组成的列表。

        参数：
            session_id: 会话唯一标识。

        返回：
            BaseMessage 列表，可能以一条历史摘要 SystemMessage 开头。

        调用顺序：RAG pipeline -> ChatHistoryStore.get_context_messages()。
        """
        recent = self.get_messages(session_id, limit=self.settings.history_recent_messages)
        summary = self.get_summary(session_id)
        # 有摘要时压缩旧对话为一条 SystemMessage，保留最近消息原文，平衡上下文长度与信息完整性
        if summary:
            # 将摘要包装为 SystemMessage 作为背景事实，让 LLM 了解历史背景
            return [SystemMessage(content=f"历史摘要：{summary}")] + recent
        return recent

    def as_pairs(self, session_id: str, limit: int | None = None) -> list[dict[str, str]]:
        """把原始消息转换成前端展示需要的问答对。

        规则是把 HumanMessage 和后面的 AIMessage 配成一组；末尾没有配对的消息会被忽略。

        参数：
            session_id: 会话唯一标识。
            limit: 最大读取消息数；None 表示读取全部。

        返回：
            形如 {"question": str, "answer": str} 的列表。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.as_pairs()。
        """
        messages = self.get_messages(session_id, limit=limit)
        pairs: list[dict[str, str]] = []
        pending_question: str | None = None
        for message in messages:
            if isinstance(message, HumanMessage):
                pending_question = str(message.content)
            elif isinstance(message, AIMessage) and pending_question is not None:
                pairs.append({"question": pending_question, "answer": str(message.content)})
                pending_question = None
        return pairs

    def get_summary(self, session_id: str) -> str:
        """读取某个会话的历史摘要。

        参数：
            session_id: 会话唯一标识。

        返回：
            摘要文本；未启用摘要或摘要不存在时返回空字符串。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.get_summary()。
        """
        if not self.settings.history_summary_enabled:
            return ""
        with self.engine.begin() as conn:
            row = conn.execute(
                text(f"SELECT summary FROM {self.settings.chat_summary_table_name} WHERE session_id=:session_id"),
                {"session_id": session_id},
            ).fetchone()
        return str(row[0]) if row and row[0] else ""

    def refresh_summary_if_needed(self, session_id: str) -> None:
        """当历史消息足够多时，通过 LLM 增量刷新会话摘要。

        执行流程：
          1. 未启用摘要时直接返回。
          2. 读取该会话全部消息；数量不足阈值时不生成摘要。
          3. 只抽取“最近消息窗口”之前的旧消息，最近消息保留原文。
          4. 如果旧消息窗口为空，直接返回。
          5. 读取已有摘要；没有则使用“无”。
          6. 把旧消息格式化成中文对话文本。
          7. 用非流式 LLM 根据“已有摘要 + 新增历史”生成更新摘要。
          8. 按 history_summary_max_chars 截断摘要。
          9. 摘要非空时保存到 MySQL。

        参数：
            session_id: 会话唯一标识。
        """
        if not self.settings.history_summary_enabled:
            return
        messages = self.get_messages(session_id)
        if len(messages) < self.settings.history_summary_after_messages:
            return

        older_messages = messages[: -self.settings.history_recent_messages]
        if not older_messages:
            return
        current_summary = self.get_summary(session_id) or "无"
        history_text = format_messages(older_messages)
        # 获取非流式 LLM 客户端
        llm = get_chat_model(streaming=False)
        # 调用 LLM 生成增量摘要
        response = llm.invoke(
            [
                SystemMessage(content=HISTORY_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"已有摘要：\n{current_summary}\n\n"
                        f"新增历史：\n{history_text}\n\n"
                        f"请输出不超过 {self.settings.history_summary_max_chars} 字的更新摘要。"
                    )
                ),
            ]
        )
        summary = str(response.content).strip()[: self.settings.history_summary_max_chars]
        if summary:
            self.save_summary(session_id, summary)

    def save_summary(self, session_id: str, summary: str) -> None:
        """通过 MySQL upsert 插入或更新会话摘要。

        参数：
            session_id: 会话唯一标识。
            summary: 要保存的摘要文本。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.save_summary()。
        """
        sql = (
            f"INSERT INTO {self.settings.chat_summary_table_name} (session_id, summary, updated_at) "
            "VALUES (:session_id, :summary, CURRENT_TIMESTAMP) "
            "ON DUPLICATE KEY UPDATE summary=VALUES(summary), updated_at=CURRENT_TIMESTAMP"
        )
        with self.engine.begin() as conn:
            conn.execute(text(sql), {"session_id": session_id, "summary": summary})

    def delete_summary(self, session_id: str) -> None:
        """删除某个会话的派生摘要，通常由 clear() 调用。

        参数：
            session_id: 会话唯一标识。

        调用顺序：问答历史或反馈存储 -> ChatHistoryStore.delete_summary()。
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {self.settings.chat_summary_table_name} WHERE session_id=:session_id"),
                {"session_id": session_id},
            )


@lru_cache(maxsize=1)
def get_history_store() -> ChatHistoryStore:
    """返回进程级单例 ChatHistoryStore。

    lru_cache 只复用存储对象本身；具体消息仍然按 session_id 从 MySQL 读取。

    调用顺序：问答历史或反馈存储 -> get_history_store()。
    """
    return ChatHistoryStore()


def format_messages(messages: Iterable[BaseMessage]) -> str:
    """把 LangChain 消息列表格式化成中文对话文本，供 Prompt 使用。

    映射规则：
      - HumanMessage -> "用户：<content>"
      - AIMessage -> "助手：<content>"
      - 其他消息 -> "<type>：<content>"

    参数：
        messages: LangChain BaseMessage 可迭代对象。

    返回：
        使用换行分隔的中文对话文本。

    调用顺序：问答历史或反馈存储 -> format_messages()。
    """
    parts: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            parts.append(f"用户：{message.content}")
        elif isinstance(message, AIMessage):
            parts.append(f"助手：{message.content}")
        else:
            parts.append(f"{message.type}：{message.content}")
    return "\n".join(parts)

