"""用户反馈持久化层 — 反馈作为长期质量资产存入 MySQL。

设计决策：
- 在线链路只记录反馈，不立即改变检索或答案。用户误点率约 5-15%，实时反馈会引入
  噪声，需要人工审核后再进入训练集或权重调整。
- 反馈的 sources 以 JSON 数组形式存储，保留来源元数据用于后续质量分析。
- 使用 `lru_cache` 保证整个进程生命周期内只创建一个 FeedbackStore 实例和对应的
  数据库连接池。

调用顺序：API 层 -> memory.feedback。
"""

from __future__ import annotations
import json
from functools import lru_cache
from typing import Any

from sqlalchemy import text

from qa_core.config.logging_config import get_logger
from qa_core.memory.base import _MySqlStore

logger = get_logger(__name__)

class FeedbackStore(_MySqlStore):
    """答案反馈表的轻量 SQL 适配器，不使用 LangChain 组件。

    提供 add_feedback（保存反馈）和 list_bad_feedback（读取低质量反馈）两个方法。
    反馈数据只做持久化不介入检索流程，让反馈作为独立的长期质量资产存在。

    调用顺序：API 层 -> FeedbackStore。
    """

    def __init__(self) -> None:
        """加载配置，并延迟到首次使用时再创建数据库引擎。

        构造对象不连接 MySQL，首次调用 add_feedback() 或 list_bad_feedback() 时
        才通过父类 _MySqlStore.engine 属性创建数据库连接。

        调用顺序：get_feedback_store() -> FeedbackStore.__init__()。
        """
        super().__init__()

    def add_feedback(
        self,
        *,
        session_id: str | None,
        scenario_id: str | None = None,
        tenant_id: str | None = None,
        dataset_id: str | None = None,
        question: str,
        answer: str,
        rating: str,
        comment: str | None,
        sources: list[dict[str, Any]],
    ) -> int:
        """保存一条用户评分，成功时返回数据库主键。写入失败向上抛出异常。（★★ 理解）

        参数：
            session_id: 会话 ID。
            scenario_id: 场景 ID。
            tenant_id: 租户 ID。
            dataset_id: 数据集 ID。
            question: 用户问题。
            answer: 助手回答。
            rating: 评分（如 "useful" / "not_useful"）。
            comment: 用户评论文本。
            sources: 答案来源列表（JSON 数组）。

        返回：
            数据库主键 ID。

        调用顺序：API 层 -> FeedbackStore.add_feedback()。
        """
        # 反馈仅持久化到 MySQL，不立即修改检索权重或排序——用户误点率约 5-15%，
        # 实时反馈会引入噪声，需要人工审核后再进入训练集或权重调整
        sql = f"""
        INSERT INTO {self.settings.feedback_table_name}
            (session_id, scenario_id, tenant_id, dataset_id, question, answer, rating, comment, sources_json)
        VALUES
            (:session_id, :scenario_id, :tenant_id, :dataset_id, :question, :answer, :rating, :comment, :sources_json)
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                text(sql),
                {
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                    "tenant_id": tenant_id,
                    "dataset_id": dataset_id,
                    "question": question,
                    "answer": answer,
                    "rating": rating,
                    "comment": comment,
                    # sources 以 JSON 数组形式存储，保留来源元数据用于后续质量分析
                    "sources_json": json.dumps(sources, ensure_ascii=False),
                },
            )
            return int(result.lastrowid or 0)

    def list_bad_feedback(
        self,
        *,
        limit: int = 200,
        scenario_id: str | None = None,
        rating: str = "not_useful",
    ) -> list[dict[str, Any]]:
        """读取需要复盘的低质量反馈。（★★ 理解）

        只提供事实数据，正式入评测集由脚本二次审核。此方法供管理后台和复盘脚本
        使用，不直接修改检索或答案流程。

        参数：
            limit: 最大返回条数（上限 1000）。
            scenario_id: 可选，按场景过滤。
            rating: 评分类型（默认 "not_useful"）。

        返回：
            包含 id、session_id、question、answer、rating、comment、sources、created_at
            的字典列表。

        调用顺序：管理后台或复盘脚本 -> FeedbackStore.list_bad_feedback()。
        """
        # 限制查询上限 1000 条，防止超大 limit 导致 OOM 或 MySQL 扫全表
        safe_limit = max(1, min(int(limit), 1000))
        filters = ["rating = :rating"]
        params: dict[str, Any] = {"rating": rating, "limit": safe_limit}
        if scenario_id:
            # 按场景过滤时追加条件，支持多场景独立复盘
            filters.append("scenario_id = :scenario_id")
            params["scenario_id"] = scenario_id
        sql = f"""
        SELECT
            id, session_id, scenario_id, tenant_id, dataset_id,
            question, answer, rating, comment, sources_json, created_at
        FROM {self.settings.feedback_table_name}
        WHERE {" AND ".join(filters)}
        ORDER BY id DESC
        LIMIT :limit
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                # 将 JSON 字符串反序列化为列表，便于前端和复盘脚本直接使用
                payload["sources"] = json.loads(payload.pop("sources_json") or "[]")
            except json.JSONDecodeError:
                payload["sources"] = []
            payload["created_at"] = str(payload.get("created_at") or "")
            result.append(payload)
        return result


@lru_cache(maxsize=1)
def get_feedback_store() -> FeedbackStore:
    """返回 API 处理器共用的反馈存储单例。

    lru_cache 保证整个进程生命周期内只创建一个 FeedbackStore 实例和对应的数据库
    连接池。但反馈数据不驻留在该对象内存中——数据始终从 MySQL 实时查询。

    调用顺序：API 层 -> get_feedback_store()。
    """
    return FeedbackStore()

