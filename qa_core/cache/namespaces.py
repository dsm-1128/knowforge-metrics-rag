"""缓存 namespace 与 epoch 管理。

设计决策：
- 通过 epoch 自增实现缓存失效：每次版本发布或回滚时推进 epoch，旧的 Redis key
  虽然仍然存在，但新的缓存 key 包含新的 epoch 值，不会命中旧缓存。旧 key 在 TTL
  过期后自然淘汰，无需扫描删除全量 Redis key。
- namespace 按 scenario_id + tenant_id + dataset_id 三个维度隔离，不同租户/数据集的
  缓存 epoch 互不影响。
- 使用 INSERT ... ON DUPLICATE KEY UPDATE 模式实现"不存在时自动创建"的语义，
  避免 SELECT-then-INSERT 的竞态条件。

调用顺序：CacheManager.namespace_epoch() / invalidate_scenario() -> namespaces。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from qa_core.governance.data_scope import DEFAULT_DATASET_ID, DEFAULT_TENANT_ID
from qa_core.memory.base import _MySqlStore, safe_sql_identifier


CACHE_NAMESPACE_TABLE = "cache_namespaces"


def bump_cache_epoch_for_scenario_with_conn(conn, scenario_id: str) -> int:
    """在外部事务中推进某个场景下所有 namespace 的 epoch。（★★★ 核心）

    执行流程：
      1. UPDATE ... SET cache_epoch = cache_epoch + 1 对所有已有 namespace 自增 epoch。
      2. 如果 rowcount == 0（该场景还没有任何 namespace），插入默认 namespace：
         - 初始 epoch = 2（模拟一次 UPDATE +1 的效果，保证首次 bump 后新缓存 key
           中的 epoch 为 2，而不是 1，与正常场景的行为一致）。

    为什么 epoch 从 2 开始：
    正常的 bump 流程是 UPDATE ... SET cache_epoch = cache_epoch + 1，所以首次 bump
    后 epoch 从 1 变成 2。如果场景没有 namespace，直接插入 epoch=1 的记录，那么新
    缓存 key 中的 epoch=1 可能与旧缓存 key 混用。插入 epoch=2 保证无论是否有过
    之前的 namespace，bump 后的 epoch 语义一致。

    参数：
        conn: 外部事务的数据库连接（SQLAlchemy Connection）。
        scenario_id: 场景 ID。

    返回：
        受影响的 namespace 数量（总是 >= 1）。

    调用顺序：CacheNamespaceStore.bump_scenario_epoch() -> bump_cache_epoch_for_scenario_with_conn()。
    """
    table = safe_sql_identifier(CACHE_NAMESPACE_TABLE, label="Cache namespace table")
    # ── 步骤 1：对所有已有 namespace 自增 epoch ──
    result = conn.execute(
        text(
            f"""
            UPDATE {table}
            SET cache_epoch = cache_epoch + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE scenario_id = :scenario_id
            """
        ),
        {"scenario_id": scenario_id},
    )
    # ── 步骤 2：该场景尚无 namespace，创建默认记录 ──
    if int(result.rowcount or 0) == 0:
        # 该场景尚无 namespace → 创建默认 namespace，epoch 从 2 开始
        # （模拟正常 UPDATE +1：如果没有这一步，新插入的 epoch=1 会与旧缓存 key 冲突）
        conn.execute(
            text(
                f"""
                INSERT INTO {table}
                    (scenario_id, tenant_id, dataset_id, cache_epoch)
                VALUES
                    (:scenario_id, :tenant_id, :dataset_id, 2)
                """
            ),
            {
                "scenario_id": scenario_id,
                "tenant_id": DEFAULT_TENANT_ID,
                "dataset_id": DEFAULT_DATASET_ID,
            },
        )
        return 1
    return int(result.rowcount or 0)


class CacheNamespaceStore(_MySqlStore):
    """缓存 namespace 元数据存储，负责 epoch 的读取、推进和列出。

    每个 namespace 由 scenario_id + tenant_id + dataset_id 三元组唯一确定，
    对应一个独立的缓存版本号（epoch）。版本发布时通过 bump_scenario_epoch()
    推进所有 namespace 的 epoch，使旧缓存自然失效。

    调用顺序：CacheManager -> CacheNamespaceStore。
    """

    def __init__(self) -> None:
        """初始化缓存 namespace 存储。

        MySQL 连接由父类 _MySqlStore 延迟创建，构造时不会连接数据库。

        调用顺序：CacheManager._namespace_store() -> CacheNamespaceStore.__init__()。
        """
        super().__init__()
        # 使用 safe_sql_identifier 校验表名合法性，防止配置项注入 SQL
        self.table = safe_sql_identifier(CACHE_NAMESPACE_TABLE, label="Cache namespace table")

    def get_epoch(self, *, scenario_id: str, tenant_id: str, dataset_id: str) -> int:
        """读取 namespace epoch；不存在时创建初始纪录并返回 1。（★★ 理解）

        使用 INSERT ... ON DUPLICATE KEY UPDATE cache_epoch = cache_epoch 的
        MySQL 技巧：
        - 如果记录不存在 → INSERT 创建 epoch=1 的初始记录
        - 如果记录已存在 → UPDATE 不改变 epoch 值（自赋值），不产生实际修改

        这个模式保证"不存在时自动创建"的语义，且不使用 SELECT-then-INSERT 的竞态方案。
        在并发环境下，两个请求同时 INSERT 会有一个遇到 DUPLICATE KEY 错误并回退为
        UPDATE，保证数据一致性。

        参数：
            scenario_id: 场景 ID。
            tenant_id: 租户 ID。
            dataset_id: 数据集 ID。

        返回：
            当前 epoch 整数值（初始为 1）。

        调用顺序：CacheManager.namespace_epoch() -> CacheNamespaceStore.get_epoch()。
        """
        with self.engine.begin() as conn:
            # ── 步骤 1：INSERT ... ON DUPLICATE KEY UPDATE ──
            # 保证记录存在，epoch 不改变
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.table}
                        (scenario_id, tenant_id, dataset_id, cache_epoch)
                    VALUES
                        (:scenario_id, :tenant_id, :dataset_id, 1)
                    ON DUPLICATE KEY UPDATE
                        cache_epoch = cache_epoch
                    """
                ),
                {"scenario_id": scenario_id, "tenant_id": tenant_id, "dataset_id": dataset_id},
            )
            # ── 步骤 2：查询实际 epoch 值 ──
            row = conn.execute(
                text(
                    f"""
                    SELECT cache_epoch
                    FROM {self.table}
                    WHERE scenario_id=:scenario_id
                      AND tenant_id=:tenant_id
                      AND dataset_id=:dataset_id
                    """
                ),
                {"scenario_id": scenario_id, "tenant_id": tenant_id, "dataset_id": dataset_id},
            ).mappings().fetchone()
        return int(row["cache_epoch"] or 1)

    def bump_scenario_epoch(self, scenario_id: str) -> int:
        """推进场景下所有 namespace 的 epoch。（★★ 理解）

        使用独立事务执行 bump 操作，不影响其他场景的 epoch。推进后该场景下的
        所有旧缓存 key 都不会再被命中（key 中包含旧的 epoch 值）。

        参数：
            scenario_id: 场景 ID。

        返回：
            受影响的 namespace 数量。

        调用顺序：CacheManager.invalidate_scenario() -> CacheNamespaceStore.bump_scenario_epoch()。
        """
        with self.engine.begin() as conn:
            return bump_cache_epoch_for_scenario_with_conn(conn, scenario_id)

    def list_namespaces(self, *, scenario_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """返回 namespace 列表，用于管理端查看缓存视图。（★★ 理解）

        按 updated_at 降序排列，最新的 namespace 排在前面。可选的 scenario_id
        参数用于过滤特定场景的 namespace。

        参数：
            scenario_id: 可选，用于过滤特定场景的 namespace。
            limit: 最大返回数量（上限 500）。

        返回：
            包含 scenario_id、tenant_id、dataset_id、cache_epoch 和 updated_at
            的字典列表。

        调用顺序：管理 API -> CacheManager.status() -> CacheNamespaceStore.list_namespaces()。
        """
        # 原因：限制查询上限 500 条，防止超大 limit 导致 MySQL 扫全表或 OOM
        safe_limit = max(1, min(int(limit or 100), 500))
        where = "WHERE scenario_id=:scenario_id" if scenario_id else ""
        params = {"scenario_id": scenario_id} if scenario_id else {}
        sql = f"""
        SELECT scenario_id, tenant_id, dataset_id, cache_epoch, updated_at
        FROM {self.table}
        {where}
        ORDER BY updated_at DESC
        LIMIT {safe_limit}
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(row) for row in rows]
