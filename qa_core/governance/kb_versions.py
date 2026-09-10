"""知识库多版本管理。

MySQL 保存版本控制面数据，Milvus 保存可检索的 FAQ/文档 chunk。版本切换只更新
MySQL 中的 active 指针，不批量复制 Milvus 数据，因此仍然保持 O(1) 激活和快速回滚。

核心能力：
- 生成带时间戳和配置 hash 的版本号。
- 跟踪版本生命周期：STAGED -> ACTIVE -> ARCHIVED。
- 按“请求参数 > MySQL active 指针”的优先级解析检索版本；未显式指定时只认数据库 active 指针。
- 记录每个版本的 FAQ/文档入库统计。
- 为引用式增量版本分配单调 version_seq，供 Milvus 有效期视图过滤。

典型调用顺序：
- 检索解析：业务入口 -> resolve_kb_version_record() -> resolve_version() -> active_version_candidate() -> get()。
- 新版本入库：入库脚本 -> ensure_version() -> generate_kb_version() -> _next_version_seq_with_conn()
  -> _upsert_version_with_conn()。
- 版本上线：管理 API/脚本 -> activate_version() -> _upsert_version_with_conn()
  -> _set_active_pointer_with_conn() -> _record_activation_with_conn()。
- 入库统计：入库流程 -> record_ingest_result() -> get()/ensure_version() -> _upsert_version()。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from qa_core.cache.manager import get_cache_manager
from qa_core.cache.namespaces import bump_cache_epoch_for_scenario_with_conn
from qa_core.common import utc_file_stamp, utc_now
from qa_core.config.settings import get_settings
from qa_core.governance.kb_version_models import (
    KB_VERSION_SELECT_COLUMNS,
    KB_VERSION_STATUS_ACTIVE,
    KB_VERSION_STATUS_ARCHIVED,
    KB_VERSION_STATUS_STAGED,
    KnowledgeBaseVersion,
    json_dumps,
)
from qa_core.memory.base import _MySqlStore, safe_sql_identifier
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import stable_hash

KB_VERSIONS_TABLE = "kb_versions"
KB_ACTIVE_TABLE = "kb_active_versions"
KB_ACTIVATION_TABLE = "kb_version_activations"

# 三张表各自负责一件事：
# - kb_versions 保存每个版本的完整元信息和统计；
# - kb_active_versions 保存每个场景当前在线版本和上一个版本指针；
# - kb_version_activations 保存每次激活/回滚流水。
# 表结构统一定义在 qa_core/storage/runtime_schema.sql，这里只做业务读写。


def _resolve_version_scenario(scenario_id: str | None = None):
    """解析知识库版本所属的场景配置，避免模块循环依赖。

    版本号生成、版本表初始化和 metadata 注入都必须使用同一个场景定义。
    这里集中调用 registry，避免治理模块自己维护一份 collection、source 或
    场景默认值，从而减少配置漂移。

    调用顺序：generate_kb_version()/KnowledgeBaseVersionStore.__init__()/version_metadata()
    -> _resolve_version_scenario() -> resolve_scenario()。
    """
    return resolve_scenario(scenario_id)


def generate_kb_version(prefix: str = "kb", scenario_id: str | None = None) -> str:
    """生成含时间戳和配置 hash 的知识库版本号。（★★ 理解）

    时间戳负责可读性和大致排序，配置 hash 负责把模型、切分规则和 collection
    变化体现在版本号中。它不是数据库里的唯一版本序号，真正用于有效窗口比较
    的是后续分配的 `version_seq`。

    调用顺序：`ensure_version(create_new=True 或首次无 active 版本)`
    -> `generate_kb_version()`。
    """
    settings = get_settings()
    scenario = _resolve_version_scenario(scenario_id)
    stamp = utc_file_stamp()
    # 配置 hash 纳入 embedding_model_version、reranker_model_version 和 chunk_schema_version，
    # 确保模型或切分策略变更时自动生成不同的版本号，避免新旧配置产生的 chunk 混在同一版本中
    config_hash = stable_hash(
        scenario.scenario_id,
        settings.embedding_model_version,
        settings.reranker_model_version,
        settings.chunk_schema_version,
        scenario.doc_collection,
        scenario.faq_collection,
    )[:8]
    return f"{prefix}_{scenario.scenario_id}_{stamp}_{config_hash}"


class KnowledgeBaseVersionStore(_MySqlStore):
    """知识库版本状态机的 MySQL 存储实现。

    常见入口：
    - 查询当前版本：resolve_version()/resolve_active_version()
    - 创建版本：ensure_version()
    - 切换版本：activate_version()
    - 归档版本：archive_version()
    - 查看管理面板数据：as_payload()

    调用顺序：治理或版本管理入口 -> KnowledgeBaseVersionStore。
    """

    def __init__(self, scenario_id: str | None = None) -> None:
        """绑定一个业务场景。MySQL 表结构由启动期 bootstrap 统一初始化。

        Store 只绑定场景，不在构造时缓存 active 版本。这样每次通过 `get()`、
        `active_version_candidate()` 或 `resolve_version()` 读取时，都能看到当前
        MySQL 控制面的最新状态。

        调用顺序：get_kb_version_store() -> KnowledgeBaseVersionStore.__init__()。
        """
        super().__init__()
        # Store 以场景为边界读写版本表，同一套代码可服务多个业务场景。
        self.scenario = _resolve_version_scenario(scenario_id)
        # 表名只在初始化阶段做一次安全校验，后续 SQL 拼接只使用校验后的表名。
        self.version_table = safe_sql_identifier(KB_VERSIONS_TABLE, label="KB versions table")
        self.active_table = safe_sql_identifier(KB_ACTIVE_TABLE, label="KB active table")
        self.activation_table = safe_sql_identifier(KB_ACTIVATION_TABLE, label="KB activation table")

    def list_versions(self) -> list[KnowledgeBaseVersion]:
        """按创建时间倒序返回当前场景全部版本。

        调用顺序：as_payload()/管理 API -> list_versions() -> KnowledgeBaseVersion.from_row()。
        """
        # 管理界面需要看到当前场景的完整版本列表，所以只按 scenario_id 过滤，不按状态过滤。
        sql = f"""
        SELECT {KB_VERSION_SELECT_COLUMNS}
        FROM {self.version_table}
        WHERE scenario_id = :scenario_id
        ORDER BY created_at DESC, kb_version DESC
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), {"scenario_id": self.scenario.scenario_id}).mappings().all()
        # Store 对外统一返回数据类，调用方不需要关心 MySQL 行对象和 JSON 字段解析。
        return [KnowledgeBaseVersion.from_row(row) for row in rows]

    def get(self, kb_version: str | None) -> KnowledgeBaseVersion | None:
        """按版本号读取版本记录。

        调用顺序：resolve_version()/ensure_version()/activate_version()/统计写入 -> get()。
        """
        if not kb_version:
            # 空版本号直接返回 None，不执行 SQL 查询，避免无效查询浪费连接
            return None
        # 版本号在不同场景下可能重复，所以读取时必须同时限定 scenario_id 和 kb_version。
        sql = f"""
        SELECT {KB_VERSION_SELECT_COLUMNS}
        FROM {self.version_table}
        WHERE scenario_id = :scenario_id AND kb_version = :kb_version
        """
        with self.engine.begin() as conn:
            row = conn.execute(
                text(sql),
                {"scenario_id": self.scenario.scenario_id, "kb_version": kb_version},
            ).mappings().fetchone()
        return KnowledgeBaseVersion.from_row(row) if row else None

    def exists(self, kb_version: str | None) -> bool:
        """判断版本是否存在于 MySQL。

        调用顺序：脚本或测试 -> exists() -> get()。
        """
        # exists 只表达布尔语义，具体读取和空值处理统一复用 get()。
        return self.get(kb_version) is not None

    def active_version_candidate(self) -> str:
        """返回 MySQL 中声明的 active 版本号（允许为空）。

        调用顺序：resolve_version()/ensure_version()/archive_version() -> active_version_candidate()
        -> _active_pointer()。
        """
        # MySQL active 指针是在线版本的唯一权威来源，由 activate_version() 维护；
        # STAGED 版本即使已经完成入库，也不能被这里当作线上版本返回。
        active, _ = self._active_pointer()
        return active

    def resolve_active_version(self, requested: str | None = None) -> str:
        """解析检索应使用的知识库版本号。（★★ 理解）

        调用顺序：resolve_active_kb_version()/旧调用方 -> resolve_active_version()
        -> resolve_version()。
        """
        # 兼容只需要版本号字符串的调用方；完整检索链路应优先使用 resolve_version()。
        return self.resolve_version(requested).kb_version

    def resolve_version(self, requested: str | None = None) -> KnowledgeBaseVersion:
        """解析检索应使用的版本记录，包含引用式增量所需 version_seq。

        调用顺序：resolve_kb_version_record()/resolve_active_version()/as_payload()
        -> resolve_version() -> get()/active_version_candidate()。
        """
        # 优先级：请求参数 > MySQL active 指针。显式请求不存在时直接失败，
        # 不自动降级到 active，避免调用方以为读取了指定版本、实际却读到了旧版本。
        if requested:
            candidate = requested.strip()
            record = self.get(candidate)
            if record is None:
                # 请求的版本不存在时直接报错，拒绝降级到 active 版本以免静默返回过时数据
                raise ValueError(f"请求的知识库版本不存在：{candidate}")
            return record
        active = self.active_version_candidate()
        if not active:
            # 没有显式请求版本，也没有 active 指针时，检索无法确定数据视图，必须中断。
            raise ValueError(f"场景 {self.scenario.scenario_id} 没有 active 知识库版本")
        record = self.get(active)
        if record is None:
            raise ValueError(f"active 知识库版本不存在于版本表：{active}")
        return record

    def ensure_version(
        self,
        kb_version: str | None = None,
        *,
        create_new: bool = False,
        description: str = "",
        created_by: str = "local",
    ) -> KnowledgeBaseVersion:
        """确保版本记录存在（不自动覆盖已有版本）。（★★★ 核心）

        调用顺序：入库脚本/record_ingest_result()/record_incremental_base()
        -> ensure_version() -> get() -> _next_version_seq_with_conn() -> _upsert_version_with_conn()。
        """
        # 入口语义分三种：
        # 1. create_new=True：为一次新入库创建新版本；
        # 2. 传入 kb_version：确保这个指定版本存在；
        # 3. 两者都没有：复用当前 active 版本；没有 active 时生成 STAGED 候选版本。
        candidate = (kb_version or "").strip()
        if create_new:
            # 请求新建版本：如果未指定版本号则自动生成，否则使用指定版本号（不覆盖已有）
            candidate = candidate or generate_kb_version(scenario_id=self.scenario.scenario_id)
        elif not candidate:
            # 未指定版本号且不新建：使用当前 active 版本；无 active 版本时只生成 STAGED 候选版本
            active = self.active_version_candidate()
            candidate = active or generate_kb_version(scenario_id=self.scenario.scenario_id)

        existing = self.get(candidate)
        if existing is not None:
            # 版本已存在时直接返回。重要的是：ensure_version 不会覆盖已有版本，
            # 版本的状态、统计和 active 指针分别由后续专用方法维护。
            return existing

        settings = self.settings
        with self.engine.begin() as conn:
            # 单调递增 version_seq：用于后续引用式增量的有效期视图过滤
            # （valid_from_seq/valid_to_seq）。分配和插入放在同一事务中，避免并发
            # 创建版本拿到相同序号。
            version_seq = self._next_version_seq_with_conn(conn)
            record = KnowledgeBaseVersion(
                kb_version=candidate,
                scenario_id=self.scenario.scenario_id,
                version_seq=version_seq,
                status=KB_VERSION_STATUS_STAGED,
                description=description,
                activated_at=None,
                doc_collection=self.scenario.doc_collection,
                faq_collection=self.scenario.faq_collection,
                embedding_model_version=settings.embedding_model_version,
                reranker_model_version=settings.reranker_model_version,
                chunk_schema_version=settings.chunk_schema_version,
                created_by=created_by,
                stats={"created_reason": "ingest_or_manual"},
            )
            self._upsert_version_with_conn(conn, record)
        return record

    def activate_version(
        self,
        kb_version: str,
        *,
        reason: str = "",
        activated_by: str = "system",
    ) -> KnowledgeBaseVersion:
        """把指定版本切为当前在线检索版本。（★★★ 核心）

        调用顺序：管理 API/管理脚本 -> activate_version() -> get()
        -> _get_version_with_conn() -> _upsert_version_with_conn()
        -> _set_active_pointer_with_conn() -> _record_activation_with_conn()。
        """
        # 先做事务外存在性检查，给调用方更直接的错误信息；真正更新时仍会在
        # 事务内重新读取 active 指针，避免检查与提交之间使用过时状态。
        record = self.get(kb_version)
        if record is None:
            raise ValueError(f"知识库版本不存在：{kb_version}")

        now = utc_now()
        with self.engine.begin() as conn:
            # 版本切换的事务边界包含四个动作：锁定 active 指针、更新版本状态、
            # 刷新 active/previous 指针、写入切换流水。四者必须一起提交，否则
            # 可能出现版本状态已经 ACTIVE 但 active 指针仍指向旧版本的半完成状态。
            # FOR UPDATE 锁定行，防止并发激活导致 active 指针不一致。
            pointer = conn.execute(
                text(f"SELECT active_kb_version FROM {self.active_table} WHERE scenario_id=:scenario_id FOR UPDATE"),
                {"scenario_id": self.scenario.scenario_id},
            ).mappings().fetchone()
            previous = str(pointer["active_kb_version"] or "") if pointer else ""
            previous_record = self._get_version_with_conn(conn, previous) if previous else None
            previous_seq = int(previous_record.version_seq or 0) if previous_record else 0
            # 判别是“激活”还是“回滚”：目标 version_seq 小于当前版本时，
            # 说明是回到历史版本，审计流水需要使用 rollback 动作。
            action = "rollback" if previous_record and int(record.version_seq or 0) < previous_seq else "activate"
            # 先收口旧 ACTIVE 状态，再写目标 ACTIVE，保证一个场景最多只有一个
            # 逻辑上的在线版本。
            conn.execute(
                text(
                    f"""
                    UPDATE {self.version_table}
                    SET status=:status
                    WHERE scenario_id=:scenario_id AND status=:active_status
                    """
                ),
                {
                    "status": KB_VERSION_STATUS_STAGED,
                    "scenario_id": self.scenario.scenario_id,
                    "active_status": KB_VERSION_STATUS_ACTIVE,
                },
            )
            # 将目标版本设为 ACTIVE
            record.status = KB_VERSION_STATUS_ACTIVE
            record.activated_at = now
            self._upsert_version_with_conn(conn, record)
            # 更新 active 指针：previous_kb_version 保存切换前的版本，支持快速
            # 回滚到上一个版本；目标版本本身重复激活时保留已有 previous 指针。
            self._set_active_pointer_with_conn(
                conn,
                kb_version,
                previous if previous != kb_version else self._active_pointer_with_conn(conn)[1],
            )
            if previous != kb_version:
                # 记录激活/回滚流水，用于多级版本回退审计和问题追踪。
                self._record_activation_with_conn(
                    conn,
                    from_version=previous,
                    to_version=kb_version,
                    from_version_seq=previous_seq,
                    to_version_seq=record.version_seq,
                    action=action,
                    reason=reason,
                    activated_by=activated_by,
                )
            # 版本发布/回滚会改变线上可见知识库视图，必须推进缓存 epoch，
            # 使旧检索缓存自然失效。
            bump_cache_epoch_for_scenario_with_conn(conn, self.scenario.scenario_id)
        # 事务提交后重新读取一次，保证返回值包含数据库触发的 updated_at 等最终字段。
        get_cache_manager().l1_cache.clear()
        return self.get(kb_version) or record

    def archive_version(self, kb_version: str) -> KnowledgeBaseVersion:
        """归档非 active 版本，仅改状态不删 Milvus 数据。

        调用顺序：管理 API/管理脚本 -> archive_version() -> active_version_candidate()
        -> get() -> _upsert_version()。
        """
        if self.active_version_candidate() == kb_version:
            # 禁止归档当前正使用的 active 版本：归档后检索链路会找不到版本，导致 500 错误
            raise ValueError("不能归档当前 active 知识库版本")
        record = self.get(kb_version)
        if record is None:
            raise ValueError(f"知识库版本不存在：{kb_version}")
        # 归档只改变控制面状态，不删除 Milvus 数据，后续仍可用于审计或人工恢复。
        record.status = KB_VERSION_STATUS_ARCHIVED
        record.archived_at = utc_now()
        self._upsert_version(record)
        return record

    def record_ingest_result(
        self,
        kb_version: str,
        *,
        content_type: str,
        count: int,
        source: str | None = None,
        extra_stats: dict[str, Any] | None = None,
    ) -> KnowledgeBaseVersion:
        """记录某次入库结果统计。

        调用顺序：FAQ/文档入库流程 -> record_ingest_result() -> get()/ensure_version()
        -> _upsert_version()。
        """
        # 归档只改变控制面生命周期，不改变 Milvus 数据，也不改变 active 指针。
        record = self.get(kb_version)
        if record is None:
            # 版本记录不存在时自动创建：首次入库前可能尚未调用 ensure_version
            record = self.ensure_version(kb_version)
        if source and source not in record.sources:
            # sources 保存参与构建该版本的数据来源，重复来源不重复写入。
            record.sources.append(source)
        # 统计字段分别记录最后一次入库数量、累计入库次数和累计总条数，
        # 后续可通过版本信息面板查看本次构建结果与历史累计值。
        key = f"last_{content_type}_count"
        runs_key = f"{content_type}_ingest_runs"
        total_key = f"total_{content_type}_written"
        record.stats[key] = count
        record.stats[runs_key] = int(record.stats.get(runs_key, 0)) + 1
        record.stats[total_key] = int(record.stats.get(total_key, 0)) + count
        if extra_stats:
            # extra_stats 用于补充入库流程产生的额外指标，例如跳过数量、耗时、
            # 失效 chunk 数量或错误摘要；这里采用 merge，不覆盖基础计数字段。
            record.stats.update(extra_stats)
        # 最后写入时间用于管理界面判断版本是否刚刚刷新过。
        record.stats["last_ingested_at"] = utc_now()
        self._upsert_version(record)
        return record

    def record_incremental_base(self, kb_version: str, base_kb_version: str) -> KnowledgeBaseVersion:
        """记录引用式增量版本所基于的基础版本。

        调用顺序：引用式增量入库流程 -> record_incremental_base()
        -> get()/ensure_version() -> _upsert_version()。
        """
        record = self.get(kb_version)
        if record is None:
            # 增量版本可能由脚本直接传入，缺少版本行时先补齐控制面记录。
            record = self.ensure_version(kb_version)
        # 目标版本记录它引用的基础版本，便于管理面板和审计追踪；实际 chunk
        # 可见性仍由 Milvus/MySQL 中的 valid_from_seq/valid_to_seq 决定。
        record.stats["incremental_base_kb_version"] = base_kb_version
        record.stats["incremental_mode"] = "reference_delta_validity_window"
        self._upsert_version(record)
        return record

    def as_payload(self) -> dict[str, Any]:
        """返回 API 和脚本可以直接打印的版本管理视图。

        调用顺序：管理 API/管理脚本 -> as_payload() -> _active_pointer()
        -> resolve_active_version() -> list_activation_history() -> list_versions()。
        """
        active, previous = self._active_pointer()
        # 管理页在尚未初始化 active 版本时也要能打开，因此保留一个可为空的有效版本字段。
        try:
            effective_active = self.resolve_active_version()
        except ValueError:
            # 管理页在尚未初始化 active 版本时也要能打开，因此这里展示为空而不是抛出。
            effective_active = None
        # payload 同时返回指针、实际生效版本、流水和版本列表，方便 API 一次响应管理面板。
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.display_name,
            "active_version": active or None,
            "effective_active_version": effective_active,
            "previous_version": previous or None,
            "metadata_store": "mysql",
            "versioning_mode": "reference_incremental",
            "activation_history": self.list_activation_history(limit=20),
            "versions": [item.as_dict() for item in self.list_versions()],
        }

    def list_activation_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """返回当前场景最近的版本激活/回滚流水。

        调用顺序：as_payload()/管理 API -> list_activation_history()。
        """
        # 限制最大返回条数，管理界面只展示最近流水，避免一次接口拉出过多历史记录。
        safe_limit = max(1, min(int(limit or 20), 200))
        # 流水按 id 倒序读取，最新激活/回滚记录排在最前面。
        sql = f"""
        SELECT
            id, scenario_id, from_kb_version, to_kb_version,
            from_version_seq, to_version_seq, action, reason,
            activated_by, created_at
        FROM {self.activation_table}
        WHERE scenario_id=:scenario_id
        ORDER BY id DESC
        LIMIT {safe_limit}
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), {"scenario_id": self.scenario.scenario_id}).mappings().all()
        return [dict(row) for row in rows]

    def _active_pointer(self) -> tuple[str, str]:
        """读取当前场景 active/previous 指针。

        调用顺序：active_version_candidate()/as_payload() -> _active_pointer()
        -> _active_pointer_with_conn()。
        """
        with self.engine.begin() as conn:
            return self._active_pointer_with_conn(conn)

    def _active_pointer_with_conn(self, conn) -> tuple[str, str]:
        """在当前连接或事务中读取 active/previous 指针。

        调用顺序：_active_pointer()/ensure_version()/activate_version()
        -> _active_pointer_with_conn()。
        """
        row = conn.execute(
            text(
                f"""
                SELECT active_kb_version, previous_kb_version
                FROM {self.active_table}
                WHERE scenario_id = :scenario_id
                """
            ),
            {"scenario_id": self.scenario.scenario_id},
        ).mappings().fetchone()
        # 返回 (active, previous)：active 表示当前在线版本，previous 表示前一个版本，
        # 用于快速回滚。首次使用时两条记录都为空字符串
        if not row:
            # 未激活过任何版本时，调用方统一收到空字符串而不是 None。
            return "", ""
        return str(row["active_kb_version"] or ""), str(row["previous_kb_version"] or "")

    def _set_active_pointer_with_conn(self, conn, active: str, previous: str) -> None:
        """在当前事务中更新 active/previous 指针。

        调用顺序：activate_version()
        -> _set_active_pointer_with_conn()。
        """
        # INSERT ON DUPLICATE KEY UPDATE：每个 scenario 在 kb_active_versions 表中有且仅有一行，
        # active 指针始终指向当前在线版本，previous 指向最近一次激活前的版本
        sql = f"""
        INSERT INTO {self.active_table}
            (scenario_id, active_kb_version, previous_kb_version)
        VALUES
            (:scenario_id, :active_kb_version, :previous_kb_version)
        ON DUPLICATE KEY UPDATE
            active_kb_version=VALUES(active_kb_version),
            previous_kb_version=VALUES(previous_kb_version),
            updated_at=CURRENT_TIMESTAMP
        """
        conn.execute(
            text(sql),
            {
                "scenario_id": self.scenario.scenario_id,
                "active_kb_version": active,
                "previous_kb_version": previous,
            },
        )

    def _record_activation_with_conn(
        self,
        conn,
        *,
        from_version: str,
        to_version: str,
        from_version_seq: int,
        to_version_seq: int,
        action: str,
        reason: str,
        activated_by: str,
    ) -> None:
        """写入版本激活/回滚流水，支撑多级回滚审计。

        调用顺序：activate_version()
        -> _record_activation_with_conn()。
        """
        sql = f"""
        INSERT INTO {self.activation_table}
            (
                scenario_id, from_kb_version, to_kb_version,
                from_version_seq, to_version_seq, action,
                reason, activated_by, created_at
            )
        VALUES
            (
                :scenario_id, :from_kb_version, :to_kb_version,
                :from_version_seq, :to_version_seq, :action,
                :reason, :activated_by, :created_at
            )
        """
        conn.execute(
            text(sql),
            {
                "scenario_id": self.scenario.scenario_id,
                "from_kb_version": from_version or "",
                "to_kb_version": to_version or "",
                "from_version_seq": int(from_version_seq or 0),
                "to_version_seq": int(to_version_seq or 0),
                "action": action,
                "reason": reason,
                "activated_by": activated_by or "system",
                "created_at": utc_now(),
            },
        )

    def _get_version_with_conn(self, conn, kb_version: str | None) -> KnowledgeBaseVersion | None:
        """在当前事务中读取版本记录。

        调用顺序：activate_version() -> _get_version_with_conn()。
        """
        if not kb_version:
            # 激活前没有 previous 版本时直接返回 None，由调用方把 previous_seq 视为 0。
            return None
        row = conn.execute(
            text(
                f"""
                SELECT {KB_VERSION_SELECT_COLUMNS}
                FROM {self.version_table}
                WHERE scenario_id=:scenario_id AND kb_version=:kb_version
                """
            ),
            {"scenario_id": self.scenario.scenario_id, "kb_version": kb_version},
        ).mappings().fetchone()
        return KnowledgeBaseVersion.from_row(row) if row else None

    def _upsert_version(self, record: KnowledgeBaseVersion) -> None:
        """用独立事务写入版本记录。

        调用顺序：archive_version()/record_ingest_result()/record_incremental_base()
        -> _upsert_version() -> _upsert_version_with_conn()。
        """
        # 非事务调用场景使用这个薄封装；已经在事务中的流程直接调用 _upsert_version_with_conn。
        with self.engine.begin() as conn:
            self._upsert_version_with_conn(conn, record)

    def _upsert_version_with_conn(self, conn, record: KnowledgeBaseVersion) -> None:
        """在当前事务中创建或更新版本记录。

        调用顺序：ensure_version()/activate_version()/_upsert_version()
        -> _upsert_version_with_conn()。
        """
        # 版本表写入统一走 upsert：创建、激活、归档和统计更新共用同一套字段映射。
        # 这里不生成业务默认值，只把 KnowledgeBaseVersion 数据类的当前状态落库；
        # 默认值和状态转换应由 ensure/activate/archive 等上层方法决定。
        sql = f"""
        INSERT INTO {self.version_table}
            (
                scenario_id, kb_version, status, description, created_at,
                version_seq,
                activated_at, archived_at, doc_collection, faq_collection,
                embedding_model_version, reranker_model_version, chunk_schema_version,
                created_by, sources_json, stats_json
            )
        VALUES
            (
                :scenario_id, :kb_version, :status, :description, :created_at,
                :version_seq,
                :activated_at, :archived_at, :doc_collection, :faq_collection,
                :embedding_model_version, :reranker_model_version, :chunk_schema_version,
                :created_by, :sources_json, :stats_json
            )
        ON DUPLICATE KEY UPDATE
            status=VALUES(status),
            version_seq=VALUES(version_seq),
            description=VALUES(description),
            activated_at=VALUES(activated_at),
            archived_at=VALUES(archived_at),
            doc_collection=VALUES(doc_collection),
            faq_collection=VALUES(faq_collection),
            embedding_model_version=VALUES(embedding_model_version),
            reranker_model_version=VALUES(reranker_model_version),
            chunk_schema_version=VALUES(chunk_schema_version),
            created_by=VALUES(created_by),
            sources_json=VALUES(sources_json),
            stats_json=VALUES(stats_json),
            updated_at=CURRENT_TIMESTAMP
        """
        conn.execute(
            text(sql),
            # sources/stats 以 JSON 字符串存储，读取时由 KnowledgeBaseVersion.from_row() 还原。
            {
                "scenario_id": record.scenario_id or self.scenario.scenario_id,
                "kb_version": record.kb_version,
                "version_seq": int(record.version_seq or 0),
                "status": record.status,
                "description": record.description,
                "created_at": record.created_at,
                "activated_at": record.activated_at,
                "archived_at": record.archived_at,
                "doc_collection": record.doc_collection,
                "faq_collection": record.faq_collection,
                "embedding_model_version": record.embedding_model_version,
                "reranker_model_version": record.reranker_model_version,
                "chunk_schema_version": record.chunk_schema_version,
                "created_by": record.created_by,
                "sources_json": json_dumps(record.sources),
                "stats_json": json_dumps(record.stats),
            },
        )

    def _next_version_seq_with_conn(self, conn) -> int:
        """返回当前场景下一个单调版本序号。

        调用顺序：ensure_version() -> _next_version_seq_with_conn()。
        """
        row = conn.execute(
            text(
                f"""
                SELECT version_seq
                FROM {self.version_table}
                WHERE scenario_id=:scenario_id
                ORDER BY version_seq DESC
                LIMIT 1
                FOR UPDATE
                """
            ),
            {"scenario_id": self.scenario.scenario_id},
        ).mappings().fetchone()
        # version_seq 从 1 开始单调递增，每次新版本 +1。FOR UPDATE 在事务中
        # 锁定最新行，防止并发入库产生相同的 version_seq；序号一旦分配不回收，
        # 即使版本后来归档，历史窗口仍然保持可解释。
        return int(row["version_seq"] or 0) + 1 if row else 1


def get_kb_version_store(scenario_id: str | None = None) -> KnowledgeBaseVersionStore:
    """返回新的版本表访问对象（不缓存，保证看到最新 MySQL 状态）。

    调用顺序：API/脚本/便捷函数 -> get_kb_version_store()
    -> KnowledgeBaseVersionStore.__init__()。
    """
    # 不做单例缓存，避免长生命周期对象持有旧配置或旧数据库状态。
    return KnowledgeBaseVersionStore(scenario_id=scenario_id)


def resolve_active_kb_version(requested: str | None = None, scenario_id: str | None = None) -> str:
    """解析当前请求应使用的知识库版本号。

    调用顺序：检索入口 -> resolve_active_kb_version()
    -> get_kb_version_store() -> resolve_active_version()。
    """
    # 只返回版本号字符串，适合启动摘要和日志字段；真正检索时也会将它作为版本边界。
    # Store 不做长生命周期缓存，确保这里读取到的是 MySQL 中最新的 active 状态。
    return get_kb_version_store(scenario_id).resolve_active_version(requested)


def resolve_kb_version_record(requested: str | None = None, scenario_id: str | None = None) -> KnowledgeBaseVersion:
    """解析当前请求应使用的版本记录，供检索构造有效期视图。

    调用顺序：检索入口 -> resolve_kb_version_record()
    -> get_kb_version_store() -> resolve_version()。
    """
    # 返回完整版本记录，检索层可直接拿到 version_seq 构造 Milvus 有效期过滤。
    return get_kb_version_store(scenario_id).resolve_version(requested)


def version_metadata(
    kb_version: str | None,
    scenario_id: str | None = None,
    *,
    version_seq: int | None = None,
) -> dict[str, Any]:
    """构建写入 FAQ/chunk metadata 的版本字段，记录模型版本和切分方案。

    该函数是 FAQ 和普通文档共用的版本字段出口。普通文档依赖
    `valid_from_seq/valid_to_seq`，FAQ 虽然按 `kb_version_exact` 过滤，也保留
    这两个公共字段以保持 metadata 结构一致。

    调用顺序：入库流程 -> version_metadata() -> _resolve_version_scenario()。
    """
    settings = get_settings()
    scenario = _resolve_version_scenario(scenario_id)
    resolved_seq = int(version_seq or 0)
    # 这些字段会写入每条 Milvus chunk metadata。普通文档检索使用有效期窗口，
    # FAQ 检索使用精确 kb_version，但两者都共享同一个版本字段生成器。
    # valid_to_seq=0 表示尚未过期（引用式增量中的“无限”标记）；后续目标版本
    # 发现文件变化或删除时，再将它更新为目标 version_seq。
    return {
        "scenario_id": scenario.scenario_id,
        "kb_version": kb_version or "",
        "valid_from_seq": resolved_seq,
        "valid_to_seq": 0,
        "embedding_model_version": settings.embedding_model_version,
        "reranker_model_version": settings.reranker_model_version,
        "chunk_schema_version": settings.chunk_schema_version,
    }
