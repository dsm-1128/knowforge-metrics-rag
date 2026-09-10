"""增量文档索引用的 MySQL 清单。

清单记录本地文件指纹和对应 Milvus chunk id。增量入库时通过它判断文件是否需要重建，
文件删除清理时也通过它定位旧 chunk。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from qa_core.common import utc_now
from qa_core.memory.base import _MySqlStore, safe_sql_identifier
from qa_core.utils import stable_hash

INDEX_MANIFEST_TABLE = "kb_document_manifests"
MANIFEST_SELECT_COLUMNS = (
    "manifest_key, scenario_id, source, path, fingerprint, "
    "chunk_ids_json, updated_at, kb_version, "
    "embedding_model_version, chunk_schema_version"
)


def _json_dumps_list(value: list[str]) -> str:
    """序列化 chunk id 列表。

    调用顺序：入库脚本或索引服务 -> _json_dumps_list()。
    """
    return json.dumps(value or [], ensure_ascii=False)


def _json_loads_list(value: Any) -> list[str]:
    """反序列化 chunk id 列表。

    调用顺序：入库脚本或索引服务 -> _json_loads_list()。
    """
    if isinstance(value, list):
        # SQLAlchemy RowMapping 已自动解析 JSON 列的情况，直接转字符串列表
        return [str(item) for item in value]
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        # JSON 解析失败时返回空列表而非抛出异常，保证查询历史脏数据时不会阻断入口流程
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


@dataclass
class ManifestRecord:
    """一个已入库文件及其对应的 Milvus chunk id 列表。

    manifest 是“文件级索引”，不是 chunk 正文：
      - `fingerprint` 用来判断磁盘文件是否变化；
      - `chunk_ids` 用来定位目标版本或基线版本对应的向量；
      - embedding/chunk schema 版本用来判断旧结果是否仍兼容当前处理配置。

    调用顺序：`IndexManifest.get()` / `iter_records()`
    -> `ManifestRecord`。
    """

    key: str
    scenario_id: str
    source: str
    path: str
    fingerprint: str
    chunk_ids: list[str]
    updated_at: str
    kb_version: str = ""
    embedding_model_version: str = ""
    chunk_schema_version: str = ""

    @classmethod
    def from_row(cls, row: Any) -> "ManifestRecord":
        """从 SQLAlchemy RowMapping 恢复 manifest 记录。

        数据库中的 chunk id 以 JSON 字符串保存，读取时统一还原成
        `list[str]`，让上层增量逻辑不必感知 SQL 类型和 JSON 解析细节。

        调用顺序：入库脚本或索引服务 -> ManifestRecord.from_row()。
        """
        payload = dict(row)
        payload["key"] = payload.pop("manifest_key")
        payload["chunk_ids"] = _json_loads_list(payload.pop("chunk_ids_json", "[]"))
        return cls(**payload)


class IndexManifest(_MySqlStore):
    """供入库脚本使用的 MySQL 清单。

    每条记录的逻辑主键由 `scenario_id + source + kb_version + 绝对路径`
    共同决定。同一物理文件在不同知识库版本中必须拥有不同 manifest，
    因为目标版本可能复用基线 chunk，也可能为同一文件生成一组新 chunk。

    调用顺序：入库脚本或索引服务 -> IndexManifest。
    """

    def __init__(self) -> None:
        """初始化 MySQL manifest。表结构由 bootstrap 统一初始化。

        调用顺序：入库脚本或索引服务 -> IndexManifest.__init__()。
        """
        super().__init__()
        self.table_name = safe_sql_identifier(INDEX_MANIFEST_TABLE, label="Index manifest table")

    @staticmethod
    def key(
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> str:
        """根据来源和绝对文件路径生成稳定的清单键。

        使用绝对路径是为了消除当前工作目录不同造成的重复记录；加入场景、
        source 和版本是为了避免不同业务分类或不同版本共享同一文件名时互相覆盖。
        返回稳定 hash 而不是直接拼接路径，既控制键长度，也避免路径中的特殊字符
        进入数据库主键。

        调用顺序：入库脚本或索引服务 -> IndexManifest.key()。
        """
        return stable_hash(scenario_id or "", source, kb_version or "", str(Path(file_path).resolve()))

    def get(
        self,
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> ManifestRecord | None:
        """如果文件曾经入库，返回对应清单记录。

        `kb_version` 决定查询的是目标版本还是增量基线版本。调用方通常会先
        查询目标记录，再按需查询基线记录，两个查询结果共同决定“跳过、复用还是重建”。
        找不到记录返回 `None` 是正常的新增文件信号，不表示数据库异常。

        调用顺序：入库脚本或索引服务 -> IndexManifest.get()。
        """
        key = self.key(source, file_path, kb_version, scenario_id)
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    f"""
                    SELECT {MANIFEST_SELECT_COLUMNS}
                    FROM {self.table_name}
                    WHERE manifest_key = :manifest_key
                    """
                ),
                {"manifest_key": key},
            ).mappings().fetchone()
        # 返回 None 表示该文件从未入库或在当前 kb_version 下没有记录，调用方据此决定是增量跳过还是全量重建
        return ManifestRecord.from_row(row) if row else None

    def update(
        self,
        source: str,
        file_path: str | Path,
        fingerprint: str,
        chunk_ids: list[str],
        *,
        scenario_id: str = "",
        kb_version: str = "",
        embedding_model_version: str = "",
        chunk_schema_version: str = "",
    ) -> None:
        """记录一次成功入库及其生成的 chunk id。

        只有向量写入成功后才应调用本方法。写入采用 upsert，保证同一文件在
        同一版本重复处理时，manifest 始终指向最新 fingerprint 和 chunk ids。
        跨版本复用也要写目标版本 manifest，原因是“引用关系”本身就是目标版本
        的构建结果，下一次运行必须能够在目标版本命中它。

        调用顺序：入库脚本或索引服务 -> IndexManifest.update()。
        """
        key = self.key(source, file_path, kb_version, scenario_id)
        # 使用 INSERT ON DUPLICATE KEY 实现幂等更新：同一文件在同一版本下重跑时
        # 覆盖指纹和 chunk id，避免保留上一轮已经失效的结果。
        sql = f"""
        INSERT INTO {self.table_name}
            (
                manifest_key, scenario_id, source, path, fingerprint,
                chunk_ids_json, updated_at, kb_version,
                embedding_model_version, chunk_schema_version
            )
        VALUES
            (
                :manifest_key, :scenario_id, :source, :path, :fingerprint,
                :chunk_ids_json, :updated_at, :kb_version,
                :embedding_model_version, :chunk_schema_version
            )
        ON DUPLICATE KEY UPDATE
            scenario_id=VALUES(scenario_id),
            source=VALUES(source),
            path=VALUES(path),
            fingerprint=VALUES(fingerprint),
            chunk_ids_json=VALUES(chunk_ids_json),
            updated_at=VALUES(updated_at),
            kb_version=VALUES(kb_version),
            embedding_model_version=VALUES(embedding_model_version),
            chunk_schema_version=VALUES(chunk_schema_version),
            row_updated_at=CURRENT_TIMESTAMP
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(sql),
                {
                    "manifest_key": key,
                    "scenario_id": scenario_id,
                    "source": source,
                    "path": str(Path(file_path).resolve()),
                    "fingerprint": fingerprint,
                    "chunk_ids_json": _json_dumps_list(chunk_ids),
                    "updated_at": utc_now(),
                    "kb_version": kb_version,
                    "embedding_model_version": embedding_model_version,
                    "chunk_schema_version": chunk_schema_version,
                },
            )

    def remove(
        self,
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> ManifestRecord | None:
        """从清单中移除一个文件，并返回其旧记录。

        调用顺序：入库脚本或索引服务 -> IndexManifest.remove()。
        """
        key = self.key(source, file_path, kb_version, scenario_id)
        return self.remove_by_key(key)

    def iter_records(
        self,
        *,
        scenario_id: str | None = None,
        source: str | None = None,
        kb_version: str | None = None,
    ) -> list[ManifestRecord]:
        """按条件列出清单记录。支持按 scenario_id/source/kb_version 过滤。

        参数：
            scenario_id: 场景 ID 过滤条件。
            source: 业务分类过滤条件。
            kb_version: 知识库版本过滤条件。

        返回：
            匹配条件的 ManifestRecord 列表，按更新时间倒序排列。

        删除检测会按基线版本读取记录，然后和本次目录遍历得到的 `seen_paths`
        比较；因此这个查询不仅用于管理，还承担“磁盘文件删除感知”的控制面职责。

        调用顺序：入库脚本或索引服务 -> IndexManifest.iter_records()。
        """
        # ── 动态构建 WHERE 子句 ──
        filters: list[str] = []
        params: dict[str, Any] = {}
        if scenario_id:
            filters.append("scenario_id = :scenario_id")
            params["scenario_id"] = scenario_id
        if source:
            filters.append("source = :source")
            params["source"] = source
        if kb_version:
            filters.append("kb_version = :kb_version")
            params["kb_version"] = kb_version
        # 没有过滤条件时列出表中全部记录；正式入库会传入场景、source 和基线版本，
        # 把扫描范围限制在当前增量任务需要的集合内。
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        sql = f"""
        SELECT {MANIFEST_SELECT_COLUMNS}
        FROM {self.table_name}
        {where}
        ORDER BY updated_at DESC, manifest_key ASC
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [ManifestRecord.from_row(row) for row in rows]

    def remove_by_key(self, key: str) -> ManifestRecord | None:
        """按 manifest key 删除记录。

        先读后删是刻意设计：调用方可能需要旧记录中的 chunk ids 做向量清理
        或审计。如果只执行 DELETE，就会丢失后续清理所需的定位信息。

        调用顺序：入库脚本或索引服务 -> IndexManifest.remove_by_key()。
        """
        with self.engine.begin() as conn:
            # 先 SELECT 获取旧记录：调用方需要知道被删除的文件对应哪些 chunk id，用于后续清理 Milvus
            row = conn.execute(
                text(
                    f"""
                    SELECT {MANIFEST_SELECT_COLUMNS}
                    FROM {self.table_name}
                    WHERE manifest_key = :manifest_key
                    """
                ),
                {"manifest_key": key},
            ).mappings().fetchone()
            if not row:
                # 记录不存在时返回 None，调用方据此判断无需清理
                return None
            conn.execute(
                text(f"DELETE FROM {self.table_name} WHERE manifest_key = :manifest_key"),
                {"manifest_key": key},
            )
        return ManifestRecord.from_row(row)
