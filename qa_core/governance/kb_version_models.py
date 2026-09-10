"""知识库版本的数据结构和字段序列化工具。

定义版本生命周期状态常量和版本数据模型。所有数据库 JSON 字段通过统一的
json_dumps/json_loads_dict/json_loads_list 进行序列化和反序列化，确保
JSON 格式一致（ensure_ascii=False, sort_keys=True）且坏数据有容错处理。

版本状态机：STAGED（已创建/暂存） -> ACTIVE（在线检索） -> ARCHIVED（归档只读）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from qa_core.common import utc_now

KB_VERSION_STATUS_STAGED = "STAGED"
KB_VERSION_STATUS_ACTIVE = "ACTIVE"
KB_VERSION_STATUS_ARCHIVED = "ARCHIVED"
KB_VERSION_SELECT_COLUMNS = (
    "scenario_id, kb_version, version_seq, status, description, created_at, "
    "activated_at, archived_at, doc_collection, faq_collection, "
    "embedding_model_version, reranker_model_version, chunk_schema_version, "
    "created_by, sources_json, stats_json"
)


def json_dumps(value: Any) -> str:
    """按项目统一方式序列化 JSON 字段。（★★ 理解）

    参数：
        value: 要序列化的值（None 或空值也会被转换为 {}）。

    返回：
        统一格式的 JSON 字符串（ensure_ascii=False, sort_keys=True）。

    调用顺序：KnowledgeBaseVersionStore._upsert_version_with_conn() -> json_dumps()。
    """
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def json_loads_dict(value: Any) -> dict[str, Any]:
    """把数据库 JSON/TEXT 字段恢复为 dict，坏数据按空字典处理。

    参数：
        value: 数据库读取的 JSON 值（可能是 dict、str 或 None）。

    返回：
        解析后的 dict，解析失败时返回空字典。

    调用顺序：KnowledgeBaseVersion.from_row() -> json_loads_dict()。
    """
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def json_loads_list(value: Any) -> list[str]:
    """把数据库 JSON/TEXT 字段恢复为字符串列表。

    参数：
        value: 数据库读取的 JSON 值（可能是 list、str 或 None）。

    返回：
        解析后的字符串列表，解析失败时返回空列表。

    调用顺序：KnowledgeBaseVersion.from_row() -> json_loads_list()。
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


@dataclass
class KnowledgeBaseVersion:
    """可检索知识库版本的元数据。（★★★ 核心）

    表示一个版本的完整快照，包含：
    - 标识信息：kb_version（版本号）、version_seq（单调序列号）、scenario_id（场景）。
    - 状态追踪：status（STAGED/ACTIVE/ARCHIVED）、activated_at、archived_at。
    - 配置快照：doc_collection、faq_collection、embedding_model_version 等。
    - 统计信息：sources（数据来源）、stats（入库统计字典）。

    调用顺序：KnowledgeBaseVersionStore -> KnowledgeBaseVersion。
    """

    kb_version: str
    scenario_id: str = ""
    version_seq: int = 0
    status: str = KB_VERSION_STATUS_STAGED
    description: str = ""
    created_at: str = field(default_factory=utc_now)
    activated_at: str | None = None
    archived_at: str | None = None
    doc_collection: str = ""
    faq_collection: str = ""
    embedding_model_version: str = ""
    reranker_model_version: str = ""
    chunk_schema_version: str = ""
    created_by: str = "local"
    sources: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeBaseVersion":
        """从 dict 恢复版本对象，老记录缺失的字段自动使用默认值。

        参数：
            payload: 包含版本字段的字典（可能缺失某些可选字段）。

        返回：
            KnowledgeBaseVersion 实例，缺失字段使用 dataclass 默认值。

        调用顺序：from_row() -> from_dict()。
        """
        fields = cls.__dataclass_fields__
        data = {name: payload.get(name) for name in fields if name in payload}
        version = cls(**data)
        # 容错处理：老记录的 sources/stats 可能被显示设为 None 而不是空列表/字典
        if version.sources is None:
            version.sources = []
        if version.stats is None:
            version.stats = {}
        version.version_seq = int(version.version_seq or 0)
        return version

    @classmethod
    def from_row(cls, row: Any) -> "KnowledgeBaseVersion":
        """从 SQLAlchemy RowMapping 恢复版本对象。

        从行记录中提取字段，将 JSON 串字段（sources_json/stats_json）解析为
        原生 Python 对象后，传入 from_dict() 构建版本对象。

        参数：
            row: SQLAlchemy 行映射对象（RowMapping），包含 KB_VERSION_SELECT_COLUMNS 对应字段。

        返回：
            KnowledgeBaseVersion 实例。

        调用顺序：KnowledgeBaseVersionStore.list_versions()/get() -> from_row()。
        """
        payload = dict(row)
        payload["sources"] = json_loads_list(payload.get("sources_json"))
        payload["stats"] = json_loads_dict(payload.get("stats_json"))
        payload.pop("sources_json", None)
        payload.pop("stats_json", None)
        return cls.from_dict(payload)

    def as_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的版本信息。

        使用 dataclasses.asdict() 将 dataclass 递归转换为普通 dict。

        返回：
            包含完整版本字段的字典。

        调用顺序：管理 API/脚本 -> KnowledgeBaseVersion.as_dict()。
        """
        return asdict(self)
