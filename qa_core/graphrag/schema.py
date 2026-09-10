"""GraphRAG 轻量图谱索引的数据模型。

定义 V2 GraphRAG 的核心数据类：图实体（GraphEntity）、图关系（GraphRelation）、
图路径（GraphPath）和图索引（GraphIndex）。所有数据类均声明为 frozen（不可变），
以确保图谱数据在构建后的引用安全性和可哈希性。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphEntity:
    """图谱中的实体节点。

    表示从文档或 FAQ 中抽取出的一个业务概念实体，例如"VPN 故障"、"入职流程"等。
    实体通过名称去重合并，相同的名称对应同一个实体 ID。

    属性:
        entity_id: 实体的唯一标识，由实体名称稳定哈希生成。
        name: 实体名称，例如"VPN 故障"、"报销流程"。
        entity_type: 实体类型分类（process/policy/incident/asset/concept）。
        source_refs: 该实体出现的资料来源引用列表（来源路径或 FAQ 行号）。
        evidence_count: 该实体在多少处资料中被观测到，用于排序和评分。

    调用顺序：GraphRAG 入库或查询流程 -> GraphEntity。
    """
    entity_id: str
    name: str
    entity_type: str
    source_refs: list[str] = field(default_factory=list)
    evidence_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """将实体转换为字典，用于 JSON 序列化。

        调用顺序：GraphRAG 入库或查询流程 -> GraphEntity.as_dict()。
        """
        return asdict(self)


@dataclass(frozen=True)
class GraphRelation:
    """图谱中的关系边。

    表示两个实体之间的一种语义关系，例如"审批流程 需要 提交材料"。
    关系通过 relation_id 去重合并（同一对实体 + 同一关系类型视为同一条关系）。

    属性:
        relation_id: 关系的唯一标识，由 source + relation + target 三段拼接生成。
        source: 关系起点实体名称。
        relation: 关系类型标签（requires/approves/depends_on/troubleshoots/governs/related_to）。
        target: 关系终点实体名称。
        evidence: 支持该关系的原文证据片段（从文档句子中截取）。
        source_ref: 该关系首次出现的资料来源引用。
        support: 该关系被多少处不同资料支持，用于可信度评分。

    调用顺序：GraphRAG 入库或查询流程 -> GraphRelation。
    """
    relation_id: str
    source: str
    relation: str
    target: str
    evidence: str
    source_ref: str
    support: int = 1

    def as_dict(self) -> dict[str, Any]:
        """将关系转换为字典，用于 JSON 序列化。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRelation.as_dict()。
        """
        return asdict(self)


@dataclass(frozen=True)
class GraphPath:
    """图谱中的一条推理路径。

    表示从查询问题出发，在图谱中检索到的一条连贯关系链，
    用于为最终答案提供可解释的图谱证据。

    属性:
        path_id: 路径的唯一标识。
        nodes: 路径上经过的实体节点名称列表，按访问顺序排列。
        relations: 路径上经过的关系类型标签列表。
        evidence: 路径上各关系对应的原文证据片段列表。
        score: 路径的综合可信度评分（0~1），越高越可靠。
        explanation: 自然语言描述的路径解释，用于人类阅读。

    调用顺序：GraphRAG 入库或查询流程 -> GraphPath。
    """
    path_id: str
    nodes: list[str]
    relations: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    score: float = 0.0
    explanation: str = ""

    def as_dict(self) -> dict[str, Any]:
        """将路径转换为字典，用于 JSON 序列化。

        调用顺序：GraphRAG 入库或查询流程 -> GraphPath.as_dict()。
        """
        return asdict(self)


@dataclass(frozen=True)
class GraphIndex:
    """一次完整的图谱索引结果。

    包含对某个业务场景下所有资料进行实体和关系抽取后的完整图谱快照。
    每次 rebuild 生成一个新的 GraphIndex，通过 index_id 区分。

    属性:
        index_id: 索引的唯一标识，由场景 ID 和 UUID 前缀拼接而成。
        scenario_id: 所属业务场景 ID。
        created_at: 索引创建时间（UTC ISO 格式）。
        entities: 抽取出的所有实体列表，按 evidence_count 降序排列。
        relations: 抽取出的所有关系列表，按 support 降序排列。
        sources: 被索引的资料来源摘要列表，每个来源包含引用路径、类型和预览。

    调用顺序：GraphRAG 入库或查询流程 -> GraphIndex。
    """
    index_id: str
    scenario_id: str
    created_at: str
    entities: list[GraphEntity] = field(default_factory=list)
    relations: list[GraphRelation] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """将索引转换为字典，用于 JSON 序列化。

        调用顺序：GraphRAG 入库或查询流程 -> GraphIndex.as_dict()。
        """
        return asdict(self)
