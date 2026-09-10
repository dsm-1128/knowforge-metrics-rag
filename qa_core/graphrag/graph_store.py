"""JSON 文件持久化的 GraphRAG 索引构建与路径检索服务。

GraphRAGStore 是 V2.1 GraphRAG 闭包的核心实现，负责：
  1. 从 FAQ CSV 和文档文件中读取资料，抽取实体与关系，构建图谱索引。
  2. 将图谱索引以 JSON 文件持久化到本地磁盘。
  3. 接收用户查询，匹配相关实体，检索关系路径，融合生成回答证据。

设计要点：
  - 不依赖任何图数据库，所有数据存储为 JSON 文件，挂掉不丢数据。
  - 每次 rebuild 重新构建完整索引，不维护增量更新（资料量级小，重建成本低）。
  - 路径检索采用基于邻居发现的广度优先策略，而非 Dijkstra/BFS 全图搜索。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from qa_core.common import read_json, utc_now, write_json
from qa_core.config.settings import PROJECT_ROOT
from qa_core.graphrag.extractor import (
    build_entity,
    extract_entity_names,
    extract_relations,
    faq_rows,
    readable_text_from_file,
)
from qa_core.graphrag.schema import GraphEntity, GraphIndex, GraphPath, GraphRelation
from qa_core.scenarios.registry import resolve_scenario


# GraphRAG 索引文件的根目录，位于 reports/v2/graphrag 下。
GRAPH_ROOT = Path("reports") / "v2" / "graphrag"


class GraphRAGStore:
    """V2.1 GraphRAG 的轻量持久化图存储。

    提供三个核心 public 方法：
      - rebuild(): 读取场景资料，完整重建图谱索引。
      - query(): 针对用户问题检索图谱路径并融合答案。
      - list_paths(): 列出场景中代表性高的关系路径。

    使用示例:
        store = GraphRAGStore()
        store.rebuild("it_helpdesk")
        result = store.query("VPN 故障和账号权限之间有什么排查路径？")

    调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore。
    """

    def __init__(self, root: Path | None = None) -> None:
        """初始化 GraphRAG 存储。

        参数:
            root: 图谱索引文件的存储根目录。默认使用 GRAPH_ROOT。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore.__init__()。
        """
        self.root = root or GRAPH_ROOT
        # 确保存储目录存在
        self.root.mkdir(parents=True, exist_ok=True)

    def rebuild(self, scenario_id: str | None = None, *, max_files: int = 200) -> dict[str, Any]:
        """读取指定场景的资料，完整重建图谱索引。

        流程：
          1. 解析场景配置，获取 FAQ CSV 路径和数据目录。
          2. 逐行读取 FAQ，对每行抽取实体和关系，合并到全局列表。
          3. 遍历场景数据目录下的所有文件，对每个文件抽取实体和关系。
          4. 按证据数和关系支持度排序，构建最终的 GraphIndex。
          5. 将索引写入 JSON 文件并返回。

        参数:
            scenario_id: 场景 ID。为 None 时使用默认场景。
            max_files: 最大处理文件数，防止一次处理过多文件。

        返回:
            包含完整图谱索引和摘要统计的字典。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore.rebuild()。
        """
        scenario = resolve_scenario(scenario_id)
        # 全局去重容器：实体按名称去重，关系按 relation_id（source+relation+target）去重
        entities_by_name: dict[str, GraphEntity] = {}
        relation_map: dict[str, GraphRelation] = {}
        sources: list[dict[str, Any]] = []
        # 收集场景特有的补充术语（场景名称、来源标签等），用于增强实体抽取覆盖
        extra_terms = [*scenario.valid_sources, *scenario.source_labels.values(), scenario.display_name]

        # ---- 第一阶段：处理 FAQ 数据（高频来源，先处理确保主要实体被收录） ----
        for index, (question, answer) in enumerate(faq_rows(Path(scenario.faq_csv_path)), start=1):
            source_ref = f"faq:{scenario.scenario_id}:{index}"
            text = f"{question}\n{answer}"
            self._merge_text(
                text,
                source_ref=source_ref,
                source_type="faq",
                entities_by_name=entities_by_name,
                relation_map=relation_map,
                sources=sources,
                extra_terms=extra_terms,
            )

        # ---- 第二阶段：处理文档文件（文件系统中的 PDF/MD/TXT 等） ----
        data_root = Path(scenario.data_root)
        if data_root.exists():
            for file_index, path in enumerate(sorted(data_root.rglob("*")), start=1):
                # max_files 保护：控制单次重建处理的文件数量上限
                if file_index > max_files:
                    break
                if not path.is_file():
                    continue
                source_ref = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
                text = readable_text_from_file(path)
                self._merge_text(
                    text,
                    source_ref=source_ref,
                    source_type="document",
                    entities_by_name=entities_by_name,
                    relation_map=relation_map,
                    sources=sources,
                    extra_terms=[*extra_terms, path.stem],
                )

        # ---- 构建最终索引 ----
        graph = GraphIndex(
            index_id=f"graph_{scenario.scenario_id}_{uuid4().hex[:8]}",
            scenario_id=scenario.scenario_id,
            created_at=utc_now(),
            # 实体按 evidence_count 降序，同名按字母序
            entities=sorted(entities_by_name.values(), key=lambda item: (-item.evidence_count, item.name)),
            # 关系按 support 降序，同分按起点+终点字母序
            relations=sorted(relation_map.values(), key=lambda item: (-item.support, item.source, item.target)),
            # 仅保留最近 500 条来源摘要，避免索引文件过大
            sources=sources[-500:],
        )
        payload = {
            **graph.as_dict(),
            "summary": {
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
                "source_count": len(graph.sources),
            },
        }
        write_json(self._path(scenario.scenario_id), payload)
        return payload

    def query(self, question: str, scenario_id: str | None = None, *, max_paths: int = 5) -> dict[str, Any]:
        """针对用户问题在图谱中检索关系路径并融合答案。

        流程：
          1. 加载（或按需重建）场景的图谱索引。
          2. 从问题中抽取候选实体名称。
          3. 匹配图谱中已有的实体，筛选出命中的实体。
          4. 基于命中的实体，检索经过这些实体的关系路径。
          5. 若无检索结果，降级为展示全图代表性路径。
          6. 融合路径信息，生成回答结论和后续行动建议。

        参数:
            question: 用户问题文本。
            scenario_id: 场景 ID。
            max_paths: 返回的最大路径数，默认 5。

        返回:
            包含查询实体、匹配结果、路径集合和融合答案的完整结果字典。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore.query()。
        """
        scenario = resolve_scenario(scenario_id)
        # 加载或自动重建图谱
        graph = self._load_or_rebuild(scenario.scenario_id)
        entities = [GraphEntity(**item) for item in graph.get("entities", [])]
        relations = [GraphRelation(**item) for item in graph.get("relations", [])]
        # 从问题中抽取查询实体
        query_entities = extract_entity_names(question, extra_terms=scenario.valid_sources)
        # 在图谱中匹配查询实体
        matched = self._match_entities(question, query_entities, entities)
        # 基于匹配实体检索路径
        paths = self._retrieve_paths(question, matched, relations, max_paths=max_paths)
        # 降级保护：若未检索到路径但有关系数据，取代表性路径
        if not paths and relations:
            paths = self._fallback_paths(relations, max_paths=max_paths)
        # 融合路径信息生成回答建议
        fused = self._fuse_answer(question, paths)
        result = {
            "mode": "graphrag",
            "scenario_id": scenario.scenario_id,
            "question": question,
            "query_entities": query_entities,
            "matched_entities": [item.as_dict() for item in matched],
            "paths": [path.as_dict() for path in paths],
            "path_count": len(paths),
            "fusion": fused,
            "graph_summary": graph.get("summary", {}),
            "index_id": graph.get("index_id", ""),
        }
        # 将查询记录追加到查询历史文件
        self._append_query(result)
        return result

    def list_paths(self, scenario_id: str | None = None, *, limit: int = 20) -> dict[str, Any]:
        """列出场景中代表性较高的关系路径。

        用于图谱可视化或人工审查，不依赖具体查询问题。
        通过 _fallback_paths 选取 top-N 关系展示。

        参数:
            scenario_id: 场景 ID。
            limit: 最大返回路径数（1~100），默认 20。

        返回:
            包含路径列表和统计摘要的字典。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore.list_paths()。
        """
        scenario = resolve_scenario(scenario_id)
        graph = self._load_or_rebuild(scenario.scenario_id)
        relations = [GraphRelation(**item) for item in graph.get("relations", [])]
        paths = self._fallback_paths(relations, max_paths=max(1, min(limit, 100)))
        return {
            "scenario_id": scenario.scenario_id,
            "items": [path.as_dict() for path in paths],
            "count": len(paths),
            "graph_summary": graph.get("summary", {}),
            "index_id": graph.get("index_id", ""),
        }

    def _merge_text(
        self,
        text: str,
        *,
        source_ref: str,
        source_type: str,
        entities_by_name: dict[str, GraphEntity],
        relation_map: dict[str, GraphRelation],
        sources: list[dict[str, Any]],
        extra_terms: list[str],
    ) -> None:
        """将一段文本中的实体和关系合并到全局容器中。

        核心去重逻辑：
          - 实体按 name 去重：相同名称的实体合并 source_refs 并增加 evidence_count。
          - 关系按 relation_id 去重：相同 ID 的关系增加 support 计数。

        参数:
            text: 待处理的文本内容。
            source_ref: 来源引用标识。
            source_type: 来源类型（faq/document）。
            entities_by_name: 全局实体字典（name -> GraphEntity），会被原地修改。
            relation_map: 全局关系字典（relation_id -> GraphRelation），会被原地修改。
            sources: 来源摘要列表，会被追加。
            extra_terms: 用于辅助实体抽取的额外术语。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._merge_text()。
        """
        # 从文本中抽取实体名称（领域术语匹配 + 中文词元关键词匹配）
        names = extract_entity_names(text, extra_terms=extra_terms)
        if not names:
            return
        # 为每个名称构建实体对象（自动推断 entity_type）
        local_entities = [build_entity(name, source_ref) for name in names]
        # 合并实体：若名称已存在，合并 source_refs 并递增 evidence_count
        # 同一来源多次出现只记录一次 source_ref 但 evidence_count 加1
        for entity in local_entities:
            existing = entities_by_name.get(entity.name)
            if existing:
                refs = sorted({*existing.source_refs, source_ref})
                entities_by_name[entity.name] = GraphEntity(
                    entity_id=existing.entity_id,
                    name=existing.name,
                    entity_type=existing.entity_type,
                    source_refs=refs,
                    evidence_count=existing.evidence_count + 1,
                )
            else:
                entities_by_name[entity.name] = entity
        # 抽取关系并去重合并：相同 relation_id 的关系递增 support 计数
        for relation in extract_relations(text, local_entities, source_ref):
            existing = relation_map.get(relation.relation_id)
            if existing:
                # 已有相同关系（同 source + relation + target），递增 support
                relation_map[relation.relation_id] = GraphRelation(
                    relation_id=existing.relation_id,
                    source=existing.source,
                    relation=existing.relation,
                    target=existing.target,
                    evidence=existing.evidence,
                    source_ref=existing.source_ref,
                    support=existing.support + 1,
                )
            else:
                relation_map[relation.relation_id] = relation
        # 记录来源摘要（用于溯源和审计），保留前 240 字符作为预览
        sources.append(
            {
                "source_ref": source_ref,
                "source_type": source_type,
                "entity_names": names[:10],
                "preview": (text or "")[:240],
            }
        )

    def _match_entities(
        self,
        question: str,
        query_entities: list[str],
        entities: list[GraphEntity],
    ) -> list[GraphEntity]:
        """将查询实体与图谱中的实体进行匹配。

        匹配条件（满足任一即可）：
          - 实体名称直接在问题文本中出现。
          - 实体名称在 query_entities 列表中（即被抽取为候选）。
          - 查询候选名称为实体名称的子串（模糊匹配）。

        参数:
            question: 用户问题文本。
            query_entities: 从问题中抽取的候选实体名称列表。
            entities: 图谱中所有实体列表。

        返回:
            匹配上的实体列表，按 evidence_count 降序排列，最多 8 个。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._match_entities()。
        """
        text = question
        matched: list[GraphEntity] = []
        for entity in entities:
            # 匹配条件（满足任一即可）：
            # 1. 实体名称直接出现在问题文本中
            # 2. 实体名称在 query_entities 抽取结果中
            # 3. 查询候选名称为实体名称的子串（模糊匹配兜底）
            if entity.name in text or entity.name in query_entities or any(name in entity.name for name in query_entities):
                matched.append(entity)
        # 按 evidence_count 降序排列，取前 8 个（证据越多的实体越可信）
        return sorted(matched, key=lambda item: (-item.evidence_count, item.name))[:8]

    def _retrieve_paths(
        self,
        question: str,
        matched: list[GraphEntity],
        relations: list[GraphRelation],
        *,
        max_paths: int,
    ) -> list[GraphPath]:
        """基于匹配实体检索关系路径。

        检索策略：
          1. 为所有关系建立按节点索引（起点和终点均可检索）。
          2. 遍历每个匹配实体，查找与其直接相连的关系。
          3. 为每条关系计算综合评分：基础分 0.6 + support 加成 + 两端匹配加成 + 问题匹配加成。
          4. 按评分降序返回 top-N 路径。

        参数:
            question: 用户问题文本（用于计算关系名匹配加分）。
            matched: 匹配到的实体列表。
            relations: 图谱中所有关系列表。
            max_paths: 最大返回路径数。

        返回:
            检索到的 GraphPath 列表，按评分降序排列。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._retrieve_paths()。
        """
        names = {entity.name for entity in matched}
        # 构建节点 -> 关系列表的倒排索引（双向：起点和终点均可检索）
        by_node: dict[str, list[GraphRelation]] = defaultdict(list)
        for relation in relations:
            by_node[relation.source].append(relation)
            by_node[relation.target].append(relation)
        paths: list[GraphPath] = []
        for name in names:
            for relation in by_node.get(name, []):
                nodes = [relation.source, relation.target]
                # 综合评分算法：
                # 基础分 0.6 + support 贡献（每个 support +0.06，最多 5 个 support 贡献 0.3）
                score = 0.6 + min(relation.support, 5) * 0.06
                # 两端实体均命中查询：说明路径与问题高度相关，额外加 0.15
                if relation.source in names and relation.target in names:
                    score += 0.15
                # 关系类型（requires/approves 等）出现在问题中：额外加 0.1
                if relation.relation in question:
                    score += 0.1
                paths.append(
                    GraphPath(
                        path_id=f"path_{len(paths) + 1}",
                        nodes=nodes,
                        relations=[relation.relation],
                        evidence=[relation.evidence],
                        score=round(min(score, 0.99), 3),
                        explanation=f"{relation.source} 通过 {relation.relation} 关系连接到 {relation.target}。",
                    )
                )
                # max_paths 限制：收集到足够路径后提前返回
                if len(paths) >= max_paths:
                    return paths
        # 按评分降序排列，截取 top-N
        return sorted(paths, key=lambda item: item.score, reverse=True)[:max_paths]

    def _fallback_paths(self, relations: list[GraphRelation], *, max_paths: int) -> list[GraphPath]:
        """降级路径检索：直接取 support 最高的前 N 条关系作为路径。

        当 query() 中实体匹配为空或检索不到路径时，调用此方法
        确保始终有路径数据可以展示，避免空响应。

        参数:
            relations: 图谱中所有关系列表（预期已按 support 降序排列）。
            max_paths: 最大返回路径数。

        返回:
            降级路径列表（无查询特定匹配时使用）。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._fallback_paths()。
        """
        paths: list[GraphPath] = []
        # 直接取 support 最高的前 N 条关系作为降级路径
        # 无查询特定匹配时使用，确保始终有路径数据可以展示
        for relation in relations[:max_paths]:
            paths.append(
                GraphPath(
                    path_id=f"path_{len(paths) + 1}",
                    nodes=[relation.source, relation.target],
                    relations=[relation.relation],
                    evidence=[relation.evidence],
                    # 降级路径评分较低（基础 0.5），但仍保留 support 区分度
                    score=round(0.5 + min(relation.support, 5) * 0.05, 3),
                    explanation=f"图谱中存在 {relation.source} -> {relation.target} 的 {relation.relation} 关系。",
                )
            )
        return paths

    def _fuse_answer(self, question: str, paths: list[GraphPath]) -> dict[str, Any]:
        """融合路径信息，生成面向用户的回答结论和建议。

        参数:
            question: 用户问题文本。
            paths: 检索到的路径列表（可能为空）。

        返回:
            包含 status（状态）、summary（摘要）和 next_action（建议行动）的字典。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._fuse_answer()。
        """
        if not paths:
            return {
                "status": "insufficient_graph_context",
                "summary": "当前图谱没有找到足够的关系路径。",
                "next_action": "先重建图谱或补充关系型资料。",
            }
        top = paths[0]
        return {
            "status": "path_found",
            "summary": f"问题“{question}”命中 {len(paths)} 条关系路径，首要路径为：{' -> '.join(top.nodes)}。",
            "next_action": "结合路径证据和文本检索结果做最终回答。",
            "top_path_id": top.path_id,
            "top_score": top.score,
        }

    def _load_or_rebuild(self, scenario_id: str) -> dict[str, Any]:
        """加载场景的图谱索引 JSON，若不存在或为空则自动重建。

        参数:
            scenario_id: 场景 ID。

        返回:
            图谱索引字典。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._load_or_rebuild()。
        """
        path = self._path(scenario_id)
        payload = read_json(path, default={})
        if isinstance(payload, dict) and payload.get("entities"):
            return payload
        return self.rebuild(scenario_id)

    def _append_query(self, result: dict[str, Any]) -> None:
        """将一次查询记录追加到查询历史 JSON 文件。

        仅保留最近 300 条查询记录，用于审计和问题复现。
        查询历史文件位于 reports/v2/graphrag/queries.json。

        参数:
            result: 查询结果字典（包含问题、路径、答案等信息）。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._append_query()。
        """
        path = self.root / "queries.json"
        items = read_json(path, default=[])
        if not isinstance(items, list):
            items = []
        items.append({**result, "queried_at": utc_now()})
        write_json(path, items[-300:])

    def _path(self, scenario_id: str) -> Path:
        """返回指定场景的索引 JSON 文件路径。

        路径格式: reports/v2/graphrag/{scenario_id}_graph.json

        参数:
            scenario_id: 场景 ID。

        返回:
            Path 对象。

        调用顺序：GraphRAG 入库或查询流程 -> GraphRAGStore._path()。
        """
        return self.root / f"{scenario_id}_graph.json"


def get_graphrag_store() -> GraphRAGStore:
    """获取默认的 GraphRAGStore 单例。

    返回:
        GraphRAGStore 实例。

    调用顺序：GraphRAG 入库或查询流程 -> get_graphrag_store()。
    """
    return GraphRAGStore()


def rebuild_graphrag_index(scenario_id: str | None = None, *, max_files: int = 200) -> dict[str, Any]:
    """便捷函数：重建指定场景的 GraphRAG 索引。

    参数:
        scenario_id: 场景 ID。为 None 时使用默认场景。
        max_files: 最大处理文件数。

    返回:
        完整的图谱索引数据字典。

    调用顺序：GraphRAG 入库或查询流程 -> rebuild_graphrag_index()。
    """
    return get_graphrag_store().rebuild(scenario_id, max_files=max_files)


def query_graphrag(question: str, scenario_id: str | None = None, *, max_paths: int = 5) -> dict[str, Any]:
    """便捷函数：查询 GraphRAG 图谱路径。

    参数:
        question: 用户问题。
        scenario_id: 场景 ID。
        max_paths: 最大返回路径数。

    返回:
        包含查询结果和路径的字典。

    调用顺序：GraphRAG 入库或查询流程 -> query_graphrag()。
    """
    return get_graphrag_store().query(question, scenario_id, max_paths=max_paths)


def list_graphrag_paths(scenario_id: str | None = None, *, limit: int = 20) -> dict[str, Any]:
    """便捷函数：列出场景中的 GraphRAG 关系路径。

    参数:
        scenario_id: 场景 ID。
        limit: 最大返回路径数。

    返回:
        包含路径列表和统计的字典。

    调用顺序：GraphRAG 入库或查询流程 -> list_graphrag_paths()。
    """
    return get_graphrag_store().list_paths(scenario_id, limit=limit)
