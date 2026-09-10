"""确定性实体和关系抽取器 —— V2 GraphRAG 的轻量抽取引擎。

设计原则：轻量、可复现、无外部依赖。
  1. 不需要图数据库：直接使用内存字典和 JSON 文件持久化。
  2. 不需要 LLM：基于领域术语词典和正则规则的确定性匹配。
  3. 可替换接口：后续若需要引入模型基抽取器，只需替换本模块的 public 函数签名即可。

抽取流程：
  1. 从 FAQ CSV 和文档文件中读取文本内容。
  2. 通过领域术语匹配 + 中文关键词匹配抽取候选实体名称。
  3. 根据实体名称中是否包含流程/制度/故障等关键词判定实体类型。
  4. 对句子按标点分割后，对同句中出现的多实体两两组合抽取关系。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from qa_core.graphrag.schema import GraphEntity, GraphRelation

# 领域核心术语词典 —— 覆盖 HR、财务、合规、IT 等常见企业场景。
# 这些术语会优先被识别为图实体候选。
DOMAIN_TERMS = (
    "VPN",
    "账号",
    "权限",
    "审批",
    "工单",
    "新人",
    "入职",
    "试用期",
    "转正",
    "报销",
    "预算",
    "发票",
    "合同",
    "合规",
    "隐私",
    "个人信息",
    "数据",
    "审计",
    "供应商",
    "整改",
    "设备",
    "告警",
    "巡检",
    "安全",
    "理赔",
    "保单",
    "免责",
    "工程",
    "图纸",
    "质量",
    "进度",
)

# 关系推断模式：每条关系类型对应一组中文触发关键词。
# 在句子中匹配到关键词时，将该句子的实体对标记为对应的关系类型。
RELATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("requires", r"需要|必须|应当|提交|提供|确认"),      # 依赖/必要条件
    ("approves", r"审批|批准|复核|审核"),                # 审批关系
    ("depends_on", r"依赖|影响|关联|前置|条件"),          # 前置依赖
    ("troubleshoots", r"故障|排查|修复|恢复|重启|检查"),  # 故障处理
    ("governs", r"制度|规范|规则|合规|禁止|不得"),         # 制度治理
)

# 中文句子分隔符：按句号、问号、感叹号、分号、中文分号和换行符切分句子。
SENTENCE_SPLIT = re.compile(r"[。！？；;\n\r]+")
# 常见分隔符分词模式：用于从复杂文件名或文本中切分词元。
TOKEN_SPLIT = re.compile(r"[_\-\s./\\]+")


def readable_text_from_file(path: Path, *, max_chars: int = 6000) -> str:
    """从文件路径中读取可读文本内容。

    策略：对于已知文本格式（md/txt/csv/toml/json）直接读取前 max_chars 字符；
    对于二进制文件（图片/PDF 等），使用文件名的最后三级路径作为替代文本描述。

    参数:
        path: 文件路径。
        max_chars: 文本文件最大读取字符数，默认 6000。

    返回:
        可读文本字符串（二进制文件返回路径描述）。

    调用顺序：GraphRAG 入库或查询流程 -> readable_text_from_file()。
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".csv", ".toml", ".json"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except OSError:
            return ""
    # 二进制文件：用文件名组合作为文本替代，提取最后三级路径
    return " ".join(path.with_suffix("").parts[-3:])


def faq_rows(path: Path) -> Iterable[tuple[str, str]]:
    """读取 FAQ CSV 文件，逐行返回 (问题, 答案) 元组。

    兼容常见的列名命名方式：question/standard_question/q、
    answer/a，按优先级依次尝试读取。仅使用标准库 csv 模块，无第三方依赖。

    参数:
        path: CSV 文件路径。

    返回:
        (问题, 答案) 元组的可迭代对象。文件不存在或读取失败时返回空列表。

    调用顺序：GraphRAG 入库或查询流程 -> faq_rows()。
    """
    if not path.exists():
        return []
    rows: list[tuple[str, str]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                # 按优先级尝试不同的列名，兼容中英文列名
                question = str(row.get("question") or row.get("standard_question") or row.get("q") or "")
                answer = str(row.get("answer") or row.get("a") or "")
                if question or answer:
                    rows.append((question, answer))
    except OSError:
        return []
    return rows


def extract_entity_names(text: str, *, extra_terms: Iterable[str] = ()) -> list[str]:
    """从文本中抽取候选实体名称。

    抽取策略分两轮：
      第一轮：在文本中搜索领域术语词典（DOMAIN_TERMS）和额外术语（extra_terms）
              中的精确匹配，命中即收录为候选实体。
      第二轮：将文本按常见分隔符分词后，对每个中文词元检查是否包含
              流程/制度/审批等业务关键词，符合条件的收录为补充候选。

    参数:
        text: 待抽取的文本内容。
        extra_terms: 该场景特有的补充术语列表（如场景名称、来源标签等）。

    返回:
        抽取到的候选实体名称列表，最多 12 个，按发现顺序排列。

    调用顺序：GraphRAG 入库或查询流程 -> extract_entity_names()。
    """
    candidates: list[str] = []
    haystack = text or ""
    # 第一轮：领域术语精确匹配（DOMAIN_TERMS + 场景特有术语）
    # 术语匹配不区分位置，只要文本中出现即收录为候选实体
    for term in [*DOMAIN_TERMS, *extra_terms]:
        cleaned = str(term).strip()
        if cleaned and cleaned in haystack and cleaned not in candidates:
            candidates.append(cleaned)
    # 第二轮：中文词元关键词匹配
    # 按常见分隔符（_ - / 空格等）拆分文本，对每个词元单独评估
    for token in TOKEN_SPLIT.split(haystack):
        normalized = token.strip(" ，。！？；:：()[]【】'\"")
        # 长度限制 2~12 且必须包含中文字符（排除标点和英文单词）
        if 2 <= len(normalized) <= 12 and re.search(r"[一-鿿]", normalized):
            # 仅包含业务关键词的词元才有资格成为实体（避免噪声实体）
            if any(keyword in normalized for keyword in ("流程", "制度", "规范", "审批", "故障", "权限", "合同", "数据")):
                if normalized not in candidates:
                    candidates.append(normalized)
    # 最多返回 12 个候选实体，按发现顺序排列（无权重排序）
    return candidates[:12]


def entity_type(name: str) -> str:
    """根据实体名称的关键词特征推断实体类型。

    规则判定优先级：process > policy > incident > asset > concept。

    参数:
        name: 实体名称。

    返回:
        实体类型标签：process（流程）/ policy（制度）/ incident（事件）/
                      asset（资产）/ concept（概念）。

    调用顺序：GraphRAG 入库或查询流程 -> entity_type()。
    """
    # 实体类型判定优先级：process > policy > incident > asset > concept
    # 流程类：包含"流程"、"审批"、"工单"等关键词 → 表示业务操作流程
    if any(keyword in name for keyword in ("流程", "审批", "工单")):
        return "process"
    # 制度类：包含"制度"、"规范"、"合规"等关键词 → 表示管理制度或规范
    if any(keyword in name for keyword in ("制度", "规范", "合规", "隐私", "合同")):
        return "policy"
    # 事件类：包含"故障"、"告警"、"VPN"、"设备"等关键词 → 表示 IT 事件或设备
    if any(keyword in name for keyword in ("故障", "告警", "VPN", "设备")):
        return "incident"
    # 资产类：包含"账号"、"权限"、"数据"等关键词 → 表示数字资产或信息资产
    if any(keyword in name for keyword in ("账号", "权限", "数据", "个人信息")):
        return "asset"
    # 默认降级：概念类（不含特定关键词的泛化实体）
    return "concept"


def build_entity(name: str, source_ref: str, evidence_count: int = 1) -> GraphEntity:
    """从实体名称快速构建 GraphEntity 对象。

    参数:
        name: 实体名称。
        source_ref: 来源引用（文件路径或 FAQ 行号）。
        evidence_count: 该实体的证据计数，默认 1。

    返回:
        GraphEntity 实例（自动计算 entity_id 和 entity_type）。

    调用顺序：GraphRAG 入库或查询流程 -> build_entity()。
    """
    return GraphEntity(
        entity_id=stable_entity_id(name),
        name=name,
        entity_type=entity_type(name),
        source_refs=[source_ref],
        evidence_count=evidence_count,
    )


def stable_entity_id(name: str) -> str:
    """从实体名称生成稳定的实体 ID。

    将名称中的非字母数字和中文字符替换为下划线，确保同一名称始终
    产生相同的 ID，从而在多次重建索引时保持实体一致性。

    参数:
        name: 实体名称。

    返回:
        形如 "entity_VPN_故障" 的稳定实体 ID。

    调用顺序：GraphRAG 入库或查询流程 -> stable_entity_id()。
    """
    safe = re.sub(r"[^0-9A-Za-z一-鿿]+", "_", name).strip("_")
    return f"entity_{safe[:40] or 'unknown'}"


def infer_relation(sentence: str) -> str:
    """根据句子中的触发关键词推断关系类型。

    遍历 RELATION_PATTERNS 中的模式，返回第一个匹配的关系类型。
    如果没有任何模式匹配，默认返回 "related_to"（泛关联）。

    参数:
        sentence: 待分析的句子文本。

    返回:
        关系类型标签（requires/approves/depends_on/troubleshoots/governs/related_to）。

    调用顺序：GraphRAG 入库或查询流程 -> infer_relation()。
    """
    for relation, pattern in RELATION_PATTERNS:
        if re.search(pattern, sentence or "", flags=re.IGNORECASE):
            return relation
    return "related_to"


def extract_relations(text: str, entities: list[GraphEntity], source_ref: str) -> list[GraphRelation]:
    """从文本中为给定的实体列表抽取关系。

    抽取逻辑：
      1. 将文本按句号等分隔符拆分为句子。
      2. 对每个句子，找出其中出现的实体名称。
      3. 如果同一句子中包含至少两个实体，则按出现顺序两两建立关系。
      4. 关系类型通过句子中的触发关键词推断。
      5. 如果没有任何句子包含多实体，且实体数 >= 2，则自动建立泛关联关系。

    参数:
        text: 待抽取的文本内容。
        entities: 在文本中已识别的实体列表。
        source_ref: 来源引用，标记关系的出处。

    返回:
        抽取到的 GraphRelation 列表。

    调用顺序：GraphRAG 入库或查询流程 -> extract_relations()。
    """
    # 构建实体名称到实体对象的快速查找字典（O(1) 查询）
    by_name = {entity.name: entity for entity in entities}
    relations: list[GraphRelation] = []
    # 按句子依次处理：句号/问号/感叹号/分号拆分，每个句子独立处理
    for sentence in SENTENCE_SPLIT.split(text or ""):
        # 找出该句子中出现的实体名称（顺序保持原始出现顺序）
        names = [name for name in by_name if name in sentence]
        # 同一个句子中至少需要出现两个实体才能建立关系
        if len(names) < 2:
            continue
        # 按句子中实体出现顺序两两组合建立关系（滑动窗口）
        for index in range(len(names) - 1):
            source = names[index]
            target = names[index + 1]
            relation = infer_relation(sentence)
            # 关系 ID 由 source + relation + target 三项组合，天然去重
            relation_id = f"rel_{by_name[source].entity_id}_{relation}_{by_name[target].entity_id}"
            relations.append(
                GraphRelation(
                    relation_id=relation_id,
                    source=source,
                    relation=relation,
                    target=target,
                    evidence=sentence.strip()[:300],
                    source_ref=source_ref,
                )
            )
    # 降级保护：没有任何句子含多实体时，用泛关联 related_to 兜底
    # 最多建立 3 条泛关联关系（防止噪声过多）
    if not relations and len(entities) >= 2:
        for index in range(min(len(entities) - 1, 3)):
            left = entities[index]
            right = entities[index + 1]
            relations.append(
                GraphRelation(
                    relation_id=f"rel_{left.entity_id}_related_to_{right.entity_id}",
                    source=left.name,
                    relation="related_to",
                    target=right.name,
                    evidence=f"{left.name} 与 {right.name} 出现在同一资料中。",
                    source_ref=source_ref,
                )
            )
    return relations
