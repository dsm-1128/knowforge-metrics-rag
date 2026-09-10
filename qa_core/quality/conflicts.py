"""FAQ 标准答案与正文资料之间的语义冲突检测。

在知识库版本激活前，自动检测 FAQ 答案与正文资料之间的三类冲突：
  1. no_related_document：FAQ 有标准答案，但正文中找不到相关片段。
  2. numeric_mismatch：FAQ 和正文包含不同数字，可能存在价格/时间/比例冲突。
  3. polarity_mismatch：FAQ 和正文的肯定/否定倾向矛盾。

设计决策：
- 使用 jieba 分词 + 关键词匹配检测相关性，不依赖向量检索或 LLM。
- 冲突报告在激活 kb_version 前生成，作为质量门禁的依据。
- 表格行使用独立的关联检测逻辑，因为表格行内容简短，需要降低匹配阈值。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import jieba
from langchain_core.documents import Document

from qa_core.document_metadata import is_table_metadata
from qa_core.quality.faq import read_faq_records


NUMERIC_FACT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:元|块|小时|天|周|个月|月|年|%|折|次|人|条|分钟)?")
VERSION_NUMBER_RE = re.compile(r"v?20\d{2}\.\d{1,2}", re.IGNORECASE)
NEGATIVE_RE = re.compile(r"(不支持|不能|不可以|不可|不应|不得|无需|不需要|禁止|无法|不会|未开放|不允许|不代表|不一定|资料不足|未经|不得作为)")
POSITIVE_RE = re.compile(r"(支持|可以|可用|需要|允许|开放|能够|必须|会|建议|应当|应|负责)")


def _keyword_tokens(text: str) -> list[str]:
    """抽取用于 FAQ 与正文关联的中文关键词。使用 jieba.cut_for_search 搜索分词。

    调用顺序：质量门禁流程 -> _keyword_tokens()。
    """
    # 通用停用词表：这些词在中文 FAQ 中高频出现但缺乏区分度，排除后减少噪声命中
    stop_words = {
        "什么",
        "怎么",
        "如何",
        "是否",
        "可以",
        "需要",
        "相关",
        "业务",
        "问题",
        "进行",
        "支持",
        "通常",
        "应该",
        "说明",
    }
    result: list[str] = []
    for token in jieba.cut_for_search((text or "").lower()):
        token = token.strip()
        # 过滤规则：单字词无区分度、停用词不参与匹配、已添加的 token 去重
        if len(token) < 2 or token in stop_words or token in result:
            continue
        result.append(token)
    # 限制最大 12 个关键词，避免过长短语列表导致高阈值误报
    return result[:12]


def _numbers(text: str) -> set[str]:
    """抽取文本中的数字事实，用于发现价格、时间、比例等口径不一致。

    调用顺序：质量门禁流程 -> _numbers()。
    """
    # 先剔除版本号（如 "2024.01"）避免被误识别为数字事实，保留纯业务数字用于口径比对
    cleaned_text = VERSION_NUMBER_RE.sub("", text or "")
    return {item.replace(" ", "") for item in NUMERIC_FACT_RE.findall(cleaned_text) if item.strip()}


def _polarity(text: str) -> str:
    """判断文本是否包含明显正向或否定口径。同时出现两者时返回 mixed_or_unknown。

    调用顺序：质量门禁流程 -> _polarity()。
    """
    # 通过正则分别检测肯定和否定关键词，两类词同时出现时无法判断倾向，返回 mixed_or_unknown 避免误判
    has_negative = bool(NEGATIVE_RE.search(text or ""))
    has_positive = bool(POSITIVE_RE.search(text or ""))
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    return "mixed_or_unknown"


def _related_threshold(keywords: list[str]) -> int:
    """根据关键词数量决定 FAQ 与正文相关性的最低命中数。关键词越多，阈值越高。

    调用顺序：质量门禁流程 -> _related_threshold()。
    """
    if len(keywords) >= 6:
        return 3
    if len(keywords) >= 3:
        return 2
    return 1


def _has_table_specific_overlap(content: str, keywords: list[str]) -> bool:
    """判断表格行是否命中了足够具体的 FAQ 关键词。至少命中一个长度>=3 的关键词或 2+ 普通关键词。

    调用顺序：质量门禁流程 -> _has_table_specific_overlap()。
    """
    # 表格行可能只有字段名和值，所以降低匹配门槛：一个长词或两个短词即视为有关联
    # 避免表格行因为内容简短而被误判为与 FAQ 无关
    normalized = content.lower()
    hits = [keyword for keyword in keywords if keyword and keyword in normalized]
    return any(len(keyword) >= 3 for keyword in hits) or len(hits) >= 2


def _check_record_conflicts(record, indexed_chunks, conflicts: list, limit: int) -> bool:
    """检查一条 FAQ 记录与所有 indexed chunks 的冲突。
    返回 True 表示应继续检查下一条记录，False 表示已达到 limit 应停止。

    三类冲突检测：
        1：no_related_document
            判断条件：找不到达到关联门槛的正文（根据faq在doc中找是否有关联的正文）
            含义：faq可能缺少资料依据
        2：numeric_mismatch
            判断条件：Faq与正文都有数字，但是集合没有交集
            含义：金额/时间/比例等口径可能冲突
        3：polarity_mismatch
            判断条件：一方是肯定，另一方否定
            含义：支持范围/制度要求可能冲突
    调用顺序：质量门禁流程 -> _check_record_conflicts()。
    """
    question = record["question"]
    answer = record["answer"]
    # 问题或答案为空时无法做任何冲突检测，跳过该条记录
    if not question or not answer:
        return True
    # 将 FAQ 的问题和答案合并提取关键词，用于在正文资料中定位相关片段
    keywords = _keyword_tokens(f"{question} {answer}")
    related = []
    # 根据关键词数量动态调整匹配阈值：关键词越多，需要命中的次数越高，减少偶然匹配
    threshold = _related_threshold(keywords)
    for item in indexed_chunks:
        # source 过滤：如果 FAQ 指定了业务分类且 chunk 也有分类，两者不同则不认为相关
        if record["source"] and item["source"] and record["source"] != item["source"]:
            continue
        content = item["content"]
        # 表格行内容简短，需使用专门的表格关联检测逻辑，避免因为行内容短而漏关联
        if is_table_metadata(item["metadata"]) and not _has_table_specific_overlap(content, keywords):
            continue
        # 统计当前 chunk 中命中了几个关键词，命中数达到阈值即视为正文中找到了 FAQ 依据
        hit_count = sum(1 for keyword in keywords if keyword and keyword in content.lower())
        if hit_count >= threshold:
            related.append((hit_count, item))
    # 按命中次数降序排列，只取前 5 个最相关的 chunk 做后续冲突分析
    related.sort(key=lambda pair: pair[0], reverse=True)
    top_related = [item for _, item in related[:5]]
    # 如果正文资料中完全找不到相关片段，说明 FAQ 可能缺乏文档依据
    if not top_related:
        conflicts.append(
            {
                "issue": "no_related_document",
                "row": record["row"],
                "question": question,
                "source": record["source"],
                "reason": "FAQ 有标准答案，但正文资料中没有找到明显相关片段，需要确认 FAQ 是否有文档依据。",
            }
        )
        return True

    # 对每个最相关 chunk 做两维度冲突检查：数字口径冲突 + 肯定/否定倾向冲突
    answer_numbers = _numbers(answer)
    answer_polarity = _polarity(answer)
    for item in top_related:
        content = item["content"]
        doc_numbers = _numbers(content)
        # numeric_mismatch：FAQ 和正文同时包含数字但无交集，说明价格/时间/比例等关键事实不一致
        if answer_numbers and doc_numbers and answer_numbers.isdisjoint(doc_numbers):
            conflicts.append(
                {
                    "issue": "numeric_mismatch",
                    "row": record["row"],
                    "question": question,
                    "source": record["source"],
                    "faq_numbers": sorted(answer_numbers),
                    "document_numbers": sorted(doc_numbers),
                    "file_name": item["file_name"],
                    "content_preview": content[:160],
                    "reason": "FAQ 答案和相关正文出现不同数字，可能存在价格、时间、比例或数量口径冲突。",
                }
            )
        doc_polarity = _polarity(content)
        # polarity_mismatch：FAQ 肯定但正文否定（或反之），说明口径模板冲突需要人工复核
        if {answer_polarity, doc_polarity} == {"positive", "negative"}:
            conflicts.append(
                {
                    "issue": "polarity_mismatch",
                    "row": record["row"],
                    "question": question,
                    "source": record["source"],
                    "faq_polarity": answer_polarity,
                    "document_polarity": doc_polarity,
                    "file_name": item["file_name"],
                    "content_preview": content[:160],
                    "reason": "FAQ 答案和相关正文存在肯定/否定口径差异，需要人工确认。",
                }
            )
        # 达到上限立即停止，避免报告文件过大
        if len(conflicts) >= limit:
            return False
    return True


def detect_faq_document_conflicts(faq_csv: str | Path, chunks: list[Document], *, limit: int = 100) -> dict[str, Any]:
    """检测 FAQ 标准答案和正文资料之间的潜在冲突（no_related_document/numeric_mismatch/polarity_mismatch）。

    调用顺序：质量门禁流程 -> detect_faq_document_conflicts()。
    """
    # 原因： FAQ 标准答案由业务人员维护，正文资料由另外的团队提供——两套来源可能口径不一致（价格/否定/支持范围），激活前自动比对可以提前发现冲突而不依赖人工逐条复核
    records = read_faq_records(faq_csv)
    conflicts: list[dict[str, Any]] = []
    # 将 Document 列表转为简单的字典列表，便于后续在 _check_record_conflicts 中快速读取
    indexed_chunks = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        indexed_chunks.append(
            {
                "content": chunk.page_content or "",
                "metadata": metadata,
                "source": metadata.get("source"),
                "file_name": metadata.get("file_name") or metadata.get("source"),
            }
        )

    for record in records:
        # 一旦冲突数量达到 limit 上限则立即终止检查，避免处理全部记录造成不必要的性能开销
        if not _check_record_conflicts(record, indexed_chunks, conflicts, limit):
            break

    summary = Counter(item["issue"] for item in conflicts)
    return {
        "checked_faq_count": len([item for item in records if item.get("question") and item.get("answer")]),
        "conflict_count": len(conflicts),
        "issue_counts": dict(summary),
        "items": conflicts[:limit],
    }

