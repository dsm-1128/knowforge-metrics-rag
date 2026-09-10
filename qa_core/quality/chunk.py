"""Chunk 质量检测规则——不依赖 LLM，适合入库前后高频运行。

提供纯规则驱动的 chunk 质量检测能力，在入库前后快速发现常见质量问题：
  - empty：chunk 内容为空，无任何语义价值。
  - too_short：chunk 过短，可能只包含标题、页码或碎片文本。
  - too_long：chunk 过长，影响召回精度并增加 prompt 成本。
  - low_unique_ratio：字符重复比例过高，可能是 OCR 噪声或解析失败内容。
  - duplicate_content：正文重复，可能造成重复召回和答案引用噪声。

设计决策：
- 所有检测规则基于纯字符串分析和统计，不调用任何外部模型。
- 两遍扫描设计：第一遍统计哈希计数和字符唯一率，第二遍基于计数标记重复。
- 对表格元数据（table_row）放宽过短和低唯一率阈值，因为表格行本身就短且可能
  包含大量重复分隔符。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_table_metadata
from qa_core.utils import stable_hash


def _content_hash(content: str) -> str:
    """对 chunk 正文生成短 hash，用于重复内容检测。

    调用顺序：质量门禁流程 -> _content_hash()。
    """
    return stable_hash(content.strip())[:16]


def analyze_chunk_quality(chunks: list[Document]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """检测 chunk 级别的常见质量问题：empty/too_short/too_long/low_unique_ratio/duplicate_content。（★★★ 核心）

    执行流程：
      1. 第一遍遍历：统计每个 chunk 的文本长度、内容哈希、字符唯一率。
      2. 按规则逐个检查：空内容、过短、过长、唯一率低。
      3. 第二遍遍历：基于第一遍的哈希计数标记重复内容。
      4. 汇总统计数据：总数、重复数、最短/最长/平均长度。

    参数：
        chunks: 待检测的 Document 列表（切分后的 chunk）。

    返回：
        (issues_list, stats_dict) 元组。
        issues_list 包含每个质量问题的详情（索引、文件名、问题类型和原因）。
        stats_dict 包含汇总统计（总数、重复数、长度分布等）。

    调用顺序：质量门禁流程 -> analyze_chunk_quality()。
    """
    settings = get_settings()
    # ── 第一遍遍历：统计长度、哈希和唯一率 ──
    issues: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    lengths: list[int] = []
    for index, chunk in enumerate(chunks, start=1):
        content = (chunk.page_content or "").strip()
        metadata = dict(chunk.metadata)
        # file_name 优先从 metadata 的 file_name 字段取，没有则退到 source 字段
        file_name = metadata.get("file_name") or metadata.get("source") or "unknown"
        length = len(content)
        lengths.append(length)
        digest = _content_hash(content)
        seen[digest] = seen.get(digest, 0) + 1
        base = {
            "index": index,
            "file_name": file_name,
            "source": metadata.get("source"),
            "chunk_id": metadata.get("chunk_id"),
            "length": length,
            "content_preview": content[:120],
        }
        # 空 chunk：完全无内容，不可能提供任何语义，直接标记为最低级问题并跳过后续检查
        if not content:
            issues.append({**base, "issue": "empty", "reason": "chunk 为空，不能提供有效语义。"})
            continue
        # 过短检测：长度 < 30 且不是表格元数据（表格行可能本身就短），判定为碎片
        if length < 30 and not is_table_metadata(metadata):
            issues.append({**base, "issue": "too_short", "reason": "chunk 过短，可能只包含标题、页码或碎片。"})
        # 过长检测：超过 parent_chunk_size 的 2 倍或 2000 字符的上限，影响召回精度和 prompt 成本
        if length > max(settings.parent_chunk_size * 2, 2000):
            issues.append({**base, "issue": "too_long", "reason": "chunk 过长，会降低召回精度并增加 prompt 成本。"})
        # 去空白后的字符唯一性比例：去除所有空白后计算唯一字符占比，用于检测 OCR 噪声/表格线/重复符号
        compact = "".join(content.split())
        unique_ratio = round(len(set(compact)) / len(compact), 4) if compact else 0.0
        # low_unique_ratio：长度 >= 50 且唯一率 < 8% 且不是表格元数据时告警
        if length >= 50 and unique_ratio < 0.08 and not is_table_metadata(metadata):
            issues.append(
                {
                    **base,
                    "issue": "low_unique_ratio",
                    "unique_ratio": unique_ratio,
                    "reason": "字符重复比例过高，可能是 OCR 噪声、表格线或解析失败内容。",
                }
            )

    # ── 第二遍遍历：基于哈希计数找出重复内容 ──
    # seen[digest] > 1 表示该内容出现多次，标记为重复内容
    duplicate_count = 0
    for index, chunk in enumerate(chunks, start=1):
        content = (chunk.page_content or "").strip()
        digest = _content_hash(content)
        if content and seen.get(digest, 0) > 1:
            duplicate_count += 1
            metadata = dict(chunk.metadata)
            issues.append(
                {
                    "index": index,
                    "file_name": metadata.get("file_name") or metadata.get("source") or "unknown",
                    "source": metadata.get("source"),
                    "chunk_id": metadata.get("chunk_id"),
                    "length": len(content),
                    "issue": "duplicate_content",
                    "reason": "chunk 正文重复，可能造成重复召回和答案引用噪声。",
                    "content_preview": content[:120],
                }
            )

    stats = {
        "chunk_count": len(chunks),
        "duplicate_chunk_count": duplicate_count,
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
        "avg_chunk_length": round(sum(lengths) / max(len(lengths), 1), 2),
        "low_quality_issue_count": len(issues),
    }
    return issues, stats
