"""OCR 复核资料的轻量识别与 metadata 解析。

提供已人工复核的 OCR Markdown 文本的识别和元数据提取能力。
OCR 复核稿通过固定标记（如"复核状态：已复核"）标注其复核状态，
本模块通过检测这些标记来判断文本是否为已复核 OCR 产物，并从中
提取 OCR 置信度、原始文件路径等元数据。

设计决策：
- 不依赖 OCR 引擎本身，纯文本特征检测，轻量且稳定。
- 标记匹配支持中英文两种格式，兼容国内和国际化团队的复核标注习惯。
- 提取出的 metadata 会由 document_normalizer 合并到 Document 元数据中，
  使 chunk 在检索时能携带 OCR 复核状态。
"""

from __future__ import annotations

import re
from typing import Any

OCR_REVIEWED_CONTENT_TYPE = "ocr_reviewed_text"
OCR_REVIEW_MARKERS = (
    "复核状态：已复核",
    "人工复核：通过",
    "review_status: approved",
    "review_status: reviewed",
    "reviewed: true",
)


def is_reviewed_ocr_text(text: str) -> bool:
    """判断文本是否是已人工复核的 OCR Markdown。（★★ 理解）

    通过检查文本中是否包含 OCR_REVIEW_MARKERS 中的任一标记来判断。
    标记支持中英文多种格式，可同时满足国内和国际化团队的复核标注习惯。

    参数：
        text: 待检查的文本内容（通常是 OCR Markdown 文件全文）。

    返回：
        若文本包含任一复核标记则返回 True，否则返回 False。

    调用顺序：入库脚本或索引服务 -> is_reviewed_ocr_text()。
    """
    # 原因：只要有一个标记命中就认为是已复核，严格全匹配容易漏判（不同团队使用的标记格式不同）
    return any(marker in text for marker in OCR_REVIEW_MARKERS)


def parse_ocr_review_metadata(text: str) -> dict[str, Any]:
    """从 OCR Markdown 中提取可复核的轻量 metadata。（★★ 理解）

    参数：
        text: OCR Markdown 全文（包含复核标记和元数据行）。

    返回：
        字典，包含 content_type、review_status 和可选的 ocr_confidence、
        ocr_source_path。如果文本不是已复核 OCR 则返回空字典。

    调用顺序：入库脚本或索引服务 -> parse_ocr_review_metadata()。
    """
    if not is_reviewed_ocr_text(text):
        return {}

    # ── 步骤 1：设定复核状态核心元数据 ──
    metadata: dict[str, Any] = {
        "content_type": OCR_REVIEWED_CONTENT_TYPE,
        "review_status": "reviewed",
    }
    # ── 步骤 2：正则提取 OCR 置信度 ──
    # 置信度标记可能出现在 OCR 候选稿的任意位置，使用正则搜索而非位置依赖的解析
    # 原因：OCR 报告中的置信度行固定以"OCR 平均置信度："开头，正则提取兼容中英文冒号
    confidence_match = re.search(r"OCR 平均置信度[：:]\s*([0-9.]+)", text)
    if confidence_match:
        metadata["ocr_confidence"] = float(confidence_match.group(1))
    # 原因：原始文件路径行以"原始文件："开头，提取后用于后续溯源和定位
    source_match = re.search(r"原始文件[：:]\s*(.+)", text)
    if source_match:
        metadata["ocr_source_path"] = source_match.group(1).strip()
    return metadata
