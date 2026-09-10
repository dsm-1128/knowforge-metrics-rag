"""文档 metadata 的公共判断和展示函数。

这个模块只放和 `Document.metadata` 直接相关的轻量纯函数。它不连接 Milvus，
不依赖 RAG 主流程，也不读取配置，所以入库、质量检测、Schema 展示和答案引用都能
安全复用，避免把公共判断放进 `pipeline.context` 后造成循环导入。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


def is_table_metadata(metadata: dict[str, Any]) -> bool:
    """判断 metadata 是否表示表格资料。

    使用场景：
    - 入库切分：表格行已经是完整业务单元，不能再按普通正文切碎；
    - 质量检测：表格行通常较短，不能按普通 chunk 的最小长度规则误报；
    - 检索上下文：表格类问题需要优先保留 sheet、行号、列名等定位信息；
    - 答案引用：表格来源展示时要补充工作表和行号，方便用户回查原始证据。

    为什么只看 `content_type` 前缀：
    当前表格 loader 会写入 `content_type=table_row`。后续如果增加 `table_cell`、
    `table_summary` 等类型，统一用 `table` 前缀即可被识别，避免每个调用点反复维护
    一组枚举值。

    调用顺序：上游业务入口 -> is_table_metadata()。
    """
    return str(metadata.get("content_type") or "").lower().startswith("table")


def is_table_document(doc: Document) -> bool:
    """判断 LangChain Document 是否来自表格资料。

    这是 is_table_metadata() 的便捷包装，直接从 Document 对象读取 metadata 字典。
    使用场景与 is_table_metadata 一致：入库切分、质量检测、检索上下文和答案引用中
    需要区分表格行和普通正文时使用。

    参数：
        doc: LangChain Document 对象。

    返回：
        True 表示该文档的 metadata 中 content_type 以 "table" 开头。

    调用顺序：上游业务入口 -> is_table_document() -> is_table_metadata()。
    """
    return is_table_metadata(doc.metadata)


def is_reviewed_ocr_metadata(metadata: dict[str, Any]) -> bool:
    """判断 metadata 是否表示已人工复核的 OCR 文本资料。

    调用顺序：上游业务入口 -> is_reviewed_ocr_metadata()。
    """
    return (
        str(metadata.get("content_type") or "").lower() == "ocr_reviewed_text"
        and str(metadata.get("review_status") or "").lower() == "reviewed"
    )


def format_source_label(metadata: dict[str, Any]) -> str:
    """生成前端和答案引用使用的来源标签。

    普通文档只展示文件名、标准问题或 source；表格资料额外展示 sheet 和行号。这样做的
    目的不是美化文本，而是让答案中的 `[1]` 能追溯到具体文件、工作表和行，满足企业
    RAG 场景对可解释性和可复核性的要求。

    调用顺序：上游业务入口 -> format_source_label()。
    """
    label = str(metadata.get("file_name") or metadata.get("standard_question") or metadata.get("source") or "unknown")
    if is_reviewed_ocr_metadata(metadata):
        confidence = metadata.get("ocr_confidence")
        suffix = " / OCR 已复核"
        if confidence not in (None, ""):
            suffix += f" / 置信度：{confidence}"
        return label + suffix
    if not is_table_metadata(metadata):
        return label
    sheet = str(metadata.get("sheet_name") or "").strip()
    row_number = metadata.get("row_number")
    parts = [label]
    if sheet:
        parts.append(f"工作表：{sheet}")
    if row_number not in (None, ""):
        parts.append(f"第 {row_number} 行")
    return " / ".join(parts)
