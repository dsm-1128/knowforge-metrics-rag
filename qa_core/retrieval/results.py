"""检索层返回给 RAG 主流程的结果对象。

这些对象属于内部数据结构，不是 FastAPI 对外接口模型。单独放在 retrieval 包里，可以让
检索、重排、上下文构建都依赖同一个轻量结果类型，同时避免 `schemas.py` 同时承担
API Schema 和核心业务模型两种职责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document

from qa_core.document_metadata import format_source_label, is_table_metadata


@dataclass
class RetrievalHit:
    """Milvus 检索并可选重排后的一条文档命中。

    调用顺序：检索准备或检索执行 -> RetrievalHit。
    """

    document: Document
    score: float = 0.0


@dataclass
class RetrievalResult:
    """包含命中结果、耗时和来源元数据的诊断集合。

    调用顺序：检索准备或检索执行 -> RetrievalResult。
    """

    hits: list[RetrievalHit] = field(default_factory=list)
    query: str = ""
    source_type: Literal["faq", "doc"] = "doc"
    elapsed_ms: float = 0.0

    @property
    def top_score(self) -> float:
        """返回最高命中分数；无命中时返回 0。

        调用顺序：检索准备或检索执行 -> RetrievalResult.top_score()。
        """
        # 边界情况：hits 为空时返回 0.0 而非 IndexError
        return self.hits[0].score if self.hits else 0.0

    @property
    def top_document(self) -> Document | None:
        """返回最佳命中文档；没有命中时返回 None。

        调用顺序：检索准备或检索执行 -> RetrievalResult.top_document()。
        """
        # 边界情况：hits 为空时返回 None，调用方需处理此情形
        return self.hits[0].document if self.hits else None

    def source_payloads(self, limit: int = 5) -> list[dict[str, Any]]:
        """把命中结果转换成前端可直接使用的紧凑 JSON 结构。

        使用场景：
        - API 返回 `sources`，让前端展示答案依据；
        - 状态页和 Trace 查看召回明细；
        - 评测脚本复核命中内容。

        为什么在检索结果对象里做转换：
        命中分数、source_type、正文片段和表格定位信息都来自检索结果本身。集中在这里生成
        payload，可以避免 API 层、调试接口和主链路各自拼一套来源结构。

        调用顺序：检索准备或检索执行 -> RetrievalResult.source_payloads()。
        """
        payloads: list[dict[str, Any]] = []
        for hit in self.hits[:limit]:
            metadata = dict(hit.document.metadata)
            # 判断该命中是否为表格类型数据（如 Excel 表格行），是则提取表格定位信息
            # 原因：表格类型的检索结果在前端需要以表格形式展示而非纯文本，
            # 需要额外的 table_id、行号、列宽等结构化信息
            table = (
                {
                    "table_id": metadata.get("table_id"),
                    "sheet_name": metadata.get("sheet_name"),
                    "row_number": metadata.get("row_number"),
                    "row_count": metadata.get("row_count"),
                    "column_count": metadata.get("column_count"),
                    "table_headers": metadata.get("table_headers"),
                }
                if is_table_metadata(metadata)
                else None
            )
            payload = {
                "score": hit.score,
                "source_type": self.source_type,
                "content": hit.document.page_content[:500],
                "metadata": metadata,
                "citation": format_source_label(metadata),
            }
            # 仅在表格类型命中时才附加 table 字段，避免非表格数据携带空结构
            if table is not None:
                payload["table"] = table
            payloads.append(payload)
        return payloads
