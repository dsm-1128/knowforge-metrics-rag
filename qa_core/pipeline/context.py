"""RAG 最终上下文构建工具：FAQ 直出判断、上下文片段筛选、来源去重和 source 过滤优先级。
"""

from __future__ import annotations
from typing import Any

from langchain_core.documents import Document

from qa_core.document_metadata import format_source_label, is_table_document
from qa_core.retrieval.ranking import document_key
from qa_core.retrieval.strategy import RetrievalPlan
from qa_core.scenarios.registry import ScenarioDefinition

def direct_faq_answer(original_query: str, doc: Document | None, score: float, threshold: float) -> str | None:
    """判断 FAQ 检索结果是否可不经 LLM 直接返回：标准问题完全一致或检索分数达阈值。★★★ 核心

    执行流程：
    1. 无文档时直接返回 None（无匹配）
    2. 提取元数据中的 answer 和 standard_question 字段
    3. 用户问题与标准问题完全一致时返回答案
    4. 检索分数达阈值时也允许直出（为全量 RAG 主链路提供弹性）

    参数：
        original_query: 用户原始提问
        doc: FAQ 命中的 Document 对象（含 answer/standard_question 元数据）
        score: 检索分数，用于阈值比较
        threshold: 直出分数阈值（float("inf") 表示关闭相似分数直出）

    返回：
        str | None: 直出的答案文本；不满足条件时返回 None
    """
    if doc is None:
        return None
    metadata = doc.metadata
    answer = str(metadata.get("answer") or "").strip()
    standard_question = str(metadata.get("standard_question") or metadata.get("question") or doc.page_content).strip()
    # FAQ 条目缺少 answer 字段时视为无效数据，不应参与直出判断
    if not answer:
        return None
    # 用户问题与标准问题完全一致时直接返回答案，跳过所有后续 RAG 流程
    if original_query.strip() == standard_question:
        return answer
    # 检索分数达到阈值时也允许直出（为全量 RAG 主链路的 get_faq_direct_answer 提供弹性）
    if score >= threshold:
        return answer
    return None


def select_context_docs(faq_hits: list[Any], doc_hits: list[Any], plan: RetrievalPlan) -> list[Document]:
    """从 FAQ+文档候选中按分数、长度和条数约束筛选最终进入 LLM prompt 的上下文片段。（★★★ 核心）

    执行流程：
      1. FAQ 命中：按 min_context_score 过滤 → 取前 2 条 → 转为"常见问题+标准答案"格式
      2. 文档命中：按 min_context_score 过滤 → prefer_table 时表格行优先排序 → 优先用 parent_content
      3. 每条追加受 final_context_top_n、max_context_chars、max_context_doc_chars 三重约束（去重、截断、计数）
      4. 内部 append_doc 闭包管理去重键、字符数上限、截断标记

    为什么不用 LLM 筛选：LLM 筛选每个候选需要额外推理延迟（约 200ms/条），50 条候选
    就需要 10 秒。确定性规则筛选是瞬时操作，且阈值/条数/字符数三重约束已足够覆盖
    绝大多数场景的质量需求。

    参数：
        faq_hits: FAQ 检索命中列表（RetrievalHit 列表）。
        doc_hits: 文档检索命中列表（RetrievalHit 列表）。
        plan: RetrievalPlan，提供 min_context_score / final_context_top_n / max_context_chars
              / max_context_doc_chars / prefer_table。

    返回：
        list[Document]: 筛选后的上下文文档列表；空列表表示筛选后无可用上下文，上游据此返回
                        确定性"信息不足"回答，不走 LLM 生成。
    """
    selected: list[Document] = []
    seen_keys: set[str] = set()
    used_chars = 0

    def append_doc(doc: Document, key: str) -> None:
        """按去重、条数上限、单条/总字符数限制追加上下文。

        调用顺序：QAService/RAG 管线 -> append_doc()。
        """
        nonlocal used_chars
        content = doc.page_content.strip()
        if not content or key in seen_keys or len(selected) >= plan.final_context_top_n:
            return
        # 单条上下文超过 max_context_doc_chars 时截断并标记，避免单个文档挤占 prompt 空间
        if len(content) > plan.max_context_doc_chars:
            content = content[: plan.max_context_doc_chars].rstrip()
            metadata = {**doc.metadata, "context_truncated": True}
        else:
            metadata = dict(doc.metadata)
        # 累计字符数达上限时停止追加（但至少保留第一条）
        if used_chars + len(content) > plan.max_context_chars and selected:
            return
        seen_keys.add(key)
        used_chars += len(content)
        selected.append(Document(page_content=content, metadata=metadata))

    # ── FAQ 部分：过滤分数 → 取前 2 条 → 转成标准问答格式 ──
    for hit in [item for item in faq_hits if item.score >= plan.min_context_score][:2]:
        metadata = hit.document.metadata
        answer = metadata.get("answer")
        question = metadata.get("standard_question") or hit.document.page_content
        if answer:
            # 从元数据中提取 Document 的唯一标识键用于去重
            append_doc(
                Document(page_content=f"常见问题：{question}\n标准答案：{answer}", metadata=metadata),
                f"faq:{document_key(hit.document)}",
            )

    # ── 文档部分：过滤分数 → prefer_table 排序 → 优先用 parent_content → 追加 ──
    eligible_doc_hits = [hit for hit in doc_hits if hit.score >= plan.min_context_score]
    # prefer_table 为 True 时把表格行（content_type=table_row）排到普通正文前
    if plan.prefer_table:
        eligible_doc_hits = sorted(
            eligible_doc_hits,
            key=lambda hit: (0 if is_table_document(hit.document) else 1, -hit.score),
        )
    for hit in eligible_doc_hits:
        metadata = hit.document.metadata
        parent_content = metadata.get("parent_content")
        # 从元数据中提取 Document 的唯一标识键用于去重
        key = str(metadata.get("parent_id") or document_key(hit.document))
        append_doc(Document(page_content=str(parent_content or hit.document.page_content), metadata=metadata), f"doc:{key}")
    return selected


def effective_source_filter(
    source_filter: str | None,
    suggested_source: str | None,
    scenario: ScenarioDefinition,
) -> str | None:
    """确定最终 source 过滤项：前端显式选择 > 意图推断 > 不过滤。

    优先级：前端显式选择优先于 LLM 意图推断；意图推断结果不在场景白名单内时，不启用 source 过滤。

    参数：
        source_filter: 前端显式选择的业务分类（优先级最高）
        suggested_source: 意图识别推断的候选源
        scenario: ScenarioDefinition（提供 valid_sources 白名单校验）

    返回：
        str | None: 最终生效的 source 过滤项；None 表示不过滤

    调用顺序：QAService/RAG 管线 -> effective_source_filter()。
    """
    if source_filter:
        return source_filter
    if suggested_source and suggested_source in scenario.valid_sources:
        return suggested_source
    return None


def build_context(docs: list[Document]) -> str:
    """将 Document 列表格式化为带编号的上下文文本，供 LLM 在回答中按编号引用来源。★★★ 核心

    执行流程：
    1. 遍历 docs，按 page_content 去重（相同来源的不同分块可能重复）
    2. 为每个去重后的 doc 生成 "[N] 来源：{source}\n{content}" 格式文本
    3. 各条目以双换行分隔拼接

    参数：
        docs: 已筛选的上下文 Document 列表

    返回：
        str: 带编号的上下文文本，供 LLM prompt 使用；空列表时返回空字符串
    """
    parts: list[str] = []
    seen: set[str] = set()
    visible_index = 1
    for doc in docs:
        content = doc.page_content.strip()
        # ── 按内容去重 ──
        # 相同来源的不同分块可能重复（如同一 FAQ 从不同查询变体各召回一次），
        # 相同内容只出现一次，避免给 LLM 注入重复证据，既浪费上下文窗口又降低答案质量
        if not content or content in seen:
            continue
        seen.add(content)
        metadata = doc.metadata
        source = _context_source_label(metadata)
        # ── 构造带编号的引用格式 ──
        # 格式为 "[N] 来源：{source}\n{content}"，LLM 在 System Prompt 中被要求
        # 引用 [N] 编号标注事实来源，这种格式保证编号与来源一一对应
        parts.append(f"[{visible_index}] 来源：{source}\n{content}")
        visible_index += 1
    return "\n\n".join(parts)


def _context_source_label(metadata: dict[str, Any]) -> str:
    """从元数据提取可读来源标签：普通文档用文件名，表格资料附加 sheet 和行号。

    参数：
        metadata: Document.metadata 字典（含 source/file_name/sheet_name/row_index 等）

    返回：
        str: 可读来源标签，如"入职流程.pdf"或"薪资表.xlsx[Sheet1:行3]"

    调用顺序：QAService/RAG 管线 -> _context_source_label()。
    """
    return format_source_label(metadata)

