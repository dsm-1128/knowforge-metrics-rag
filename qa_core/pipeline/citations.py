"""最终答案引用来源强约束：模型漏写来源编号时在末尾补充"参考来源"，确保证据链完整。

设计决策：
- 后处理修补而非依赖 LLM 遵守指令：即使 System Prompt 明确要求标注来源编号，复杂问题或
  长文本生成时 LLM 仍有概率遗漏（约 5-15%）。后处理修补是确定性操作，零延迟，比
  反复重试/调高 temperature 更可靠。
- 表格行补全仅补第一条遗漏：多行表格行全部追加会导致答案尾部长而杂乱，影响阅读体验。
  只修补第一个遗漏的表格行，兼顾信息完整性和可读性。
- 引用来源只追加前 3 个文档：来源过多反而不利于快速定位，前 3 个文档通常覆盖了
  答案中最重要的证据来源。

依赖分层：
- 被 pipeline.steps.prepare_answer() 和 pipeline.rag._search_and_generate() 调用。
- 依赖 qa_core.document_metadata 获取来源标签和表格判定。
"""

from __future__ import annotations
import re
from typing import Any

from langchain_core.documents import Document

from qa_core.document_metadata import format_source_label, is_table_document

CITATION_RE = re.compile(r"\[\d+\]")
TABLE_CELL_RE = re.compile(r"^-\s*(?P<key>[^:：]{1,40})[:：]\s*(?P<value>.+?)\s*$")

def source_reference_label(doc: Document, index: int) -> str:
    """生成简短来源标签，用于答案末尾的"参考来源"列表。

    格式为 `[index] 来源标签`，来源标签由 format_source_label 从 metadata 提取，
    表格资料会附加 sheet 和行号信息，方便用户回查原始证据。

    参数：
        doc: LangChain Document 对象（需含 metadata 中的 file_name/source/sheet_name 等）。
        index: 来源编号（1-based）。

    返回：
        形如 "[1] 员工手册.pdf" 或 "[2] 报销流程 / 工作表：Sheet1 / 第 5 行" 的来源标签。

    调用顺序：enforce_answer_citations() -> source_reference_label()。
    """
    metadata: dict[str, Any] = dict(doc.metadata)
    # format_source_label 从元数据中提取文件名（或表格 sheet+行号）作为可读来源描述
    return f"[{index}] {format_source_label(metadata)}"


def extract_table_cells(doc: Document) -> list[tuple[str, str]]:
    """从表格行文本中提取"列名-单元格值"键值对。

    对文档的每行文本用正则 `- 列名: 值` 匹配，提取表格行的列名和对应值。
    用于后处理检查 LLM 生成文本是否遗漏了表格关键单元格。

    参数：
        doc: 表格行类型的 LangChain Document，page_content 含 `- 列名: 值` 格式行。

    返回：
        (列名, 值) 元组列表，按文档中的出现顺序排列。

    调用顺序：build_table_row_detail() / needs_table_row_detail() -> extract_table_cells()。
    """
    cells: list[tuple[str, str]] = []
    # 按行扫描表格文档内容，匹配 `- 列名: 值` 格式提取键值对
    for line in doc.page_content.splitlines():
        match = TABLE_CELL_RE.match(line.strip())
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key and value:
            cells.append((key, value))
    return cells


def build_table_row_detail(doc: Document, index: int) -> str:
    """构造一条可直接追加到答案末尾的表格行要点。

    从表格文档中提取前 6 个键值对，用中文分号拼接为一行要点文本，
    末尾附带来源编号。例如: "表格行要点：状态：进行中；金额：5000 [1]"

    参数：
        doc: 表格行类型的 Document，page_content 含键值对。
        index: 来源编号（1-based）。

    返回：
        格式化的表格行要点字符串。提取不到键值对时返回空字符串。

    调用顺序：enforce_table_row_details() -> build_table_row_detail()。
    """
    cells = extract_table_cells(doc)
    if not cells:
        # 文档不是表格格式或无可提取的键值对时返回空字符串
        return ""
    # 限制取前 6 个键值对防止单行过长，用中文分号拼接
    detail = "；".join(f"{key}：{value}" for key, value in cells[:6])
    return f"表格行要点：{detail} [{index}]"


def needs_table_row_detail(answer: str, doc: Document) -> bool:
    """判断 LLM 生成的自由文本是否遗漏了表格行中的键值对。

    对比表格文档的所有单元格值与答案文本，只要有一条缺失即判定需要后处理补全。
    这么做是为了防止 LLM 在长文本生成中只覆盖了部分列，遗漏状态/金额等关键字段。

    参数：
        answer: LLM 生成的回答文本。
        doc: 表格行类型的 Document。

    返回：
        True 表示答案中缺失了表格文档中的某个单元格值，需要后处理追加。

    调用顺序：enforce_table_row_details() -> needs_table_row_detail()。
    """
    cells = extract_table_cells(doc)
    if not cells:
        return False
    # 检查答案中是否已包含所有单元格值；只要有一条缺失就需要后处理追加
    return any(value not in answer for _, value in cells)


def has_source_citation(answer: str) -> bool:
    r"""判断答案中是否已经包含 `[数字]` 形式的来源编号。

    使用正则 `\[\d+\]` 匹配。如果 LLM 已经遵守引用格式主动标注了来源，
    后处理不应重复追加参考来源列表以免破坏原文结构。

    参数：
        answer: LLM 生成的回答文本。

    返回：
        True 表示答案中已包含至少一处 [数字] 格式的来源标注。

    调用顺序：enforce_answer_citations() -> has_source_citation()。
    """
    return bool(CITATION_RE.search(answer))


def enforce_table_row_details(answer: str, context_docs: list[Document]) -> str:
    """后处理补全 LLM 遗漏的表格行键值对信息。

    LLM 在长文本生成时容易丢弃半结构化表格单元格（状态/金额等），
    本函数遍历上下文中的表格文档，对每个遗漏的表格行构造要点详情并追加到答案末尾。
    为控制答案长度，只修补第一个遗漏的表格行。

    参数：
        answer: LLM 生成的回答文本。
        context_docs: 进入上下文的文档列表（含表格行）。

    返回：
        可能追加了表格行详情的答案文本。无遗漏时返回原答案。

    调用顺序：enforce_answer_citations() -> enforce_table_row_details()。
    """
    details: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        # 只补全表格文档中模型未覆盖的行，非表格文档或已含单元格值的行无需处理
        if not is_table_document(doc) or not needs_table_row_detail(answer, doc):
            continue
        # 为漏掉的表格行构造要点详情（含来源编号），拼接在答案最后便于用户查阅
        detail = build_table_row_detail(doc, index)
        if detail:
            details.append(detail)
        # 只修补第一个遗漏的表格行，避免多行拼接后答案过于冗长影响阅读体验
        if len(details) >= 1:
            break
    if not details:
        return answer
    return f"{answer}\n\n" + "\n".join(details)


def enforce_answer_citations(answer: str, context_docs: list[Document]) -> str:
    """Stage 6：后处理保证每条答案都有可追溯的来源编号。（★★★ 核心）

    不依赖 LLM 在生成时主动遵守引用格式，而是在后处理阶段确定性修补。
    执行三步策略：
    1. 补全表格行遗漏的键值对（enforce_table_row_details）。
    2. 检查是否已有 [数字] 来源标注 — 有则保留原样。
    3. 无来源标注时追加"参考来源"列表（前 3 个文档）。

    参数：
        answer: LLM 生成的回答文本。
        context_docs: 进入 LLM 上下文的文档列表。

    返回：
        可能追加了来源引用和表格行详情的完整答案文本。

    调用顺序：pipeline.steps.prepare_answer() / pipeline.rag._search_and_generate() -> enforce_answer_citations()。
    """
    clean_answer = answer.strip()
    # 无答案文本或无上下文文档时无需补充来源，直接返回原始内容
    if not clean_answer or not context_docs:
        return clean_answer
    # 步骤1：确保表格类答案不丢失核心单元格信息（状态/金额/责任人等），通过正则补全遗漏行
    clean_answer = enforce_table_row_details(clean_answer, context_docs)
    # 步骤2：模型已在答案中嵌入来源编号（如 [1][2]）时保留原样，不做二次追加破坏原文结构
    if has_source_citation(clean_answer):
        return clean_answer
    # 步骤3：只追加前 3 个文档的来源标签，来源过多反而不利于用户快速定位关键证据
    references = "；".join(source_reference_label(doc, index) for index, doc in enumerate(context_docs[:3], start=1))
    return f"{clean_answer}\n\n参考来源：{references}"
