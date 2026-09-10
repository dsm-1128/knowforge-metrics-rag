"""文档切分策略：parent-child 多粒度切片，平衡检索精度与上下文完整性。

把标准化后的 Document 按 parent-child 双层策略切分：
  - parent 块：较大的文本片段（默认 1024 字符），提供完整上下文窗口。
  - child 块：较小的子片段（默认 256 字符），用于精确语义匹配。
  检索时子块命中后携带父块上下文一同返回，实现在精确召回率和上下文
  完整性之间的平衡。

设计决策：
- 表格行和已复核 OCR 文本不经过父子切分——它们已经是治理后的完整证据单元。
- Markdown 文件先通过标题切分器结构化（按 #/##/### 分层），再进入递归切分。
- chunk_id 和 parent_id 由内容稳定哈希生成，相同内容产生相同 id，支持增量重建。

依赖分层：
- langchain_text_splitters：MarkdownHeaderTextSplitter、RecursiveCharacterTextSplitter。
- qa_core.config.settings：parent_chunk_size / child_chunk_size 等配置。
- qa_core.utils.stable_hash：基于内容生成稳定 id。
"""

from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_reviewed_ocr_metadata, is_table_metadata
from qa_core.utils import stable_hash

CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。", "！", "？", "；",
    ";", ".", "!", "?",
    "，", ",",
    " ",
    "",
    # 原因： 中英文混排文档需要同时支持中文句号/感叹号/问号和英文句点/分号作为切分边界，递归切分器按 separator 顺序优先匹配大粒度分隔符
]

def chunk_identity(page_content: str, metadata: dict) -> tuple[str, str]:
    """基于正文和标准元数据生成 parent_id 与 chunk_id。

    ID 同时纳入场景、知识库版本、embedding 模型版本和 chunk schema 版本：
    即使两个版本的文本完全相同，也不会发生跨版本主键覆盖。普通文档还区分
    父块内容和子块内容，因此同一父块下的多个子块能够各自稳定定位；表格行
    则把 table/sheet/row 维度纳入身份，避免不同表格行混淆。

    只要文件指纹和处理配置没有变化，重复运行就会得到相同 ID，这使 manifest
    可以安全地用于跳过判断，也使变化文件能够通过旧 ID 定位并清理。

    调用顺序：入库脚本或索引服务 -> chunk_identity()。
    """
    parent_content = str(metadata.get("parent_content") or page_content or "").strip()
    if is_table_metadata(metadata):
        # 表格行的 parent_id 额外包含 table_id、sheet_name、row_number，确保同一张表的同一行
        # 在多次入库时 id 稳定，且不同行的 chunk 不会混淆；行列关系由 table_id + row_number 唯一锁定
        parent_id = stable_hash(
            metadata.get("scenario_id"),
            metadata.get("kb_version"),
            metadata.get("embedding_model_version"),
            metadata.get("chunk_schema_version"),
            metadata.get("doc_id"),
            metadata.get("table_id"),
            metadata.get("sheet_name"),
            metadata.get("row_number"),
            parent_content,
        )
        # 表格行的子块 id 只基于 parent_id + parent_content 生成，因为表格行不会进一步切分子块
        chunk_id = stable_hash(parent_id, parent_content)
        return parent_id, chunk_id

    parent_id = stable_hash(
        metadata.get("scenario_id"),
        metadata.get("kb_version"),
        metadata.get("embedding_model_version"),
        metadata.get("chunk_schema_version"),
        metadata.get("doc_id"),
        parent_content,
    )
    # 普通文档的 chunk_id 使用子块自身的 page_content（而非 parent_content）参与 hash，
    # 使得同一父块下不同子块拥有不同 chunk_id
    chunk_id = stable_hash(parent_id, page_content)
    return parent_id, chunk_id


def split_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """将文档切成可检索的子块并保留父块上下文。（★★★ 核心）

    子块用于精确召回，父块上下文保存在 metadata.parent_content 中。
    检索时子块命中后携带父块上下文一同返回。

    执行流程：
      1. 对每个 Document，根据 file_type 和 metadata 选择切分策略。
      2. 表格行和已复核 OCR：不切分，整行/整段作为唯一块。
      3. Markdown：先按标题切分（#/##/###），再递归文本切分。
      4. 其他文本：直接按 parent_chunk_size 递归切分。
      5. 对每个父块生成 child 子块，计算稳定 parent_id 和 chunk_id。

    参数：
        documents: 标准化后的 LangChain Document 列表。

    返回：
        (chunks_list, ids_list) 元组。chunks_list 包含所有子块，每个块
        携带 parent_content（父块全文）和 parent_id/chunk_id。
        ids_list 是所有子块的 chunk_id 列表，与 chunks_list 一一对应。

    调用顺序：入库脚本或索引服务 -> split_documents()。
    """
    # 原因：parent-child 分别切分使子块保持精确命中，而父块提供完整上下文窗口。
    # 配置从 settings 读取，chunk_schema_version 会随策略变化而变化，供 manifest
    # 判断旧切分结果是否仍可复用。
    settings = get_settings()
    markdown_headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_overlap,
        separators=CHINESE_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_overlap,
        separators=CHINESE_SEPARATORS,
    )

    # 主处理循环：逐个标准化 Document 执行切分。一个 loader 页面可能变成多个
    # parent/child chunk，因此不能把“Document 数量”当成最终入库数量。
    chunks: list[Document] = []
    ids: list[str] = []
    for doc in documents:
        file_type = str(doc.metadata.get("file_type", "")).lower()
        parent_docs: list[Document]
        if is_table_metadata(doc.metadata) or is_reviewed_ocr_metadata(doc.metadata):
            # 表格行和已复核 OCR 文本都是治理后的完整证据单元。
            # 表格不能被拆散行列关系；OCR 复核稿不能丢失复核状态、置信度和原始文件说明。
            parent_content = doc.page_content.strip()
            # 空行跳过：表格中全空行或 OCR 空白页没有检索价值，不生成 chunk 以节省 Milvus 存储
            if not parent_content:
                continue
            metadata = dict(doc.metadata)
            metadata["parent_content"] = parent_content
            parent_id, chunk_id = chunk_identity(parent_content, metadata)
            metadata.update(
                {
                    "parent_id": parent_id,
                    "chunk_id": chunk_id,
                }
            )
            # 表格/OCR 不经过父子切分：直接以整行/整段作为唯一块，parent_id
            # 与 chunk_id 都由治理后的完整证据单元生成。
            chunks.append(Document(page_content=parent_content, metadata=metadata))
            ids.append(chunk_id)
            continue
        elif file_type == ".md":
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
            # Markdown 标题会先转成结构化元数据，再进入递归切分，能提升来源标签和上下文质量。
            # 如果 Markdown 解析失败，说明资料格式需要修复；入库阶段应该暴露异常并进入
            # 异常文件报告，而不是悄悄按普通文本切分，造成章节 metadata 丢失。
            header_docs = header_splitter.split_text(doc.page_content)
            for header_doc in header_docs:
                # Markdown 标题切分器会生成新的 Document，这里把原始文件 metadata
                # 补回去，避免切分后丢失 source、file_name、doc_id 等关键字段。
                header_doc.metadata.update(doc.metadata)
            parent_docs = parent_splitter.split_documents(header_docs)
        else:
            # 非 Markdown 普通文本：直接按父块大小切分，不再经过标题解析
            parent_docs = parent_splitter.split_documents([doc])

        # 父块循环：每个 parent_doc 先保留完整上下文，再拆成用于向量命中的 child。
        for parent_doc in parent_docs:
            parent_content = parent_doc.page_content
            # parent_id 和 chunk_id 都纳入 kb_version、embedding_model_version 和 chunk_schema_version。
            # 这样同一个文件在两个知识库版本里可以同时存在，不会因为内容相同而主键冲突。
            child_docs = child_splitter.split_documents([parent_doc])
            # child_splitter 返回的每个子块都必须独立写入 Milvus；父块全文放在
            # metadata.parent_content，召回时可恢复更完整的上下文。
            for child_doc in child_docs:
                # chunk_id 由父块和子块内容共同决定。同一文件未变化时 id 稳定；文件变化时
                # id 会变化，配合 manifest 删除旧 chunk 后重建。
                metadata = dict(child_doc.metadata)
                metadata["parent_content"] = parent_content
                parent_id, chunk_id = chunk_identity(child_doc.page_content, metadata)
                metadata.update(
                    {
                        "parent_id": parent_id,
                        "chunk_id": chunk_id,
                    }
                )
                chunks.append(Document(page_content=child_doc.page_content, metadata=metadata))
                ids.append(chunk_id)
    return chunks, ids


