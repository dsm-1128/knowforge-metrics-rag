"""文档索引入库层。

提供完整的文档入库链路：文件加载（document_loaders）-> 元数据标准化（document_normalizer）
-> 文档切分（chunking）-> 表格资料处理（table_documents）-> OCR 识别与复核
（ocr_documents/ocr_review）-> 图片风险检测（image_risk）-> FAQ 入库（faq_ingestion）
-> Milvus 写入编排（service）-> 入库清单维护（manifest）-> 旧 chunk 清理（cleanup）。

设计决策：
- 入库和在线问答完全分离：本层只负责离线写入，不参与在线检索生成。
- parent-child 切分策略：子块精确命中，父块提供上下文窗口。
- 增量构建：通过 manifest + fingerprint 跳过未变更文件。

依赖分层：
- qa_core.config.settings：全局配置读取。
- qa_core.scenarios.registry：场景定义和 valid_sources 白名单。
- qa_core.governance：数据域隔离、知识库版本管理。
- qa_core.retrieval.factory：Milvus/FAISS 存储工厂。
"""

