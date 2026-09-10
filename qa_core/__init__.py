"""
KnowForge RAG Platform 核心问答链路包。

架构分层（自上而下）：
    api/          → FastAPI HTTP/WebSocket 路由层（请求接入、限流、事件转发）
    application/  → 应用编排层（QAService：请求适配 + pipeline 委托）
    pipeline/     → RAG 主流程编排（stream_query + debug_retrieval + 各阶段步骤）
    prompts/      → 提示词模板管理与 Prompt Profile 选择器
    retrieval/    → Milvus 混合检索（dense+sparse）、重排、过滤、结果处理
    indexing/     → 离线文档入库（loader → normalizer → chunking → Milvus 写入）
    quality/      → 入库质量报告（chunk/FAQ 质量、冲突检测）
    governance/   → 知识库版本管理、数据域隔离、chunk 版本控制
    intent/       → 检索意图分类（规则 + BERT 模型融合网关）
    cache/        → 两级缓存（L1 进程级 TTL + L2 Redis JSON）+ MySQL namespace epoch
    memory/       → 对话历史持久化（MySQL）+ 摘要/反馈记忆
    llm/          → ChatOpenAI 客户端（含 LangSmith callback）
    config/       → Pydantic Settings 运行时配置 + 门禁规则 + 启动校验
    scenarios/    → 多业务场景注册表与配置解析
    storage/      → MySQL schema 初始化
    observability/→ LangSmith 可观测性适配

典型请求链路：
    浏览器 → WebSocket /api/stream
    → api/chat.py → QAService.stream_query()
    → pipeline/rag.py stream_query()
        → Stage 0: create_query_context（场景、数据域、session）
        → Stage 1: decide_route（直答/FAQ精确/检索）
        → Stage 2: prepare_retrieval（意图、改写、检索计划、查询变体）
        → Stage 3: search_faq（Milvus FAQ 检索）
        → Stage 4: search_doc（Milvus 文档检索）
        → Stage 5: prepare_answer（上下文构建）
        → Stage 6: stream_llm_answer（LLM 流式生成）
        → Stage 7: save_history + trace + end event

离线入库链路：
    脚本 scripts/rebuild_kb_version.py
    → indexing/service.py ingest_directory()
        → load_file → normalize_documents → split_documents → Milvus add_documents
        → quality/ingestion.py 质量报告
        → governance/kb_versions.py 版本激活
"""
