"""RAG 编排层包：封装从用户查询到最终答案的完整流式问答管线。

业务流程图 Stage 契约（见 docs/animation/business-flow.html）：
Stage 0 create_query_context -> Stage 1 decide_route -> Stage 2 prepare_retrieval
-> Stage 3 search_faq -> Stage 4 search_doc -> Stage 5 prepare_answer
-> Stage 6 stream_llm_answer -> Stage 7 finish_success/save_history/Trace。

包含模块：
- rag.py：主流程编排（stream_query / debug_retrieval），协调 Stage 0-7 管线。
- steps.py：业务步骤（路由决策、检索准备、上下文构建、LLM 流式生成）。
- retrieval_steps.py：检索执行步骤（FAQ 检索、文档检索）。
- runtime.py：单次请求状态上下文（RAGQueryContext），阶段计时和 LangSmith Trace。
- events.py：WebSocket 事件构造器（start/status/token/end/error）。
- citations.py：答案引用来源后处理修补。
- context.py：上下文筛选与格式化。
- confidence.py：最终答案置信度计算。
- query_input.py：用户查询归一化（问候/礼貌前缀剥离）。
- query_variants.py：检索查询扩展（同义替换）。
- rewrite.py：追问改写（指代消解）。

调用方典型用法：from qa_core.pipeline.rag import stream_query
"""
