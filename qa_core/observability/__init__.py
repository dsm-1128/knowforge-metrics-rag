"""可观测性适配器：可选的 LangSmith Tracing 数据上报。

本地评估报告和质量门禁是默认的质量循环（local evaluation reports / eval_sets / gates）。
此包只包含 RAG 运行时用来附加业务元数据的 LangSmith 适配器。当 LANGSMITH_TRACING
启用时，pipeline 收尾阶段调用 record_query_trace() 将问答请求记录到 LangSmith。

设计决策：
- LangSmith 是可选 Trace 旁路：未启用或网络失败时只记录日志，不影响用户请求。
- 本地 Evaluation 和 Gate 不依赖 LangSmith，始终使用 local reports、eval_sets
  和 gate 脚本作为默认质量循环。

调用顺序：pipeline 收尾或管理 API -> observability。
"""

