"""模型适配层：通过 OpenAI 兼容接口创建 LLM 客户端的工厂。

子模块说明：
- client：LLM 客户端工厂，通过 ChatOpenAI 兼容层接入 DashScope 等模型服务，
  避免在主链路里绑定特定厂商 SDK。提供 get_chat_model()、validate_llm_connectivity()
  和 LLM 运行时状态监控。

设计决策：
- 切换模型时只改配置（LLM_MODEL / DASHSCOPE_BASE_URL），不改业务代码。
- 使用 lru_cache 缓存流式/非流式两个客户端实例，避免每次调用都重建 TCP 连接。
- LLM 连通性是旁路监控：启动时不阻断（由 preflight 校验 API Key），运行时通过
  /health 和管理 API 暴露状态。

调用顺序：pipeline 阶段或启动配置 -> llm。
"""

