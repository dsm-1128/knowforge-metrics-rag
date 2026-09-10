"""FastAPI 路由层。

`app.py` 只负责创建 FastAPI 应用、挂载静态资源和注册路由。具体 HTTP/WebSocket
接口放在本包中，避免入口文件继续膨胀。

当前拆分边界：
- `pages.py`：页面入口、健康检查、会话创建；
- `chat.py`：问答、历史、反馈、检索调试、WebSocket 流式输出；
- `admin.py`：追踪、入库报告、评测报告等只读管理诊断；
- `kb_versions.py`：知识库版本查看、回滚、归档；
- `dependencies.py`：管理令牌、限流等协议层横切能力；
- `error_handlers.py`：统一异常处理器；
- `service_context.py`：API 层传给 QAService 的公共调用上下文。

为什么采用这种拆法：
- 页面、聊天、管理、版本这些接口变化频率不同，放在一起会让 `app.py` 重新变成
  难维护的大文件；
- RAG 主链路仍然由 QAService 和 pipeline 层承载，API 层只做协议适配；
- QAService 和 pipeline 仍然是业务核心，API 包只做参数转换和协议适配。

设计原则：
- 路由函数不做业务逻辑，只做参数校验、协议适配和异常转换。
- 所有业务编排委托给 QAService 和 pipeline 层。
- WebSocket 端点不走 FastAPI HTTP 异常机制，自行发送 type=error 事件。
"""
