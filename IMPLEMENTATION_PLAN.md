# 独立指标与数据字典平台实施计划

依据 DESIGN.md 及用户本轮明确要求执行。原项目不修改；新项目端口 8001。

## 架构与接口约定

提取 qa_core/config/scenarios 与必要入库脚本；独立 app.py 只挂载新的 webapp 路由，不挂载旧的无账号路由。MySQL 保存账号、登录令牌及会话归属，RAG 复用现有 MySQL/Milvus 引擎。登录 Cookie 为 HttpOnly，令牌服务端保存摘要，每个请求重新读取授权；写请求校验 Origin 和 CSRF。默认角色 viewer、knowledge_admin、tenant_admin。

前端只使用下列接口：
- POST /api/auth/login {username,password}；GET /api/auth/me 返回 {user:{id,username,display_name,tenant_id,role,dataset_id,visibility},csrf_token}；POST /api/auth/logout。
- GET /api/categories 返回 {categories:[{id,name,description,questions}]}。
- GET /api/sessions 返回 {sessions:[{id,title,created_at}]}；POST /api/sessions {title?} 返回同一会话对象；GET /api/sessions/{id}/messages 返回 {messages:[{role,content}]}；DELETE /api/sessions/{id}。
- WebSocket /api/stream?csrf_token=...：Cookie 鉴权，同源 Origin；发送 {query,session_id,source_filter:null或分类ID}，接收原 RAG start/status/token/end/error 事件。
- GET /api/admin/users 返回 {users:[user]}；POST /api/admin/users {username,display_name,password,role,dataset_id,visibility}，租户从管理员身份确定；PATCH /api/admin/users/{id} {enabled}。
- GET /api/status 返回 {rag_ready,detail}；不返回凭据、内部路径。

单个账号第一版绑定一个数据集和一个可见范围。所有数据域均由服务端账号派生。账号管理为租户管理员专用；知识库入库与激活通过显式 CLI 完成，避免提供跨租户的全局激活操作。

## 实施步骤

- [x] 1. 在新目录提取独立运行核心，生成配置模板、启动/初始化命令。
- [x] 2. 用失败测试驱动账号、登录会话、租户与会话归属检查。
- [x] 3. 实现新的登录页、工作台、业务分类页、问答、账号信息与账号管理 UI。
- [x] 4. 接通受保护 HTTP/WebSocket 路由与原 RAG 服务，禁止客户端指定身份字段。
- [x] 5. 回归鉴权、CSRF、跨用户会话访问、跨租户管理、资料质量；独立审查。
- [x] 6. 编写独立 README，验证从新路径导入、启动和运行；记录外部服务限制。

## 验证原则

生产持久化使用 MySQL；单元测试可以使用隔离 SQLite 数据库测试账号业务，不作为 RAG 运行替代方案。不得伪造在线检索结果或成功激活状态。未配置外部服务时允许登录/工作台启动，问答明确报告未就绪。演示资料不自动暴露给任意租户。

验证记录：15 项测试通过；Docker Compose 配置通过；真实 MySQL schema 初始化通过；资料质量门禁通过；独立 Milvus 中 8 FAQ/33 chunks 入库并激活。HTTP 登录页与健康检查通过。尚未填写 API Key、创建用户自定管理员密码；未执行真实浏览器视觉验收与模型生成端到端验证。
