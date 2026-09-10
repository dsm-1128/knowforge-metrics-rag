# KnowForge · 企业指标口径与数据字典知识平台

独立项目路径：`C:\Projects\knowforge-metrics-rag`。本项目包含自己的 Python 核心、页面、配置与指标场景资料，不需要启动原平台。模型权重可通过 `.env` 指向已有文件；迁移电脑时需要重新设置模型路径。

## 页面与权限

- 登录页：账号密码登录，密码使用 scrypt 摘要，登录令牌由服务端管理。
- 工作台与知识问答：流式回答、来源引用、个人会话历史。
- 业务分类独立页面：指标口径、数据字典、版本管理、数据血缘，分类可以直接进入对应范围的问答。
- 账号中心：显示账号绑定的租户、数据集、可见范围和角色，普通用户不能自行修改。
- 租户管理员：创建本租户用户，启用或停用账号；停用会撤销登录令牌。

角色为 `viewer`、`knowledge_admin`、`tenant_admin`。当前知识入库和版本激活由服务器 CLI 操作，页面不提供上传或激活功能。账号绑定一个数据集和一个可见范围。`public` 资料仍受租户与数据集限制；`internal` 可访问 public/internal，`private` 可访问 public/private，两者不是递增等级。服务端根据账号设置检索条件，拒绝浏览器提交租户、角色等身份字段。

## Windows 运行步骤

以下命令全部在 **PowerShell** 中运行，工作目录始终为新项目。首次安装建议 Python 3.12；不要使用机器默认的 Python 3.7。

### 1. 进入新项目并选择解释器

```powershell
cd C:\Projects\knowforge-metrics-rag
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py --version
# 新环境首次安装时执行；已有完整 rag 环境可跳过安装
& $py -m pip install -r requirements.txt
```

### 2. 配置本项目

```powershell
& $py scripts/configure.py --models-dir C:\Models
notepad .env
```

当前目录已生成 `.env`，再次运行 configure 不会覆盖。填写自己的 `DASHSCOPE_API_KEY`。三个模型路径需要分别指向 BGE-M3、BGE reranker 和已训练的 intent classifier。配置生成器不会复制原项目 API Key。若要完全脱离原目录，可把权重复制到本项目 `models`，再修改 `.env` 的模型路径。

### 3. 启动独立基础服务

打开 Docker Desktop，等待引擎就绪，然后运行：

```powershell
docker compose up -d
docker compose ps
```

MySQL 使用 `127.0.0.1:13308`，数据库 `knowforge_metrics`；Milvus 使用 `127.0.0.1:19531`；数据卷前缀为 `knowforge-metrics`。首次 MySQL 初始化需等待健康状态变为 healthy。Redis 默认关闭。启动后不要随意修改 MYSQL_PASSWORD，已有数据卷中的密码不会自动随配置改变。

### 4. 初始化账号

```powershell
& $py scripts/init_project.py --username admin --display-name '平台管理员' --tenant-id default --dataset-id default --visibility public
```

按提示输入两次自定密码（12–256 个字符，输入时不显示）。没有预置通用密码。账号表目前已初始化，仍需要这一步创建自己的登录账号。首次管理员属于 `default/default/public`，与下面入库范围一致。

### 5. 检查资料质量

```powershell
& $py scripts/quality/check_ingestion_quality_gate.py --scenario metrics_dictionary --kb-version metrics_dictionary_offline_check --output reports/offline_quality.json
```

输出 `"ok": true` 表示质量检查通过。它不等于已经入库或模型服务可用。

### 6. 构建并激活独立知识库

```powershell
& $py scripts/rebuild_kb_version.py --scenario metrics_dictionary --new-version --quality-gate --activate --tenant-id default --dataset-id default --visibility public --description 'KnowForge Metrics initial knowledge base'
```

成功输出应包含 `Rebuilt knowledge base version` 和 `activated=True`。知识库集合为 `knowforge_metrics_faq_v1`、`knowforge_metrics_doc_v1`。这一步只处理本项目资料和独立数据库。

新增租户/数据集时，入库参数必须与账号范围一致。版本激活指针仍为场景级，规划多个租户的知识版本时需要在同一目标版本中完成各范围入库后统一激活；不要把单租户版本直接当作所有租户的完整版本。

### 7. 启动网站

```powershell
& $py -m uvicorn app:app --host 127.0.0.1 --port 8001 --ws-max-size 20000
```

浏览器打开 **http://127.0.0.1:8001**。
## 验证与部署边界

```powershell
& $py -m pytest tests -q
node --check static/app.js
docker compose config --quiet
```

测试覆盖密码/令牌、登录、CSRF、跨用户会话、跨租户账号管理、WebSocket 服务端身份绑定。SQLite 仅用于隔离单元测试，实际运行使用 MySQL。

默认是本机单进程部署。对外服务需要 HTTPS，设置 `COOKIE_SECURE=true` 和准确的 `APP_ORIGINS`，并配置反向代理。当前内存限流与生成并发控制适用于单个 worker。不要把 `.env`、模型、运行日志或数据库凭据提交到版本库。

## 本次交付验证记录

2026-09-10：15 项测试通过；独立 MySQL 表初始化成功；质量门禁通过；8 条 FAQ、33 个文档片段已入库并激活，版本为 `kb_metrics_dictionary_20260910_033910_a1eeee40`。当前电脑可跳过上述第 5、6 步，资料变化后再执行。登录页 HTTP 200，账号数据库可用。

尚需用户填写 API Key，并通过第 4 步设置管理员密码。真实模型生成尚未验证，浏览器视觉与完整交互尚未验收；页面静态资源与 JavaScript 语法已检查。


## 中文回答与分类提示

各分类问答页提供“试着提问”卡片，切换知识范围会更新问题与输入提示。回答按“已确认 / 需确认 / 建议”分区显示；分区表示回答内容的性质，不代表完成了人工审批。模型仅在资料支持时列出事实，缺失项明确说明；常见英文术语附中文解释，表名、字段名和来源文件保留原文。FAQ 直答同样格式化，无需重新入库。历史消息在打开时按其已有标题排版；旧的纯文本历史不会自动生成新的业务内容。

## LangSmith 追踪

本地 `.env` 使用以下四项配置（不要将真实密钥写入示例模板）：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的LangSmith密钥
LANGSMITH_PROJECT=knowforge-metrics-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

开关只能填写 true/false，不能填密钥。修改后重启网站进程。当前已通过既有问答追踪适配器上传并读取一条合成测试记录，记录标识保存在 `reports/langsmith_verification.txt`。

在 LangSmith 的项目列表打开 `knowforge-metrics-rag`，查看 `qa_stream_query` 记录，可检查问题、答案摘要、知识库版本、分类、租户/数据集、来源与各阶段耗时。前端来源区域提供本轮追踪编号，方便查找。追踪开启后问答与来源摘录会发送到配置的 LangSmith 服务；账号密码不会作为追踪字段发送。完整真实问答的追踪请在重启后发起一次提问验证，合成连通性测试不代表已验证所有模型链路。

配置依据：[LangSmith 官方追踪说明](https://docs.langchain.com/langsmith/trace-with-langchain)。
