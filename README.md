# KnowForge · 企业指标口径与数据字典知识平台

独立项目路径：`C:\Projects\knowforge-metrics-rag`。本项目包含自己的 Python 核心、页面、配置与指标场景资料，不需要启动原平台。模型权重可通过 `.env` 指向已有文件；迁移电脑时需要重新设置模型路径。

## 页面与权限

- 登录页：账号密码登录，密码使用 scrypt 摘要，登录令牌由服务端管理。
- 工作台与知识问答：流式回答、来源引用、个人会话历史。
- 业务分类独立页面：指标口径、数据字典、版本管理、数据血缘，分类可以直接进入对应范围的问答。
- 账号中心：显示账号绑定的租户、数据集、可见范围和角色，普通用户不能自行修改。
- 租户管理员：创建本租户用户，启用或停用账号；停用会撤销登录令牌。

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

### 5. 检查资料质量

```powershell
& $py scripts/quality/check_ingestion_quality_gate.py --scenario metrics_dictionary --kb-version metrics_dictionary_offline_check --output reports/offline_quality.json
```

### 6. 构建并激活独立知识库

```powershell
& $py scripts/rebuild_kb_version.py --scenario metrics_dictionary --new-version --quality-gate --activate --tenant-id default --dataset-id default --visibility public --description 'KnowForge Metrics initial knowledge base'
```

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
