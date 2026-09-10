"""配置层：运行时配置、规则配置、日志配置和前置校验。

子模块说明：
- settings：通过进程环境变量和 .env 文件加载的运行时配置，负责读取配置值。
- rules：从 config/rules.toml 加载的规则配置，负责 FAQ 快路径、查询变体、
  intent 规则分数和检索策略等确定性规则。
- logging_config：日志配置，qa_core 主链路使用的统一日志入口。
- preflight：主链路运行环境前置校验，Milvus、MySQL、本地模型、LLM Key 等
  基础条件不满足时直接报错阻断启动。

调用顺序：进程启动时 -> config。
"""

