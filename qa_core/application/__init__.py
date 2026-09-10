"""应用服务层包。

提供问答 RAG 编排服务的定义和工厂方法。包含：
- service.py：核心 QAService，实现流式问答和检索调试的完整工作流。
- factory.py：lru_cache 管理的进程级单例工厂。

请从具体子模块导入，例如 `from qa_core.application.service import QAService`。
"""