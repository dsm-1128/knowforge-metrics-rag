"""运行期 MySQL schema bootstrap。（★★★ 核心）

API 启动、入库脚本和版本运维脚本在业务读写前显式调用这里；业务 Store 不再在方法
内部夹带 DDL 副作用。统一在启动期完成所有运行时表的建表操作。

设计决策：
- 所有运行时表名的 SQL 防注入标识符转义集中在此处完成（通过 safe_sql_identifier），
  确保无论表名来自配置还是常量，都不会产生 SQL 注入风险。
- 使用 pool_pre_ping=True 确保每次取连接前发送轻量 SELECT 1 探测，避免复用失效连接。
- 返回处理摘要（数据库名、表清单、SQL 文件路径和语句数量），供调用方日志输出。
"""

from __future__ import annotations

from sqlalchemy import create_engine

from qa_core.config.settings import get_settings
from qa_core.cache.namespaces import CACHE_NAMESPACE_TABLE
from qa_core.governance.chunk_versions import KB_CHUNK_VERSIONS_TABLE
from qa_core.governance.kb_versions import KB_ACTIVATION_TABLE, KB_ACTIVE_TABLE, KB_VERSIONS_TABLE
from qa_core.indexing.manifest import INDEX_MANIFEST_TABLE
from qa_core.memory.base import safe_sql_identifier
from qa_core.storage.mysql_schema import RUNTIME_SCHEMA_SQL, run_runtime_schema_sql


DEFAULT_CHAT_MESSAGES_TABLE = "chat_messages"


def bootstrap_mysql_schema() -> dict[str, object]:
    """初始化运行期 MySQL 表结构，并返回本次处理摘要。（★★★ 核心）

    执行流程：
      1. 读取配置中的 MySQL 连接 URI，创建带 pool_pre_ping 的数据库引擎。
      2. 对所有运行时表名做 SQL 防注入标识符转义（反引号包裹）。
      3. 调用 run_runtime_schema_sql() 将转义后的表名注入 SQL 模板并逐条执行 DDL。
      4. 返回包含数据库名、表清单、SQL 文件路径和语句数量的处理摘要。

    返回：
        字典，包含：
        - mysql_database: MySQL 数据库名。
        - tables: 已建表的表名列表。
        - schema_sql: SQL 模板文件路径。
        - statement_count: 实际执行的 SQL 语句数量。

    这个函数只负责“启动期建表”，不负责版本初始化、active 指针设置或业务数据
    写入。把 DDL 副作用集中到这里后，`IndexManifest`、`ChunkVersionIndex`
    和 `KnowledgeBaseVersionStore` 的构造函数都可以保持为纯访问对象。

    调用顺序：启动期 schema bootstrap -> bootstrap_mysql_schema()。
    """
    # 使用集中配置创建数据库连接，保证控制面表和业务 Store 指向同一个 MySQL 数据库。
    settings = get_settings()
    # pool_pre_ping=True 确保每次取连接前发送轻量 SELECT 1 探测，避免复用失效连接导致 2006 错误
    engine = create_engine(settings.mysql_sync_uri, pool_pre_ping=True)

    # 步骤 1：对所有运行时表名做 SQL 防注入标识符转义。
    # 表名来自配置或常量而非用户输入，但仍通过 safe_sql_identifier 增加安全层
    version_table = safe_sql_identifier(KB_VERSIONS_TABLE, label="KB versions table")
    active_table = safe_sql_identifier(KB_ACTIVE_TABLE, label="KB active table")
    activation_table = safe_sql_identifier(KB_ACTIVATION_TABLE, label="KB activation table")
    cache_namespace_table = safe_sql_identifier(CACHE_NAMESPACE_TABLE, label="Cache namespace table")
    chunk_version_table = safe_sql_identifier(KB_CHUNK_VERSIONS_TABLE, label="KB chunk versions table")
    manifest_table = safe_sql_identifier(INDEX_MANIFEST_TABLE, label="Index manifest table")
    feedback_table = safe_sql_identifier(settings.feedback_table_name, label="Feedback table")
    summary_table = safe_sql_identifier(settings.chat_summary_table_name, label="Chat summary table")
    chat_messages_table = safe_sql_identifier(DEFAULT_CHAT_MESSAGES_TABLE, label="Chat messages table")

    # 步骤 2：把安全表名映射交给 SQL runner，统一渲染并逐条执行 DDL。
    # 这样业务 Store 开始查询 active 版本、会话和反馈之前，表结构已经准备完成。
    statement_count = run_runtime_schema_sql(
        engine,
        {
            "KB_VERSIONS_TABLE": version_table,
            "KB_ACTIVE_TABLE": active_table,
            "KB_ACTIVATION_TABLE": activation_table,
            "CACHE_NAMESPACE_TABLE": cache_namespace_table,
            "KB_CHUNK_VERSIONS_TABLE": chunk_version_table,
            "INDEX_MANIFEST_TABLE": manifest_table,
            "FEEDBACK_TABLE": feedback_table,
            "CHAT_SUMMARY_TABLE": summary_table,
            "CHAT_MESSAGES_TABLE": chat_messages_table,
        },
    )

    # 调用方只需要摘要用于日志和诊断，不需要持有本次 bootstrap 创建的临时 engine。
    return {
        "mysql_database": settings.mysql_database,
        "tables": [
            version_table,
            active_table,
            activation_table,
            cache_namespace_table,
            chunk_version_table,
            manifest_table,
            feedback_table,
            summary_table,
            chat_messages_table,
        ],
        "schema_sql": str(RUNTIME_SCHEMA_SQL),
        "statement_count": statement_count,
    }
