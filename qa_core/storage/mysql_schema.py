"""MySQL schema SQL runner.（★★ 理解）

表结构 DDL 定义在 ``runtime_schema.sql`` 文件中。本模块只负责渲染表名占位符
并在启动时或脚本 bootstrap 时执行该文件。

设计决策：
- DDL 与代码分离：建表语句放在独立的 .sql 文件中，便于 DBA 审查和版本对比。
- 表名占位符机制：使用 ``{{TABLE_NAME}}`` 语法在运行时注入真实表名，
  允许同一套 SQL 模板服务不同的表名配置。
- 引号感知的 SQL 拆分：_split_sql_statements() 逐字符扫描，正确区分引号内和
  引号外的分号，避免将存储过程或字符串中的分号误当作语句边界。
- 安全：调用方通过 safe_sql_identifier() 转义表名后才注入模板，防止 SQL 注入。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from sqlalchemy import text


RUNTIME_SCHEMA_SQL = Path(__file__).with_name("runtime_schema.sql")
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def run_runtime_schema_sql(engine, variables: Mapping[str, str]) -> int:
    """执行 runtime_schema.sql 并返回实际执行的 SQL 语句数量。（★★★ 核心）

    执行流程：
      1. 检查 runtime_schema.sql 文件是否存在，丢失时直接阻断启动。
      2. 读取 SQL 模板，将其中 {{TABLE_NAME}} 占位符替换为运行时确定的真实表名。
      3. 去除行注释并按分号拆分，得到可逐条执行的 SQL 语句。
      4. 在事务中逐条执行 DDL，任一语句失败会整体回滚。

    参数：
        engine: SQLAlchemy 数据库引擎实例。
        variables: 表名映射字典，key 为占位符名称，value 为安全转义后的表名。

    返回：
        实际执行的 SQL 语句数量。

    异常：
        RuntimeError: runtime_schema.sql 文件不存在。

    调用顺序：bootstrap_mysql_schema() -> run_runtime_schema_sql()。
    """
    # SQL 模板是运行时表结构的唯一来源，缺失时不能让服务带着半套表结构启动。
    if not RUNTIME_SCHEMA_SQL.exists():
        # SQL 模板文件丢失时直接阻断，避免服务启动后缺少核心表导致业务层报"表不存在"
        raise RuntimeError(f"MySQL runtime schema SQL file is missing: {RUNTIME_SCHEMA_SQL}")

    # 步骤 1：渲染 SQL 模板中的表名占位符。
    # 原因：表名在运行时才能确定（配置或常量），通过占位符注入避免硬编码
    rendered = _render_sql(RUNTIME_SCHEMA_SQL.read_text(encoding="utf-8"), variables)
    # 步骤 2：拆分 SQL 为可逐条执行的语句。
    # 去除行注释并按分号拆分，支持字符串字面量中的分号不被误拆分
    statements = _split_sql_statements(rendered)
    # 步骤 3：在事务中逐条执行 DDL。
    # engine.begin() 自动提交事务，任一语句失败会整体回滚
    with engine.begin() as conn:
        # engine.begin() 保证全部 DDL 在同一事务边界内执行，失败时由上下文回滚。
        for statement in statements:
            # 每条语句独立提交给 SQLAlchemy，便于数据库驱动正确处理 DDL。
            conn.execute(text(statement))
    return len(statements)


def _render_sql(source: str, variables: Mapping[str, str]) -> str:
    """渲染 SQL 模板中的表名占位符，将 {{NAME}} 替换为实际表名。

    参数：
        source: SQL 模板字符串，包含 {{PLACEHOLDER_NAME}} 形式的占位符。
        variables: 占位符名称到真实表名的映射字典。

    返回：
        替换完成后的 SQL 字符串。

    异常：
        RuntimeError: 模板中出现了 variables 未提供的占位符。

    调用顺序：run_runtime_schema_sql() -> _render_sql()。
    """

    def replace(match: re.Match[str]) -> str:
        """替换 SQL 模板中的占位符（{{NAME}}）为实际值。

        参数：
            match: 正则匹配结果，group(1) 为占位符名称（大写字母、数字、下划线）。

        返回：
            替换后的实际表名字符串（已由调用方通过 safe_sql_identifier() 转义）。

        异常：
            RuntimeError: 占位符在 variables 中找不到对应值。

        调用顺序：启动期 schema bootstrap -> replace()。
        """
        name = match.group(1)
        if name not in variables:
            # 模板中出现了调用方未提供的占位符，阻止生成不完整的 DDL，避免漏建表
            raise RuntimeError(f"MySQL schema placeholder is not provided: {name}")
        return variables[name]

    # 全局搜索 {{VARIABLE_NAME}} 模式，每匹配到一处就回调 replace 执行替换
    return _PLACEHOLDER_RE.sub(replace, source)


def _split_sql_statements(sql: str) -> list[str]:
    """将 SQL 文件内容拆成可逐条执行的语句。（★★ 理解）

    逐字符扫描，正确区分引号内分号和引号外分号，避免误将字符串字面量中的分号
    当作语句分隔。支持单引号、双引号和转义字符。

    参数：
        sql: 去除行注释后的 SQL 文本。

    返回：
        可逐条执行的 SQL 语句列表（已去除尾部分号）。

    调用顺序：run_runtime_schema_sql() -> _split_sql_statements()。
    """
    content = _strip_line_comments(sql)
    statements: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escape_next = False

    # 逐字符扫描，区分引号内分号和引号外分号
    # 原因：避免将存储过程中的分号或字符串字面量中的分号误当作语句分隔
    for char in content:
        current.append(char)
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if char == ";" and not in_single_quote and not in_double_quote:
            # 遇到引号外的分号时截断一条完整语句，去尾分号后加入结果列表
            statement = "".join(current).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            current = []

    # 文件末尾可能无尾部分号，剩余内容也是一条完整语句
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _strip_line_comments(sql: str) -> str:
    """移除 SQL 文件中的空行和 -- 行注释。

    参数：
        sql: 原始 SQL 文本（含注释行和空行）。

    返回：
        仅包含 DDL 语句行的纯文本。

    调用顺序：_split_sql_statements() -> _strip_line_comments()。
    """
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        # 过滤 `--` 开头的行注释和空行，保留实际 DDL 语句行
        if not stripped or stripped.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines)
