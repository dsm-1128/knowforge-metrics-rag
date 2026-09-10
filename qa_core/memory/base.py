"""MySQL 存储基类：延迟创建 SQLAlchemy 引擎并统一管理 MySQL 连接。

设计决策：
- SQLAlchemy 引擎延迟创建：首次访问 engine 属性时才连接数据库，避免构造对象时
  阻塞或报错（如 MySQL 未启动）。
- pool_pre_ping=True：每次从连接池取出连接前先执行 SELECT 1 校验连接是否可用，
  避免长时间空闲后连接失效导致的"MySQL server has gone away"错误。
- safe_sql_identifier() 校验表名/索引名，避免配置项注入 SQL。

调用顺序：问答历史或反馈存储 -> base。
"""

from __future__ import annotations

import re

from sqlalchemy import create_engine

from qa_core.config.settings import get_settings


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_sql_identifier(value: str, *, label: str = "SQL identifier") -> str:
    """校验可拼接到 SQL 中的表名/索引名，避免配置项注入 SQL。

    只允许字母、数字和下划线开头的标识符，拒绝包含空格、引号、分号等 SQL 特殊字符
    的输入。所有配置项中拼接到 SQL 的表名都必须经过此函数校验。

    参数：
        value: 表名或索引名。
        label: 错误信息中的名称描述（如 "Cache namespace table"）。

    返回：
        校验通过的原值（不会转义——因为只允许安全字符，无需转义）。

    调用顺序：CacheNamespaceStore / FeedbackStore / ChatHistoryStore -> safe_sql_identifier()。
    """
    if not _SQL_IDENTIFIER_RE.fullmatch(value or ""):
        raise ValueError(f"{label} 不合法：{value!r}")
    return value


class _MySqlStore:
    """MySQL 存储的轻量基类，统一加载项目配置并延迟创建 SQLAlchemy 引擎。

    子类只需要关心自己的表名和业务方法，无需重复处理 MySQL 连接和配置加载。
    所有子类共享同一个数据库连接池，避免每个存储类独立创建连接消耗资源。

    调用顺序：问答历史或反馈存储 -> _MySqlStore。
    """

    def __init__(self) -> None:
        """初始化 MySQL 存储基类。

        加载全局配置并延迟创建引擎（首次访问 engine 属性时才连接数据库）。
        构造对象时不会立即连接 MySQL，避免启动时因 MySQL 不可用而崩溃。

        调用顺序：ChatHistoryStore / FeedbackStore / CacheNamespaceStore -> _MySqlStore.__init__()。
        """
        self.settings = get_settings()
        self._engine = None  # 首次访问 engine 属性时延迟创建

    @property
    def engine(self):
        """延迟创建带连接健康检查（pool_pre_ping）的 SQLAlchemy 同步引擎。

        create_engine 本身是轻量操作（不建立 TCP 连接），真正的连接在第一次 SQL
        执行时创建。pool_pre_ping=True 确保取出连接前校验连接有效性，避免使用
        已断开的连接。

        返回：
            SQLAlchemy Engine 实例。

        调用顺序：子类业务方法 -> _MySqlStore.engine()。
        """
        if self._engine is None:
            # 延迟创建 SQLAlchemy 引擎（带连接健康检查 pool_pre_ping）
            # 原因：pool_pre_ping=True 避免"MySQL server has gone away"错误
            self._engine = create_engine(self.settings.mysql_sync_uri, pool_pre_ping=True)
        return self._engine
