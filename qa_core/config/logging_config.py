"""问答服务的统一日志配置，qa_core 主链路使用的唯一日志入口。

设计决策：
- 使用 RotatingFileHandler 控制日志文件大小（20MB），避免单文件无限增长导致
  磁盘空间耗尽。保留 5 个备份文件，兼顾历史留存和磁盘空间控制。
- 同时输出到文件和控制台：文件用于运维复盘，控制台用于 Docker logs 和本地开发。
- 同一个 logger name 只添加一次 handler，避免热重载或测试重复导入时重复输出。

调用顺序：进程启动时 -> logging_config。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from qa_core.config.settings import PROJECT_ROOT, get_settings


def get_logger(name: str = "MultiScenarioRAG") -> logging.Logger:
    """返回同时输出到文件和控制台的已配置日志器。（★★ 理解）

    同一个 logger name 只添加一次 handler，避免热重载或测试重复导入时重复输出。
    日志统一写入 PROJECT_ROOT/logs/，目录自动创建。

    参数：
        name: 日志器名称，建议使用模块的 __name__ 或统一名称（如 "MultiScenarioRAG"）。

    返回：
        已配置的 logging.Logger 实例，包含文件和控制台两个 handler。

    调用顺序：启动配置或前置校验 -> get_logger()。
    """
    # 获取运行时配置中的日志级别，支持 DEBUG/INFO/WARNING/ERROR 四档控制台和文件输出
    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level.upper())
    if logger.handlers:
        # 已有 handler 说明该 logger 在本进程已初始化过，避免重复绑定导致日志行重复输出
        return logger

    # 日志统一写入 PROJECT_ROOT/logs/，确保目录存在否则运行时炸
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # 格式含时间戳、级别、日志器名、消息体，便于日志链路回溯
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    # 文件处理器：20MB 回滚 + 5 个备份，兼顾历史留存和磁盘空间控制
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器：Docker 场景下 stdout 被容器运行时接管，本地开发直接看终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 文件用于运维复盘，控制台用于 Docker logs 和本地开发
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

