"""qa_core 内部复用的轻量公共工具：UTC 时间、JSON 文件读写、文件更新时间等。

此模块放置所有不依赖其他 qa_core 模块的纯函数工具。设计原则：
- 零依赖或只依赖 Python 标准库。
- 不连接外部服务（MySQL / Redis / LLM）。
- 不读取运行时配置（get_settings）。

这样所有其他 qa_core 模块都可以安全导入 common 而不会产生循环依赖。

调用顺序：所有 qa_core 模块 -> common。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """返回 UTC ISO 时间字符串，用于版本管理、报告和 trace 的时间统一排序。

    所有时间相关操作统一使用 UTC，避免时区不一致导致的排序和比较问题。

    返回：
        UTC ISO 格式字符串，如 "2026-07-26T10:30:00.123456"。

    调用顺序：上游业务入口 -> utc_now()。
    """
    return datetime.now(timezone.utc).isoformat()


def utc_file_stamp() -> str:
    """返回适合放进文件名和版本号的 UTC 时间戳（YYYYMMDD_HHMMSS）。

    与 utc_now() 不同，此函数输出的格式不含冒号等文件系统不兼容字符，
    适合直接用作文件名或版本号后缀。

    返回：
        格式化的 UTC 时间戳字符串，如 "20260726_103000"。

    调用顺序：上游业务入口 -> utc_file_stamp()。
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def path_updated_at(path: str | Path) -> str:
    """返回文件修改时间对应的 UTC ISO 字符串。

    参数：
        path: 文件路径。

    返回：
        UTC ISO 格式的修改时间字符串。

    调用顺序：上游业务入口 -> path_updated_at()。
    """
    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


def read_json(path: str | Path, default: Any = None) -> Any:
    """读取 JSON 文件，读取失败时返回 default。

    捕获 json.JSONDecodeError 和 OSError，确保文件不存在或 JSON 格式错误时
    不会抛出异常。适用于"尝试读取，不存在就用默认值"的场景。

    参数：
        path: JSON 文件路径。
        default: 读取失败时返回的默认值。

    返回：
        解析后的 JSON 对象；读取失败时返回 default。

    调用顺序：上游业务入口 -> read_json()。
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_json_dict(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取对象型 JSON 文件，非对象或读取失败时返回字典默认值。

    与 read_json 的区别：此函数强制返回字典类型。如果 JSON 内容是数组或基本类型，
    会返回 default。适用于"预期读取一个 JSON 对象"的场景。

    参数：
        path: JSON 文件路径。
        default: 读取失败或内容非对象时的默认字典。

    返回：
        解析后的字典；内容非 dict 或读取失败时返回 default。

    调用顺序：上游业务入口 -> read_json_dict()。
    """
    payload = read_json(path, default=None)
    if isinstance(payload, dict):
        return payload
    return dict(default or {})


def write_json(path: str | Path, payload: Any) -> str:
    """按项目统一格式（ensure_ascii=False, indent=2）写入 JSON 文件。

    目录不存在时自动创建。ensure_ascii=False 保证中文等非 ASCII 字符
    以原始形式存储。indent=2 让输出的 JSON 文件可读性好且适合版本控制。

    参数：
        path: 输出文件路径。
        payload: 待序列化的 Python 对象。

    返回：
        写入的文件路径字符串。

    调用顺序：上游业务入口 -> write_json()。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def list_json_reports(root: Path, glob_pattern: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """列出目录下最近的 JSON 报告，每个条目包含 path/file_name/updated_at/payload。

    按修改时间降序排列，最新的报告排在前面。可选的 limit 参数控制返回条数。

    参数：
        root: 报告目录路径。
        glob_pattern: 文件匹配模式（如 "*.json"）。
        limit: 最大返回条数（默认 20）。

    返回：
        包含 path、file_name、updated_at 和 payload 的字典列表。

    调用顺序：管理 API 或报告查看 -> list_json_reports()。
    """
    if not root.exists():
        return []
    reports: list[dict[str, Any]] = []
    # 按修改时间降序排列，最新的报告排在前面
    for path in sorted(root.glob(glob_pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json_dict(path)
        reports.append(
            {
                "path": str(path),
                "file_name": path.name,
                "updated_at": path_updated_at(path),
                "payload": payload,
            }
        )
        if limit and len(reports) >= limit:
            break
    return reports
