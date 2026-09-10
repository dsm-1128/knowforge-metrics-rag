"""脚本层公共工具。

`scripts/` 里的文件越来越多，如果每个脚本都重复处理 JSON 读写、UTF-8 输出、命令执行
和报告保存，阅读时会被样板代码淹没。本模块只放"脚本基础设施"，不放 RAG 业务
规则；业务规则仍留在各自脚本里。

使用方式：
    from scripts.common import read_json_file, write_json_file, run_command_step
    report = run_command_step("验收步骤 A", ["python", "-m", "pytest"])
"""

from __future__ import annotations

# json: 标准 JSON 序列化（用于中文友好输出，ensure_ascii=False）
import json

# os: 操作系统接口（子进程环境变量继承）
import os

# subprocess: 子进程管理（subprocess.run 执行外部命令）
import subprocess

# sys: 系统功能（sys.stdout.reconfigure 设置 UTF-8 编码）
import sys

# time: 时间功能（time.perf_counter 高精度计时）
import time

# dataclasses: 数据类定义（CommandStepResult）
from dataclasses import dataclass

# datetime: 日期时间（UTC 时间戳）
from datetime import datetime, timezone

# pathlib.Path: 文件路径操作
from pathlib import Path

# typing.Any: 任意类型
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandStepResult:
    """一条命令式验收步骤的执行结果。

    Attributes:
        name: 步骤名称（如"检查环境变量"、"运行测试用例"）
        command: 实际执行的命令列表（subprocess.run 的 args 参数）
        ok: 是否成功（returncode == 0）
        elapsed_ms: 执行耗时，单位毫秒
        stdout_preview: 标准输出预览（自动截断，避免日志过大）
        stderr_preview: 标准错误预览（自动截断）
        returncode: 子进程返回码

    调用顺序：命令行入口 -> CommandStepResult。
    """

    name: str
    command: list[str]
    ok: bool
    elapsed_ms: float
    stdout_preview: str
    stderr_preview: str
    returncode: int


def configure_utf8_stdio() -> None:
    """把脚本标准输出统一成 UTF-8。

    Windows PowerShell 的默认编码可能不是 UTF-8，一键验收里又会输出中文 JSON。这里集中
    处理，避免每个脚本单独写一遍编码保护。

    执行流程：
        1. 检查 sys.stdout 是否支持 reconfigure
        2. 支持则将编码设为 utf-8，errors 采用 replace 容错模式
        3. 对 sys.stderr 执行同样的操作
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_json_file(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件并返回 dict 对象。

    参数:
        path: JSON 文件路径（字符串或 Path 对象）

    返回:
        解析后的字典对象

    异常:
        FileNotFoundError: 文件不存在时抛出
        json.JSONDecodeError: JSON 格式错误时抛出

    调用顺序：命令行入口 -> read_json_file()。
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    """返回 UTC 时间字符串（ISO 8601 格式）。

    发布、验收和质量报告都用这个函数生成时间，避免每个脚本重复导入 datetime，也避免
    一部分报告使用本地时间、一部分报告使用 UTC。

    返回:
        形如 "2026-07-11T08:30:00.123456+00:00" 的 ISO 格式字符串

    调用顺序：命令行入口 -> utc_now()。
    """
    return datetime.now(timezone.utc).isoformat()


def write_json_file(path: str | Path, payload: dict[str, Any]) -> str:
    """把对象写成中文友好的 JSON 文件，并返回写入路径。

    参数:
        path: 输出文件路径
        payload: 要序列化的字典数据

    返回:
        写入后的文件路径字符串

    执行流程:
        1. 确保目标目录存在（自动创建父目录）
        2. 以 UTF-8 编码写入，ensure_ascii=False 保留中文
        3. 缩进 2 空格便于人工阅读
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def print_json(payload: dict[str, Any]) -> None:
    """按统一格式打印 JSON 到标准输出。

    参数:
        payload: 要打印的字典数据

    说明:
        与 write_json_file 格式保持一致（ensure_ascii=False, indent=2），
        方便在终端直接查看中文内容。

    调用顺序：命令行入口 -> print_json()。
    """
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_optional_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    """当调用方提供路径时写 JSON；未提供时什么都不做。

    参数:
        path: 输出文件路径（为 None 时跳过写入）
        payload: 要序列化的字典数据

    使用场景:
        验收脚本中，仅在 --report 参数指定时才生成报告文件。

    调用顺序：命令行入口 -> write_optional_json()。
    """
    if path:
        write_json_file(path, payload)


def preview_text(text: str, limit: int = 1200) -> str:
    """截断长输出，避免验收报告被命令日志撑爆。

    参数:
        text: 原始文本内容
        limit: 保留的最大字符数（默认 1200）

    返回:
        截断后的文本。若原始内容超过限制，末尾追加 "\\n..." 表示被截断

    执行流程:
        1. 去除首尾空白
        2. 长度检查：不超过 limit 直接返回
        3. 超过 limit 则截断并追加省略标记
    """
    compact = (text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "\n..."


def run_command_step(name: str, command: list[str], *, preview_limit: int = 1200) -> CommandStepResult:
    """运行一个验收命令并记录结果。

    这里不用 shell 拼接命令，是为了减少 Windows 下路径、引号和转义问题。每个步骤独立
    执行，某一步失败后仍继续跑后续步骤，最后统一汇总失败项。

    参数:
        name: 步骤名称（用于报告展示）
        command: 命令及参数列表（如 ["python", "-m", "pytest"]）
        preview_limit: 输出截断长度

    返回:
        CommandStepResult 数据类实例

    执行流程:
        1. 记录开始时间（time.perf_counter 高精度计时）
        2. 通过 subprocess.run 执行命令，设置 UTF-8 编码并捕获输出
        3. 计算耗时并截断输出预览
        4. 打包返回 CommandStepResult
    """
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return CommandStepResult(
        name=name,
        command=command,
        ok=completed.returncode == 0,
        elapsed_ms=elapsed_ms,
        stdout_preview=preview_text(completed.stdout, preview_limit),
        stderr_preview=preview_text(completed.stderr, preview_limit),
        returncode=completed.returncode,
    )
