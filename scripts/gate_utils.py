"""门禁脚本公共判断工具。

入库质量门禁和评测门禁的业务指标不同，但失败项结构、最大/最小阈值判断、必填项判断
完全一样。抽到这里后，两个门禁脚本只保留"检查哪些指标"的业务含义。

使用方式：
    from scripts.gate_utils import to_count, add_max_failure, add_min_failure, add_required_failure
    failures = []
    add_max_failure(failures, metric="p99_latency_ms", actual=2500, maximum=2000,
                    message="P99 延迟超过 2000ms")
"""

from __future__ import annotations

# typing.Any: 任意类型（用于接收报告中的各种值类型：列表、字典、数字等）
from typing import Any


def to_count(value: Any) -> int:
    """把报告里的列表、字典或数字统一转成可比较的数量。

    参数:
        value: 任意类型的输入值（可能是 list、dict、int、None 等）

    返回:
        整数计数值。None 返回 0；列表/字典/集合返回其长度；数字返回 int(value)

    执行流程:
        1. None 检查 → 直接返回 0
        2. 容器类型检查（list/tuple/set/dict）→ 返回 len(value)
        3. 数字类型 → 尝试转 int
        4. 转换失败 → 返回 0（容错处理）
    """
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def add_max_failure(
    failures: list[dict[str, Any]],
    *,
    metric: str,
    actual: float,
    maximum: float,
    message: str,
) -> None:
    """当实际值高于最高允许值时追加失败项。

    参数:
        failures: 失败项列表（调用方维护，函数直接向其追加）
        metric: 指标名称（如 "p99_latency_ms"）
        actual: 实际测量值
        maximum: 最高允许阈值
        message: 失败描述信息（中文，可直接展示在报告中）

    失败项格式:
        {"metric": metric, "actual": actual, "threshold": maximum, "message": message}

    调用顺序：命令行入口 -> add_max_failure()。
    """
    if actual > maximum:
        failures.append({"metric": metric, "actual": actual, "threshold": maximum, "message": message})


def add_min_failure(
    failures: list[dict[str, Any]],
    *,
    metric: str,
    actual: float,
    minimum: float,
    message: str,
) -> None:
    """当实际值低于最低要求时追加失败项。

    参数:
        failures: 失败项列表
        metric: 指标名称
        actual: 实际测量值
        minimum: 最低要求阈值
        message: 失败描述信息

    失败项格式:
        {"metric": metric, "actual": actual, "threshold": minimum, "message": message}

    调用顺序：命令行入口 -> add_min_failure()。
    """
    if actual < minimum:
        failures.append({"metric": metric, "actual": actual, "threshold": minimum, "message": message})


def add_required_failure(
    failures: list[dict[str, Any]],
    *,
    metric: str,
    actual: Any,
    enabled: bool,
    message: str,
) -> None:
    """当必填项缺失时追加失败项。

    与阈值函数不同，这里不做大小比较，而是检查值是否存在（truthiness）。
    enabled 参数用于条件开启：某些门禁项在特定场景下才执行检查。

    参数:
        failures: 失败项列表
        metric: 指标名称
        actual: 实际值（预期应非空）
        enabled: 是否启用该项检查（False 时跳过）
        message: 失败描述信息

    执行流程:
        1. 检查 enabled 是否为 True
        2. 检查 actual 的 truthiness（空字符串、None、空列表等均为缺失）
        3. 两者都满足时追加失败项
    """
    if enabled and not actual:
        failures.append({"metric": metric, "actual": actual, "threshold": "required", "message": message})
