"""Milvus 检索过滤表达式构造。

这里把 source_filter、kb_version 和 DataScope 合并成一个 Milvus boolean expr，
供 similarity_search_with_score 调用。所有字符串都会转义，避免把用户输入直接拼进
Milvus 表达式造成过滤条件注入。
"""

from __future__ import annotations

from qa_core.governance.data_scope import DataScope, escape_expr_value
from qa_core.governance.kb_versions import get_kb_version_store


def validate_source_filter(
    source_filter: str | None,
    valid_sources: list[str],
) -> None:
    """校验 source_filter 白名单。source_filter='hr', valid_sources=['hr','legal'] -> pass; source_filter='ai', ... -> ValueError。

    参数：
        source_filter: 要校验的 source 值。
        valid_sources: 当前场景允许的 source 白名单。

    异常：
        ValueError: source_filter 不在 valid_sources 中。

    调用顺序：检索准备或检索执行 -> validate_source_filter()。
    """
    # 当 source_filter 不为 None 时，必须存在于 valid_sources 白名单中
    # 原因：防止前端传递了当前业务场景下不存在的分类值，导致检索范围异常
    # 注意：source_filter 为 None 表示前端未选择任何分类过滤，此处不做校验，由业务自行处理
    if source_filter and source_filter not in valid_sources:
        raise ValueError(f"无效的业务分类：{source_filter}")


def build_source_expr(
    source_filter: str | None,
    kb_version: str | None = None,
    data_scope: DataScope | None = None,
    *,
    scenario_id: str | None = None,
    source_type: str | None = None,
) -> str | None:
    """合并 source/kb_version/data_scope 为 Milvus 布尔表达式。

    示例：
        ``build_source_expr("hr", "v1")`` returns
        ``'source == "hr" and kb_version == "v1"'``.

    执行流程：
      1. source_filter 存在时添加 source == "<value>"。
      2. FAQ 按 kb_version 精确过滤；文档按 valid_from_seq/valid_to_seq 构造引用式版本视图。
      3. data_scope 存在时追加租户、数据集、可见级别和角色过滤。
      4. 所有子句用 and 连接；没有任何约束时返回 None。

    参数：
        source_filter: 业务分类过滤项。
        kb_version: 知识库版本。
        data_scope: 数据隔离范围。

    返回：
        Milvus boolean expr；没有约束时返回 None。
    """
    clauses: list[str] = []
    # 逐项拼接过滤子句：业务分类 + KB 版本 + 数据域，每项值做转义防止 Milvus 表达式注入
    # Milvus boolean expr 最终用 and 连接所有子句，实现组合过滤

    if source_filter:
        # 对 source 值做转义，防止包含引号或其他特殊字符破坏 Milvus 表达式结构
        safe_source = escape_expr_value(str(source_filter))
        clauses.append(f'source == "{safe_source}"')

    # KB 版本过滤分两种场景：
    # 1. 文档检索（source_type == "doc"）：使用 valid_from_seq/valid_to_seq 区间匹配，支持版本快照
    # 2. FAQ 检索或其他：直接用精确的 kb_version 字段匹配
    if kb_version and source_type == "doc" and scenario_id:
        # 文档场景：将语义版本号（如 "v1.2"）解析为递增序列号 version_seq
        # 过滤逻辑：文档的 valid_from_seq <= 当前版本序列号 < valid_to_seq
        # valid_to_seq == 0 表示后续版本无限制（即始终有效）
        version = get_kb_version_store(scenario_id).resolve_version(kb_version)
        clauses.append(
            f"(valid_from_seq <= {int(version.version_seq)} and "
            f"(valid_to_seq == 0 or valid_to_seq > {int(version.version_seq)}))"
        )
    elif kb_version:
        # FAQ 等非文档场景：字段直接等于版本号，精确过滤
        safe_version = escape_expr_value(str(kb_version))
        clauses.append(f'kb_version == "{safe_version}"')

    if data_scope is not None:
        # 数据域隔离：追加租户、数据集、可见级别和角色的过滤子句
        # data_scope.expr_clauses() 返回多个子句列表（如 tenant_id == "xxx" OR tenant_id == "" 等）
        clauses.extend(data_scope.expr_clauses())

    # 没有任何过滤约束时返回 None，而非空字符串 ""，调用方可通过 None 判断是否需要剪枝
    return " and ".join(clauses) if clauses else None

