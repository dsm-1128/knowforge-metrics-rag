"""数据域隔离模型与检索过滤工具。（★★★ 核心）

DataScope 是检索安全边界的基础模型，统一携带 tenant_id、dataset_id、visibility
和用户角色。入库时这些字段会写入 Milvus metadata；检索时会转换成 Milvus expr，
确保多租户、多数据集和角色权限不会互相串数据。

设计决策：
- 使用 frozen dataclass 确保数据域在构造后不可变，避免在检索链中被意外修改。
- 默认值机制：所有字段都有默认值，确保非登录态用户至少可以访问 public 数据。
- 向下兼容：兼容 user_role（单值，来自 JWT token）和 user_roles（多值，来自 API 参数）
  两套角色传参方式，通过 from_request 合并去重。
- Milvus expr 构建：visibility 策略确保 public 始终可见，internal/private 只对同级及以上开放。
- 角色过滤使用 OR 语义：用户任一角色命中 allowed_roles 即可访问，适用于多角色用户场景。
- 字符串值通过 escape_expr_value 转义，防止 Milvus expr 注入。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
DEFAULT_TENANT_ID = "default"
DEFAULT_DATASET_ID = "default"
DEFAULT_VISIBILITY = "public"
DEFAULT_USER_ROLE = "public"

def _clean_token(value: str | None, default: str) -> str:
    """清洗短标识，去除空白并提供默认值。

    参数：
        value: 原始输入值，可能为空或只有空白字符。
        default: 清洗后为空时使用的默认值。

    返回：
        清洗后的非空字符串。

    调用顺序：DataScope.from_request() -> _clean_token()。
    """
    cleaned = str(value or "").strip()
    # 空字符串回退到默认值：确保 tenant_id、dataset_id 等标识永远不会为空字符串，
    # 避免 Milvus expr 过滤时空串匹配到错误的数据
    return cleaned or default


def _clean_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """清洗字符串列表，顺序去重。

    参数：
        values: 原始字符串列表、元组或 None。

    返回：
        清洗后且按原顺序去重的列表。

    调用顺序：DataScope.from_request() -> _clean_list()。
    """
    result: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        # 跳过空字符串和重复项，保持原始顺序（不排序，保证角色优先级顺序不变）
        if item and item not in result:
            result.append(item)
    return result


def escape_expr_value(value: str) -> str:
    """转义 Milvus expr 中的字符串值。（★★ 理解）

    Milvus 的布尔表达式使用双引号括起字符串值，当字符串本身包含双引号时
    需要转义，否则会导致表达式语法错误。

    参数：
        value: 原始字符串值。

    返回：
        可安全放入 Milvus 双引号表达式中的字符串（双引号被转义为 \"）。

    调用顺序：DataScope.expr_clauses() -> escape_expr_value()。
    """
    return str(value).replace('"', '\\"')


@dataclass(frozen=True)
class DataScope:
    """入库或检索时使用的数据域，包含租户、数据集、可见级别和角色，用于 Milvus 过滤。（★★★ 核心）

    数据隔离是安全边界：不同租户、数据集之间的数据绝对不能互相可见；可见级别
    （public/internal/private）和角色控制确保用户只能访问授权范围内的知识库内容。

    字段说明：
      - tenant_id：租户标识，默认 default。
      - dataset_id：数据集标识，默认 default。
      - visibility：可见级别，支持 public/internal/private。
      - user_roles：当前用户拥有的角色列表。

    调用顺序：API/入库入口 -> DataScope -> Milvus expr。
    """

    tenant_id: str = DEFAULT_TENANT_ID
    dataset_id: str = DEFAULT_DATASET_ID
    visibility: str = DEFAULT_VISIBILITY
    user_roles: list[str] = field(default_factory=lambda: [DEFAULT_USER_ROLE])

    @classmethod
    def from_request(
        cls,
        *,
        tenant_id: str | None = None,
        dataset_id: str | None = None,
        visibility: str | None = None,
        user_roles: list[str] | tuple[str, ...] | None = None,
        user_role: str | None = None,
    ) -> "DataScope":
        """从 API 请求构建数据域，user_role 和 user_roles 合并去重。（★★ 理解）

        兼容两套角色传参方式：user_role（单值，来自 JWT token）和 user_roles（多值，来自 API 显式参数）；
        二者合并去重确保一个用户有多个角色时仍能正确匹配。

        参数：
            tenant_id: 请求中的租户标识。为空时使用默认值 "default"。
            dataset_id: 请求中的数据集标识。为空时使用默认值 "default"。
            visibility: 请求中的可见级别。为空时使用默认值 "public"。
            user_roles: 请求中的多角色列表。会与 user_role 合并。
            user_role: 请求中的单个角色，会合并进 user_roles。

        返回：
            清洗后的不可变 DataScope 实例。

        入库阶段把这个对象转换成 metadata，检索阶段再用同样的字段构造
        Milvus expr。也就是说，这里的默认值不是展示层默认值，而是实际的数据
        隔离边界；调用方应在任务开始时解析一次并沿链路复用。

        调用顺序：resolve_data_scope() -> DataScope.from_request()。
        """
        roles = _clean_list(user_roles)
        single_role = _clean_token(user_role, "")
        # 合并 user_role（JWT 单值）和 user_roles（API 多值）：
        # 用户可能同时从 token 和显式参数获得角色，交集去重确保不丢失权限也不重复
        if single_role and single_role not in roles:
            roles.append(single_role)
        if not roles:
            # 无角色时使用默认角色，确保非登录态用户至少可以访问 public 数据
            roles = [DEFAULT_USER_ROLE]
        return cls(
            tenant_id=_clean_token(tenant_id, DEFAULT_TENANT_ID),
            dataset_id=_clean_token(dataset_id, DEFAULT_DATASET_ID),
            visibility=_clean_token(visibility, DEFAULT_VISIBILITY),
            user_roles=roles,
        )

    def metadata(self, *, allowed_roles: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
        """返回写入 Milvus metadata 的数据域字段。

        参数：
            allowed_roles: 可访问该数据的角色列表。为空时使用当前 DataScope 的 user_roles。

        返回：
            包含 tenant_id、dataset_id、visibility、allowed_roles 的 metadata 字典。

        调用顺序：入库流程 -> DataScope.metadata() -> Milvus metadata。
        """
        # allowed_roles 用于入库：指定哪些角色可以检索到该文档。
        # 如果入库时未显式传入，使用当前请求的 user_roles 作为默认可访问角色
        roles = _clean_list(allowed_roles) or list(self.user_roles)
        return {
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "visibility": self.visibility,
            "allowed_roles": roles,
        }

    def as_dict(self) -> dict[str, Any]:
        """返回 API 诊断使用的结构化数据域。

        返回：
            包含 tenant_id、dataset_id、visibility、user_roles 的字典。

        调用顺序：检索诊断 API -> DataScope.as_dict()。
        """
        return {
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "visibility": self.visibility,
            "user_roles": list(self.user_roles),
        }

    def expr_clauses(self) -> list[str]:
        """转换为 Milvus 过滤子句。（★★★ 核心）

        构建 Milvus 布尔表达式子句列表，用于检索时过滤不可见数据。

        执行流程：
          1. 添加 tenant_id 过滤（精确匹配）。
          2. 添加 dataset_id 过滤（精确匹配）。
          3. 构建 visibility 过滤：public 始终可见，internal/private 只在请求级别允许时加入。
          4. 构建角色过滤：用户任一角色命中 allowed_roles 即可访问（OR 语义）。
          5. 返回所有 Milvus expr 子句。

        示例：
            ``DataScope(visibility="private", user_roles=["admin"]).expr_clauses()``
            returns::

                [
                    'tenant_id == "default"',
                    'dataset_id == "default"',
                    '(visibility == "public" or visibility == "private")',
                    '(array_contains(allowed_roles, "admin"))',
                ]

        返回：
            Milvus 布尔表达式子句列表。

        调用顺序：检索流程 -> DataScope.expr_clauses() -> Milvus search(expr=...)。
        """
        clauses = [
            f'tenant_id == "{escape_expr_value(self.tenant_id)}"',
            f'dataset_id == "{escape_expr_value(self.dataset_id)}"',
        ]
        # 可见级别策略：public 始终可见（任何角色都能看到公开内容）；
        # 内部/私有级别需要增加对应匹配，确保用户不能越级看到高保密内容
        allowed_visibility = ["public"]
        if self.visibility in {"internal", "private"}:
            # 当前用户只能看到 public 级别加上自身可见级别的内容；
            # 不允许 private 用户看到 internal 内容，反之亦然
            allowed_visibility.append(self.visibility)
        visibility_expr = " or ".join(f'visibility == "{escape_expr_value(item)}"' for item in allowed_visibility)
        clauses.append(f"({visibility_expr})")

        # 角色过滤：用户必须至少拥有一个 allowed_roles 中的角色才能访问该文档；
        # 任意角色匹配即可（OR 语义），适用于一个用户有多个业务角色的场景
        role_exprs = [f'array_contains(allowed_roles, "{escape_expr_value(role)}")' for role in self.user_roles]
        if role_exprs:
            clauses.append(f"({' or '.join(role_exprs)})")
        return clauses


def resolve_data_scope(
    *,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    user_roles: list[str] | tuple[str, ...] | None = None,
    user_role: str | None = None,
) -> DataScope:
    """构建当前请求或入库任务的数据域。（★★ 理解）

    便捷函数，封装 DataScope.from_request() 调用。统一在检索链路和入库链路的
    入口处调用，确保整条链路使用同一套租户/数据集/可见性/角色隔离规则。

    参数：
        tenant_id: 租户标识。
        dataset_id: 数据集标识。
        visibility: 可见级别。
        user_roles: 多角色列表。
        user_role: 单个角色。

    返回：
        清洗后的 DataScope 实例。

    离线总流程在 FAQ、文档和质量报告开始前调用一次，并把返回对象传给下游。
    这样三类数据会写入完全一致的 tenant/dataset/visibility/roles；若每个子函数
    自行解析默认值，容易出现 FAQ 与文档隔离边界不一致。

    调用顺序：检索/入库入口 -> resolve_data_scope() -> DataScope.from_request()。
    """
    return DataScope.from_request(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_roles=user_roles,
        user_role=user_role,
    )
