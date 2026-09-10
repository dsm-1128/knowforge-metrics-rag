"""规则配置加载：从 config/rules.toml 加载 FAQ 快路径、查询变体、intent 规则分数和检索策略。

设计决策：
- 每次 get_rule_config() 调用都重新读取 TOML 文件，而非在启动时加载到内存。
  这样本地规则编辑在下一个请求或测试运行时就生效，匹配 scenario.toml 的热加载工作流。
- 所有规则 dataclass 使用 frozen=True，保证规则在运行时不可变，防止意外修改导致
  检索/分类行为异常。
- 参数校验在加载时完成（fail-fast）：配置缺失、类型错误或值超出范围时直接抛出
  ValueError，不等到运行期才暴露问题。

调用顺序：进程启动时 / 检索阶段 -> rules。
"""

from __future__ import annotations

import re
import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path

from qa_core.config.settings import PROJECT_ROOT


DEFAULT_RULE_CONFIG_PATH = PROJECT_ROOT / "config" / "rules.toml"


@dataclass(frozen=True)
class FaqFastPathRules:
    """FAQ 快路径规则：判断短查询是否值得先在 FAQ 中探查。

    FAQ 快路径是一个性能优化：对于明显带有 FAQ 特征（包含 hints 关键词）的短查询，
    先以低代价在 FAQ embedding 中检索，命中时直接返回 FAQ 答案，跳过文档检索和
    LLM 调用。

    max_chars  # 字段说明：快路径查询的最大字符数
    hints  # 字段说明：FAQ 快路径触发关键词列表

    调用顺序：检索准备阶段 -> FaqFastPathRules。
    """

    max_chars: int
    hints: tuple[str, ...]

    def hint_matches(self, query: str) -> bool:
        """判断查询是否包含任何 FAQ 快路径关键词。

        hints 中的每个关键词通过 re.escape 转义后构造正则，以 re.IGNORECASE 匹配，
        保证"发票"和"发票吗"都能命中。

        参数：
            query: 用户查询文本。

        返回：
            True 表示查询属于 FAQ 快路径，可以优先在 FAQ 中探查。

        调用顺序：检索准备阶段 -> FaqFastPathRules.hint_matches()。
        """
        if not self.hints:
            return False
        # 原因：re.escape 转义每个 hint，避免 hint 包含正则特殊字符时导致匹配错误
        pattern = re.compile("|".join(re.escape(item) for item in self.hints), re.IGNORECASE)
        return bool(pattern.search(query or ""))


@dataclass(frozen=True)
class QueryVariantReplacementRule:
    """查询变体替换规则：根据关键词条件对查询进行确定性替换。

    每条规则包含 when_any（命中任一关键词即触发）和 when_all（全部关键词都命中才触发）
    两个条件维度，以及一个或多个替换对（find → replace）。

    when_any  # 字段说明：命中任一关键词即触发规则的列表
    when_all  # 字段说明：全部关键词都命中才触发规则的列表
    replacements  # 字段说明：替换对列表，每对为 (查找文本, 替换文本)
    ignore_case  # 字段说明：是否忽略大小写匹配，默认 False

    调用顺序：检索准备阶段 -> QueryVariantReplacementRule。
    """

    when_any: tuple[str, ...]
    when_all: tuple[str, ...]
    replacements: tuple[tuple[str, str], ...]
    ignore_case: bool = False

    def matches(self, query: str) -> bool:
        """判断当前查询是否适用此替换规则。

        匹配逻辑：
        1. ignore_case 为 True 时，所有比较在 lower() 化后进行。
        2. when_any 非空时，任一关键词在查询中出现即算命中。
        3. when_all 非空时，所有关键词都在查询中出现才算命中。
        4. 至少一个条件维度非空且命中，规则才适用。

        参数：
            query: 用户查询文本。

        返回：
            True 表示此替换规则应应用于当前查询。

        调用顺序：检索准备阶段 -> QueryVariantReplacementRule.matches()。
        """
        # 原因：ignore_case 时统一转小写比较，减少模式数量
        source = query.lower() if self.ignore_case else query
        any_terms = tuple(item.lower() for item in self.when_any) if self.ignore_case else self.when_any
        all_terms = tuple(item.lower() for item in self.when_all) if self.ignore_case else self.when_all
        if any_terms and not any(term in source for term in any_terms):
            return False
        if all_terms and not all(term in source for term in all_terms):
            return False
        return bool(any_terms or all_terms)


@dataclass(frozen=True)
class QueryVariantRules:
    """查询变体规则：确定性的查询改写/扩展规则集合。

    short_structured_max_chars  # 字段说明：短结构化查询的最大字符数，超过此值不跳过改写
    short_structured_markers  # 字段说明：短结构化查询标记词列表（如"的"、"是"等）
    replacements  # 字段说明：替换规则列表

    调用顺序：检索准备阶段 -> QueryVariantRules。
    """

    short_structured_max_chars: int
    short_structured_markers: tuple[str, ...]
    replacements: tuple[QueryVariantReplacementRule, ...]

    def is_short_structured_question(self, query: str) -> bool:
        """判断查询是否为短结构化问题，可以跳过改写。

        短结构化问题的判断标准：
        1. 去除首尾空格后不为空。
        2. 长度不超过 short_structured_max_chars。
        3. 包含任意一个 short_structured_markers 标记。

        短结构化问题通常已经足够明确（如"报销流程是什么"），不需要通过 LLM 改写
        来提升召回率。

        参数：
            query: 用户查询文本。

        返回：
            True 表示查询足够明确，可以跳过改写阶段。

        调用顺序：检索准备阶段 -> QueryVariantRules.is_short_structured_question()。
        """
        compact = query.strip()
        if not compact or len(compact) > self.short_structured_max_chars:
            return False
        return any(marker in compact for marker in self.short_structured_markers)


@dataclass(frozen=True)
class RetrievalStrategyRules:
    """检索策略守卫线：FAQ 直出阈值、短查询守卫线、规则分阈值和上下文窗口大小。

    faq_direct_floor  # 字段说明：FAQ 直接命中的最低阈值
    faq_direct_discount  # 字段说明：FAQ 直出分数折扣系数
    short_query_guard_threshold  # 字段说明：短查询守卫阈值
    low_rule_score_threshold  # 字段说明：低规则分阈值
    medium_rule_score_threshold  # 字段说明：中规则分阈值
    medium_rule_score_direct_threshold  # 字段说明：中规则分直接通过阈值
    pricing_direct_threshold  # 字段说明：费用类问题的直接通过阈值
    compliance_direct_threshold  # 字段说明：合规类问题的直接通过阈值
    low_rule_score_direct_threshold  # 字段说明：低规则分直接通过阈值
    follow_up_faq_top_k_min  # 字段说明：追问场景 FAQ 最小召回数
    strong_faq_doc_top_k_min  # 字段说明：强 FAQ 文档最小召回数
    knowledge_context_top_n_min  # 字段说明：知识查询上下文最小条数
    guard_context_top_n_min  # 字段说明：守卫上下文最小条数
    table_context_top_n_min  # 字段说明：表格查询上下文最小条数

    调用顺序：检索计划阶段 -> RetrievalStrategyRules。
    """

    faq_direct_floor: float
    faq_direct_discount: float
    short_query_guard_threshold: float
    low_rule_score_threshold: float
    medium_rule_score_threshold: float
    medium_rule_score_direct_threshold: float
    pricing_direct_threshold: float
    compliance_direct_threshold: float
    low_rule_score_direct_threshold: float
    follow_up_faq_top_k_min: int
    strong_faq_doc_top_k_min: int
    knowledge_context_top_n_min: int
    guard_context_top_n_min: int
    table_context_top_n_min: int


@dataclass(frozen=True)
class IntentRuleScoreRules:
    """意图规则分数：用于对确定性意图规则候选项排序。

    分数含义：
    - strong_faq：仅命中 FAQ 关键词时的规则分，最低。
    - knowledge：命中知识库关键词时的规则分。
    - source_question_shape：命中业务域 + 标准问法（"怎么办/需要什么"）。
    - direct_faq_shape：命中业务域 + 直接 FAQ 问法（"是什么/可以吗"），最高。

    分数必须严格递增：strong_faq < knowledge < source_question_shape < direct_faq_shape。

    strong_faq  # 字段说明：强 FAQ 关键词匹配分数
    knowledge  # 字段说明：知识查询关键词匹配分数
    source_question_shape  # 字段说明：来源+标准问法匹配分数
    direct_faq_shape  # 字段说明：来源+直接 FAQ 问法匹配分数

    调用顺序：意图识别阶段 -> IntentRuleScoreRules。
    """

    strong_faq: float
    knowledge: float
    source_question_shape: float
    direct_faq_shape: float


@dataclass(frozen=True)
class RuleConfig:
    """运行时规则集合，检索准备阶段由 pipeline 模块共享访问。

    包含四个子规则集：
    - faq_fast_path：FAQ 快路径规则。
    - query_variants：查询变体规则。
    - intent_rule_scores：意图规则分数。
    - retrieval_strategy：检索策略守卫线。

    faq_fast_path  # 字段说明：FAQ 快路径规则
    query_variants  # 字段说明：查询变体规则
    intent_rule_scores  # 字段说明：意图规则分数
    retrieval_strategy  # 字段说明：检索策略守卫线

    调用顺序：启动配置或前置校验 -> RuleConfig。
    """

    faq_fast_path: FaqFastPathRules
    query_variants: QueryVariantRules
    intent_rule_scores: IntentRuleScoreRules
    retrieval_strategy: RetrievalStrategyRules


def get_rule_config(path: str | Path | None = None) -> RuleConfig:
    """从 TOML 文件加载路由规则。（★★★ 核心）

    文件在每次调用时重新读取，而非在启动时加载到内存。这样本地规则编辑在下一个
    请求或测试运行时就生效，匹配 scenario.toml 的热加载工作流。生产环境建议通过
    配置管理工具（如 Consul）同步更新 rules.toml。

    参数：
        path: 规则文件路径，None 时使用默认路径 config/rules.toml。

    返回：
        完整的 RuleConfig 对象，包含所有子规则集。

    调用顺序：启动配置或前置校验 -> get_rule_config()。
    """
    config_path = Path(path) if path else DEFAULT_RULE_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    # ── 步骤 1：解析 TOML ──
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    # ── 步骤 2：加载 FAQ 快路径规则 ──
    faq_payload = dict(payload.get("faq_fast_path") or {})
    max_chars = int(faq_payload.get("max_chars") or 0)
    hints = tuple(str(item).strip() for item in faq_payload.get("hints", ()) if str(item).strip())
    # 校验：max_chars 和 hints 都必须非空，否则 FAQ 快路径无法正常工作
    if max_chars <= 0:
        raise ValueError(f"faq_fast_path.max_chars 必须大于 0：{config_path}")
    if not hints:
        raise ValueError(f"faq_fast_path.hints 不能为空：{config_path}")
    # ── 步骤 3：加载其他子规则 ──
    query_variants = _load_query_variant_rules(payload, config_path)
    intent_rule_scores = _load_intent_rule_score_rules(payload, config_path)
    retrieval_strategy = _load_retrieval_strategy_rules(payload, config_path)
    # ── 步骤 4：组合为完整 RuleConfig ──
    return RuleConfig(
        faq_fast_path=FaqFastPathRules(max_chars=max_chars, hints=hints),
        query_variants=query_variants,
        intent_rule_scores=intent_rule_scores,
        retrieval_strategy=retrieval_strategy,
    )


def _load_query_variant_rules(payload: dict, config_path: Path) -> QueryVariantRules:
    """从 TOML payload 解析查询变体规则。

    参数：
        payload: TOML 解析后的完整字典。
        config_path: 配置文件路径，用于错误信息提示。

    返回：
        QueryVariantRules 对象。

    调用顺序：get_rule_config() -> _load_query_variant_rules()。
    """
    variant_payload = dict(payload.get("query_variants") or {})
    max_chars = int(variant_payload.get("short_structured_max_chars") or 0)
    markers = _clean_tuple(variant_payload.get("short_structured_markers", ()))
    # 校验：short_structured_max_chars 和 markers 都必须非空
    if max_chars <= 0:
        raise ValueError(f"query_variants.short_structured_max_chars 必须大于 0：{config_path}")
    if not markers:
        raise ValueError(f"query_variants.short_structured_markers 不能为空：{config_path}")

    replacements = tuple(
        _parse_replacement_rule(item, config_path)
        for item in variant_payload.get("replacements", ())
    )
    if not replacements:
        raise ValueError(f"query_variants.replacements 不能为空：{config_path}")

    return QueryVariantRules(
        short_structured_max_chars=max_chars,
        short_structured_markers=markers,
        replacements=replacements,
    )


def _parse_replacement_rule(payload: dict, config_path: Path) -> QueryVariantReplacementRule:
    """解析一条查询变体替换规则。

    参数：
        payload: 包含单条替换规则配置的字典。
        config_path: 配置文件路径，用于错误信息提示。

    返回：
        QueryVariantReplacementRule 对象。

    调用顺序：_load_query_variant_rules() -> _parse_replacement_rule()。
    """
    rule_payload = dict(payload or {})
    when_any = _clean_tuple(rule_payload.get("when_any", ()))
    when_all = _clean_tuple(rule_payload.get("when_all", ()))
    # 原因：replace 中的每对必须是 (find, replace) 二元组
    replacements = tuple(
        (str(pair[0]).strip(), str(pair[1]).strip())
        for pair in rule_payload.get("replace", ())
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[0]).strip() and str(pair[1]).strip()
    )
    # 校验：至少配置 when_any 或 when_all
    if not when_any and not when_all:
        raise ValueError(f"query_variants.replacements 中每条规则必须配置 when_any 或 when_all：{config_path}")
    if not replacements:
        raise ValueError(f"query_variants.replacements 中每条规则必须配置 replace：{config_path}")
    return QueryVariantReplacementRule(
        when_any=when_any,
        when_all=when_all,
        replacements=replacements,
        ignore_case=bool(rule_payload.get("ignore_case", False)),
    )


def _load_intent_rule_score_rules(payload: dict, config_path: Path) -> IntentRuleScoreRules:
    """从 TOML payload 解析确定性意图规则分数。

    校验规则：
    - 所有必须的 key 都存在。
    - 每个分数都在 0-1 范围内。
    - 分数严格递增：strong_faq < knowledge < source_question_shape < direct_faq_shape。

    参数：
        payload: TOML 解析后的完整字典。
        config_path: 配置文件路径，用于错误信息提示。

    返回：
        IntentRuleScoreRules 对象。

    调用顺序：get_rule_config() -> _load_intent_rule_score_rules()。
    """
    raw = dict(payload.get("intent_rule_scores") or {})
    required_keys = (
        "strong_faq",
        "knowledge",
        "source_question_shape",
        "direct_faq_shape",
    )
    missing = [key for key in required_keys if key not in raw]
    if missing:
        raise ValueError(f"intent_rule_scores 缺少配置项 {missing}：{config_path}")

    scores = {key: _as_float(raw[key], f"intent_rule_scores.{key}", config_path) for key in required_keys}
    for key, value in scores.items():
        if not 0 < value <= 1:
            raise ValueError(f"intent_rule_scores.{key} 必须在 0 到 1 之间：{config_path}")

    # 原因：分数必须严格递增，确保意图分类优先级一致。如果配置违反了这一顺序，
    # 会在运行期产生"低分优先"的违反直觉的行为
    if not (
        scores["strong_faq"]
        < scores["knowledge"]
        < scores["source_question_shape"]
        < scores["direct_faq_shape"]
    ):
        raise ValueError(
            "intent_rule_scores 必须满足 strong_faq < knowledge < "
            f"source_question_shape < direct_faq_shape：{config_path}"
        )

    return IntentRuleScoreRules(
        strong_faq=scores["strong_faq"],
        knowledge=scores["knowledge"],
        source_question_shape=scores["source_question_shape"],
        direct_faq_shape=scores["direct_faq_shape"],
    )


def _load_retrieval_strategy_rules(payload: dict, config_path: Path) -> RetrievalStrategyRules:
    """从 TOML payload 解析检索策略守卫线。

    faq_direct_discount 必须在 0-1 之间（不包含端点值），其他浮点数在 0-1 之间
    （包含上限）。low_rule_score_threshold 必须小于 medium_rule_score_threshold。

    参数：
        payload: TOML 解析后的完整字典。
        config_path: 配置文件路径，用于错误信息提示。

    返回：
        RetrievalStrategyRules 对象。

    调用顺序：get_rule_config() -> _load_retrieval_strategy_rules()。
    """
    raw = dict(payload.get("retrieval_strategy") or {})
    required_float_keys = (
        "faq_direct_floor",
        "faq_direct_discount",
        "short_query_guard_threshold",
        "low_rule_score_threshold",
        "medium_rule_score_threshold",
        "medium_rule_score_direct_threshold",
        "pricing_direct_threshold",
        "compliance_direct_threshold",
        "low_rule_score_direct_threshold",
    )
    required_int_keys = (
        "follow_up_faq_top_k_min",
        "strong_faq_doc_top_k_min",
        "knowledge_context_top_n_min",
        "guard_context_top_n_min",
        "table_context_top_n_min",
    )
    missing = [key for key in (*required_float_keys, *required_int_keys) if key not in raw]
    if missing:
        raise ValueError(f"retrieval_strategy 缺少配置项 {missing}：{config_path}")

    # 解析并校验浮点参数
    floats = {key: _as_float(raw[key], f"retrieval_strategy.{key}", config_path) for key in required_float_keys}
    # 解析并校验整型参数
    ints = {key: _as_positive_int(raw[key], f"retrieval_strategy.{key}", config_path) for key in required_int_keys}
    # faq_direct_discount 必须在 (0, 1) 开区间，即不能等于 0 或 1
    if not 0 < floats["faq_direct_discount"] < 1:
        raise ValueError(f"retrieval_strategy.faq_direct_discount 必须在 0 到 1 之间：{config_path}")
    # 其他浮点参数在 (0, 1] 闭区间，即可以等于 1
    for key in required_float_keys:
        if key == "faq_direct_discount":
            continue
        if not 0 < floats[key] <= 1:
            raise ValueError(f"retrieval_strategy.{key} 必须在 0 到 1 之间：{config_path}")
    # low 必须严格小于 medium，否则检索计划的难度分级逻辑不符合预期
    if floats["low_rule_score_threshold"] >= floats["medium_rule_score_threshold"]:
        raise ValueError(
            f"retrieval_strategy.low_rule_score_threshold 必须小于 medium_rule_score_threshold：{config_path}"
        )

    return RetrievalStrategyRules(
        faq_direct_floor=floats["faq_direct_floor"],
        faq_direct_discount=floats["faq_direct_discount"],
        short_query_guard_threshold=floats["short_query_guard_threshold"],
        low_rule_score_threshold=floats["low_rule_score_threshold"],
        medium_rule_score_threshold=floats["medium_rule_score_threshold"],
        medium_rule_score_direct_threshold=floats["medium_rule_score_direct_threshold"],
        pricing_direct_threshold=floats["pricing_direct_threshold"],
        compliance_direct_threshold=floats["compliance_direct_threshold"],
        low_rule_score_direct_threshold=floats["low_rule_score_direct_threshold"],
        follow_up_faq_top_k_min=ints["follow_up_faq_top_k_min"],
        strong_faq_doc_top_k_min=ints["strong_faq_doc_top_k_min"],
        knowledge_context_top_n_min=ints["knowledge_context_top_n_min"],
        guard_context_top_n_min=ints["guard_context_top_n_min"],
        table_context_top_n_min=ints["table_context_top_n_min"],
    )


def _as_float(value: object, name: str, config_path: Path) -> float:
    """将配置值解析为浮点数，失败时提供有意义的错误信息。

    参数：
        value: 配置值。
        name: 配置项名称（如 "intent_rule_scores.strong_faq"）。
        config_path: 配置文件路径。

    返回：
        浮点数。

    调用顺序：_load_intent_rule_score_rules() / _load_retrieval_strategy_rules() -> _as_float()。
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字：{config_path}") from exc


def _as_positive_int(value: object, name: str, config_path: Path) -> int:
    """将配置值解析为正整数，失败时提供有意义的错误信息。

    参数：
        value: 配置值。
        name: 配置项名称。
        config_path: 配置文件路径。

    返回：
        正整数。

    调用顺序：_load_retrieval_strategy_rules() -> _as_positive_int()。
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数：{config_path}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} 必须大于 0：{config_path}")
    return parsed


def _clean_tuple(items: object) -> tuple[str, ...]:
    """过滤并返回非空字符串元组。

    从配置中读取的列表可能包含空字符串或仅空白字符串。该方法统一做 .strip()
    处理后过滤空值，保证后续逻辑不会遇到空字符串。

    参数：
        items: 原始配置值（通常为列表）。

    返回：
        非空字符串元组。

    调用顺序：配置解析辅助函数 -> _clean_tuple()。
    """
    return tuple(str(item).strip() for item in items or () if str(item).strip())
