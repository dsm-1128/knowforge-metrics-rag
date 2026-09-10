"""检索查询扩展工具：为同一检索意图生成少量同义检索表达（如"Webhook" → "回调"），不改变问题含义。

设计决策：
- 两层扩展策略：先尝试低成本本地规则替换（零延迟），规则匹配不足时再回退 LLM 扩展。
  这样短问题/高频术语可以完全避免 LLM 调用，兼顾延迟与覆盖率。
- 长问题和追问不受 short_structured 限制：长问题含多个业务词，同义替换提升召回明显；
  追问的指代已经过 rewrite_module 补全，不应再加限制，否则丢失同义召回机会。
- 规则变体由配置驱动（rules.toml），新增业务术语无需修改 Python 代码，非开发人员
  即可通过更新配置文件来扩展同义替换。

依赖分层：
- 被 pipeline.steps.prepare_retrieval() 在 Stage 2 调用。
- 依赖 qa_core.prompts.constants.QUERY_VARIANT_SYSTEM_PROMPT 作为 LLM 扩展的 System Prompt。
- 输出注入 retrieval.plan.use_query_variants，后续 search_faq / search_doc 使用多变体检索。
"""

from __future__ import annotations
import re
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from qa_core.config.logging_config import get_logger
from qa_core.config.rules import QueryVariantReplacementRule, get_rule_config
from qa_core.prompts.constants import QUERY_VARIANT_SYSTEM_PROMPT
from qa_core.config.settings import get_settings
from qa_core.llm.client import get_chat_model
logger = get_logger(__name__)

FOLLOW_UP_REWRITE_MARKERS = ("追问：", "追问:")

class QueryVariants(BaseModel):
    """LLM 输出检索表达时使用的 Pydantic 结构化模型，避免模型输出解释性文本。

    调用顺序：QAService/RAG 管线 -> QueryVariants。
    """

    queries: list[str] = Field(default_factory=list, description="等价检索表达")


def generate_query_variants(query: str, *, enabled: bool, allow_short_structured: bool = False) -> list[str]:
    """为同一检索意图生成少量同义表达（如"流程"→"SOP"），提升召回而不改变问题含义。（★★★ 核心）

    执行流程：
      1. 功能禁用（enabled=False）或无可变体空间时仅返回原问题，避免无关变体稀释召回精度。
      2. 短结构化问题（如"报销流程"含充分信息）直接返回原问题，无需 LLM 扩展。
      3. 第一层：用确定性本地规则（零成本）为高频业务术语生成同义变体。
        规则生成足够变体时直接返回，跳过第二层 LLM 调用，兼顾延迟与成本。
      4. 第二层：规则未覆盖的新领域词或罕见表达回退 LLM 扩展，避免召回覆盖率因规则缺失而下降。
      5. 结果去重后返回，原问题始终排在第一位。

    参数：
        query: 用户问题文本（已通过 normalize_user_query 归一化）。
        enabled: 功能启用标记（来自检索计划 plan.use_query_variants）。
        allow_short_structured: 是否允许短结构化问题也生成变体（追问改写后放宽此限制）。

    返回：
        list[str]: 同义检索表达列表，第一条始终为原问题。禁用时返回 [原问题]。

    调用顺序：QAService/RAG 管线 Stage 2 -> prepare_retrieval() -> generate_query_variants()。
    """
    # 加载应用全局设置（retrieval_variant_max 等检索配置）
    settings = get_settings()
    cleaned = query.strip()
    # 功能禁用或无可变体空间时仅用原问题检索，避免无关变体稀释召回精度
    if not enabled or not cleaned or settings.retrieval_variant_max <= 0:
        return [cleaned]

    # 普通短结构化问题保持克制；已追问改写的问题仍允许规则变体，避免上下文锚点丢失同义召回机会。
    if (
        _looks_like_short_structured_question(cleaned)
        and not allow_short_structured
        and not _is_rewritten_follow_up_query(cleaned)
    ):
        return [cleaned]

    # 第一层：先用确定性本地规则（零成本）为高频业务术语生成同义变体
    # 规则生成足够变体时直接返回，跳过第二层 LLM 调用，兼顾延迟与成本
    heuristic_variants = _heuristic_variants(cleaned, settings.retrieval_variant_max)
    if len(heuristic_variants) > 1:
        return heuristic_variants

    variants = [cleaned]
    # 第二层：规则未覆盖的新领域词或罕见表达回退 LLM 扩展，避免召回覆盖率因规则缺失而下降
    model = get_chat_model(streaming=False).with_structured_output(QueryVariants)
    # 调用 LLM 生成等价检索表达（如同义词、不同说法），不改变用户问题含义
    result = model.invoke(
        [
            SystemMessage(content=QUERY_VARIANT_SYSTEM_PROMPT),
            HumanMessage(content=f"原问题：{cleaned}\n最多生成 {settings.retrieval_variant_max} 条检索表达。"),
        ]
    )
    for item in result.queries:
        candidate = item.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
        if len(variants) >= settings.retrieval_variant_max + 1:
            break
    return variants


def _heuristic_variants(query: str, max_extra: int) -> list[str]:
    """用配置中的确定性规则为高频业务知识说法生成同义变体。（★★ 理解）

    规则变体是零成本操作（纯字符串替换），相比 LLM 扩展节省每次 200-500ms 推理延迟。
    当前规则覆盖：Webhook→回调、合同→协议、审批→审核等高频一对一同义替换。
    规则定义在 config/rules.toml 的 query_variants.replacements 列表中，
    新增替换无需修改 Python 代码。

    参数：
        query: 用户问题文本。
        max_extra: 允许生成的最大额外变体数量（不含原问题）。

    返回：
        list[str]: 变体列表，第一条始终为原问题。

    调用顺序：QAService/RAG 管线 -> generate_query_variants() -> _heuristic_variants()。
    """
    variants = [query]
    rules = get_rule_config().query_variants

    def add(candidate: str) -> None:
        """在保持顺序和上限的前提下，追加非空不重复变体。

        调用顺序：QAService/RAG 管线 -> add()。
        """
        candidate = candidate.strip()
        if candidate and candidate not in variants and len(variants) < max_extra + 1:
            variants.append(candidate)

    for rule in rules.replacements:
        if not rule.matches(query):
            continue
        for old, new in rule.replacements:
            add(_replace_term(query, old, new, rule))
    return variants


def _looks_like_short_structured_question(query: str) -> bool:
    """判断问题的常见同义说法是否已被配置规则覆盖，无需进一步 LLM 扩展。

    调用顺序：QAService/RAG 管线 -> _looks_like_short_structured_question()。
    """
    return get_rule_config().query_variants.is_short_structured_question(query)


def _is_rewritten_follow_up_query(query: str) -> bool:
    """判断是否为追问改写产物，例如"报销流程是什么；追问：那审批呢"。

    调用顺序：QAService/RAG 管线 -> _is_rewritten_follow_up_query()。
    """
    return any(marker in query for marker in FOLLOW_UP_REWRITE_MARKERS)


def _replace_term(query: str, old: str, new: str, rule: QueryVariantReplacementRule) -> str:
    """在 query 中执行一条配置替换，可选忽略大小写。（★★ 理解）

    规则变体使用字符串替换而非正则，因为同义替换场景是精确短语匹配（"Webhook"→"回调"），
    不需要正则的模糊匹配能力。大小写忽略仅在 rule 显式标注 ignore_case=True 时启用，
    用于覆盖中文业务术语的大小写不一致场景（如"HR"→"人力资源"）。

    参数：
        query: 用户问题文本。
        old: 被替换的源文本（配置规则中的原词）。
        new: 替换后的目标文本（配置规则中的同义词）。
        rule: 替换规则对象（含 ignore_case 标记）。

    返回：
        str: 替换后的文本；无匹配时返回原文本。

    调用顺序：QAService/RAG 管线 -> _heuristic_variants() -> _replace_term()。
    """

    if not rule.ignore_case:
        return query.replace(old, new)
    return re.sub(re.escape(old), new, query, flags=re.IGNORECASE)

