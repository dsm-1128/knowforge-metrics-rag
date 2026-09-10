"""Prompt 模板选择器：根据意图、问题类别和业务场景选择最终回答模板。

这个模块只做配置分发，不查数据库、不调用模型。这样 Prompt 策略可以独立调整，
不会把模板选择逻辑散落到 RAG 主链路中。
"""

from __future__ import annotations
from qa_core.intent.question_category import infer_question_category
from qa_core.prompts.metrics_dictionary import METRICS_DICTIONARY_RULES
from qa_core.prompts.profiles import CATEGORY_PROMPT_PROFILES, DEFAULT_PROMPT_PROFILE, PROMPT_PROFILES, PromptProfile
from qa_core.scenarios.registry import ScenarioDefinition

def _scenario_prompt_context(scenario: ScenarioDefinition) -> dict[str, str]:
    """把场景配置转换成 Prompt 模板需要的变量。

    参数：
        scenario: 当前业务场景。

    返回：
        包含 assistant_name、business_domain、industry、support_contact、phone 的字典。

    调用顺序：回答准备阶段 -> _scenario_prompt_context()。
    """
    # 从场景配置中提取 prompt 模板需要的插值变量，所有 system prompt 中的 {assistant_name} 等占位符都依赖此字典填充
    return {
        "assistant_name": scenario.assistant_name,
        "business_domain": scenario.business_domain,
        "industry": scenario.industry,
        "support_contact": scenario.support_contact,
        "phone": scenario.support_contact,
    }


def build_answer_prompt_profile(
    intent: str,
    scenario: ScenarioDefinition,
    query: str,
) -> PromptProfile:
    """根据意图和问题类别选择最终回答模板，并注入场景变量。（★★★ 核心）

    执行流程：
      1. 用 infer_question_category 推断风险类别（pricing/compliance/troubleshooting/summary/other）。
      2. 三级回退选择策略：风险类别专用模板 > 意图专属模板 > 默认通用模板。
      3. 从场景配置填充 system_template 中的插值变量（{assistant_name} 等）。

    选择优先级：
      - 风险类问题模板（CATEGORY_PROMPT_PROFILES）：费用、合规、排障、总结等类别
        优先使用专用模板（即使意图是 KNOWLEDGE_QUERY，如果问题含费用关键词也使用费用模板）。
      - 意图专属模板（PROMPT_PROFILES）：FAQ_QUERY、KNOWLEDGE_QUERY、FOLLOW_UP。
      - 默认模板（DEFAULT_PROMPT_PROFILE）：前两者都没有命中时使用通用安全回答模板。

    为什么风险类别优先级高于意图：费用、合规类问题的答案需要特别保守的口径约束
    （如"必须区分已确认和未确认"），这些约束独立于意图分类。如果用户问"报销流程"
    （KNOWLEDGE_QUERY+费用风险），应该用 pricing_guard 模板而非 knowledge_answer 模板。

    参数：
        intent: 意图识别结果（如 FAQ_QUERY、KNOWLEDGE_QUERY、FOLLOW_UP）。
        scenario: 当前业务场景（用于注入 {assistant_name}、{business_domain} 等变量）。
        query: 用户归一化后的业务有效问题（用于 infer_question_category 判断风险类别）。

    返回：
        已完成场景变量填充的 PromptProfile（system_template 中的占位符已被替换）。

    调用顺序：回答准备阶段 -> prepare_retrieval() -> build_answer_prompt_profile()。
    """
    # 判断 RAG 回答风险类别（费用/合规/排障/总结等），优先于通用意图选择专用模板
    question_category = infer_question_category(query)
    # 三级回退策略：风险类别专用模板 → 意图专属模板 → 默认通用模板
    # CATEGORY_PROMPT_PROFILES 命中时优先于 PROMPT_PROFILES，确保费用/合规等强口径问题使用保守模板
    profile = CATEGORY_PROMPT_PROFILES.get(question_category) or PROMPT_PROFILES.get(intent, DEFAULT_PROMPT_PROFILE)
    # 注入场景上下文变量到 system_template（{assistant_name} 等占位符），模板本身只做最简替换
    context = _scenario_prompt_context(scenario)
    system_template = profile.system_template.format(**context)
    if scenario.scenario_id == "metrics_dictionary":
        system_template += METRICS_DICTIONARY_RULES
    return PromptProfile(
        name=profile.name,
        system_template=system_template,
        user_template=profile.user_template,
        reason=profile.reason,
    )
