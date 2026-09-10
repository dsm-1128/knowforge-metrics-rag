"""最终回答 Prompt 档位定义：按意图维度和风险分类维度组织提示词模板。

本模块只定义模板和档位（PromptProfile），不负责选择。模板选择逻辑在 selector.py 中，
按"风险类别专用 > 意图专属 > 默认"三级优先级决定最终使用的档位。

设计决策：
- 意图维度（PROMPT_PROFILES）：根据意图识别结果（FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP）
  选择对应模板。FAQ 问题强调"短、准、直接"，知识查询允许结构化回答，追问强调不重复历史。
- 风险分类维度（CATEGORY_PROMPT_PROFILES）：独立于意图，费用/合规/排障/总结类问题使用
  更保守的模板（如费用类必须区分已确认/未确认）。即使意图识别为 KNOWLEDGE_QUERY，
  如果问题被分类为"pricing"，也会优先使用 pricing 模板。
- 默认模板（DEFAULT_PROMPT_PROFILE）：未知或新增意图回退到通用安全回答模板，
  保证系统对新场景也能产出合理回答而非直接崩溃。
"""
from __future__ import annotations
from dataclasses import dataclass

from qa_core.prompts.constants import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    COMPLIANCE_ANSWER_SYSTEM_PROMPT,
    FAQ_ANSWER_SYSTEM_PROMPT,
    FAQ_ANSWER_USER_TEMPLATE,
    FOLLOW_UP_ANSWER_SYSTEM_PROMPT,
    FOLLOW_UP_ANSWER_USER_TEMPLATE,
    KNOWLEDGE_ANSWER_SYSTEM_PROMPT,
    KNOWLEDGE_ANSWER_USER_TEMPLATE,
    PRICING_ANSWER_SYSTEM_PROMPT,
    RISK_CONTROL_USER_TEMPLATE,
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
    TROUBLESHOOTING_ANSWER_SYSTEM_PROMPT,
    TROUBLESHOOTING_USER_TEMPLATE,
)

@dataclass(frozen=True)
class PromptProfile:
    """最终回答阶段使用的提示词档位，创建后不可修改。属于规则配置，不应在请求中被临时改写。

    调用顺序：回答准备阶段 -> PromptProfile。
    """
    # 原因： frozen 使实例不可变，防止响应链路中意外修改 prompt 模板——模板变更必须通过配置代码审查，避免运行时静默篡改导致口径风险

    name: str
    system_template: str
    user_template: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        """返回可进入调试信息的精简描述（只含模板名称和选择原因，不暴露完整 prompt 文本）。

        调用顺序：回答准备阶段 -> PromptProfile.as_dict()。
        """
        # 只暴露 name 和 reason，不暴露 system_template/user_template 的完整内容，避免 log 中泄漏 prompt 细节
        return {"name": self.name, "reason": self.reason}


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "FAQ_QUERY": PromptProfile(
        name="faq_answer",
        system_template=FAQ_ANSWER_SYSTEM_PROMPT,
        user_template=FAQ_ANSWER_USER_TEMPLATE,
        reason="FAQ 类问题优先复用标准答案，控制回答长度和业务口径。",
    ),
    "KNOWLEDGE_QUERY": PromptProfile(
        name="knowledge_answer",
        system_template=KNOWLEDGE_ANSWER_SYSTEM_PROMPT,
        user_template=KNOWLEDGE_ANSWER_USER_TEMPLATE,
        reason="业务知识咨询需要整合文档资料，允许按流程、规则、步骤或说明结构化回答。",
    ),
    "FOLLOW_UP": PromptProfile(
        name="follow_up",
        system_template=FOLLOW_UP_ANSWER_SYSTEM_PROMPT,
        user_template=FOLLOW_UP_ANSWER_USER_TEMPLATE,
        reason="追问需要结合历史理解指代，但回答焦点仍限定在当前问题。",
    ),
}


CATEGORY_PROMPT_PROFILES: dict[str, PromptProfile] = {
    # 原因： 业务分类维度独立于意图维度——费用/合规/排障/总结需要额外的口径约束 prompt，与常规 FAQ/KNOWLEDGE 意图无关
    "pricing": PromptProfile(
        name="pricing_guard",
        system_template=PRICING_ANSWER_SYSTEM_PROMPT,
        user_template=RISK_CONTROL_USER_TEMPLATE,
        reason="费用、退款、优惠、发票等强口径问题必须保守回答，并区分已确认/未确认。",
    ),
    "compliance": PromptProfile(
        name="compliance_guard",
        system_template=COMPLIANCE_ANSWER_SYSTEM_PROMPT,
        user_template=RISK_CONTROL_USER_TEMPLATE,
        reason="合规、合同、隐私、审计类问题需要更严格的确认边界和风险提示。",
    ),
    "troubleshooting": PromptProfile(
        name="troubleshooting_steps",
        system_template=TROUBLESHOOTING_ANSWER_SYSTEM_PROMPT,
        user_template=TROUBLESHOOTING_USER_TEMPLATE,
        reason="故障排查类问题需要步骤化输出，同时不能编造文档外操作。",
    ),
    "summary": PromptProfile(
        name="source_bound_summary",
        system_template=SUMMARY_ANSWER_SYSTEM_PROMPT,
        user_template=SUMMARY_USER_TEMPLATE,
        reason="总结类问题需要整合更多上下文，但只能总结资料中已出现的信息。",
    ),
}


DEFAULT_PROMPT_PROFILE = PromptProfile(
    name="default_answer",
    system_template=ANSWER_SYSTEM_PROMPT,
    user_template=ANSWER_USER_TEMPLATE,
    reason="未知或新增意图使用通用安全回答模板。",
)
