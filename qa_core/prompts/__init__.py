"""提示词包：管理 RAG 管线中所有 LLM 提示词模板。

包含模块：
- constants.py：原始提示词模板常量（System Prompt 和 User Template）。
- profiles.py：按意图维度和风险分类维度组织的 Prompt 档位（PromptProfile）。
- selector.py：模板选择器，根据意图和问题类别选择最终模板。
- templates.py：统一 re-export 入口。

调用方典型用法：from qa_core.prompts import build_answer_prompt_profile
"""

