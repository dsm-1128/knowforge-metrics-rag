"""依赖上下文的查询改写：将指代模糊的追问改写为独立检索问题。

用户在对话中常使用依赖上下文的指代（如"那审批呢"、"它的流程是什么"、"费用多少"），
这些表达在单轮检索中缺乏实体锚点，直接送入 BM25/向量检索会丢失意图。改写模块将此类
表达补全为可独立检索的完整问题（如"那审批呢" → "入职流程中的审批步骤是什么"）。

设计决策：
- 使用 LLM 改写而非规则：中文指代消除（"它"、"那"、"这个"等）需要理解对话上下文
  语义，正则规则无法覆盖多样化表达。单次 LLM 推理约 200-500ms，在追问场景下投入
  产出比可接受。
- 只改写 should_rewrite=True 的情况：由上游 intent.classifier 判断当前问题是否为
  需要上下文的追问（FOLLOW_UP），避免不必要的 LLM 调用。
- 非流式调用：改写结果需要完整句法结构，流式片段拼接可能导致句式断裂或语义不完整。
- 空结果硬失败：改写为空说明 LLM 未能理解上下文，抛出 RuntimeError 让调用方感知
  而非静默回退到原始查询（会导致检索语义偏离）。

依赖分层：
- 被 pipeline.steps.prepare_retrieval() 在 Stage 2 调用。
- 上游依赖 qa_core.intent.classifier 输出 requires_rewrite 标记。
- 下游注入 context.rewritten_query，后续检索计划和查询变体均使用改写后文本。
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from qa_core.memory.history import format_messages
from qa_core.llm.client import get_chat_model
from qa_core.prompts.constants import REWRITE_SYSTEM_PROMPT

def rewrite_query_if_needed(query: str, history_messages, should_rewrite: bool) -> str:
    """将依赖上下文的追问改写为独立检索问题，确保检索不丢失对话意图。（★★★ 核心）

    执行流程：
      1. 无需改写（should_rewrite=False）或无历史时直接返回原问题。
      2. 从历史中取最近 8 条消息作为上下文（过长历史会稀释当前焦点）。
      3. 调用非流式 LLM，以 SystemPrompt+HumanMessage 形式请求改写。
      4. 改写结果为空时抛出 RuntimeError，让上游感知处理失败而非静默回退。

    参数：
        query: 用户当前提问（可能包含指代表述如"那审批呢"）。
        history_messages: 对话历史消息列表，格式由 qa_core.memory.history 定义。
        should_rewrite: 上游意图分类结果，仅 FOLLOW_UP 类型需要改写。

    返回：
        str: 改写后的独立检索问题；无需改写时返回原问题。

    异常：
        RuntimeError: LLM 改写返回空结果（LLM 未能理解上下文）。

    调用顺序：QAService/RAG 管线 Stage 2 -> prepare_retrieval() -> rewrite_query_if_needed()。
    """
    # ── 步骤 1：门控判断 ──
    # 非追问或空历史无需 LLM 调用，直接使用原问题；追问改写只改写有上下文的场景
    if not should_rewrite or not history_messages:
        return query
    # ── 步骤 2：取最近 8 条历史 ──
    # 原因：过长历史会稀释当前提问焦点，改写只需最近一轮对话的前置信息即可完成指代消解
    history_text = format_messages(history_messages[-8:])
    # ── 步骤 3：非流式 LLM 改写 ──
    # 原因：改写需要完整句法，流式拼接片段可能导致"那审批呢"→"那审批呢步"这类不完整输出
    llm = get_chat_model(streaming=False)
    response = llm.invoke(
        [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"对话历史：\n{history_text}\n\n当前问题：{query}\n\n改写后的检索问题："),
        ]
    )
    rewritten = str(response.content).strip()
    # ── 步骤 4：空结果硬失败 ──
    # 原因：改写为空说明 LLM 未能理解上下文，此时若静默回退到原始查询"它呢"，
    # BM25/向量检索会拿到无实质语义的短查询，召回质量极差且难以排查
    if not rewritten:
        raise RuntimeError("查询改写返回空结果，无法生成独立检索问题。")
    return rewritten

