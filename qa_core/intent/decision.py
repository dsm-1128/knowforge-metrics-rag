"""Intent 决策网关：融合确定性规则和 BERT 意图模型，产出可治理的最终意图结果。

设计决策：
- 非检索类意图（直答/越界/问候）由路由层 `classify_direct_intent()` 直接收口，
  不经过网关。网关只处理检索类意图（FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP）。
- 网关仲裁策略是从高到低的优先级决策树：先检查模型误判（无历史判追问），再检查
  低置信度，然后检查一致性，最后处理冲突。
- 网关输出的 IntentResult 包含规则分数、模型分数、最终置信度、候选列表、风险标签
  和策略版本号，可用于 Trace 回放和审计。

融合策略（_fuse_decision）的优先级：
1. 模型判追问但无历史 → 屏蔽模型，保持规则结果（BERT 容易把短句误判为追问）
2. 模型置信度低于阈值(0.55) → 屏蔽模型，信任规则
3. 规则和模型一致 → 取高分并加成(+0.03)
4. 规则为默认兜底(default_knowledge) → 采纳模型（规则本身信号不可靠）
5. 规则和模型冲突 → 保持规则但降分到 0.68，保守处理
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage

from qa_core.config.rules import get_rule_config
from qa_core.intent.model_classifier import BertIntentModelService, IntentModelPrediction
from qa_core.intent.question_category import infer_question_category
from qa_core.scenarios.registry import ScenarioDefinition


RETRIEVAL_INTENTS = {"FAQ_QUERY", "KNOWLEDGE_QUERY", "FOLLOW_UP"}


@dataclass(frozen=True)
class IntentDecisionPolicy:
    """V1 意图决策策略：规则/模型仲裁的固定参数。

    policy_version  # 字段说明：策略版本号，用于 Trace 回放和审计追溯
    model_min_score  # 字段说明：模型置信度最低阈值，低于此值时完全屏蔽模型输出
    agreement_score_boost  # 字段说明：规则和模型一致时给最终分的加成（取高分后 +0.03）
    conflict_final_score  # 字段说明：规则和模型冲突时的最终置信度上限

    调用顺序：意图决策阶段 -> IntentDecisionPolicy。
    """

    policy_version: str = "intent-policy-v1-bert"
    model_min_score: float = 0.55
    agreement_score_boost: float = 0.03
    conflict_final_score: float = 0.68


POLICY = IntentDecisionPolicy()


def apply_intent_decision_gateway(
    query: str,
    history: list[BaseMessage],
    scenario: ScenarioDefinition,
    rule_result: Any,
) -> Any:
    """融合规则和模型信号，返回治理化的 intent 结果。（★★★ 核心）

    return type 与 rule_result 保持相同的 dataclass 类型，这样现有 pipeline 代码
    无需修改即可获得 IntentResult.as_dict() 中的企业级诊断信息。

    网关仲裁流程：
      1. 非检索类意图（直答/越界/问候）→ 直接返回规则结果，不调用模型。
      2. 检索类意图（FAQ_QUERY/KNOWLEDGE_QUERY/FOLLOW_UP）→
         a. 调用 BERT 模型获取预测（_default_model().predict()）。
         b. 融合规则分和模型分（_fuse_decision()），决策树共 5 个优先级。
         c. 产出最终意图、置信度、策略标签、风险标签和候选列表。

    参数：
        query: 用户查询文本。
        history: 历史对话消息列表。
        scenario: 当前业务场景定义。
        rule_result: 确定性规则产出的 IntentResult（或等效 dataclass）。

    返回：
        与 rule_result 同类型的 dataclass，但 intent/final_score/reason 等字段
        已根据网关仲裁结果更新。

    调用顺序：classifier.classify_intent() -> apply_intent_decision_gateway()。
    """
    # 非检索类意图（直答/越界/问候）：不需要模型参与，直接返回规则结果
    if rule_result.intent not in RETRIEVAL_INTENTS:
        return _with_decision_fields(
            rule_result,
            final_intent=rule_result.intent,
            final_score=rule_result.rule_score,
            decision_policy="deterministic_route",
            risk_tags=_risk_tags(query, rule_result, scenario, final_intent=rule_result.intent),
            model_prediction=None,
            candidates=_candidate_payloads(rule_result, None),
        )

    # 检索类意图：加载 BERT 模型并获取预测（模型首次加载耗时约 1-3 秒，后续预测约 10-30ms）
    model_prediction = _default_model().predict(query, has_history=bool(history))
    # 融合规则和模型信号，产出最终意图、置信度和决策策略标签
    final_intent, final_score, decision_policy = _fuse_decision(
        rule_result,
        model_prediction,
        has_history=bool(history),
    )
    # 如果最终意图与规则一致，保持规则原因；否则标记为模型辅助
    reason = rule_result.reason if final_intent == rule_result.intent else f"{rule_result.reason}_model_assisted"

    return _with_decision_fields(
        rule_result,
        final_intent=final_intent,
        final_score=final_score,
        decision_policy=decision_policy,
        risk_tags=_risk_tags(query, rule_result, scenario, final_intent=final_intent),
        model_prediction=model_prediction,
        candidates=_candidate_payloads(rule_result, model_prediction),
        reason=reason,
    )


def _fuse_decision(
    rule_result: Any,
    model_prediction: IntentModelPrediction,
    *,
    has_history: bool,
) -> tuple[str, float, str]:
    """融合规则分和模型分，按优先级决策树输出最终意图和置信度。（★★★ 核心）

    决策优先级（从高到低）：
    1. 模型判追问但无历史 → 屏蔽模型，保持规则结果
       （BERT 容易把短句误判为追问，无历史上下文时追问没有意义）
    2. 模型置信度低于阈值(0.55) → 屏蔽模型，信任规则
    3. 规则和模型一致 → 两者互相印证，取高分并加成(+0.03)，上限 1.0
    4. 规则为默认兜底(default_knowledge) → 采纳模型，规则本身不可靠
    5. 规则和模型冲突 → 保持规则但降分到 0.68，保守处理

    参数：
        rule_result: 规则产出的 IntentResult。
        model_prediction: BERT 模型的预测结果。
        has_history: 当前对话是否有历史消息。

    返回：
        (最终意图, 最终置信度, 决策策略标签) 三元组。
    """
    # 优先级1：模型判追问但当前对话无历史 → 模型误判，保持规则
    # 原因：BERT 模型倾向于把短句（如"那这个呢"）误判为追问，无历史时追问没有意义
    if model_prediction.intent == "FOLLOW_UP" and not has_history:
        return rule_result.intent, rule_result.rule_score, "model_follow_up_without_history_guarded"

    # 优先级2：模型置信度不足 → 不信任模型，保持规则
    # 原因：0.55 阈值基于离线评测，置信度低于此值时模型准确率显著下降
    if model_prediction.score < POLICY.model_min_score:
        return rule_result.intent, rule_result.rule_score, "model_low_confidence_rule_kept"

    # 优先级3：规则和模型一致 → 互相印证，取两者高分并小幅加成
    # 原因：一致时说明规则和模型互相印证，加 0.03 作为正反馈，上限 1.0
    if model_prediction.intent == rule_result.intent:
        final_score = min(
            1.0,
            max(rule_result.rule_score, model_prediction.score) + POLICY.agreement_score_boost,
        )
        return rule_result.intent, final_score, "rule_model_agreed"

    # 优先级4：规则是 default_knowledge（兜底），本身不可靠 → 采纳模型
    # 原因：default_knowledge 只是"以上规则都不命中时的兜底"，信号强度弱于模型输出
    if rule_result.reason == "default_knowledge":
        return model_prediction.intent, model_prediction.score, "model_assisted_default"

    # 优先级5：规则和模型冲突且都不是兜底 → 保守策略，保持规则但降分
    # 原因：无法判断规则和模型谁更准确时，保持规则结果但降分到 0.68（保守值）
    return rule_result.intent, min(rule_result.rule_score, POLICY.conflict_final_score), "rule_model_conflict_guarded"


def _with_decision_fields(
    rule_result: Any,
    *,
    final_intent: str,
    final_score: float,
    decision_policy: str,
    risk_tags: tuple[str, ...],
    candidates: tuple[dict[str, str | float], ...],
    model_prediction: IntentModelPrediction | None,
    reason: str | None = None,
) -> Any:
    """使用 dataclasses.replace() 更新 rule_result 的决策相关字段。

    replace() 会创建一个新的 dataclass 实例（浅拷贝），不修改原始对象。
    这样网关的决策结果与规则的原始结果共享相同的类结构，对调用方完全透明。

    参数：
        rule_result: 原始规则结果 dataclass。
        final_intent: 网关最终意图。
        final_score: 网关最终置信度。
        decision_policy: 决策策略标签。
        risk_tags: 风险标签列表。
        candidates: 候选意图列表（规则 + 模型）。
        model_prediction: 模型预测结果，用于填充 model_score/model_version。
        reason: 可选的覆盖原因，None 时保持原始 reason。

    返回：
        与 rule_result 同类型的新 dataclass 实例。

    调用顺序：apply_intent_decision_gateway() -> _with_decision_fields()。
    """
    return replace(
        rule_result,
        intent=final_intent,
        reason=reason or rule_result.reason,
        requires_rewrite=rule_result.requires_rewrite or final_intent == "FOLLOW_UP",
        final_score=final_score,
        risk_tags=risk_tags,
        decision_policy=decision_policy,
        candidate_intents=candidates,
        model_score=model_prediction.score if model_prediction else None,
        model_version=model_prediction.model_version if model_prediction else None,
        policy_version=POLICY.policy_version,
    )


def _candidate_payloads(
    rule_result: Any,
    model_prediction: IntentModelPrediction | None,
) -> tuple[dict[str, str | float], ...]:
    """构建候选意图列表：规则候选始终排在第一位，模型候选随后追加。（★★ 理解）

    格式：[{intent, score, source:"rule"|"model", reason}, ...]
    用于 trace 回放时展示完整的决策依据链，方便开发者和运维人员理解每个意图决策的来源。

    参数：
        rule_result: 规则产出的 IntentResult。
        model_prediction: BERT 模型的预测结果（可能为 None）。

    返回：
        候选意图字典元组。

    调用顺序：apply_intent_decision_gateway() -> _candidate_payloads()。
    """
    # 规则候选始终存在，排在第一位
    candidates: list[dict[str, str | float]] = [
        {
            "intent": str(rule_result.intent),
            "score": round(float(rule_result.rule_score), 4),
            "source": "rule",
            "reason": str(rule_result.reason),
        }
    ]
    # 有模型预测时，追加所有模型候选（3 个意图的分数）
    # 原因：展示完整的模型概率分布，帮助理解模型为什么会做出某个预测
    if model_prediction:
        for intent, score in model_prediction.scores.items():
            candidates.append(
                {
                    "intent": intent,
                    "score": round(float(score), 4),
                    "source": "model",
                    "reason": model_prediction.reason,
                }
            )
    return tuple(candidates)


def _risk_tags(
    query: str,
    rule_result: Any,
    scenario: ScenarioDefinition,
    *,
    final_intent: str,
) -> tuple[str, ...]:
    """生成风险标签列表，供 trace 和监控系统分类追踪。

    标签维度：
    - domain: 业务场景（如 enterprise_knowledge）。
    - source: 推断的资料分类（规则推断出的）。
    - risk: 问题风险类别（pricing/compliance/troubleshooting/summary）。
    - context: 是否需要对话历史（追问类意图标记 history_required）。
    - confidence: 规则置信度是否偏低（低规则分时标记 low_rule_score）。

    参数：
        query: 用户查询文本。
        rule_result: 规则产出的 IntentResult。
        scenario: 当前业务场景定义。
        final_intent: 网关最终确定的意图。

    返回：
        风险标签元组，每个标签格式为 "维度:值"。

    调用顺序：apply_intent_decision_gateway() -> _risk_tags()。
    """
    tags = [f"domain:{scenario.scenario_id}"]
    # 规则推断出了 source 分类时记录
    if rule_result.suggested_source:
        tags.append(f"source:{rule_result.suggested_source}")
    # 推断问题风险类别（费用/合规/排障/总结），非 default 时记录
    category = infer_question_category(query)
    if category != "default":
        tags.append(f"risk:{category}")
    # 追问意图标记需要对话历史
    if final_intent == "FOLLOW_UP":
        tags.append("context:history_required")
    # 规则分低于阈值时发出低置信度警告
    if rule_result.rule_score < get_rule_config().retrieval_strategy.low_rule_score_threshold:
        tags.append("confidence:low_rule_score")
    return tuple(tags)


@lru_cache(maxsize=1)
def _default_model() -> BertIntentModelService:
    """返回进程级 BERT 意图模型单例。

    使用 lru_cache(maxsize=1) 确保整个进程生命周期只加载一次 BERT 模型。
    模型加载耗时约 1-3 秒（HuggingFace AutoModel + config），缓存后后续调用
    仅执行 tokenize + predict（约 10-30ms）。

    返回：
        BertIntentModelService 实例。

    调用顺序：apply_intent_decision_gateway() -> _default_model()。
    """
    return BertIntentModelService.from_settings()


def warmup_intent_decision_gateway() -> dict[str, object]:
    """在 API 进程中预热 BERT 意图模型，并执行一次样本预测。

    API 进程启动后在后台调用此函数，让模型加载到内存中，避免第一个用户请求
    等待模型加载（冷启动延迟约 1-3 秒）。

    返回：
        包含模型版本、标签列表、样本预测结果和策略版本的字典。

    调用顺序：API lifespan -> warmup_intent_decision_gateway()。
    """
    # 第一步：触发进程级单例加载，把 BERT 权重和 tokenizer 提前放入内存。
    model = _default_model()
    # 第二步：用业务常见问题做一次真实预测，验证 tokenize、模型推理和标签映射都正常。
    # 这一步还会消除第一个用户请求承担的冷启动延迟。
    prediction = model.predict("新人入职流程有哪些", has_history=False)
    # 第三步：只返回可记录的诊断摘要，不把模型对象或权重暴露给启动日志和状态接口。
    return {
        "model_version": model.model_version,
        "labels": list(model.labels),
        "sample_intent": prediction.intent,
        "sample_score": round(prediction.score, 4),
        "policy_version": POLICY.policy_version,
    }
