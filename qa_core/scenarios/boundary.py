"""跨场景和跨 source 边界检测。

当用户当前选择的场景或业务分类与问题内容明显不一致时，这个模块会给出可解释的
拦截/建议结果。例如用户在“企业知识库”场景里问“隐蔽工程验收资料”，系统应能判断
它更像“工程项目资料问答”场景。
"""

from __future__ import annotations

from dataclasses import dataclass

from qa_core.scenarios.registry import ScenarioDefinition, get_scenario_registry


MIN_OTHER_SCENARIO_SCORE = 12
CURRENT_SCENARIO_SAFE_SCORE = 8
OTHER_SCENARIO_SCORE_MARGIN = 4
SOURCE_MATCH_COUNT_WEIGHT = 10


@dataclass(frozen=True)
class ScenarioBoundaryDecision:
    """单次跨场景边界检测结果。

    字段说明：
      - crossed：是否判断为跨场景。
      - matched_scenario_id：命中的其他场景 ID。
      - matched_scenario_name：命中场景的展示名称。
      - matched_source：触发命中的 source key。
      - matched_source_label：触发命中的 source 展示名称。
      - reason：判断原因，写入 trace 方便排查。

    调用顺序：场景解析入口 -> ScenarioBoundaryDecision。
    """

    crossed: bool
    matched_scenario_id: str = ""
    matched_scenario_name: str = ""
    matched_source: str = ""
    matched_source_label: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, str | bool]:
        """转换为结构化字典，供 trace 和审计日志使用。

        返回：
            包含 crossed、matched_scenario_id、matched_source、reason 等字段的字典。

        调用顺序：场景解析入口 -> ScenarioBoundaryDecision.as_dict()。
        """
        return {
            "crossed": self.crossed,
            "matched_scenario_id": self.matched_scenario_id,
            "matched_scenario_name": self.matched_scenario_name,
            "matched_source": self.matched_source,
            "matched_source_label": self.matched_source_label,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SourceBoundaryDecision:
    """单次场景内 source 边界检测结果。

    字段说明：
      - mismatched：用户选择的 source 是否与问题内容明显不匹配。
      - selected_source：用户显式选择的 source。
      - selected_source_label：用户选择的 source 展示名称。
      - matched_source：根据问题内容命中的 source。
      - matched_source_label：命中 source 的展示名称。
      - reason：判断原因。

    调用顺序：场景解析入口 -> SourceBoundaryDecision。
    """

    mismatched: bool
    selected_source: str = ""
    selected_source_label: str = ""
    matched_source: str = ""
    matched_source_label: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, str | bool]:
        """转换为结构化字典，供 trace 和审计日志使用。

        返回：
            包含 mismatched、selected_source、matched_source、reason 等字段的字典。

        调用顺序：场景解析入口 -> SourceBoundaryDecision.as_dict()。
        """
        return {
            "mismatched": self.mismatched,
            "selected_source": self.selected_source,
            "selected_source_label": self.selected_source_label,
            "matched_source": self.matched_source,
            "matched_source_label": self.matched_source_label,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SourceMatch:
    """场景 source pattern 匹配后的排名结果，包含 source key、累计分数和归一化置信度。（★★ 理解）

    confidence 是归一化诊断值，不是模型概率分数。它帮助 Trace 面板判断最佳 source
    是否明显领先其他候选。当 confidence 接近 1.0 时表示匹配结果非常确定，
    低于 0.5 时说明多个分类间模糊，可能需要人工确认。

    调用顺序：场景解析入口 -> rank_source_matches() -> SourceMatch。
    """

    source: str
    score: int
    confidence: float

    def as_dict(self) -> dict[str, str | int | float]:
        """转换为可 JSON 序列化的匹配诊断结构，供 trace 和前端调试展示。

        返回：
            {"source": str, "score": int, "confidence": float} 字典。

        调用顺序：场景解析入口 -> SourceMatch.as_dict()。
        """

        return {
            "source": self.source,
            "score": self.score,
            "confidence": self.confidence,
        }


def score_source_map(query: str, scenario: ScenarioDefinition) -> dict[str, int]:
    """计算问题对当前场景内每个 source pattern 的匹配分数。

    计分公式：命中次数 * 10 + 命中文本长度总和。

    公式设计理念：
      - 命中次数 * 10：让多个业务词共同命中成为主要证据，降低单个偶然关键词的影响。
      - + 匹配文本总长度：同次数时，匹配到的原文越长说明 pattern 与 query 的语义重叠度越高，
        对跨场景判断更有参考价值。
      - source 顺序不进入相关性分数，只在完全同分时作为稳定排序条件。

    该分数是可解释的规则排序值，不是概率。权重和边界阈值需要通过标注集持续评测。

    参数：
        query: 用户原始问题。
        scenario: 当前业务场景，内部包含 source_patterns。

    返回：
        {source_key: score} 字典；没有任何命中时返回空字典。

    调用顺序：场景解析入口 -> score_source_map()。
    """
    normalized = query.strip()
    scores: dict[str, int] = {}
    for source, pattern in scenario.compiled_source_patterns().items():
        # pattern 已使用 re.IGNORECASE 编译，只匹配一次，避免原文与 lower 文本重复计数。
        matches = list(pattern.finditer(normalized))
        if not matches:
            continue
        scores[source] = len(matches) * SOURCE_MATCH_COUNT_WEIGHT + sum(len(match.group(0)) for match in matches)
    return scores


def rank_source_matches(query: str, scenario: ScenarioDefinition) -> tuple[SourceMatch, ...]:
    """对 query 在每个 source pattern 上的匹配结果排序并返回，附带归一化置信度用于 Trace 诊断。（★★★ 核心）

    执行流程：
      1. 调用 score_source_map 计算 query 在每个 source pattern 上的匹配分数。
      2. 无任何匹配时返回空元组（调用方据此判定当前场景没有相关业务分类）。
      3. 以最高分为基准计算每个 source 的置信度（当前分 / 最高分）。
      4. 按分数降序排序，完全同分时按 valid_sources 配置顺序稳定决胜。

    参数：
        query: 用户原始问题。
        scenario: 当前业务场景（提供 source_patterns 和 valid_sources）。

    返回：
        tuple[SourceMatch, ...]: 按分数降序排列的匹配结果；无任何匹配时返回空元组。

    调用顺序：场景解析入口 -> score_source_matches() / rank_source_matches()。
    """

    scores = score_source_map(query, scenario)
    # 没有任何 source pattern 命中时，返回空元组而非 None，避免调用方做 None 检查
    if not scores:
        return ()
    # 以最高分为基准计算置信度，用于前端 trace 面板解释最佳 source 是否明显领先其他候选
    best_score = max(scores.values())
    # 按 valid_sources 顺序建立索引表，用于完全同分时的稳定排序
    source_order = {source: index for index, source in enumerate(scenario.valid_sources)}
    return tuple(
        SourceMatch(
            source=source,
            score=score,
            # 置信度 = 当前得分 / 最高分，best_score 为 0 时兜底为 0.0
            confidence=round(score / best_score, 4) if best_score > 0 else 0.0,
        )
        # 相关性按分数降序；完全同分时才按 valid_sources 配置顺序稳定决胜
        for source, score in sorted(scores.items(), key=lambda item: (-item[1], source_order[item[0]]))
    )


def score_source_matches(query: str, scenario: ScenarioDefinition) -> tuple[str | None, int]:
    """返回当前场景内最匹配的 source 及其分数。

    参数：
        query: 用户原始问题。
        scenario: 当前业务场景。

    返回：
        (最佳 source 或 None, 分数)。

    调用顺序：场景解析入口 -> score_source_matches()。
    """
    # 取排名最高的 source 及其分数；空结果时返回 None 让调用方自行判断
    matches = rank_source_matches(query, scenario)
    if not matches:
        return None, 0
    best = matches[0]
    return best.source, best.score


def detect_source_boundary(
    query: str,
    current_scenario: ScenarioDefinition,
    selected_source: str | None,
) -> SourceBoundaryDecision:
    """检查用户显式选择的 source 是否与问题内容明显不匹配。

    执行流程：
      1. 用户没有选择 source 时不做提示。
      2. 根据当前场景 source_patterns 推断问题更像哪个 source。
      3. 如果没有命中或命中结果就是用户选择的 source，则继续正常流程。
      4. 如果命中的是另一个 source，则提示切换分类。

    参数：
        query: 用户原始问题。
        current_scenario: 当前业务场景。
        selected_source: 用户显式选择的 source，可能为空。

    返回：
        source 边界检测结果。
    """
    # 三级检测流程：空选择 → 已匹配 → 建议切换
    # 用户未指定分类时不做越界提示，避免空分类场景下的误报干扰
    if not selected_source:
        return SourceBoundaryDecision(mismatched=False, reason="no_selected_source")
    matched_source, score = score_source_matches(query, current_scenario)
    # 无匹配 / 已选分类最佳 → 保持当前流程，减少不必要的前端切换建议
    if not matched_source or matched_source == selected_source:
        return SourceBoundaryDecision(mismatched=False, selected_source=selected_source, reason="source_matched_or_unknown")
    # 命中另一个分类 → 主动建议切换，帮助用户快速定位
    return SourceBoundaryDecision(
        mismatched=True,
        selected_source=selected_source,
        selected_source_label=current_scenario.label_for_source(selected_source),
        matched_source=matched_source,
        matched_source_label=current_scenario.label_for_source(matched_source),
        reason=f"matched_source_score={score}",
    )


def detect_scenario_boundary(query: str, current_scenario: ScenarioDefinition) -> ScenarioBoundaryDecision:
    """检测当前问题是否明显属于另一个业务场景。

    执行流程：
      1. 先对当前场景的 source_patterns 打分。
      2. 遍历其他场景，找出最佳候选场景。
      3. 候选场景必须达到最低分，避免无依据切换。
      4. 当前场景已有可信命中时，候选还必须领先一个最小幅度。
      5. 两项条件都成立时，返回跨场景建议和命中详情。

    参数：
        query: 用户原始问题。
        current_scenario: 当前业务场景。

    返回：
        场景边界检测结果。
    """
    # 同时比较当前场景和最佳候选，避免当前场景的单个弱关键词压住其他场景的强证据。
    current_source, current_score = score_source_matches(query, current_scenario)
    best_scenario: ScenarioDefinition | None = None
    best_source = None
    best_score = 0
    for scenario in get_scenario_registry().list_scenarios():
        if scenario.scenario_id == current_scenario.scenario_id:
            continue
        source, score = score_source_matches(query, scenario)
        if source and score > best_score:
            best_scenario = scenario
            best_source = source
            best_score = score

    # 无其他场景达到匹配最低分 → 保持当前场景不切换，避免无依据的场景跳转
    if best_scenario is None or best_source is None or best_score < MIN_OTHER_SCENARIO_SCORE:
        return ScenarioBoundaryDecision(crossed=False, reason="no_strong_other_scenario")

    if (
        current_source
        and current_score >= CURRENT_SCENARIO_SAFE_SCORE
        and best_score < current_score + OTHER_SCENARIO_SCORE_MARGIN
    ):
        return ScenarioBoundaryDecision(
            crossed=False,
            reason=f"current_scenario_not_outscored: current_score={current_score}, other_score={best_score}",
        )

    # 候选达到最低门槛，且在当前场景有证据时至少领先一个 margin，才建议切换。
    return ScenarioBoundaryDecision(
        crossed=True,
        matched_scenario_id=best_scenario.scenario_id,
        matched_scenario_name=best_scenario.display_name,
        matched_source=best_source,
        matched_source_label=best_scenario.label_for_source(best_source),
        reason=f"other_scenario_score={best_score}, current_score={current_score}",
    )
