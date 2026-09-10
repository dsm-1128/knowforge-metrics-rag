"""FAQ 分类到场景 source 的轻量映射工具。

在 FAQ 入库前，将 CSV 中的业务分类（subject）字段映射到场景配置的
valid_sources 白名单。映射策略：优先精确匹配，再按正则模式模糊匹配，
均无法匹配时抛异常以提醒运维检查场景配置遗漏。

设计决策：
- 不依赖 pandas、Milvus 等重库，测试和校验代码可单独复用。
- 匹配顺序（精确 -> 模糊）保证在分类名恰好等于 source 值时不走正则，
  既提高匹配效率又避免正则误匹配。
"""

from __future__ import annotations

from qa_core.scenarios.registry import ScenarioDefinition


def normalize_faq_source(
    subject: str,
    *,
    scenario: ScenarioDefinition,
    question: str = "",
) -> str:
    """将 FAQ 分类标准化为场景的有效 source。（★★★ 核心）

    执行流程：
      1. 对 subject 做大小写归一化和去空格。
      2. 精确匹配：如果标准化后的分类名完全等于 valid_sources 中的某个值，直接返回。
      3. 模糊匹配：遍历场景配置的正则模式，subject 或 question 任意一个命中即返回。
      4. 均无法匹配则抛 ValueError，提醒运维检查场景配置是否有遗漏。

    参数：
        subject: FAQ 分类名（如"IT 故障"、"财务报销"）。
        scenario: 场景定义对象，包含 valid_sources 和编译好的正则模式。
        question: FAQ 标准问题（可选），匹配 source 时作为补充输入。

    返回：
        标准化后的 source 字符串，属于 scenario.valid_sources 成员。

    返回值是后续 source 过滤、manifest 分区、质量统计和 FAQ ID 生成的共同
    业务键。映射失败必须抛错，不能把未知分类默认塞进某个 source，否则问题
    会在入库后才表现为“检索不到”或“跨分类召回”。

    调用顺序：FAQ 入库阶段 -> normalize_faq_source()。
    """

    normalized = subject.strip().lower()
    # 先尝试精准匹配：如果分类名完全等于 valid_sources 中的某个值，直接返回，无需正则遍历
    if normalized in scenario.valid_sources:
        return normalized

    # 模糊匹配：遍历场景配置的正则模式，subject（FAQ 分类名）或 question（问题内容）
    # 任意一个命中即认为该 FAQ 属于该 source
    for source, pattern in scenario.compiled_source_patterns().items():
        if pattern.search(subject) or pattern.search(question):
            return source

    # 无法映射到任何 source：说明此 FAQ 分类不在场景配置范围内，需要先确认场景配置是否有遗漏
    raise ValueError(
        f"FAQ 分类无法映射到场景 {scenario.scenario_id} 的 valid_sources："
        f"subject={subject!r}, question={question!r}"
    )
