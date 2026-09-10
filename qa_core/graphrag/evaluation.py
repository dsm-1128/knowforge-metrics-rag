"""GraphRAG 索引质量的轻量评估工具 —— V2.1 验收测试辅助。

对一组预定义的业务问题执行 GraphRAG 查询，检查每个问题是否能
检索到至少一条路径，并统计整体通过率。评估结果用于：
  1. CI 验收门禁：判断图谱索引质量是否达标。
  2. 人工审查：快速定位哪些问题类型在图谱中缺乏覆盖。
"""

from __future__ import annotations

from typing import Any

from qa_core.graphrag.graph_store import query_graphrag


# 默认的 GraphRAG 评估问题集合，覆盖 IT 排障、HR 流程、财务报销、
# 合规风控、运维监控等主要企业场景。
DEFAULT_GRAPH_EVAL_QUESTIONS = (
    "VPN 故障和账号权限之间有什么排查路径？",
    "新人入职流程和审批之间有什么关系？",
    "报销材料和发票规则之间有什么依赖？",
    "合同合规风险和供应商审查之间如何关联？",
    "设备告警和安全巡检之间有什么处理路径？",
)


def evaluate_graphrag(scenario_id: str | None = None) -> dict[str, Any]:
    """对指定场景运行 GraphRAG 评估。

    执行流程：
      1. 遍历 DEFAULT_GRAPH_EVAL_QUESTIONS 中的每个问题。
      2. 调用 query_graphrag 检索图谱路径（限制最多 3 条路径）。
      3. 记录每个问题的路径数量和最高路径评分。
      4. 若 path_count > 0，该问题标记为 passed。
      5. 统计整体通过数，若全部通过则 ok 为 True。

    参数:
        scenario_id: 场景 ID。为 None 时使用默认场景。

    返回:
        包含以下字段的评估结果字典：
          - scenario_id: 评估的场景 ID
          - case_count: 总测试用例数
          - passed: 通过的用例数
          - items: 每个用例的详细结果（问题、路径数、是否通过、最高分）
          - ok: 是否全部通过
    """
    # ── 遍历每个预定义问题，执行 GraphRAG 查询 ──
    # ── 遍历每个预定义问题，执行 GraphRAG 查询 ──
    items = []
    for question in DEFAULT_GRAPH_EVAL_QUESTIONS:
        result = query_graphrag(question, scenario_id=scenario_id, max_paths=3)
        items.append(
            {
                "question": question,
                "path_count": result.get("path_count", 0),
                # 通过判定：至少检索到一条关系路径
                "passed": int(result.get("path_count", 0) or 0) > 0,
                # 取路径列表第一项的评分，空列表兜底防止 IndexError
                "top_score": ((result.get("paths") or [{}])[0]).get("score", 0),
            }
        )
    # ── 汇总统计 ──
    return {
        "scenario_id": scenario_id or "",
        "case_count": len(items),
        "passed": sum(1 for item in items if item["passed"]),
        "items": items,
        # 整体通过判定：所有用例均通过才返回 ok=True
        "ok": all(item["passed"] for item in items),
    }
