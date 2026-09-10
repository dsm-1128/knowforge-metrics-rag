"""意图模型治理：版本发布检查和管理后台状态展示的治理报告构建。

模块职责：
1. build_intent_model_report()：构建确定性的 intent 模型治理报告，包含模型产物状态、
   运行时状态、评测准确率、决策策略和 warmup 结果。
2. write_intent_model_report()：将治理报告写入 JSON 文件（reports/intent_model/），
   供版本发布脚本和 CI/CD 管道使用。
3. latest_intent_model_report()：读取最新的治理报告，不存在时在线生成只读报告。

治理检查的核心标准（build_intent_model_report 中的 ok 判定）：
- 模型目录存在且包含必需文件和权重。
- 模型可以正常加载和推理（warmup 通过）。
- 标签顺序与 RETRIEVAL_INTENTS 一致。
- 可选：在默认评测集上的准确率 >= 0.75。

调用顺序：管理 API 或发布脚本 -> governance。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qa_core.common import path_updated_at, read_json_dict, utc_now, write_json
from qa_core.config.settings import PROJECT_ROOT, get_settings
from qa_core.intent.decision import POLICY
from qa_core.intent.model_classifier import (
    DEFAULT_EVAL_DATASET,
    LABELS_FILENAME,
    RETRIEVAL_INTENTS,
    BertIntentModelService,
    load_default_eval_examples,
)


INTENT_MODEL_REPORT_DIR = PROJECT_ROOT / "reports" / "intent_model"
INTENT_MODEL_LATEST_REPORT = INTENT_MODEL_REPORT_DIR / "intent_model_latest.json"


def build_intent_model_report(*, evaluate: bool = True) -> dict[str, Any]:
    """构建确定性 intent 模型治理报告。（★★★ 核心）

    报告包含以下维度：
    - artifact_ok：模型目录完整性（必需文件和权重文件是否存在）。
    - runtime_ok：模型能否正常加载和推理。
    - ok：综合判定结果（artifact_ok + runtime_ok + 标签顺序正确 + 可选评测准确率 >= 0.75）。
    - model：模型路径、版本、设备、标签等元数据。
    - artifact：必需文件清单和缺失文件。
    - evaluation：在默认评测集上的准确率和混淆矩阵。
    - warmup：样本预测结果，验证模型输出合理性。
    - decision_policy：当前决策策略参数。
    - closure：相关脚本和端点的引用路径。

    参数：
        evaluate: 是否在报告包含评测结果。为 False 时跳过模型评测步骤。

    返回：
        完整的治理报告字典，可直接序列化为 JSON。

    调用顺序：管理 API / 发布脚本 -> build_intent_model_report()。
    """
    settings = get_settings()
    model_path = Path(settings.intent_model_path)
    labels_payload = read_json_dict(model_path / LABELS_FILENAME)
    required_files = ["config.json", LABELS_FILENAME, "tokenizer.json", "vocab.txt"]
    missing_files = [name for name in required_files if not (model_path / name).exists()]
    has_weights = (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists()
    artifact_ok = model_path.exists() and not missing_files and has_weights

    evaluation_payload: dict[str, Any] = {}
    warmup_payload: dict[str, Any] = {}
    runtime_ok = artifact_ok
    error = ""
    if artifact_ok:
        try:
            # 加载默认评测集（如果 evaluate=True）
            eval_examples = load_default_eval_examples() if evaluate else ()
            service = BertIntentModelService.from_settings()
            # 原因：使用"新人入职流程有哪些"作为样本查询，覆盖业务核心问题类型
            prediction = service.predict("新人入职流程有哪些", has_history=False)
            warmup_payload = {
                "model_version": service.model_version,
                "labels": list(service.labels),
                "sample_intent": prediction.intent,
                "sample_score": round(prediction.score, 4),
                "policy_version": POLICY.policy_version,
            }
            if evaluate:
                # 在默认评测集上运行完整评估
                evaluation_payload = service.evaluate(eval_examples).as_dict()
            runtime_ok = True
        except Exception as exc:  # pragma: no cover - exercised by deployment checks
            runtime_ok = False
            error = str(exc)

    accuracy = float(evaluation_payload.get("accuracy") or 0.0)
    labels = labels_payload.get("labels") if isinstance(labels_payload.get("labels"), list) else []
    # 综合判定：产物完整 + 运行正常 + 标签顺序正确 +（可选）准确率 >= 0.75
    ok = bool(artifact_ok and runtime_ok and tuple(labels) == RETRIEVAL_INTENTS and (not evaluate or accuracy >= 0.75))
    return {
        "report_type": "intent_model_governance",
        "created_at": utc_now(),
        "ok": ok,
        "artifact_ok": artifact_ok,
        "runtime_ok": runtime_ok,
        "error": error,
        "model": {
            "model_path": str(model_path),
            "model_version": settings.intent_model_version,
            "device": settings.intent_model_device,
            "max_length": settings.intent_model_max_length,
            "labels": labels,
            "expected_labels": list(RETRIEVAL_INTENTS),
            "base_model": labels_payload.get("base_model"),
            "training_examples": labels_payload.get("training_examples"),
            "eval_examples": labels_payload.get("eval_examples"),
            "training_dataset": labels_payload.get("training_dataset"),
            "eval_dataset": labels_payload.get("eval_dataset") or str(DEFAULT_EVAL_DATASET.relative_to(PROJECT_ROOT)),
            "training_dataset_sha256": labels_payload.get("training_dataset_sha256"),
            "eval_dataset_sha256": labels_payload.get("eval_dataset_sha256"),
            "updated_at": path_updated_at(model_path / LABELS_FILENAME) if (model_path / LABELS_FILENAME).exists() else "",
        },
        "artifact": {
            "required_files": required_files,
            "missing_files": missing_files,
            "has_weights": has_weights,
        },
        "evaluation": evaluation_payload,
        "warmup": warmup_payload,
        "decision_policy": {
            "policy_version": POLICY.policy_version,
            "model_min_score": POLICY.model_min_score,
            "agreement_score_boost": POLICY.agreement_score_boost,
            "conflict_final_score": POLICY.conflict_final_score,
        },
        "closure": {
            "online_gateway": "qa_core.intent.decision.apply_intent_decision_gateway",
            "training_script": "scripts/intent/train_intent_bert.py",
            "model_eval_script": "scripts/intent/demo_intent_model.py --eval-only",
            "policy_eval_script": "scripts/intent/evaluate_intent_policy.py --fail-on-critical",
            "admin_endpoint": "/api/admin/intent_model",
            "latest_report": str(INTENT_MODEL_LATEST_REPORT.relative_to(PROJECT_ROOT)),
        },
    }


def write_intent_model_report(path: str | Path = INTENT_MODEL_LATEST_REPORT, *, evaluate: bool = True) -> str:
    """将最新的 intent 模型治理报告写入 JSON 文件。

    写入路径默认为 reports/intent_model/intent_model_latest.json。
    目录不存在时自动创建。

    参数：
        path: 输出文件路径。
        evaluate: 是否在报告包含评测结果。

    返回：
        写入的文件路径字符串。

    调用顺序：发布脚本或管理 API -> write_intent_model_report()。
    """
    return write_json(path, build_intent_model_report(evaluate=evaluate))


def latest_intent_model_report() -> dict[str, Any]:
    """返回最新的 intent 模型治理报告。

    如果最新的报告文件（intent_model_latest.json）已存在，直接读取并返回。
    不存在时在线生成一个只读报告（不包含评测结果，避免模型加载耗时影响 API 响应）。

    返回：
        包含 available、file、updated_at 和 payload 的字典。

    调用顺序：管理 API -> latest_intent_model_report()。
    """
    payload = read_json_dict(INTENT_MODEL_LATEST_REPORT)
    if payload:
        return {
            "available": True,
            "file": str(INTENT_MODEL_LATEST_REPORT.relative_to(PROJECT_ROOT)),
            "updated_at": path_updated_at(INTENT_MODEL_LATEST_REPORT),
            "payload": payload,
        }
    return {
        "available": False,
        "file": None,
        "payload": build_intent_model_report(evaluate=False),
    }
