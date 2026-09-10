"""BERT 微调意图模型服务：V1 意图网关使用的本地序列分类模型。

在线 RAG 链路在确定性路由之后、检索计划构建之前调用此模块。它加载本地 HuggingFace
BertForSequenceClassification 产物，返回 FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP
三个检索类意图的预测结果。

设计决策：
- 使用 local_files_only=True 禁止 HuggingFace 联网下载，保证模型推理不依赖外网。
- 模型加载和 tokenizer 初始化在构造函数中完成，避免每次 predict 都重新加载。
- 通过 from_settings() 工厂方法从运行时配置构建服务实例，方便集成测试 mock。

调用顺序：apply_intent_decision_gateway() -> model_classifier。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RETRIEVAL_INTENTS = ("FAQ_QUERY", "KNOWLEDGE_QUERY", "FOLLOW_UP")
LABELS_FILENAME = "intent_labels.json"
INTENT_DATASET_DIR = Path(__file__).resolve().parents[2] / "eval_sets" / "intent"
DEFAULT_TRAINING_DATASET = INTENT_DATASET_DIR / "train.jsonl"
DEFAULT_EVAL_DATASET = INTENT_DATASET_DIR / "eval.jsonl"


@dataclass(frozen=True)
class IntentTrainingExample:
    """BERT 意图微调的单条监督样本。

    query  # 字段说明：用户查询文本
    label  # 字段说明：意图标签，必须属于 RETRIEVAL_INTENTS
    has_history  # 字段说明：查询时是否有对话历史

    调用顺序：训练或评测阶段 -> IntentTrainingExample。
    """

    query: str
    label: str
    has_history: bool = False


def load_intent_examples(path: str | Path) -> tuple[IntentTrainingExample, ...]:
    """从 JSONL 加载并校验意图训练或评测样本。（★★ 理解）

    数据文件属于可版本化的数据资产，业务代码只维护字段契约。加载时拒绝以下情况：
    - 空问题（query 为空字符串）。
    - 未知标签（不属于 RETRIEVAL_INTENTS）。
    - 非布尔历史标记（has_history 必须是 bool 类型）。
    - 重复样本（相同 query 和 has_history 的组合）。

    参数：
        path: JSONL 文件路径（每行一个 JSON 对象，包含 query、label、has_history 字段）。

    返回：
        IntentTrainingExample 元组。

    调用顺序：训练或评测脚本 -> load_intent_examples()。
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"意图数据集不存在：{dataset_path}")

    examples: list[IntentTrainingExample] = []
    # 原因：用 set 去重，key 为 (query.casefold(), has_history)，忽略大小写
    seen: set[tuple[str, bool]] = set()
    for line_no, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"意图数据集第 {line_no} 行不是合法 JSON：{dataset_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"意图数据集第 {line_no} 行必须是 JSON 对象：{dataset_path}")

        query = str(payload.get("query") or "").strip()
        label = str(payload.get("label") or "").strip()
        has_history = payload.get("has_history", False)
        # 校验：空问题会干扰训练，直接拒绝
        if not query:
            raise ValueError(f"意图数据集第 {line_no} 行缺少 query：{dataset_path}")
        # 校验：标签必须属于预定义的三个检索类意图
        if label not in RETRIEVAL_INTENTS:
            raise ValueError(f"意图数据集第 {line_no} 行 label 必须属于 {RETRIEVAL_INTENTS}：{dataset_path}")
        # 校验：has_history 必须是布尔值，避免字符串 "true"/"false" 混入训练数据
        if not isinstance(has_history, bool):
            raise ValueError(f"意图数据集第 {line_no} 行 has_history 必须是布尔值：{dataset_path}")

        # 去重检查：相同 query + has_history 的样本只保留第一条
        key = (query.casefold(), has_history)
        if key in seen:
            raise ValueError(f"意图数据集存在重复样本：query={query!r}, has_history={has_history}")
        seen.add(key)
        examples.append(IntentTrainingExample(query=query, label=label, has_history=has_history))

    if not examples:
        raise ValueError(f"意图数据集不能为空：{dataset_path}")
    # 校验：三个标签都必须有样本，否则模型无法学会区分全部意图
    missing_labels = sorted(set(RETRIEVAL_INTENTS) - {example.label for example in examples})
    if missing_labels:
        raise ValueError(f"意图数据集缺少标签 {missing_labels}：{dataset_path}")
    return tuple(examples)


def load_default_training_examples() -> tuple[IntentTrainingExample, ...]:
    """加载项目默认训练集（eval_sets/intent/train.jsonl）。

    返回：
        IntentTrainingExample 元组。

    调用顺序：训练脚本 -> load_default_training_examples()。
    """
    return load_intent_examples(DEFAULT_TRAINING_DATASET)


def load_default_eval_examples() -> tuple[IntentTrainingExample, ...]:
    """加载项目默认独立评测集（eval_sets/intent/eval.jsonl）。

    返回：
        IntentTrainingExample 元组。

    调用顺序：评测脚本或治理报告 -> load_default_eval_examples()。
    """
    return load_intent_examples(DEFAULT_EVAL_DATASET)


def validate_intent_dataset_split(
    training_examples: Iterable[IntentTrainingExample],
    eval_examples: Iterable[IntentTrainingExample],
) -> None:
    """拒绝训练集与评测集中的重复问题，防止评测数据泄漏。（★★ 理解）

    训练集和评测集不能有相同 (query.casefold(), has_history) 的组合。如果有重叠，
    模型的评测准确率会被高估（因为模型"见过"评估样本），导致部署到线上后的真实表现
    低于预期。

    参数：
        training_examples: 训练集样本。
        eval_examples: 评测集样本。

    调用顺序：训练或评测脚本 -> validate_intent_dataset_split()。
    """
    # 原因：用 casefold() 忽略大小写，保证 "报销流程" 和 "报销流程" 被视为重复
    training_keys = {(item.query.casefold(), item.has_history) for item in training_examples}
    eval_keys = {(item.query.casefold(), item.has_history) for item in eval_examples}
    overlap = sorted(training_keys & eval_keys)
    if overlap:
        queries = ", ".join(query for query, _ in overlap[:5])
        raise ValueError(f"意图训练集与评测集存在 {len(overlap)} 条重复样本：{queries}")


@dataclass(frozen=True)
class IntentModelPrediction:
    """模型预测结果，返回给意图决策网关使用。

    intent  # 字段说明：预测的意图标签（FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP）
    score  # 字段说明：预测标签的 softmax 概率
    scores  # 字段说明：所有标签的完整概率分布
    reason  # 字段说明：预测原因，固定为 "bert_intent_model"
    model_version  # 字段说明：模型版本号

    调用顺序：决策网关 -> IntentModelPrediction。
    """

    intent: str
    score: float
    scores: dict[str, float]
    reason: str
    model_version: str

    def as_dict(self) -> dict[str, object]:
        """转换为可 JSON 序列化的诊断数据。

        返回：
            包含 intent、score、scores、reason 和 model_version 的字典。

        调用顺序：IntentResult.as_dict() -> IntentModelPrediction.as_dict()。
        """
        return {
            "intent": self.intent,
            "score": round(self.score, 4),
            "scores": {intent: round(score, 4) for intent, score in self.scores.items()},
            "reason": self.reason,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class IntentModelEvaluation:
    """离线评测结果：在标注验证集上的准确率和混淆矩阵。

    accuracy  # 字段说明：准确率（正确预测数 / 总数）
    total  # 字段说明：评测样本总数
    correct  # 字段说明：正确预测数
    confusion_matrix  # 字段说明：混淆矩阵，格式为 matrix[真实标签][预测标签] = 计数

    调用顺序：评测脚本或治理报告 -> IntentModelEvaluation。
    """

    accuracy: float
    total: int
    correct: int
    confusion_matrix: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, object]:
        """转换为可 JSON 序列化的诊断数据。

        返回：
            包含 accuracy、total、correct 和 confusion_matrix 的字典。

        调用顺序：治理报告 -> IntentModelEvaluation.as_dict()。
        """
        return {
            "accuracy": round(self.accuracy, 4),
            "total": self.total,
            "correct": self.correct,
            "confusion_matrix": self.confusion_matrix,
        }


def format_intent_model_input(query: str, *, has_history: bool) -> str:
    """格式化模型输入文本：保持训练和推理时的输入格式一致。

    格式："有历史对话。用户问题：xxx" 或 "无历史对话。用户问题：xxx"
    训练和推理使用相同的格式化函数，保证模型识别到的模式一致。

    参数：
        query: 用户查询文本。
        has_history: 当前查询是否有对话历史。

    返回：
        格式化后的模型输入字符串。

    调用顺序：BertIntentModelService.predict() -> format_intent_model_input()。
    """
    history_marker = "有历史对话" if has_history else "无历史对话"
    return f"{history_marker}。用户问题：{query.strip()}"


class BertIntentModelService:
    """本地 BERT 序列分类模型服务，用于检索类意图识别。

    加载本地 HuggingFace BertForSequenceClassification 模型，提供 predict()
    和 evaluate() 两个方法。模型和 tokenizer 在构造函数中完成加载，后续调用
    predict() 不需重复加载。

    from_settings() 工厂方法从运行时配置构建实例，微调后的模型路径由
    INTENT_MODEL_PATH 配置控制。

    调用顺序：apply_intent_decision_gateway() -> BertIntentModelService。
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 64,
        model_version: str | None = None,
    ) -> None:
        """加载本地 BERT 模型和 tokenizer，准备推理环境。（★★★ 核心）

        执行流程：
          1. 解析设备参数（cpu / cuda / auto）。
          2. 校验模型目录存在且包含必需文件。
          3. 读取标签列表并校验顺序与 RETRIEVAL_INTENTS 一致。
          4. 加载 HuggingFace tokenizer（local_files_only=True）。
          5. 加载 HuggingFace 模型（local_files_only=True）。
          6. 校验模型标签数量与配置文件一致。
          7. 将模型移入目标设备并切换为 eval 模式。

        参数：
            model_path: 本地模型目录路径。
            device: 推理设备（cpu / cuda / auto）。
            max_length: tokenizer 的最大序列长度（默认 64，足够覆盖短查询）。
            model_version: 模型版本号，None 时从 intent_labels.json 读取。

        调用顺序：from_settings() -> BertIntentModelService.__init__()。
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        # 解析设备参数：auto 时自动检测 CUDA
        self.device = _resolve_device(device)
        # 校验模型目录完整性（必需文件和权重文件）
        _require_model_artifact(self.model_path)
        # 读取标签列表（必须与 RETRIEVAL_INTENTS 一致）
        self.labels = _load_label_sequence(self.model_path)
        # 建立 label -> index 映射，用于解释模型输出的概率分布
        self.label2id = {label: index for index, label in enumerate(self.labels)}
        # 读取模型版本号，用于 Trace 和诊断输出
        self.model_version = model_version or _read_model_version(self.model_path)

        # 加载 HuggingFace tokenizer（local_files_only=True 不走外网）
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        # 加载 HuggingFace 模型（local_files_only=True 不走外网）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        # 校验：模型配置的标签数量必须与 intent_labels.json 一致
        if int(self.model.config.num_labels) != len(self.labels):
            raise RuntimeError(
                f"意图模型标签数量不一致：model num_labels={self.model.config.num_labels}, "
                f"{LABELS_FILENAME} labels={len(self.labels)}"
            )
        # 将模型移入目标设备并切换为 eval 模式（禁用 dropout 等训练机制）
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_settings(cls) -> "BertIntentModelService":
        """从运行时配置构建模型服务实例。

        从 settings 中读取模型路径、设备、最大序列长度和版本号，传递给构造函数。
        这是推荐的实例化方式，便于集成测试时替换配置。

        返回：
            配置完成的 BertIntentModelService 实例。

        调用顺序：apply_intent_decision_gateway() -> BertIntentModelService.from_settings()。
        """
        from qa_core.config.settings import get_settings

        settings = get_settings()
        return cls(
            settings.intent_model_path,
            device=settings.intent_model_device,
            max_length=settings.intent_model_max_length,
            model_version=settings.intent_model_version,
        )

    def predict(self, query: str, *, has_history: bool = False) -> IntentModelPrediction:
        """用本地 BERT 模型预测检索类意图。（★★★ 核心）

        执行流程：
          1. 将用户查询格式化为"有/无历史对话。用户问题：xxx"的固定输入格式。
          2. Tokenize -> 截断/填充到 max_length -> 转为 PyTorch tensor。
          3. 将 tensor 移到目标设备（CPU/CUDA）。
          4. 无梯度模式下执行前向推理，取 batch[0] 的 logits。
          5. Softmax 归一化为概率分布，转回 CPU list。
          6. 构建 label -> score 映射，取最高分作为预测意图。

        参数：
            query: 用户查询文本。
            has_history: 当前查询是否有对话历史。

        返回：
            IntentModelPrediction 对象，包含预测意图、概率分布和模型版本。

        调用顺序：apply_intent_decision_gateway() -> BertIntentModelService.predict()。
        """
        # ── 步骤 1：格式化为固定输入格式 ──
        text = format_intent_model_input(query, has_history=has_history)
        # ── 步骤 2：tokenize -> truncation + padding to max_length ──
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        # ── 步骤 3：将 input_ids/attention_mask 移到目标设备 ──
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        # ── 步骤 4：无梯度推理 ──
        # torch.no_grad() 禁用梯度计算（推理模式，节省显存和计算）
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            # ── 步骤 5：softmax 归一化 ──
            # 原因：logits 是原始分数，softmax 转为概率分布（各意图概率之和 = 1）
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        # ── 步骤 6：构建 label -> score 映射，取最高分 ──
        scores = {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        intent = max(scores, key=scores.get)
        return IntentModelPrediction(
            intent=intent,
            score=scores[intent],
            scores=scores,
            reason="bert_intent_model",
            model_version=self.model_version,
        )

    def evaluate(self, examples: Iterable[IntentTrainingExample]) -> IntentModelEvaluation:
        """在标注样本上评估已加载的 BERT 模型。（★★ 理解）

        对标注样本逐条预测，统计准确率和混淆矩阵。混淆矩阵格式：
        matrix[真实标签][预测标签] = 计数。

        参数：
            examples: IntentTrainingExample 可迭代对象。

        返回：
            IntentModelEvaluation 对象，包含准确率、总数、正确数和混淆矩阵。

        调用顺序：治理报告或评测脚本 -> BertIntentModelService.evaluate()。
        """
        total = 0
        correct = 0
        # 初始化 3×3 混淆矩阵（FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP）
        matrix: dict[str, dict[str, int]] = {
            label: {predicted: 0 for predicted in RETRIEVAL_INTENTS}
            for label in RETRIEVAL_INTENTS
        }
        for example in examples:
            if example.label not in matrix:
                raise ValueError(f"unsupported intent label: {example.label}")
            prediction = self.predict(example.query, has_history=example.has_history)
            total += 1
            if prediction.intent == example.label:
                correct += 1
            # 混淆矩阵：真实标签 -> 预测标签 计数+1
            matrix[example.label][prediction.intent] += 1
        accuracy = correct / total if total else 0.0
        return IntentModelEvaluation(
            accuracy=accuracy,
            total=total,
            correct=correct,
            confusion_matrix=matrix,
        )


def _resolve_device(device: str) -> str:
    """解析设备参数：auto 时自动检测 CUDA 可用性，否则校验 cpu/cuda 合法性。

    参数：
        device: 设备参数（"cpu" / "cuda" / "auto"）。

    返回：
        解析后的设备名（"cpu" 或 "cuda"）。

    调用顺序：BertIntentModelService.__init__() -> _resolve_device()。
    """
    normalized = (device or "cpu").strip().lower()
    # auto 模式：优先 CUDA，不可用时降级为 CPU
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("INTENT_MODEL_DEVICE=cuda，但当前环境不可用 CUDA")
    if normalized not in {"cpu", "cuda"}:
        raise RuntimeError("INTENT_MODEL_DEVICE 只能是 cpu、cuda 或 auto")
    return normalized


def _require_model_artifact(model_path: Path) -> None:
    """校验模型目录存在且包含必需的配置文件、标签文件和权重文件。

    必需文件：config.json（HuggingFace）、intent_labels.json（自定义标签映射）。
    权重文件兼容两种格式：model.safetensors（推荐）或 pytorch_model.bin（旧版 HF 格式）。

    参数：
        model_path: 模型目录路径。

    调用顺序：BertIntentModelService.__init__() -> _require_model_artifact()。
    """
    if not model_path.exists():
        raise RuntimeError(f"意图 BERT 模型目录不存在：{model_path}")
    # 必需文件：HuggingFace config.json + 自定义 intent_labels.json
    required_files = ("config.json", LABELS_FILENAME)
    missing = [name for name in required_files if not (model_path / name).exists()]
    if missing:
        raise RuntimeError(f"意图 BERT 模型目录缺少文件 {missing}：{model_path}")
    # 权重文件兼容两种格式：safetensors（推荐）或 pytorch_model.bin（旧版）
    if not (model_path / "model.safetensors").exists() and not (model_path / "pytorch_model.bin").exists():
        raise RuntimeError(f"意图 BERT 模型目录缺少权重文件：{model_path}")


def _load_label_sequence(model_path: Path) -> tuple[str, ...]:
    """从 intent_labels.json 读取标签列表并校验顺序必须与 RETRIEVAL_INTENTS 一致。

    标签顺序校验非常重要：BertForSequenceClassification 的输出是 logits 数组，
    第 i 个元素对应第 i 个标签。如果标签顺序与训练时不匹配，模型的预测结果完全错误。

    参数：
        model_path: 模型目录路径。

    返回：
        标签元组，顺序与 RETRIEVAL_INTENTS 一致。

    调用顺序：BertIntentModelService.__init__() -> _load_label_sequence()。
    """
    payload = json.loads((model_path / LABELS_FILENAME).read_text(encoding="utf-8"))
    # 兼容两种格式：{"labels": [...]} 或直接的 [...]
    if isinstance(payload, dict):
        raw_labels = payload.get("labels")
    else:
        raw_labels = payload
    if not isinstance(raw_labels, list) or not raw_labels:
        raise RuntimeError(f"{LABELS_FILENAME} 必须包含非空 labels 列表：{model_path}")
    labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
    # 标签顺序必须与训练时的 RETRIEVAL_INTENTS 一致，否则模型输出 index 对不上
    if labels != RETRIEVAL_INTENTS:
        raise RuntimeError(f"意图模型标签顺序必须是 {RETRIEVAL_INTENTS}，实际为 {labels}")
    return labels


def _read_model_version(model_path: Path) -> str:
    """从 intent_labels.json 读取模型版本号，未配置时返回默认版本。

    参数：
        model_path: 模型目录路径。

    返回：
        模型版本字符串。

    调用顺序：BertIntentModelService.__init__() -> _read_model_version()。
    """
    payload = json.loads((model_path / LABELS_FILENAME).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("model_version"):
        return str(payload["model_version"])
    return "bert-intent-v1"
