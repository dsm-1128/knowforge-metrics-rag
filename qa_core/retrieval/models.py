"""检索链路模型加载器：集中管理 BGE embedding 和 CrossEncoder reranker 的进程级单例。

设计决策：
- 进程级 lru_cache 管理重资源：BGE embedding 约几百 MB，CrossEncoder 约 1GB+，
  每个请求重复加载显然不可接受。lru_cache(maxsize=1) 保证全进程只加载一次，所有
  请求共享同一模型实例（线程安全由模型推理内部保证）。
- local_files_only=True：所有模型文件必须本地存在，不允许从 HuggingFace Hub 自动
  下载。这确保了离线部署环境的稳定性和合规性（模型可审计、可版本控制）。
- CPU 是一等执行设备而非降级方案：BGE 和 CrossEncoder 在典型批大小下 CPU 推理
  延迟完全可接受（约 20-50ms/query），不硬性依赖 GPU 降低部署门槛。
- CachedEmbeddings 包装 embedding：查询缓存对用户透明，业务代码无感知。

依赖分层：
- embedding 输出被 MilvusHybridStore 在构造时传入 langchain-milvus Milvus() 的
  embedding_function 参数，用于稠密向量检索时自动向量化查询文本。
- reranker 被 MillvusHybridStore._rerank() 调用，用于检索候选的二阶段精细排序。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# 防止 transformers 将 SentencePiece 转 Tiktoken 失败导致崩溃，需在导入前设置
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from qa_core.cache.embedding import CachedEmbeddings
from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings


logger = get_logger(__name__)


def resolve_device() -> str:
    """选择本机可用推理设备（CUDA > CPU），CPU 是正常执行设备而非降级方案。

    BGE embedding 和 CrossEncoder 在典型批大小下 CPU 推理延迟完全可接受，
    所以 CPU 是一等执行设备；CUDA 只是可有可无的加速，不是必要条件。
    这样设计避免了对 GPU 环境的硬依赖，降低部署门槛。

    调用顺序：检索准备或检索执行 -> resolve_device()。
    """
    # 检查 CUDA 是否可用：优先使用 GPU 加速推理以降低检索延迟
    # 当前环境即使无 GPU 也能以 CPU 正常运行（推理延迟在可接受范围内）
    # 原因：不硬性依赖 GPU，降低生产部署门槛，本地开发也无需 GPU
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embeddings():
    """返回已缓存的 BGE 向量模型，用于 Milvus 稠密向量检索。

    模型加载涉及从磁盘读取权重文件到 GPU/CPU 内存（数百 MB），开销巨大。
    lru_cache 保证整个进程生命周期只加载一次，所有请求共享同一个模型实例。

    调用顺序：检索准备或检索执行 -> get_embeddings()。
    """
    # 加载应用全局设置（embedding 模型路径等配置）。
    # 预热和在线检索共用这个缓存入口，保证不会各自创建一套模型。
    settings = get_settings()
    model_path = Path(settings.embedding_model_path)
    if not model_path.exists():
        # 模型文件不存在时直接抛错而非静默下载，确保部署环境一致性和离线稳定性
        raise RuntimeError(f"Embedding model path does not exist: {model_path}")
    # 创建 BGE HuggingFaceEmbeddings 实例，用于把用户问题转换成稠密向量。
    # local_files_only=True 强制仅从本地加载，不使用 HuggingFace Hub 远程文件
    # normalize_embeddings=True 启用 L2 归一化，使得余弦相似度等价于内积，Milvus 索引更高效
    embeddings = HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": resolve_device(), "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return CachedEmbeddings(embeddings)


@lru_cache(maxsize=1)
def get_reranker():
    """返回已缓存的 CrossEncoder 重排模型，用于 Milvus 召回后的二阶段精细排序。（★★★ 核心）

    执行流程：
      1. 读取配置中的 reranker 模型路径。
      2. 校验模型权重文件路径和 vocab 文件是否存在（缺少任一文件即启动失败）。
      3. 创建 CrossEncoder 实例（CrossEncoder 的 tokenizer 需要 vocab 文件
         sentencepiece.bpe.model——如果 vocab 文件缺失，即使模型权重文件存在也无法运行）。
      4. lru_cache 确保只加载一次，所有检索请求复用同一个重排模型实例。

    参数：
        （无参数，配置全部来自全局 settings）

    返回：
        CrossEncoder: 已加载的重排模型实例，实现了 predict() 方法。

    调用顺序：检索准备或检索执行 -> get_reranker()。
    """
    # 加载应用全局设置（reranker 模型路径等配置）。
    # 该入口与 warmup_retrieval_stack() 共用 lru_cache，预热后的实例会被正式检索复用。
    settings = get_settings()
    model_path = Path(settings.reranker_model_path)
    # ── 步骤 1：校验模型路径 ──
    # 原因：所有模型文件必须本地存在，不允许从 HuggingFace Hub 自动下载，
    # 确保离线部署环境的稳定性和可审计性
    if not model_path.exists():
        raise RuntimeError(f"Reranker model path does not exist: {model_path}")
    # ── 步骤 2：校验 vocab 文件 ──
    # CrossEncoder 的 tokenizer 需要 vocab 文件（sentencepiece.bpe.model）
    # 如果 vocab 文件缺失，即使模型权重文件存在也无法正常运行
    vocab_file = model_path / "sentencepiece.bpe.model"
    if not vocab_file.exists():
        raise RuntimeError(f"Reranker tokenizer vocab file does not exist: {vocab_file}")
    # ── 步骤 3：创建 CrossEncoder 实例 ──
    # local_files_only=True 确保不联网加载，适用于离线部署环境
    # use_fast=False 使用原版 tokenizer（与 BGE 训练阶段对齐），而非 HuggingFace 的 fast tokenizer
    # 原因：新版 transformers 读 tokenizer_config.json 相对路径可能拼错，显式传完整路径
    return CrossEncoder(
        str(model_path),
        device=resolve_device(),
        local_files_only=True,
        tokenizer_kwargs={
            "use_fast": False,
            "vocab_file": str(vocab_file),
        },
    )
