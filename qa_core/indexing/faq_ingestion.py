"""FAQ CSV 入库链路。

将 FAQ CSV 文件转换为可写入 Milvus 的 Document 对象并提供完整的入库编排。
FAQ 的 page_content 存储标准问题，标准答案放在 metadata.answer 中，
检索时用问题匹配，召回后将答案作为上下文返回给用户。

设计决策：
- FAQ 采用按版本快照重建模式：先删后写，确保新旧版本 FAQ 不混合。
- 使用 pandas 读取 CSV 而非 csv.DictReader：自动处理 BOM/编码推断/空值填充，
  兼容中英文列名（问题/question、答案/answer）。
- FAQ ID 由 scenario_id + kb_version + source + question 的稳定哈希生成，
  同一标准问题不同答案时加入 answer 参与哈希避免 ID 冲突。

依赖分层：
- qa_core.scenarios.registry：场景定义和 valid_sources 白名单。
- qa_core.governance：数据域隔离和知识库版本管理。
- qa_core.retrieval.factory：Milvus FAQ 集合写入。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.documents import Document

from qa_core.config.logging_config import get_logger
from qa_core.quality.faq import _resolve_csv_source
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store, version_metadata
from qa_core.indexing.source_normalization import normalize_faq_source
from qa_core.retrieval.factory import get_faq_store
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import stable_hash
logger = get_logger(__name__)


def _resolve_faq_context(
    scenario_id: str | None,
    tenant_id: str | None,
    dataset_id: str | None,
    visibility: str | None,
    allowed_roles: list[str] | None,
    *,
    scenario: Any | None = None,
    data_scope: Any | None = None,
) -> tuple[Any, Any]:
    """统一解析 FAQ 入库需要的场景和数据域上下文。

    FAQ 的“解析 Document”和“正式写入 Milvus”都需要同一套场景、数据范围。
    因此函数支持两种调用方式：
      - 离线总流程先完成前置解析，再把对象传进来，避免每条 FAQ 或每个阶段
        重复解析；
      - 独立调用 `faq_documents_from_csv()` 或 `ingest_faq_csv()` 时，只传
        标识参数，由这里补齐默认对象。

    这个复用边界很重要：上游传入对象时，下面所有 FAQ 记录都必须使用同一个
    `scenario_id`、collection 和 DataScope，不能在子函数内部偷偷解析成另一套上下文。

    调用顺序：`ingest_faq_csv()` / `faq_documents_from_csv()`
    -> `_resolve_faq_context()`。
    """
    resolved_scenario = (
        scenario if scenario is not None else resolve_scenario(scenario_id)
    )
    resolved_data_scope = (
        data_scope
        if data_scope is not None
        else resolve_data_scope(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            visibility=visibility,
            user_roles=allowed_roles,
        )
    )
    return resolved_scenario, resolved_data_scope


def faq_documents_from_csv(
    csv_path: str,
    kb_version: str | None = None,
    version_seq: int | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    *,
    scenario: Any | None = None,
    data_scope: Any | None = None,
) -> tuple[list[Document], list[str]]:
    """把 FAQ CSV 转换为可写入 Milvus 的 Document 对象列表。（★★★ 核心）

    每条 FAQ 行生成一个 Document：`page_content=标准问题`，metadata 包含
    标准答案、source、数据域隔离信息和版本元数据。重复行（相同问题+答案）
    自动跳过。

    FAQ 和普通文档采用不同的版本策略：
      - FAQ 是完整快照，记录直接带有当前 `kb_version`；
      - 检索时使用 `kb_version_exact` 精确过滤；
      - 不通过普通文档的 `valid_from_seq/valid_to_seq` 做跨版本引用。

    因此本函数只负责生成“当前版本应该存在的完整 FAQ 集合”，不负责删除旧
    FAQ；删除和写入由 `ingest_faq_csv()` 统一完成。

    参数：
        csv_path: FAQ CSV 文件路径，支持中文列名（问题/答案）和英文列名（question/answer）。
        kb_version: 知识库版本号（可选）。
        version_seq: 版本序号，用于引用式增量的有效期视图（可选）。
        scenario_id: 业务场景标识（可选，默认从 ACTIVE_SCENARIO_ID 读取）。
        tenant_id: 租户 ID（可选）。
        dataset_id: 数据集 ID（可选）。
        visibility: 可见级别（可选）。
        allowed_roles: 允许检索的角色列表（可选）。
        scenario: 已解析好的场景对象；上游已解析时可直接复用。
        data_scope: 已解析好的 DataScope；上游已解析时可直接复用。

    返回：
        (documents_list, faq_ids_list) 元组。
        documents_list: 待写入 Milvus 的 Document 列表。
        faq_ids_list: 对应的 FAQ ID 列表，用于后续删除和统计。

    调用顺序：入库脚本 -> ingest_faq_csv() -> faq_documents_from_csv()。
    """
    scenario, data_scope = _resolve_faq_context(
        scenario_id,
        tenant_id,
        dataset_id,
        visibility,
        allowed_roles,
        scenario=scenario,
        data_scope=data_scope,
    )
    version_meta = version_metadata(kb_version, scenario.scenario_id, version_seq=version_seq)
    # 第一步：读取整张 CSV。pandas 会统一处理 BOM、空值和表头访问，避免
    # 中英文列名、空单元格导致逐行解析分支失控。
    data = pd.read_csv(csv_path, encoding="utf-8")
    docs: list[Document] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    for _, row in data.iterrows():
        # 第二步：兼容中文和英文列名，并把空值统一成空字符串。
        # FAQ 只有问题和答案都存在时才构成可检索记录。
        question = str(row.get("问题") or row.get("question") or "").strip()
        answer = str(row.get("答案") or row.get("answer") or "").strip()
        subject = _resolve_csv_source(dict(row))
        if not question or not answer:
            continue

        # 第三步：把 CSV 中的分类字段归一化到当前场景的 source 白名单。
        # 归一化结果用于检索过滤和 FAQ ID，原始 subject 仍保留用于审计。
        source = normalize_faq_source(subject, scenario=scenario, question=question)
        # 第四步：用业务稳定字段生成 ID。版本号纳入 ID，确保不同快照的
        # FAQ 不会因为问题文本相同而互相覆盖。
        faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question)
        if faq_id in seen_ids:
            # 同一版本、同一 source 下问题相同但答案不同，加入答案区分记录。
            faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question, answer)
        if faq_id in seen_ids:
            # 加入答案后仍然冲突，说明是完全重复行，跳过以保持快照幂等。
            continue
        seen_ids.add(faq_id)
        docs.append(
            Document(
                # page_content 只放标准问题，向量检索围绕用户问题匹配；
                # answer 放入 metadata，召回后作为结构化答案上下文使用。
                page_content=question,
                metadata={
                    "faq_id": faq_id,
                    "scenario_id": scenario.scenario_id,
                    # FAQ 采用按版本快照重建模式；这里保留公共版本字段，
                    # 但在线过滤以 kb_version_exact 为准，不走文档 validity window。
                    "source_type": "faq",
                    "record_type": "faq",
                    "versioning_mode": "snapshot",
                    "version_filter_mode": "kb_version_exact",
                    **data_scope.metadata(allowed_roles=allowed_roles),
                    "standard_question": question,
                    "answer": answer,
                    "source": source,
                    "subject_name": subject,
                    "status": "published",
                    **version_meta,
                },
            )
        )
        ids.append(faq_id)
    return docs, ids


def ingest_faq_csv(
    csv_path: str,
    *,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    kb_version: str | None = None,
    create_new_version: bool = False,
    description: str = "",
    scenario: Any | None = None,
    data_scope: Any | None = None,
    target_version: Any | None = None,
    version_store: Any | None = None,
) -> int:
    """从 CSV 重新构建 FAQ 记录并写入 Milvus FAQ 混合集合。（★★★ 核心）

    完整入库流程：
    1. 解析或复用场景配置，获取或创建知识库版本记录。
    2. 调用 faq_documents_from_csv 将 CSV 转换为 Document 列表。
    3. 先删后写（delete_ids + add_documents）：FAQ 采用整体替换策略，
       确保 FAQ ID 包含 kb_version，新旧版本不混合。
    4. 记录入库统计到版本控制面。

    参数：
        csv_path: FAQ CSV 文件路径，支持中英文列名。
        scenario_id: 业务场景标识（可选）。
        tenant_id: 租户 ID（可选）。
        dataset_id: 数据集 ID（可选）。
        visibility: 可见级别（可选）。
        allowed_roles: 允许检索的角色列表（可选）。
        kb_version: 知识库版本号（可选，不传时使用 active 版本或自动生成）。
        create_new_version: 是否强制创建新版本，默认 False。
        description: 版本描述（创建新版本时使用）。
        scenario/data_scope/target_version/version_store：
            可由上游编排器传入的已解析上下文。传入后直接复用，
            独立调用时可以省略。

    返回：
        成功写入的 FAQ 记录数。

    调用顺序：入库脚本或索引服务 -> ingest_faq_csv() -> faq_documents_from_csv()。
    """
    scenario, data_scope = _resolve_faq_context(
        scenario_id,
        tenant_id,
        dataset_id,
        visibility,
        allowed_roles,
        scenario=scenario,
        data_scope=data_scope,
    )
    version_store = (
        version_store
        if version_store is not None
        else get_kb_version_store(scenario.scenario_id)
    )
    # 第一步：确保目标版本已存在。总流程传入 target_version 时直接复用，
    # 独立调用时才由 ensure_version 根据参数选择已有版本或创建新版本。
    version = (
        target_version
        if target_version is not None
        else version_store.ensure_version(
            kb_version,
            create_new=create_new_version,
            description=description,
            created_by="ingest_faq_csv",
        )
    )
    active_kb_version = version.kb_version
    # 第二步：只做 CSV -> Document 的转换，不在转换函数里操作 Milvus。
    # 这样解析失败不会留下半写入状态，也能让测试独立验证快照内容。
    docs, ids = faq_documents_from_csv(
        csv_path,
        active_kb_version,
        scenario_id=scenario.scenario_id,
        version_seq=version.version_seq,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        allowed_roles=allowed_roles,
        scenario=scenario,
        data_scope=data_scope,
    )
    store = get_faq_store(scenario.faq_collection)
    # 第三步：FAQ 使用快照式先删后写。删除的是本次生成出来的同一组 IDs；
    # ID 包含版本号，所以不会误删其他版本，但可以让目标版本重复执行幂等。
    store.delete_ids(ids)
    # 第四步：写入完整快照。这里不调用普通文档的 manifest/validity-window
    # 逻辑，因为 FAQ 不做引用式增量。
    store.add_documents(docs, ids=ids)
    # 第五步：记录版本统计。FAQ 入库统计只代表本次快照写入量，版本激活仍由
    # 外层质量门禁决定。
    version_store.record_ingest_result(active_kb_version, content_type="faq", count=len(docs))
    logger.info("Ingested %s FAQ records from %s, kb_version: %s", len(docs), csv_path, active_kb_version)
    return len(docs)

