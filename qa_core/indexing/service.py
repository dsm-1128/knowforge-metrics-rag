"""本地业务文档入库编排服务。

这个文件负责“离线入库链路”，不参与在线问答时的实时生成。它把一个业务场景目录下的
本地资料按固定流程写入 Milvus：解析场景配置 → 确认数据隔离范围 → 确认知识库版本 →
加载文件 → 标准化元数据 → 切分 chunk → 写入向量库 → 更新本地索引清单。

这里单独成一个 service 文件，是为了把入库流程和在线 QAService 分开：
- 在线问答只负责检索、重排、Prompt 和流式返回；
- 离线入库只负责资料治理、增量构建、版本记录和 Milvus 写入。
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.governance.chunk_versions import ChunkVersionIndex
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import get_document_loader_spec, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.indexing.manifest import IndexManifest
from qa_core.retrieval.factory import get_doc_store
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import file_fingerprint, normalize_source_from_path


logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentIngestContext:
    """单次目录入库时所有文件共享的上下文。

    这个对象把“目录级不变量”集中保存起来，避免每处理一个文件都重新解析
    场景、数据范围和知识库版本。目录中的所有文件必须使用同一个
    `kb_version`、`version_seq`、Milvus collection、manifest 和 chunk 版本索引，
    否则同一次入库可能出现文件之间版本不一致的问题。

    字段可以分成三组：
      1. 治理上下文：source、scenario、data_scope、kb_version、kb_version_seq。
      2. 写入依赖：doc_store、manifest、chunk_index。
      3. 增量策略：force、incremental_base_kb_version、
         incremental_base_version_seq。

    调用顺序：`rebuild_kb_version.main()` 或独立索引服务
    -> `ingest_directory()` -> `DocumentIngestContext`。
    """

    source: str
    kb_version: str
    kb_version_seq: int
    scenario: Any
    data_scope: Any
    allowed_roles: list[str] | None
    doc_store: Any
    manifest: IndexManifest
    chunk_index: ChunkVersionIndex
    force: bool
    incremental_base_kb_version: str | None = None
    incremental_base_version_seq: int = 0

    @property
    def scenario_id(self) -> str:
        """返回当前上下文的场景标识。

        调用顺序：入库脚本或索引服务 -> DocumentIngestContext.scenario_id()。
        """
        return self.scenario.scenario_id


@dataclass(frozen=True)
class FileIngestResult:
    """单个文件入库后的统计结果。

    一个文件只会落入以下几种结果之一：
      - `skipped=True`：目标版本中已经存在完全匹配的 manifest，不需要任何写入。
      - `reused_chunks > 0`：跨版本内容未变化，目标版本 manifest 引用了基线 chunk。
      - `reembedded_chunks > 0`：文件新增、内容变化或 schema 变化，需要重新解析和切分。

    `expired_chunks` 单独统计被目标版本收口的基线 chunk，因为失效动作不等于
    新写入动作；`total_chunks` 只表示目标版本当前拥有的有效 chunk 数。

    调用顺序：`_ingest_single_file()` -> `FileIngestResult`。
    """

    reembedded_chunks: int = 0
    skipped: bool = False
    reused_chunks: int = 0
    expired_chunks: int = 0

    @property
    def total_chunks(self) -> int:
        """返回该文件入库的 chunk 总数（重新写入 + 引用复用）。

        调用顺序：入库脚本或索引服务 -> FileIngestResult.total_chunks()。
        """
        return self.reembedded_chunks + self.reused_chunks


@dataclass
class DirectoryIngestStats:
    """目录入库统计，用于写入知识库版本 stats。

    这里的统计是“本次目录入库动作”的统计，不是 Milvus 中的物理行数：
      - `reembedded_chunks`：本次真正重新生成向量的 chunk；
      - `reused_chunks`：通过 manifest 跨版本引用的 chunk；
      - `expired_chunks`：因为文件变化或删除而关闭可见窗口的基线 chunk；
      - `skipped_files`：同一目标版本中完全没有变化、直接跳过的文件。

    调用顺序：`ingest_directory()` -> `DirectoryIngestStats`。
    """

    reembedded_chunks: int = 0
    reused_chunks: int = 0
    expired_chunks: int = 0
    skipped_files: int = 0

    @property
    def total_chunks(self) -> int:
        """返回该目录入库的 chunk 总数（重新写入 + 引用复用）。

        调用顺序：入库脚本或索引服务 -> DirectoryIngestStats.total_chunks()。
        """
        return self.reembedded_chunks + self.reused_chunks

    def add(self, result: FileIngestResult) -> None:
        """累加一个文件入库结果到目录统计中。

        参数：
            result: 单个文件的入库结果，含各类型 chunk 计数和跳过标记。

        调用顺序：入库脚本或索引服务 -> DirectoryIngestStats.add()。
        """
        if result.skipped:
            self.skipped_files += 1
        self.reembedded_chunks += result.reembedded_chunks
        self.reused_chunks += result.reused_chunks
        self.expired_chunks += result.expired_chunks

    def as_version_stats(self, incremental_base_kb_version: str | None) -> dict[str, int | str]:
        """将入库统计转换为版本记录用的 stats 字典。

        参数：
            incremental_base_kb_version: 增量基准版本号，为空时对应字段为空字符串。

        返回：
            含重新写入数、复用数、失效数、跳过文件数和基准版本的字典。

        调用顺序：入库脚本或索引服务 -> DirectoryIngestStats.as_version_stats()。
        """
        return {
            "last_doc_reembedded_count": self.reembedded_chunks,
            "last_doc_reused_count": self.reused_chunks,
            "last_doc_expired_count": self.expired_chunks,
            "last_doc_skipped_file_count": self.skipped_files,
            "last_doc_incremental_base_kb_version": incremental_base_kb_version or "",
        }


def _walk_files(root: Path):
    """递归产出目录中的文件路径。

    这里只负责遍历，不做文件类型判断，也不在这里读取内容。文件类型判断要交给
    loader registry，这样质量报告和正式入库可以共享同一套后缀注册规则。

    调用顺序：入库脚本或索引服务 -> _walk_files()。
    """
    for current_root, _, files in os.walk(root):
        for file_name in files:
            yield Path(current_root) / file_name


def _manifest_matches_current_settings(record, fingerprint: str, settings) -> bool:
    """判断 manifest 记录是否仍可用于当前文件、embedding 模型和 chunk schema。

    这里是严格等值匹配，不做“字段种类没少即可复用”的宽松兼容：
      1. fingerprint 一致，表示本地文件未变化。
      2. embedding_model_version 一致，表示旧向量仍处于同一向量空间。
      3. chunk_schema_version 一致，表示切分规则和 chunk metadata 契约未变化。

    `valid_from_seq / valid_to_seq` 属于 chunk metadata 契约。引入或调整这类会影响
    检索过滤语义的字段时，应提升 CHUNK_SCHEMA_VERSION，让旧 manifest 自动失配并重建。

    该判断是增量入库的第一道门：只有三项都相等，才允许跳过解析或复用旧向量。
    任意一项不匹配都必须回到“加载 -> 标准化 -> 切分 -> 写入”的重建路径，
    从而避免旧模型向量或旧切分结构混入当前版本。

    调用顺序：`_ingest_single_file()` -> `_manifest_matches_current_settings()`。
    """
    return bool(
        record
        and record.fingerprint == fingerprint
        and record.embedding_model_version == settings.embedding_model_version
        and record.chunk_schema_version == settings.chunk_schema_version
    )


def _record_manifest(
    context: DocumentIngestContext,
    path: Path,
    fingerprint: str,
    chunk_ids: list[str],
    settings,
) -> None:
    """把一次成功写入或跨版本复用结果写回 MySQL manifest。

    重建文件和复用基线文件都必须写目标版本 manifest：
      - 重建时记录新生成的 chunk ids；
      - 复用时记录基线 chunk ids 在目标版本中的引用关系。

    这样下一次入库才能以“目标版本 + 文件路径”为粒度判断是否可以跳过，
    同时删除文件时也能遍历基线 manifest 找到需要收口的 chunk。manifest 是
    增量决策索引，不是向量正文的替代品。

    调用顺序：
    `'_ingest_single_file()'` -> `'_record_manifest()'`，
    或 `'_rebuild_file_chunks()'` -> `'_record_manifest()'`。
    """
    context.manifest.update(
        context.source,
        path,
        fingerprint,
        chunk_ids,
        scenario_id=context.scenario_id,
        kb_version=context.kb_version,
        embedding_model_version=settings.embedding_model_version,
        chunk_schema_version=settings.chunk_schema_version,
    )


def _expire_base_record_for_target(context: DocumentIngestContext, base_record) -> int:
    """让基线 chunk 从目标版本开始失效（valid_to_seq 收口）。

    当文件在目标版本中已删除或内容变化时，基准版本的 chunk 不应继续在目标版本
    视图中可见。本函数通过设置 valid_to_seq = 目标版本序号，让这些 chunk 从目标
    版本开始被检索过滤掉（valid_from_seq <= active_seq < valid_to_seq 才可见）。

    注意：这里不删除 Milvus 行，只更新 `valid_to_seq`。旧版本仍可通过自己的
    valid_from_seq/valid_to_seq 窗口访问这些 chunk，保证版本回滚时数据不丢。

    参数：
        context: 单次目录入库的共享上下文，含目标版本序号和基准版本序号。
        base_record: 基准版本中待失效的 manifest 记录，含 chunk_ids。

    返回：
        失效的 chunk 数量。base_record 为空或 chunk_ids 为空时返回 0。

    执行流程：
      1. 调用 doc_store.expire_documents_for_version() 更新 Milvus 侧 valid_to_seq。
      2. 调用 chunk_index.expire_chunks() 同步更新 MySQL 侧 chunk 版本索引。
      3. 返回失效数量，用于 DirectoryIngestStats 统计。

    调用顺序：_ingest_single_file() / _expire_missing_base_records()
    -> _expire_base_record_for_target()。
    """
    if not base_record or not base_record.chunk_ids:
        return 0
    # 第一步先收口 Milvus 的可见窗口。物理数据保留，避免破坏旧版本检索和回滚。
    expired = context.doc_store.expire_documents_for_version(
        base_record.chunk_ids,
        valid_to_seq=context.kb_version_seq,
    )
    # 第二步同步 MySQL 索引。Milvus 负责向量检索，MySQL 负责版本治理，两边
    # 必须使用同一个目标 version_seq，否则后续审计和可见性判断会不一致。
    context.chunk_index.expire_chunks(
        base_record.chunk_ids,
        scenario_id=context.scenario_id,
        source=context.source,
        kb_version=base_record.kb_version,
        valid_from_seq=context.incremental_base_version_seq,
        valid_to_seq=context.kb_version_seq,
        file_path=base_record.path,
    )
    return expired


def _rebuild_file_chunks(
    context: DocumentIngestContext,
    path: Path,
    fingerprint: str,
    existing,
    settings,
    *,
    expired_chunks: int = 0,
) -> FileIngestResult:
    """重新加载、标准化、切分并写入一个已变化或新增的文件。

    这个函数只处理“确定需要重建”的文件，调用方已经完成 fingerprint 和
    manifest 判断。因此这里可以专注执行一条确定性的处理链：
      1. 删除目标版本自己原先写入的同文件 chunk；
      2. 使用 loader registry 解析原始文件；
      3. 补齐场景、来源、版本、数据范围和权限 metadata；
      4. 按统一 chunk schema 切分并生成稳定 chunk id；
      5. 写入 Milvus；
      6. 写入 MySQL chunk 版本索引和 manifest。

    这里不能先写 manifest 再写向量：如果解析或写入中途失败，manifest 会
    伪装成“文件已经完成”，下一次入库反而可能错误跳过该文件。

    调用顺序：`_ingest_single_file()` -> `_rebuild_file_chunks()`。
    """
    # 目标版本重跑时只删除目标版本自己写入的 chunk；跨版本复用的基线 chunk
    # 不属于目标版本的物理写入，不能被这里误删。
    if existing and existing.chunk_ids:
        if hasattr(context.doc_store, "delete_ids_for_kb_version"):
            context.doc_store.delete_ids_for_kb_version(existing.chunk_ids, context.kb_version)
        else:
            context.doc_store.delete_ids(existing.chunk_ids)
    # loader 返回的 Document 只代表“解析出了文本”。进入企业检索前还必须补齐
    # 标准 metadata，后续的权限隔离、版本过滤、来源追溯都依赖这些字段。
    docs = normalize_documents(
        load_file(path),
        path,
        context.source,
        kb_version=context.kb_version,
        scenario_id=context.scenario_id,
        version_seq=context.kb_version_seq,
        data_scope=context.data_scope,
        allowed_roles=context.allowed_roles,
        scenario=context.scenario,
    )
    # 切分器同时生成 chunk 文本和稳定 id；空文档不会产生可检索 chunk。
    chunks, ids = split_documents(docs)
    if not chunks:
        return FileIngestResult()
    # 先写向量正文，再登记治理索引。只有正文写成功后，manifest 才能代表一次
    # 完整入库，避免“索引记录存在但向量不存在”的半成品状态。
    context.doc_store.add_documents(chunks, ids=ids)
    # MySQL 侧记录每个 chunk 的 valid_from_seq，供版本可见性和审计使用。
    context.chunk_index.upsert_chunks(
        ids,
        scenario_id=context.scenario_id,
        source=context.source,
        kb_version=context.kb_version,
        valid_from_seq=context.kb_version_seq,
        file_path=str(path.resolve()),
    )
    # 最后写入文件级 manifest，供下一次同版本跳过和跨版本复用。
    _record_manifest(context, path, fingerprint, ids, settings)
    return FileIngestResult(reembedded_chunks=len(chunks), expired_chunks=expired_chunks)


def _ingest_single_file(path: Path, context: DocumentIngestContext) -> FileIngestResult:
    """处理单个文件的增量入库。

    这是目录入库中最重要的决策函数，按以下顺序判断：
      1. 后缀未注册：立即失败，让正式入库和质量报告遵守同一套 loader 契约；
      2. 目标版本 manifest 完全匹配：跳过，避免同一版本重复计算；
      3. 基线版本 manifest 完全匹配：只复制 manifest 引用，不复制向量行；
      4. 基线记录存在但不匹配：先关闭旧 chunk 的目标版本可见窗口，再重建；
      5. 没有可复用记录：按新增文件路径直接重建。

    `force=True` 会绕过同版本跳过和跨版本复用，但仍然沿用目标版本的写入、
    metadata 和版本索引逻辑。
    """
    if get_document_loader_spec(path) is None:
        raise ValueError(f"不支持的文档类型：{path}")
    fingerprint = file_fingerprint(path)
    settings = get_settings()
    existing = context.manifest.get(context.source, path, context.kb_version, context.scenario_id)
    # 第一分支：目标版本已经登记过同一文件，且处理配置没有变化。
    # 这是“同版本增量”的快速路径，不需要读文件、更不需要重新 embedding。
    if not context.force and _manifest_matches_current_settings(existing, fingerprint, settings):
        return FileIngestResult(skipped=True)

    base_record = None
    if context.incremental_base_kb_version and not context.force:
        # 第二步才查基线版本。目标版本优先，避免把目标版本已存在的状态
        # 错误地当成基线复用结果。
        base_record = context.manifest.get(
            context.source,
            path,
            context.incremental_base_kb_version,
            context.scenario_id,
        )
    if not existing and base_record and _manifest_matches_current_settings(base_record, fingerprint, settings):
        # 文件内容、embedding 版本和 chunk schema 都没变，可以让目标 manifest
        # 指向同一组基线 chunk，从而实现增量版本的“引用式复用”。
        _record_manifest(context, path, fingerprint, base_record.chunk_ids, settings)
        return FileIngestResult(reused_chunks=len(base_record.chunk_ids))

    expired_chunks = 0
    if base_record:
        # 只有文件确实变化或配置发生变化时，才关闭基线 chunk 在目标版本的
        # 可见窗口；旧版本窗口保持不变。
        expired_chunks = _expire_base_record_for_target(context, base_record)

    # 第三分支：新增、变化、force 或 schema 升级，回到完整重建链路。
    return _rebuild_file_chunks(context, path, fingerprint, existing, settings, expired_chunks=expired_chunks)


def _expire_missing_base_records(context: DocumentIngestContext, seen_paths: set[str]) -> int:
    """目标版本中已删除的文件不再复制，只让基线 chunk 从目标版本开始不可见。

    目录遍历只能看到当前磁盘存在的文件，因此必须在遍历结束后反向扫描基线
    manifest，找出“基线有、当前目录没有”的路径。这个补偿步骤是删除检测的
    关键，否则删除本地文件不会触发任何写入，也就不会关闭旧 chunk。

    调用顺序：入库脚本或索引服务 -> _expire_missing_base_records()。
    """
    if not context.incremental_base_kb_version or context.force:
        return 0
    expired_chunks = 0
    base_records = context.manifest.iter_records(
        scenario_id=context.scenario_id,
        source=context.source,
        kb_version=context.incremental_base_kb_version,
    )
    for record in base_records:
        # 使用绝对路径比较，避免相对路径、工作目录不同导致删除检测误判。
        if str(Path(record.path).resolve()) in seen_paths:
            continue
        expired_chunks += _expire_base_record_for_target(context, record)
    return expired_chunks


def _resolve_directory_source(root: Path, source: str | None, scenario: Any) -> str:
    """统一处理目录入库的 source 推断和白名单校验。

    解析顺序是“显式 source 优先，否则按目录名推断”，随后立即做场景白名单
    校验。入口脚本通常已经先做过场景路由，但这里保留兜底，避免直接调用
    `ingest_directory()` 时把资料写进错误分类。
    """
    resolved_source = source or normalize_source_from_path(root)
    if resolved_source not in scenario.valid_sources:
        raise ValueError(f"无效的业务分类：{resolved_source}，当前场景支持：{scenario.valid_sources}")
    return resolved_source


def _resolve_incremental_base_version_seq(version_store: Any, incremental_base_kb_version: str | None) -> int:
    """把增量基准版本转换成序号，供引用式复用和失效窗口使用。

    `kb_version` 是面向人的版本名称，`version_seq` 是治理层用于比较窗口的
    单调序号。目录处理只解析一次并放入上下文，后续所有 chunk 失效操作都使用
    同一个序号。没有基线时返回 0，表示当前不是跨版本增量构建。
    """
    if not incremental_base_kb_version:
        return 0
    base_version = version_store.get(incremental_base_kb_version)
    if base_version is None:
        raise ValueError(f"增量基准版本不存在：{incremental_base_kb_version}")
    return base_version.version_seq


def _build_directory_ingest_context(
    *,
    source: str,
    version: Any,
    scenario: Any,
    data_scope: Any,
    allowed_roles: list[str] | None,
    force: bool,
    incremental_base_kb_version: str | None,
    incremental_base_version_seq: int,
) -> DocumentIngestContext:
    """组装目录入库执行上下文，把和遍历无关的样板收拢到一起。

    这个函数不做磁盘遍历和数据库写入，只负责把已经确认的治理对象接入
    执行链。这样 `_ingest_single_file()` 可以只接收一个 context，避免在每个
    文件分支里重复获取 collection、manifest 和 chunk index。
    """
    return DocumentIngestContext(
        source=source,
        kb_version=version.kb_version,
        kb_version_seq=version.version_seq,
        scenario=scenario,
        data_scope=data_scope,
        allowed_roles=allowed_roles,
        doc_store=get_doc_store(scenario.doc_collection),
        manifest=IndexManifest(),
        chunk_index=ChunkVersionIndex(),
        force=force,
        incremental_base_kb_version=incremental_base_kb_version,
        incremental_base_version_seq=incremental_base_version_seq,
    )


def ingest_directory(
    directory_path: str,
    source: str | None = None,
    *,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    force: bool = False,
    kb_version: str | None = None,
    create_new_version: bool = False,
    description: str = "",
    incremental_base_kb_version: str | None = None,
    scenario: Any | None = None,
    data_scope: Any | None = None,
    target_version: Any | None = None,
    version_store: Any | None = None,
    incremental_base_version_seq: int | None = None,
) -> int:
    """把某个目录下的业务文档增量写入 Milvus。

    这是文档入库的主入口，通常由脚本调用，例如重建某个场景的知识库版本时会走这里。
    它不负责 FAQ CSV，FAQ 有单独的 `faq_ingestion.py`；这里专注处理普通业务文档、
    表格行、OCR 后的文本等“文档型资料”。

    主要职责：
      1. 解析或复用当前业务场景，拿到 doc_collection、valid_sources、版本清单路径等配置。
      2. 构建或复用 DataScope，把 tenant/dataset/visibility/roles 写入 metadata，支持隔离检索。
      3. 校验 source 必须属于当前场景的 valid_sources，防止跨场景数据写错集合。
      4. 确认或复用知识库版本，新旧版本可以并存，线上只检索 active 版本。
      5. 递归遍历目录，对每个文件调用 `_ingest_single_file()` 做增量判断和写入。
      6. 保存 manifest，让下次入库可以跳过未变化文件，并能删除旧 chunk。
      7. 记录本次入库统计；版本发布由 rebuild_kb_version.py 的质量门禁收口。

    参数说明：
      - directory_path：要入库的目录。
      - source：业务分类；不传时从目录名推断，例如 `finance_data` 推断为 `finance`。
      - scenario_id：目标业务场景，例如 `enterprise_knowledge`。
      - tenant_id/dataset_id/visibility/allowed_roles：数据隔离字段，会进入 Milvus metadata。
      - force：是否忽略 fingerprint，强制重建所有文件。
      - incremental_base_kb_version：跨版本增量构建的基准版本；未变化文件会引用旧 chunk，不复制 Milvus 行。
      - scenario/data_scope/target_version/version_store/incremental_base_version_seq：
        可由上游编排器传入的已解析上下文。传入后直接复用，独立调用时可以省略。

    返回：
      实际写入 Milvus 的 chunk 总数，不包含被增量跳过的文件。

    调用顺序：入库脚本或索引服务 -> ingest_directory()。
    """
    # 第 1 步：解析或复用上游已经准备好的场景和数据范围。命令行总流程会
    # 预先解析，这里的 fallback 兼容直接调用本函数的旧入口。
    scenario = scenario if scenario is not None else resolve_scenario(scenario_id)
    data_scope = (
        data_scope
        if data_scope is not None
        else resolve_data_scope(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            visibility=visibility,
            user_roles=allowed_roles,
        )
    )
    # 第 2 步：确定物理目录和业务 source；source 校验必须发生在任何写入前。
    root = Path(directory_path)
    resolved_source = _resolve_directory_source(root, source, scenario)
    # 第 3 步：确定版本存储和目标版本。上游传入 target_version 时复用同一
    # 版本对象，避免 FAQ、文档和质量报告各自创建出不同版本。
    version_store = (
        version_store
        if version_store is not None
        else get_kb_version_store(scenario.scenario_id)
    )
    version = (
        target_version
        if target_version is not None
        else version_store.ensure_version(
            kb_version,
            create_new=create_new_version,
            description=description,
            created_by="ingest_directory",
        )
    )
    active_kb_version = version.kb_version
    # 第 4 步：把可读版本号转换成 version_seq，只解析一次，供所有文件共享。
    if incremental_base_version_seq is None:
        incremental_base_version_seq = _resolve_incremental_base_version_seq(
            version_store,
            incremental_base_kb_version,
        )
    # 第 5 步：创建目录级上下文。之后每个文件只做“检查、复用或重建”，不再
    # 重复处理版本、数据范围和存储对象。
    context = _build_directory_ingest_context(
        source=resolved_source,
        version=version,
        scenario=scenario,
        data_scope=data_scope,
        allowed_roles=allowed_roles,
        force=force,
        incremental_base_kb_version=incremental_base_kb_version,
        incremental_base_version_seq=incremental_base_version_seq,
    )
    stats = DirectoryIngestStats()
    seen_paths: set[str] = set()
    # 第 6 步：遍历当前目录。每个文件的三分支决策都集中在
    # `_ingest_single_file()`，本层只负责汇总结果。
    for path in _walk_files(root):
        seen_paths.add(str(path.resolve()))
        stats.add(_ingest_single_file(path, context))
    # 第 7 步：遍历结束后补做删除检测，收口“基线存在但当前目录已删除”的文件。
    stats.expired_chunks += _expire_missing_base_records(context, seen_paths)
    # 第 8 步：把统计写回版本治理表。这里只记录入库事实，不在文档服务中
    # 激活版本；激活必须由上层质量门禁和发布流程统一决定。
    version_store.record_ingest_result(
        active_kb_version,
        content_type="doc",
        count=stats.total_chunks,
        source=resolved_source,
        extra_stats=stats.as_version_stats(incremental_base_kb_version),
    )
    logger.info(
        "文档入库完成：目标版本 chunk=%s，重新写入=%s，引用复用=%s，失效旧chunk=%s，目录=%s，跳过未变化文件=%s，kb_version=%s",
        stats.total_chunks,
        stats.reembedded_chunks,
        stats.reused_chunks,
        stats.expired_chunks,
        directory_path,
        stats.skipped_files,
        active_kb_version,
    )
    return stats.total_chunks



