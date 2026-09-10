"""入库清单与 Milvus 旧 chunk 清理。处理本地资料文件被删除后 Milvus 中残留 chunk 的清除。仅用于离线清理。"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from qa_core.common import utc_now, write_json
from qa_core.config.settings import PROJECT_ROOT
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.indexing.manifest import IndexManifest, ManifestRecord
from qa_core.retrieval.factory import get_doc_store
from qa_core.scenarios.registry import resolve_scenario

def cleanup_missing_document_chunks(
    *,
    scenario_id: str | None = None,
    source: str | None = None,
    kb_version: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """清理 manifest 中已不存在本地文件的文档 chunk。（★★ 理解）

    dry_run=True 时只预览不执行删除，防止误删。

    参数：
        scenario_id: 场景 ID，用于过滤 manifest 记录。
        source: 业务分类 source，用于过滤 manifest 记录。
        kb_version: 知识库版本。未指定时使用当前 active 版本。
        dry_run: 是否仅预览。默认为 True，设置为 False 时才实际执行删除。

    返回：
        包含检查数量、缺失数量、删除数量和文件列表的字典。

    调用顺序：入库脚本或索引服务 -> cleanup_missing_document_chunks()。
    """
    # 原因： dry_run 默认防止误删——文件"消失"可能是因为临时路径变更或权限抖动而非用户意图删除，必须先预览确认后再确认执行
    scenario = resolve_scenario(scenario_id)
    # 未指定 kb_version 时使用当前 active 版本，因为清理通常针对在线使用的版本进行
    effective_version = kb_version or resolve_active_kb_version(None, scenario.scenario_id)
    manifest = IndexManifest()
    records = manifest.iter_records(scenario_id=scenario.scenario_id, source=source, kb_version=effective_version)
    # 分类：manifest 中有记录但文件已不存在的为 missing（需清理），
    # 文件仍在的是 existing（无需清理），用于差异报告
    # ── 分类：manifest 中有记录但文件已不存在的为 missing（需清理） ──
    # 文件仍在的是 existing（无需清理），用于差异报告
    missing: list[ManifestRecord] = [record for record in records if record.path and not Path(record.path).exists()]
    existing: list[ManifestRecord] = [record for record in records if record.path and Path(record.path).exists()]
    source_counts: dict[str, int] = {}
    missing_source_counts: dict[str, int] = {}
    for record in records:
        source_counts[record.source] = source_counts.get(record.source, 0) + 1
    for record in missing:
        missing_source_counts[record.source] = missing_source_counts.get(record.source, 0) + 1
    deleted_chunk_count = 0
    deleted_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []

    # ── 非 dry_run 模式：实际执行删除 ──
    if not dry_run and missing:
        doc_store = get_doc_store(scenario.doc_collection)
        for record in missing:
            ok = doc_store.delete_ids(record.chunk_ids)
            if ok:
                # Milvus 删除成功后同步移除 manifest 记录，防止下次重复清理同一批 chunk id
                manifest.remove_by_key(record.key)
                deleted_chunk_count += len(record.chunk_ids)
                deleted_records.append(_record_payload(record))
            else:
                # 删除失败时保留记录并报告，等待人工排查（可能 Milvus 连接异常或 collection 不存在）
                failed_records.append(_record_payload(record))

    return {
        "scenario_id": scenario.scenario_id,
        "kb_version": effective_version,
        "source": source,
        "dry_run": dry_run,
        "records_checked": len(records),
        "existing_file_count": len(existing),
        "missing_file_count": len(missing),
        "source_record_counts": source_counts,
        "missing_source_counts": missing_source_counts,
        "affected_chunk_count": sum(len(record.chunk_ids) for record in missing),
        "deleted_chunk_count": deleted_chunk_count,
        "missing_files": [_record_payload(record) for record in missing],
        "deleted_records": deleted_records,
        "failed_records": failed_records,
        "recommendation": _cleanup_recommendation(missing, failed_records, dry_run),
    }


def _record_payload(record: ManifestRecord) -> dict[str, Any]:
    """把 ManifestRecord 转成脚本和 API 友好的 JSON 结构。

    调用顺序：入库脚本或索引服务 -> _record_payload()。
    """
    return {
        "key": record.key,
        "scenario_id": record.scenario_id,
        "source": record.source,
        "path": record.path,
        "kb_version": record.kb_version,
        "chunk_count": len(record.chunk_ids),
        "updated_at": record.updated_at,
    }


def _cleanup_recommendation(missing: list[ManifestRecord], failed_records: list[dict[str, Any]], dry_run: bool) -> str:
    """根据清理结果给出下一步建议。返回中文建议字符串。

    调用顺序：入库脚本或索引服务 -> _cleanup_recommendation()。
    """
    if failed_records:
        return "存在删除失败记录，请先检查 Milvus collection 和 chunk id 是否一致。"
    if missing and dry_run:
        return "发现本地已删除但 Milvus 仍保留的文档，请复核 missing_files 后使用 --apply 清理。"
    if missing:
        return "已清理缺失文件对应 chunk，建议随后运行主链路评测确认不会再召回旧资料。"
    return "未发现缺失文件对应的旧 chunk。"


def write_cleanup_report(payload: dict[str, Any], output_path: str | Path | None = None) -> Path:
    """把清理差异写成固定报告文件。默认输出到 reports/ingestion/。

    调用顺序：入库脚本或索引服务 -> write_cleanup_report()。
    """
    target = Path(output_path) if output_path else PROJECT_ROOT / "reports" / "ingestion" / "cleanup_missing_docs_latest.json"
    report = {
        "report_type": "missing_document_cleanup",
        "created_at": utc_now(),
        **payload,
    }
    write_json(target, report)
    return target
