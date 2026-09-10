"""知识库入库质量报告编排层——只解析和切分，不写 Milvus。

在激活 kb_version 之前，对候选目录中的文件执行完整的质量检测流程：
  1. 三道路径：文件后缀注册检查 -> source 白名单检查 -> 解析和切分。
  2. 聚合所有维度的检测结果：chunk 质量、FAQ 质量、FAQ-正文冲突、图片风险。
  3. 生成结构化报告和可读的中文警告列表，辅助运维决策。

设计决策：
- 所有检测结果基于纯规则，不调用 LLM，保证低成本和高稳定性。
- 单个文件解析失败不影响整体报告生成——异常被捕获并记录到 failed_files 列表。
- 质量报告存储在 reports/ingestion/<scenario_id>/ 目录，按时间戳和 kb_version 组织。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from qa_core.common import list_json_reports, utc_file_stamp, utc_now, write_json
from qa_core.config.logging_config import get_logger
from qa_core.config.settings import PROJECT_ROOT, get_settings
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import get_document_loader_spec, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.indexing.image_risk import analyze_image_risk
from qa_core.indexing.table_documents import is_table_file, looks_like_ocr_risk, looks_like_table_text
from qa_core.quality.chunk import analyze_chunk_quality
from qa_core.quality.conflicts import detect_faq_document_conflicts
from qa_core.quality.faq import analyze_faq_csv
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import normalize_source_from_path


logger = get_logger(__name__)
INGESTION_REPORT_DIR = PROJECT_ROOT / "reports" / "ingestion"


def _iter_candidate_files(root: Path) -> list[Path]:
    """列出入库候选目录下的全部文件，按路径排序保证报告稳定。

    质量报告要先看到“可能被入库的全部文件”，再逐个判定是否支持、
    是否属于当前场景 source。这里不提前过滤后缀，是为了让未支持文件也能
    出现在报告里，提示是否需要补 loader 或调整目录。

    调用顺序：质量门禁流程 -> _iter_candidate_files()。
    """
    # 目录不存在时返回空列表而非抛异常，便于在配置缺失时仍能生成部分报告
    if not root.exists():
        return []
    # 递归遍历所有子目录，只保留文件不保留目录条目；排序确保多次运行结果一致
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def _process_candidate_file(
    path: Path,
    source: str,
    scenario: Any,
    active_kb_version: str,
    scope: Any,
    allowed_roles: list[str] | None,
) -> dict[str, Any]:
    """解析单个候选文件并返回质量报告需要的中间结果。

    该函数只做“试解析 + 试切分”，不会写入 Milvus。返回值中包含：
    - ok：文件是否成功解析。
    - raw_docs：Loader 原始文档结果。
    - normalized_chunks：标准化并切分后的 chunk。
    - is_table / is_ocr_risk：是否疑似表格资料或 OCR 风险资料。
    - image_risk：图片、扫描页或图文混排资料的可见风险。
    - error：解析失败时的错误信息。

    正式入库和质量报告共用 loader、normalizer、splitter，这样质量报告看到的
    chunk 形态和真正入库的 chunk 形态保持一致；区别只是这里不调用
    `doc_store.add_documents()`、`ChunkVersionIndex.upsert_chunks()` 和 manifest 写入。

    调用顺序：质量门禁流程 -> _process_candidate_file()。
    """
    try:
        # 第一步：使用正式入库同一套 loader registry 加载原始文档，可能返回多页 Document。
        raw_docs = load_file(path)
        # loader 返回空列表说明文件被无声跳过（如空白页或空文件），标记为空文件不继续处理
        if not raw_docs:
            image_risk = analyze_image_risk(path, "")
            return {
                "ok": True,
                "raw_docs": [],
                "is_table": False,
                "is_ocr_risk": False,
                "image_risk": image_risk.as_dict() if image_risk else None,
                "normalized_chunks": [],
                "error": None,
                "path": str(path),
                "source": source,
            }
        # 合并在内存中做一次全文文本拼接，用于后续的表格/OCR 风险判断
        combined_text = "\n".join(doc.page_content or "" for doc in raw_docs)
        # 表格文件检测：先看后缀，再看内容特征（如含大量空格分隔的字段行）
        is_table = is_table_file(path) or looks_like_table_text(path, combined_text)
        # OCR 风险文件检测：扫描件或图片型 PDF 可能文本量极少，需要人工复核而非自动清洗
        is_ocr_risk = looks_like_ocr_risk(path, combined_text)
        # 图片风险检测：把嵌入图片、扫描页或独立图片的处理边界显式写入质量报告
        image_risk = analyze_image_risk(path, combined_text)
        # 标准化：添加 source、kb_version、data_scope 等 metadata。质量报告
        # 不写 Milvus，所以这里不需要分配目标 version_seq。
        normalized = normalize_documents(
            raw_docs,
            path,
            source,
            kb_version=active_kb_version,
            scenario_id=scenario.scenario_id,
            data_scope=scope,
            allowed_roles=allowed_roles,
            scenario=scenario,
        )
        # 切分：将标准化后的 Document 切分成适合检索的 chunk
        chunks, _ = split_documents(normalized)
        return {
            "ok": True,
            "raw_docs": raw_docs,
            "is_table": is_table,
            "is_ocr_risk": is_ocr_risk,
            "image_risk": image_risk.as_dict() if image_risk else None,
            "normalized_chunks": chunks,
            "error": None,
            "path": str(path),
            "source": source,
            "combined_text": combined_text,
        }
    except Exception as exc:
        # 捕获所有解析/切分阶段的异常，确保单个文件失败不影响整个质量报告的继续生成
        logger.warning("构建入库质量报告时文件解析失败：%s，错误：%s", path, exc)
        return {
            "ok": False,
            "raw_docs": None,
            "is_table": False,
            "is_ocr_risk": False,
            "image_risk": None,
            "normalized_chunks": [],
            "error": str(exc),
            "path": str(path),
            "source": source,
        }


def build_ingestion_quality_report(
    *,
    scenario_id: str | None = None,
    data_dir: str | None = None,
    faq_csv: str | None = None,
    kb_version: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    scenario: Any | None = None,
    data_scope: Any | None = None,
) -> dict[str, Any]:
    """构建一次完整的入库质量报告（只解析和切分，不写 Milvus）。（★★★ 核心）

    这是质量门禁的“眼睛”：在激活 kb_version 之前，对候选目录中的文件和 FAQ CSV
    执行完整的解析 + 切分试运行，把所有维度的潜在质量问题聚合成结构化报告，
    供 `rebuild_kb_version.py` 的质量门禁和人工运维决策使用。本函数不写 Milvus，
    因此可以在不污染线上检索数据的前提下安全地重复运行。

    执行流程：
      1. 解析或复用业务场景并构建 DataScope，把 tenant/dataset/visibility/roles 写入 metadata。
      2. 解析 kb_version：未传时回退到当前 active 版本，作为报告归属版本。
      3. 列举入库候选目录下所有文件（递归、按路径排序保证报告稳定）。
      4. 每个文件经过三道筛选门：
           - 第一道门：文件后缀未注册 LangChain loader → 记录到 unsupported_files；
           - 第二道门：文件目录不在场景 source 白名单内 → 记录到 unsupported_files；
           - 第三道门：调用 _process_candidate_file() 实际解析和切分，结果按
             解析失败 / 空文件 / 表格文件 / OCR 风险 / 图片风险 / 成功 分桶收集。
      5. 调用 analyze_chunk_quality() 聚合所有成功 chunk 的质量问题。
      6. 调用 analyze_faq_csv() 检测 FAQ CSV 质量（空问题、空答案、重复问题等）。
      7. 调用 detect_faq_document_conflicts() 检测 FAQ 标准答案与正文资料冲突。
      8. 汇总所有维度生成可读的中文 warnings 列表，辅助运维决策是否激活版本。

    参数：
        scenario_id: 业务场景 ID；为空时使用 ACTIVE_SCENARIO_ID。
        data_dir: 文档根目录；为空时使用场景配置 data_root。
        faq_csv: FAQ CSV 路径；为空时使用场景配置 faq_csv_path。
        kb_version: 报告归属的知识库版本号；为空时回退到当前 active 版本。
        tenant_id: 租户 ID（可选），会写入报告 data_scope 字段供追溯。
        dataset_id: 数据集 ID（可选），同上。
        visibility: 可见级别（可选），同上。
        allowed_roles: 允许检索的角色列表（可选），同上。
        scenario/data_scope: 可由上游编排器传入的已解析上下文。传入后直接复用，
            独立调用时可以省略。

    返回：
        dict[str, Any]，包含以下结构化字段：
        - 元数据：scenario_id / scenario_name / kb_version / data_scope /
          embedding_model_version / reranker_model_version / chunk_schema_version /
          data_root / faq_csv_path / started_at / finished_at。
        - 计数摘要：files_scanned / files_loaded_count / unsupported_files_count /
          failed_files_count / empty_files_count / table_files_count /
          ocr_risk_files_count / image_risk_files_count / image_risk_blocking_files_count。
        - chunk 维度：source_chunk_counts / chunk_quality / low_quality_chunks（最多 200 条）。
        - FAQ 维度：faq_quality / faq_document_conflicts。
        - 明细列表：files_loaded / unsupported_files / failed_files / empty_files /
          table_files / ocr_risk_files / image_risk_files。
        - 中文告警：warnings，按严重度排序的可读列表，辅助人工决策。

    设计决策：
    - 不调用 LLM：所有检测结果基于纯规则，保证低成本、高稳定、可重复。
    - 单文件失败不阻断：解析失败的异常被捕获并记入 failed_files，整体报告照常生成。
    - 不写 Milvus：只在内存中解析和切分，可作为激活前的一次“试运行”安全调用。
    - 时间戳双字段：started_at / finished_at 同时记录，便于排查耗时异常的文件。

    调用顺序：质量门禁流程 -> build_ingestion_quality_report()。
    """
    # 第一步：解析或复用场景和数据范围。离线总流程会把已经确认过的对象传进来，
    # 独立运行质量报告时才走这里的 fallback。
    scenario = scenario if scenario is not None else resolve_scenario(scenario_id)
    scope = (
        data_scope
        if data_scope is not None
        else resolve_data_scope(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            visibility=visibility,
            user_roles=allowed_roles,
        )
    )
    # 第二步：确定报告归属版本。入库脚本会传入本次候选版本；独立运行时
    # 回退到当前 active 版本，方便对线上版本做巡检。
    active_kb_version = kb_version or resolve_active_kb_version(None, scenario.scenario_id)
    settings = get_settings()
    root = Path(data_dir or scenario.data_root)
    faq_path = Path(faq_csv or scenario.faq_csv_path)
    started_at = utc_now()

    all_chunks: list[Document] = []
    source_counts: Counter[str] = Counter()
    files_loaded: list[dict[str, Any]] = []
    unsupported_files: list[dict[str, Any]] = []
    failed_files: list[dict[str, Any]] = []
    empty_files: list[dict[str, Any]] = []
    table_files: list[dict[str, Any]] = []
    ocr_risk_files: list[dict[str, Any]] = []
    image_risk_files: list[dict[str, Any]] = []
    # 第三步：先列出全部候选文件，后续再按 loader、source 和解析结果分桶。
    candidate_files = _iter_candidate_files(root)

    for path in candidate_files:
        # 每个文件经过三道筛选门：loader 注册检查 -> source 白名单检查 -> 解析切分。
        # 先检查文件后缀有没有注册 LangChain loader。
        spec = get_document_loader_spec(path)
        source = normalize_source_from_path(path.parent)
        # 第一道门：文件后缀未注册 loader。正式入库会失败，质量报告提前把它
        # 记录下来，便于在发布前修复。
        if spec is None:
            # 即使 loader 不支持，也尝试记录图片风险；有些图片型资料本来就
            # 需要独立 OCR 或人工处理，而不是简单归为“未注册格式”。
            image_risk = analyze_image_risk(path, "")
            if image_risk:
                payload = image_risk.as_dict()
                payload["source"] = source
                image_risk_files.append(payload)
            # 记录到 unsupported_files，包括路径、后缀和拒绝原因。
            unsupported_files.append({"path": str(path), "suffix": path.suffix.lower(), "reason": "未注册 LangChain loader。"})
            continue
        # 第二道门：文件目录不在当前场景的 source 白名单内。source 来自目录名
        # 归一化结果，白名单由场景配置维护。
        if source not in scenario.valid_sources:
            unsupported_files.append({"path": str(path), "source": source, "reason": "文件目录无法映射到当前场景 source 白名单。"})
            continue
        # 第三道门：实际解析和切分。这里加载文件、标准化 metadata、切分 chunk，
        # 但只在内存中试运行，不写入 Milvus。
        result = _process_candidate_file(path, source, scenario, active_kb_version, scope, allowed_roles)
        # 三道门完成后按质量维度分桶。每个桶对应后续一个门禁阈值或告警维度，
        # 例如 failed_files 对应 max_failed_files，image_risk_files 对应图片风险阈值。
        if result.get("image_risk"):
            payload = dict(result["image_risk"])
            payload["source"] = result["source"]
            image_risk_files.append(payload)
        if result["error"]:
            failed_files.append({"path": result["path"], "error": result["error"]})
            continue
        if not result["raw_docs"]:
            empty_files.append({"path": result["path"], "reason": "loader 未返回任何 Document。"})
            continue
        if result["is_table"]:
            table_files.append({
                "path": result["path"],
                "suffix": path.suffix.lower(),
                "source": result["source"],
                "raw_documents": len(result["raw_docs"]),
                "reason": "表格型资料已按表头、行号和单元格键值保留结构化语义。",
            })
        if result["is_ocr_risk"]:
            ocr_risk_files.append({
                "path": result["path"],
                "suffix": path.suffix.lower(),
                "source": result["source"],
                "reason": "疑似扫描件或 OCR 噪声，默认不执行复杂 OCR，应先人工复核或走独立 OCR 清洗流程。",
                "preview": result["combined_text"][:160],
            })
        # 收集所有成功切分的 chunk，用于后续的质量聚合分析
        all_chunks.extend(result["normalized_chunks"])
        source_counts[result["source"]] += len(result["normalized_chunks"])
        files_loaded.append({
            "path": result["path"],
            "source": result["source"],
            "raw_documents": len(result["raw_docs"]),
            "chunks": len(result["normalized_chunks"]),
        })

    # 第四步：分析文档 chunk 质量，包括过短、重复、噪声或结构异常等规则问题。
    chunk_issues, chunk_stats = analyze_chunk_quality(all_chunks)
    # 第五步：分析 FAQ CSV 本身的质量，例如空问题、空答案、重复问题和非法 source。
    faq_quality = analyze_faq_csv(faq_path, scenario.valid_sources)
    # 第六步：检测 FAQ 标准答案和正文资料的潜在冲突：
    # 1. 用 FAQ 问题和答案抽关键词，命中同 source 正文后进入候选；
    # 2. 对比数字集合、肯定/否定倾向等事实口径，找出需要人工复核的冲突。
    faq_document_conflicts = detect_faq_document_conflicts(faq_path, all_chunks)
    # 第七步：汇总所有维度的问题，生成可读中文告警。真正是否阻断激活由
    # `check_ingestion_quality_gate.py` 根据阈值判断；这里负责提供事实和说明。
    warnings: list[str] = []
    if unsupported_files:
        warnings.append("存在未支持或未纳入白名单的文件，需要确认是否应扩展 loader 或调整目录。")
    if failed_files:
        warnings.append("存在解析失败文件，需要单独修复后重新入库。")
    if chunk_issues:
        warnings.append("存在低质量 chunk，需要检查切分策略、文档格式或 OCR 质量。")
    if faq_quality.get("duplicate_questions"):
        warnings.append("FAQ 存在重复标准问题，高置信直出可能产生口径冲突。")
    if faq_quality.get("invalid_sources"):
        warnings.append("FAQ 存在不在当前场景白名单内的 source。")
    if faq_document_conflicts.get("conflict_count"):
        warnings.append("FAQ 标准答案与正文资料存在潜在冲突，需要人工复核后再激活知识库版本。")
    if table_files:
        warnings.append("检测到表格型资料，已按行列语义入库或标记，适合清单、金额、状态和字段类问题检索。")
    if ocr_risk_files:
        warnings.append("检测到疑似 OCR/扫描件风险文件，默认不执行复杂 OCR，需要人工复核后再进入 active 知识库。")
    image_blocking_count = sum(1 for item in image_risk_files if item.get("severity") == "block")
    if image_blocking_count:
        warnings.append("检测到必须 OCR/人工复核的图片资料，禁止直接激活为 active 知识库版本。")
    elif image_risk_files:
        warnings.append("检测到图文混排资料；当前只保证文本层入库，图片承载业务信息时需要单独 OCR/复核。")

    return {
        "report_type": "ingestion_quality",
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "kb_version": active_kb_version,
        "data_scope": scope.as_dict(),
        "embedding_model_version": settings.embedding_model_version,
        "reranker_model_version": settings.reranker_model_version,
        "chunk_schema_version": settings.chunk_schema_version,
        "data_root": str(root),
        "faq_csv_path": str(faq_path),
        "started_at": started_at,
        "finished_at": utc_now(),
        "files_scanned": len(candidate_files),
        "files_loaded_count": len(files_loaded),
        "unsupported_files_count": len(unsupported_files),
        "failed_files_count": len(failed_files),
        "empty_files_count": len(empty_files),
        "table_files_count": len(table_files),
        "ocr_risk_files_count": len(ocr_risk_files),
        "image_risk_files_count": len(image_risk_files),
        "image_risk_blocking_files_count": image_blocking_count,
        "source_chunk_counts": dict(source_counts),
        "chunk_quality": chunk_stats,
        "low_quality_chunks": chunk_issues[:200],
        "faq_quality": faq_quality,
        "faq_document_conflicts": faq_document_conflicts,
        "files_loaded": files_loaded,
        "unsupported_files": unsupported_files,
        "failed_files": failed_files,
        "empty_files": empty_files,
        "table_files": table_files,
        "ocr_risk_files": ocr_risk_files,
        "image_risk_files": image_risk_files,
        "warnings": warnings,
    }


def save_ingestion_quality_report(report: dict[str, Any], output: str | None = None) -> str:
    """保存入库质量报告为 JSON 文件并返回路径。（★★ 理解）

    报告默认按“场景 + 时间戳 + kb_version”组织，方便按场景回溯某次入库前后版本
    的质量快照；当调用方显式传入 output 时则按指定路径写入，便于在 CI/CD 等场景
    中把报告落到固定路径。

    参数：
        report: 由 build_ingestion_quality_report() 生成的完整报告字典。
        output: 显式输出路径；为空时自动写入
            reports/ingestion/<scenario_id>/<时间戳>_<kb_version>.json。
            kb_version 中的冒号会被替换为下划线，避免 Windows 文件名非法字符。

    返回：
        实际写入的 JSON 文件绝对路径字符串，供调用方日志输出或下游门禁引用。

    设计决策：
    - 自动按场景分目录：reports/ingestion/<scenario_id>/ 让多场景项目的报告彼此隔离。
    - 时间戳前缀 + kb_version 后缀：既能按时间排序，又能通过 kb_version 反查入库版本。
    - 冒号转下划线：Windows 文件名不允许冒号，kb_version 形如 kb_xxx_2026..._abcd 中
      不含冒号，但保留这个安全转换以防未来 kb_version 格式变化。

    调用顺序：质量门禁流程 -> save_ingestion_quality_report()。
    """
    scenario_id = str(report.get("scenario_id") or "default")
    kb_version = str(report.get("kb_version") or "preview").replace(":", "_")
    if output:
        # 调用方显式指定输出路径时直接使用，不加入自动目录结构
        path = Path(output)
    else:
        # 默认存储到 reports/ingestion/<scenario_id>/<时间戳>_<kb_version>.json，按场景和时间组织便于回溯
        stamp = utc_file_stamp()
        path = INGESTION_REPORT_DIR / scenario_id / f"{stamp}_{kb_version}.json"
    return write_json(path, report)


def list_ingestion_reports(*, scenario_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """列出最近的入库质量报告。

    调用顺序：质量门禁流程 -> list_ingestion_reports()。
    """
    root = INGESTION_REPORT_DIR
    # 如果指定了 scenario_id，则先获取所有报告再在代码中过滤；否则按 limit 限制读取数量
    fetch_limit = 0 if scenario_id else limit
    entries = list_json_reports(root, "**/*.json", limit=fetch_limit)
    reports: list[dict[str, Any]] = []
    for entry in entries:
        payload = entry["payload"]
        # scenario_id 过滤：只在显式指定时才用，空 scenario_id 场景下不过滤
        if scenario_id and payload.get("scenario_id") != scenario_id:
            continue
        # 只提取报告摘要（不包含完整的低质量 chunk 列表和文件明细），减少列表接口的响应体大小
        reports.append(
            {
                "path": entry["path"],
                "file_name": entry["file_name"],
                "scenario_id": payload.get("scenario_id"),
                "kb_version": payload.get("kb_version"),
                "updated_at": entry["updated_at"],
                "summary": {
                    "files_scanned": payload.get("files_scanned"),
                    "files_loaded_count": payload.get("files_loaded_count"),
                    "unsupported_files_count": payload.get("unsupported_files_count"),
                    "failed_files_count": payload.get("failed_files_count"),
                    "table_files_count": payload.get("table_files_count"),
                    "ocr_risk_files_count": payload.get("ocr_risk_files_count"),
                    "image_risk_files_count": payload.get("image_risk_files_count"),
                    "image_risk_blocking_files_count": payload.get("image_risk_blocking_files_count"),
                    "table_files": payload.get("table_files", [])[:5],
                    "ocr_risk_files": payload.get("ocr_risk_files", [])[:5],
                    "image_risk_files": payload.get("image_risk_files", [])[:5],
                    "chunk_quality": payload.get("chunk_quality"),
                    "faq_quality": payload.get("faq_quality"),
                    "faq_document_conflicts": payload.get("faq_document_conflicts"),
                    "warnings": payload.get("warnings", []),
                },
            }
        )
        if len(reports) >= limit:
            break
    return reports


