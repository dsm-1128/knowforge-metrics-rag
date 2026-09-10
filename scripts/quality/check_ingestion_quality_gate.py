"""入库质量门禁检查。

质量报告负责“看见问题”，质量门禁负责“阻止问题进入线上版本”。本脚本可以读取已经
生成的入库质量报告，也可以现场生成一份报告后立即判断是否允许继续激活知识库版本。

它检查的不是模型回答效果，而是入库前后最容易造成 RAG 失真的基础问题：
- 文件解析失败；
- 不支持或未纳入场景白名单的文件；
- 空文件、空 FAQ、重复 FAQ；
- 低质量 chunk；
- FAQ 标准答案和正文资料潜在冲突；
- 图片、扫描页、图文资料是否绕过 OCR 复核；
- 知识库版本号、embedding 版本、chunk schema 版本是否记录完整。
"""

from __future__ import annotations

# argparse: 命令行参数解析
import argparse

# sys: 系统功能（sys.path + sys.exit）
import sys

# dataclasses: 数据类定义（IngestionQualityThresholds）
from dataclasses import asdict, dataclass

# pathlib.Path: 文件路径操作
from pathlib import Path

# typing.Any: 任意类型
from typing import Any

# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── 导入公共模块 ──
from scripts.common import print_json, read_json_file, write_optional_json
from scripts.gate_utils import add_max_failure, add_required_failure, to_count
from qa_core.quality.ingestion import build_ingestion_quality_report, save_ingestion_quality_report


@dataclass(frozen=True)
class IngestionQualityThresholds:
    """入库质量门禁阈值。

    默认值采用严格策略：解析失败、低质量 chunk、FAQ 冲突等问题都不允许进入 active
    版本。真实企业项目里可以按资料治理阶段临时放宽某个阈值，但放宽应通过命令行显式
    写出来，不能在代码里悄悄吞掉。

    调用顺序：命令行入口 -> IngestionQualityThresholds。
    """

    max_failed_files: int = 0
    max_unsupported_files: int = 0
    max_empty_files: int = 0
    max_low_quality_issues: int = 0
    max_duplicate_chunks: int = 0
    max_empty_faq_questions: int = 0
    max_empty_faq_answers: int = 0
    max_duplicate_faq_questions: int = 0
    max_invalid_faq_sources: int = 0
    max_faq_document_conflicts: int = 0
    max_image_risk_blocking_files: int = 0
    require_faq_file: bool = True
    require_kb_version: bool = True
    require_model_versions: bool = True


def load_quality_report(path: str | Path) -> dict[str, Any]:
    """读取入库质量报告 JSON。

    使用独立函数是为了让 `rebuild_kb_version.py` 和单元测试可以复用同一套加载逻辑。

    调用顺序：命令行入口 -> load_quality_report()。
    """
    return read_json_file(path)


def summarize_report_counts(report: dict[str, Any]) -> dict[str, int]:
    """抽取入库质量门禁需要比较的核心数量。

    这里不把报告原文全部带入门禁摘要，是为了输出更稳定，也避免终端里刷出大量 chunk
    预览。要看详情时仍然打开原始 report JSON。

    参数：
        report: 由 build_ingestion_quality_report() 生成的完整质量报告字典。
            函数只读取以下三类子结构：
            - report 顶层计数字段（failed_files_count 等）；
            - report["faq_quality"] 中的 FAQ 维度计数；
            - report["chunk_quality"] 中的 chunk 维度计数；
            - report["faq_document_conflicts"] 中的冲突计数。

    返回：
        dict[str, int]，键固定，便于 evaluate_report_against_gate() 做阈值比较。
        - failed_files: 解析失败的文件数（loader 抛异常或返回空）。
        - unsupported_files: 未注册 loader 或不在 source 白名单内的文件数。
        - empty_files: loader 成功但返回 0 个 Document 的空文件数。
        - ocr_risk_files: 疑似扫描件/OCR 噪声的文件数。
        - image_risk_files: 含图片、扫描页或图文混排资料的文件数。
        - image_risk_blocking_files: 其中严重度=block（必须 OCR/复核）的文件数。
        - low_quality_issues: 空 chunk、过短 chunk、疑似 OCR 噪声等低质量问题数。
        - duplicate_chunks: 重复 chunk 数，可能造成重复召回。
        - empty_faq_questions: FAQ 行问题为空的数量。
        - empty_faq_answers: FAQ 行答案为空的数量。
        - duplicate_faq_questions: FAQ 重复标准问题的数量。
        - invalid_faq_sources: FAQ source 不在场景白名单内的数量。
        - faq_document_conflicts: FAQ 标准答案与正文资料冲突的数量。

    调用顺序：命令行入口或 rebuild_kb_version.py -> summarize_report_counts()。
    """
    # 质量报告是面向人和排障的完整 JSON，门禁只抽取固定指标。
    # 统一在这里做 to_count 归一化，避免字段缺失、None 或字符串数字让
    # 后续阈值比较出现类型分支。
    faq_quality = report.get("faq_quality") or {}
    chunk_quality = report.get("chunk_quality") or {}
    conflicts = report.get("faq_document_conflicts") or {}
    return {
        # ── 文件解析维度：来自 report 顶层计数字段 ──
        "failed_files": to_count(report.get("failed_files_count")),
        "unsupported_files": to_count(report.get("unsupported_files_count")),
        "empty_files": to_count(report.get("empty_files_count")),
        # ── 图片/OCR 风险维度：来自 report 顶层计数字段 ──
        "ocr_risk_files": to_count(report.get("ocr_risk_files_count")),
        "image_risk_files": to_count(report.get("image_risk_files_count")),
        "image_risk_blocking_files": to_count(report.get("image_risk_blocking_files_count")),
        # ── Chunk 维度：来自 report["chunk_quality"] ──
        "low_quality_issues": to_count(chunk_quality.get("low_quality_issue_count")),
        "duplicate_chunks": to_count(chunk_quality.get("duplicate_chunk_count")),
        # ── FAQ 维度：来自 report["faq_quality"] ──
        "empty_faq_questions": to_count(faq_quality.get("empty_question_rows")),
        "empty_faq_answers": to_count(faq_quality.get("empty_answer_rows")),
        "duplicate_faq_questions": to_count(faq_quality.get("duplicate_questions")),
        "invalid_faq_sources": to_count(faq_quality.get("invalid_sources")),
        # ── FAQ/正文冲突维度：来自 report["faq_document_conflicts"] ──
        "faq_document_conflicts": to_count(conflicts.get("conflict_count")),
    }


def evaluate_report_against_gate(
    report: dict[str, Any],
    thresholds: IngestionQualityThresholds,
    *,
    report_path: str = "",
) -> dict[str, Any]:
    """用阈值判断入库质量报告是否通过。（★★★ 核心）

    质量报告负责“看见问题”，本函数负责“阻止问题进入线上版本”：把报告里每个
    维度的实际数量与阈值做比较，任何一项超过阈值就追加到 failures，最终只要
    failures 非空就 ok=False，rebuild_kb_version.py 拿到 ok=False 后就不会激活版本。

    返回结构同时适合命令行打印和 `rebuild_kb_version.py` 调用：
    - `ok=True` 表示可以继续激活；
    - `failures` 给出明确失败原因；
    - `counts` 保留关键指标，便于状态页或 CI 展示。

    参数：
        report: 由 build_ingestion_quality_report() 生成的完整质量报告字典。
        thresholds: 由命令行参数构造的 IngestionQualityThresholds 实例，每个字段
            对应一项允许的最大问题数量或必填字段开关。
        report_path: 报告 JSON 的路径，写入返回结果便于 CI 或人工反查；可空。

    返回：
        dict[str, Any]，结构：
        - ok: bool，failures 为空时为 True，否则 False。
        - report_type: 固定 "ingestion_quality_gate"，标识这是门禁判定而非原始报告。
        - report_path: 入参原样回传，便于追溯。
        - scenario_id / kb_version: 从报告中取出，便于聚合统计。
        - counts: summarize_report_counts() 输出的核心数量摘要。
        - thresholds: asdict(thresholds)，把阈值对象序列化便于审计。
        - failures: 失败明细列表，每项含 metric / actual / maximum / message。

    设计决策：
    - 阈值默认全部为 0：严格策略，任何解析失败/空 FAQ/低质量 chunk 都阻断激活；
      真实治理中需要放宽时，应在命令行显式传 --max-xxx，不能在代码里悄悄吞掉。
    - add_max_failure 而非 if：统一走 add_max_failure() 助手，让每条失败的 metric /
      actual / maximum / message 结构一致，方便 CI 展示和审计聚合。
    - OCR 风险写死阈值 0：扫描件和图片型 PDF 必须先人工复核或走独立 OCR 清洗，
      不允许通过命令行放宽，避免误入库为不可用向量。

    add_max_failure和add_required_failure的区别：
        add_max_failure：写入的failure字典只有{"metric": metric, "actual": actual, "threshold": maximum, "message": message}
            一旦动态传递了threshold参数意味着最多允许actual指定的参数数量，则失败
            举例：failed_files=3，意味着超过三个失败的文件，不激活
        add_required_failure：做存在性检查，必填项检查不做大小比较，只看值是否存在

    调用顺序：命令行入口或 rebuild_kb_version.py -> evaluate_report_against_gate()。
    """
    # 先把原始报告压缩成稳定的计数摘要，再逐项套用阈值。报告生成和门禁
    # 解耦后，既可以对刚生成的报告判定，也可以对历史 JSON 重复判定。
    counts = summarize_report_counts(report)
    faq_quality = report.get("faq_quality") or {}
    failures: list[dict[str, Any]] = []

    # ── 第一组：数量阈值类（actual 超过 maximum 即失败）──
    # 这一组覆盖文件解析、chunk 质量、FAQ 质量、冲突检测等数量型指标；
    # 阈值默认为 0 表示严格策略，可通过命令行 --max-xxx 显式放宽。

    # 文件解析维度：解析失败说明 loader 异常或文件损坏，必须修复后重新入库
    add_max_failure(
        failures,
        metric="failed_files",
        actual=counts["failed_files"],
        maximum=thresholds.max_failed_files,
        message="存在解析失败文件，必须修复后重新生成知识库版本。",
    )
    # 文件白名单维度：未注册 loader 或不在 source 白名单，可能导致资料漏入库
    add_max_failure(
        failures,
        metric="unsupported_files",
        actual=counts["unsupported_files"],
        maximum=thresholds.max_unsupported_files,
        message="存在未支持或未纳入 source 白名单的文件，可能导致资料漏入库。",
    )
    # 空文件维度：loader 成功但返回 0 个 Document，通常是空白页或格式异常
    add_max_failure(
        failures,
        metric="empty_files",
        actual=counts["empty_files"],
        maximum=thresholds.max_empty_files,
        message="存在 loader 未解析出内容的空文件。",
    )
    # OCR 风险维度：扫描件默认不能直接进入 active，必须先人工复核或走独立 OCR 清洗
    # 注意：这一项的 maximum 写死为 0，不允许通过命令行放宽，防止扫描件被误入库
    add_max_failure(
        failures,
        metric="ocr_risk_files",
        actual=counts["ocr_risk_files"],
        maximum=0,
        message="存在疑似 OCR/扫描件风险文件，默认不能直接进入 active 知识库。",
    )
    # 图片阻断维度：严重度=block 的图片资料必须 OCR/人工复核，禁止直接入库
    add_max_failure(
        failures,
        metric="image_risk_blocking_files",
        actual=counts["image_risk_blocking_files"],
        maximum=thresholds.max_image_risk_blocking_files,
        message="存在必须 OCR/人工复核的图片资料，不能直接进入 active 知识库。",
    )
    # 低质量 chunk 维度：空 chunk、过短 chunk、疑似 OCR 噪声等
    add_max_failure(
        failures,
        metric="low_quality_issues",
        actual=counts["low_quality_issues"],
        maximum=thresholds.max_low_quality_issues,
        message="存在空 chunk、过短 chunk、重复 chunk 或疑似 OCR 噪声。",
    )
    # 重复 chunk 维度：可能造成重复召回和答案引用噪声
    add_max_failure(
        failures,
        metric="duplicate_chunks",
        actual=counts["duplicate_chunks"],
        maximum=thresholds.max_duplicate_chunks,
        message="存在重复 chunk，可能造成重复召回和答案引用噪声。",
    )
    # FAQ 空问题维度：FAQ 直出依赖标准问题匹配，空问题无法稳定命中
    add_max_failure(
        failures,
        metric="empty_faq_questions",
        actual=counts["empty_faq_questions"],
        maximum=thresholds.max_empty_faq_questions,
        message="FAQ 存在空问题，无法稳定命中标准问答。",
    )
    # FAQ 空答案维度：高置信命中后无法直出可靠结果
    add_max_failure(
        failures,
        metric="empty_faq_answers",
        actual=counts["empty_faq_answers"],
        maximum=thresholds.max_empty_faq_answers,
        message="FAQ 存在空答案，高置信命中后无法直出可靠结果。",
    )
    # FAQ 重复问题维度：同一标准问题不同答案可能造成口径冲突
    add_max_failure(
        failures,
        metric="duplicate_faq_questions",
        actual=counts["duplicate_faq_questions"],
        maximum=thresholds.max_duplicate_faq_questions,
        message="FAQ 存在重复标准问题，可能造成口径冲突。",
    )
    # FAQ source 维度：source 不在场景白名单，检索过滤会不稳定
    add_max_failure(
        failures,
        metric="invalid_faq_sources",
        actual=counts["invalid_faq_sources"],
        maximum=thresholds.max_invalid_faq_sources,
        message="FAQ source 不在当前场景白名单中，检索过滤会不稳定。",
    )
    # FAQ/正文冲突维度：FAQ 标准答案与正文资料冲突，会导致召回矛盾
    add_max_failure(
        failures,
        metric="faq_document_conflicts",
        actual=counts["faq_document_conflicts"],
        maximum=thresholds.max_faq_document_conflicts,
        message="FAQ 标准答案和正文资料存在潜在冲突或缺少正文依据。",
    )

    # ── 第二组：存在性必填类（关键字段缺失即失败）──
    # 这一组检查报告中必须存在的关键字段（FAQ CSV、kb_version、模型版本等）；
    # 缺失任意一项都意味着入库流程不完整或配置异常，不能激活版本。

    # FAQ CSV 必须存在：FAQ 是高置信直出的核心来源，缺失时直出能力不可验收
    add_required_failure(
        failures,
        metric="faq_file",
        actual=faq_quality.get("exists"),
        enabled=thresholds.require_faq_file,
        message="FAQ CSV 不存在，FAQ 直出能力不可验收。",
    )
    # kb_version 必填：缺失则无法追溯这次入库对应的版本，激活后无法回滚
    add_required_failure(
        failures,
        metric="kb_version",
        actual=report.get("kb_version"),
        enabled=thresholds.require_kb_version,
        message="报告缺少 kb_version，无法追溯这次入库版本。",
    )
    # embedding_model_version 必填：缺失则无法判断向量化版本，可能造成向量空间混乱
    add_required_failure(
        failures,
        metric="embedding_model_version",
        actual=report.get("embedding_model_version"),
        enabled=thresholds.require_model_versions,
        message="报告缺少 embedding_model_version，无法判断向量化版本。",
    )
    # chunk_schema_version 必填：缺失则无法判断切分方案版本，可能造成 chunk metadata 契约混乱
    add_required_failure(
        failures,
        metric="chunk_schema_version",
        actual=report.get("chunk_schema_version"),
        enabled=thresholds.require_model_versions,
        message="报告缺少 chunk_schema_version，无法判断切分方案版本。",
    )

    # 只要 failures 非空就不允许上层激活版本；结果同时保留 counts 和 thresholds，
    # 便于 CI、管理页面和人工复盘还原“为什么没有发布”。
    return {
        "ok": not failures,
        "report_type": "ingestion_quality_gate",
        "report_path": report_path,
        "scenario_id": report.get("scenario_id"),
        "kb_version": report.get("kb_version"),
        "counts": counts,
        "thresholds": asdict(thresholds),
        "failures": failures,
    }


def thresholds_from_args(args: argparse.Namespace) -> IngestionQualityThresholds:
    """把命令行参数转换成门禁阈值对象。

    该函数只做参数映射，不做业务判断。默认值和显式放宽项都原样进入
    `IngestionQualityThresholds`，真正的比较集中在 evaluate_report_against_gate，
    这样脚本入口和上层 rebuild 编排可以共享同一套门禁规则。

    调用顺序：命令行入口 -> thresholds_from_args()。
    """
    return IngestionQualityThresholds(
        max_failed_files=args.max_failed_files,
        max_unsupported_files=args.max_unsupported_files,
        max_empty_files=args.max_empty_files,
        max_low_quality_issues=args.max_low_quality_issues,
        max_duplicate_chunks=args.max_duplicate_chunks,
        max_empty_faq_questions=args.max_empty_faq_questions,
        max_empty_faq_answers=args.max_empty_faq_answers,
        max_duplicate_faq_questions=args.max_duplicate_faq_questions,
        max_invalid_faq_sources=args.max_invalid_faq_sources,
        max_faq_document_conflicts=args.max_faq_document_conflicts,
        max_image_risk_blocking_files=args.max_image_risk_blocking_files,
        require_faq_file=not args.allow_missing_faq,
        require_kb_version=not args.allow_missing_kb_version,
        require_model_versions=not args.allow_missing_model_versions,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Check ingestion quality report against strict gate thresholds.")
    parser.add_argument("--report", default="", help="已有入库质量报告路径。未提供时会现场生成报告。")
    parser.add_argument("--scenario", default=None, help="业务场景 ID，默认使用 ACTIVE_SCENARIO_ID。")
    parser.add_argument("--data-dir", default=None, help="文档根目录，默认使用场景配置 data_root。")
    parser.add_argument("--faq-csv", default=None, help="FAQ CSV 路径，默认使用场景配置 faq_csv_path。")
    parser.add_argument("--kb-version", default=None, help="报告关联的知识库版本，默认使用当前 active 版本。")
    parser.add_argument("--tenant-id", default=None, help="报告中记录的租户 ID。")
    parser.add_argument("--dataset-id", default=None, help="报告中记录的数据集 ID。")
    parser.add_argument("--visibility", default=None, help="报告中记录的可见级别。")
    parser.add_argument("--allowed-role", action="append", default=None, help="允许检索该资料的角色，可重复。")
    parser.add_argument("--output", default="", help="现场生成报告时的输出路径。")
    parser.add_argument("--gate-output", default="", help="门禁判定摘要输出路径。")
    parser.add_argument("--max-failed-files", type=int, default=0)
    parser.add_argument("--max-unsupported-files", type=int, default=0)
    parser.add_argument("--max-empty-files", type=int, default=0)
    parser.add_argument("--max-low-quality-issues", type=int, default=0)
    parser.add_argument("--max-duplicate-chunks", type=int, default=0)
    parser.add_argument("--max-empty-faq-questions", type=int, default=0)
    parser.add_argument("--max-empty-faq-answers", type=int, default=0)
    parser.add_argument("--max-duplicate-faq-questions", type=int, default=0)
    parser.add_argument("--max-invalid-faq-sources", type=int, default=0)
    parser.add_argument("--max-faq-document-conflicts", type=int, default=0)
    parser.add_argument("--max-image-risk-blocking-files", type=int, default=0)
    parser.add_argument("--allow-missing-faq", action="store_true", help="允许没有 FAQ CSV。")
    parser.add_argument("--allow-missing-kb-version", action="store_true", help="允许报告缺少 kb_version。")
    parser.add_argument("--allow-missing-model-versions", action="store_true", help="允许报告缺少模型和切分版本。")
    return parser


def main() -> None:
    """执行入库质量门禁并按结果设置退出码。

    命令行支持两种模式：
      1. 传 `--report`：读取已经落盘的质量报告，只做门禁判定；
      2. 不传 `--report`：现场试解析并切分，先保存原始报告，再做门禁判定。

    注意：这个脚本本身不激活知识库版本。它只输出 `ok` 和失败明细，
    由 `rebuild_kb_version.py` 决定是否继续调用 `activate_version()`。

    调用顺序：命令行入口 -> main()。
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.report:
        # 历史报告模式：不重新读取资料，保证复盘时使用当时生成的事实快照。
        report_path = args.report
        report = load_quality_report(report_path)
    else:
        # 现场模式：先生成报告再保存，门禁结果中的 report_path 指向同一份快照。
        report = build_ingestion_quality_report(
            scenario_id=args.scenario,
            data_dir=args.data_dir,
            faq_csv=args.faq_csv,
            kb_version=args.kb_version,
            tenant_id=args.tenant_id,
            dataset_id=args.dataset_id,
            visibility=args.visibility,
            allowed_roles=args.allowed_role,
        )
        report_path = save_ingestion_quality_report(report, args.output or None)

    # 把命令行阈值映射成对象后执行统一规则；失败时使用退出码 1 阻断 CI 或上层脚本。
    result = evaluate_report_against_gate(report, thresholds_from_args(args), report_path=report_path)
    write_optional_json(args.gate_output, result)
    print_json(result)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
