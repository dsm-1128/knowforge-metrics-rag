# -*- coding: utf-8 -*-
"""构建单个业务场景的完整知识库版本。

业务目标：
    把 FAQ 和文档写入同一个 kb_version，并在质量门禁通过后按需激活。

通用发布流程：
    1. 解析参数并校验冲突选项
    2. 解析业务场景，必要时重建 Milvus collection
    3. 创建或复用目标 kb_version
    4. 可选解析跨版本增量基准
    5. FAQ 入库
    6. 文档入库
    7. 生成质量报告；如果要激活版本，则必须执行质量门禁
    8. 按需激活版本并输出结果

增量入库流程：
    1. 选择业务场景，例如 enterprise_knowledge。
    2. 创建新的目标 kb_version，新版本先保持 STAGED，不影响线上查询。
    3. 用 --incremental-from active 或显式旧 kb_version 确定增量基准版本。
    4. 在新版本 stats 中记录 incremental_base_kb_version，方便追溯。
    5. FAQ 仍然按新版本重建；FAQ 数量小且高置信直出口径要求更高，不做跨版本引用。
    6. 文档逐文件计算 fingerprint，并同时检查 embedding_model_version 和 chunk_schema_version。
    7. 文件未变化时，从基准版本 manifest 找到旧 chunk_ids，目标版本 manifest 直接引用这些
       chunk，不复制 Milvus 行、不重新 embedding。
    8. 文件新增、内容变化、模型变化或切分策略变化时，重新执行 load_file()、
       normalize_documents()、split_documents()、add_documents()，再更新 manifest。
    9. 文件删除时给旧 chunk 写 valid_to_seq，让它从目标版本开始不可见。
    10. 生成目标版本的质量报告，检查文件解析、FAQ、chunk 和 FAQ/正文冲突。
    11. 执行质量门禁；失败则不激活，旧 active 继续服务。
    12. 门禁通过且传入 --activate 时切换 MySQL active 指针，文档检索按 active version_seq
        解释有效期视图：valid_from_seq <= active_seq 且未失效。

关键边界：
    增量入库不是线上查询时拼接多个 kb_version，而是用 version_seq 解释同一个 collection
    中 chunk 的有效期视图。未变化 chunk 不复制，变化或删除通过 valid_to_seq 收口。

典型命令示例：
    # 1. 全量重建并激活（初始化或全量重建，强制重新 embedding 所有文档）
    python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate --description "full rebuild for enterprise_knowledge"

    # 2. 增量构建并激活（日常只改了少量资料，未变化文件引用复用旧 chunk）
    python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --incremental-from active --quality-gate --activate --description "incremental rebuild from active"

    # 3. 只构建 STAGED 不激活（预演或调试，不影响线上查询）
    python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --description "staged build without activation"

    # 4. schema 变更后重建（迁移到 BM25 BuiltInFunction 时必须先 drop collection）
    python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --reset-collections --quality-gate --activate --description "rebuild after schema migration"

    # 5. 复用已有版本重新入库（上次入库失败或质量门禁没过，指定同一版本号重试）
    python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --kb-version kb_enterprise_knowledge_20260806_055203_9ed89890 --force --quality-gate --activate --description "retry failed version"

参数冲突校验（validate_args 会拒绝以下组合）：
    --activate 与 --skip-quality-report 不能同用（激活必须经过质量门禁）
    --quality-gate 与 --skip-quality-report 不能同用（门禁依赖质量报告）
    --incremental-from 与 --reset-collections 不能同用（reset 会 drop 旧向量，无法引用）
    --incremental-from 与 --force 不能同用（force 强制全量重建，与增量语义冲突）
    --incremental-from 必须配合 --new-version 或显式 --kb-version（增量需要明确目标版本）
"""

from __future__ import annotations

# ── 标准库 ──
# argparse: 命令行参数解析
import argparse
# sys: 系统功能（sys.path 修改 + sys.exit 退出码）
import sys
# pathlib.Path: 文件路径操作
from pathlib import Path
# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 导入 Milvus SDK ──
# MilvusClient: PyMilvus 客户端（用于 Collection 的 drop 操作）
from pymilvus import MilvusClient

# ── 导入核心模块 ──
# get_kb_version_store: 获取知识库版本 Store（MySQL 中的版本控制面）
from qa_core.governance.kb_versions import get_kb_version_store
# resolve_data_scope: 解析本次构建统一使用的数据域上下文
from qa_core.governance.data_scope import resolve_data_scope
# ingest_faq_csv: FAQ CSV 入库（写入 FAQ collection）
from qa_core.indexing.faq_ingestion import ingest_faq_csv
# ingest_directory: 目录批量入库（逐文件 load → normalize → split → add）
from qa_core.indexing.service import ingest_directory
# build_ingestion_quality_report: 生成入库质量报告
# save_ingestion_quality_report: 保存入库质量报告到文件
from qa_core.quality.ingestion import build_ingestion_quality_report, save_ingestion_quality_report
# ensure_milvus_database: 确保 Milvus database 存在
# langchain_connection_args: 生成 langchain-milvus 兼容的连接参数
from qa_core.retrieval.milvus_compat import ensure_milvus_database, langchain_connection_args
# resolve_scenario: 解析业务场景配置
from qa_core.scenarios.registry import resolve_scenario
from qa_core.storage.bootstrap import bootstrap_mysql_schema
# IngestionQualityThresholds: 入库质量门禁阈值
# evaluate_report_against_gate: 用阈值判断入库质量报告是否通过
from scripts.quality.check_ingestion_quality_gate import IngestionQualityThresholds, evaluate_report_against_gate


def build_parser() -> argparse.ArgumentParser:
    """构造离线入库发布脚本的命令行解析器。

    参数按离线发布链路分成五组：
      1. 输入定位：场景、文档根目录、FAQ CSV。
      2. 版本与增量：目标 kb_version、新版本、增量基准。
      3. 入库控制：是否强制重建、是否跳过 FAQ/文档、是否重置 Collection。
      4. 发布控制：质量报告、质量门禁和 active 激活。
      5. 数据隔离与质量阈值：租户、数据集、可见级别、角色和门禁上限。

    这个函数只负责“声明参数”，不读取数据库、不解析场景，也不执行入库。
    真正的参数语义校验在 `validate_args()` 中完成，业务执行统一从 `main()` 开始。

    调用顺序：命令行入口 -> build_parser()。
    """
    # argparse 负责把字符串命令转换成 Namespace；默认值保持 None，
    # 让后续场景配置可以在没有显式参数时提供默认 data_root 和 faq_csv_path。
    parser = argparse.ArgumentParser(description="Rebuild one scenario into a complete knowledge base version.")

    # ── 输入路径参数 ──
    # 路径不在这里转成 Path：main() 需要先解析场景，才能决定未传参数时的默认路径。
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root data directory. Defaults to the selected scenario data_root.",
    )
    parser.add_argument(
        "--faq-csv",
        default=None,
        help="FAQ CSV path. Defaults to the selected scenario faq_csv_path.",
    )

    # ── 场景与版本参数 ──
    # scenario 决定 collection、valid_sources、默认资料目录和 FAQ 路径。
    # kb-version 与 new-version 共同决定第 3 步是复用旧版本还是创建新版本。
    parser.add_argument("--scenario", default=None, help="Business scenario id. Defaults to ACTIVE_SCENARIO_ID.")
    parser.add_argument("--kb-version", default=None, help="Explicit knowledge base version id.")
    parser.add_argument("--new-version", action="store_true", help="Create a new staged version before ingest.")
    parser.add_argument("--force", action="store_true", help="Rebuild files even when fingerprint is unchanged.")
    parser.add_argument(
        "--incremental-from",
        default=None,
        help=(
            "Cross-version incremental document build base. Use 'active' to copy unchanged chunks "
            "from the current active version, or pass an explicit kb_version. FAQ is still rebuilt."
        ),
    )

    # ── 阶段开关和数据面控制 ──
    # skip 参数只影响对应阶段；它们不会删除已有数据，也不会改变版本状态。
    parser.add_argument("--skip-faq", action="store_true", help="Skip FAQ ingest.")
    parser.add_argument("--skip-docs", action="store_true", help="Skip document ingest.")
    parser.add_argument(
        "--reset-collections",
        action="store_true",
        help=(
            "Drop the selected scenario FAQ and document Milvus collections before ingest. "
            "Use this when schema changed, especially when migrating to BM25 BuiltInFunction hybrid search."
        ),
    )

    # ── 发布生命周期参数 ──
    # --activate 的语义是“尝试上线”，因此 validate_args() 会自动打开质量门禁。
    parser.add_argument("--skip-quality-report", action="store_true", help="Skip ingestion quality report generation. Only allowed for staged builds.")
    parser.add_argument("--quality-gate", action="store_true", help="Run strict ingestion quality gate. Activation always enables it.")
    parser.add_argument("--activate", action="store_true", help="Activate this version after successful ingest.")
    parser.add_argument("--description", default="", help="Human readable version description.")

    # ── 写入 metadata 的数据隔离参数 ──
    # 这些值会在第 1 步统一解析成 DataScope，并传给 FAQ、文档和质量报告。
    parser.add_argument("--tenant-id", default=None, help="Tenant/org id written into metadata. Defaults to default.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id written into metadata. Defaults to default.")
    parser.add_argument("--visibility", default=None, help="Visibility written into metadata: public/internal/private.")
    parser.add_argument("--allowed-role", action="append", default=None, help="Role allowed to retrieve this data. Can repeat.")

    # ── 质量门禁阈值 ──
    # 默认值为 0，表示对应质量问题一旦出现就阻断激活；
    # 是否启用某一项门禁由 `quality_thresholds_from_args()` 统一转换。
    parser.add_argument("--max-failed-files", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-unsupported-files", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-empty-files", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-low-quality-issues", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-duplicate-chunks", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-empty-faq-questions", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-empty-faq-answers", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-duplicate-faq-questions", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-invalid-faq-sources", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-faq-document-conflicts", type=int, default=0, help="Quality gate threshold.")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """校验会破坏离线发布语义的参数组合。

    参数校验发生在所有数据库和 Milvus 写入之前。这样错误命令不会先创建版本、
    删除 Collection 或写入部分数据，避免产生无法解释的 STAGED 状态。

    校验规则：
      - 激活必须依赖质量报告和质量门禁。
      - 跨版本增量必须保留旧向量，不能和 reset/force 冲突。
      - 指定增量基准时，必须有明确的目标版本，否则无法定义继承关系。

    调用顺序：命令行入口 -> validate_args()。
    """
    # 激活是发布动作，不允许绕过质量门禁；这里直接把隐含前置条件写回 args，
    # 让 main() 后面只需要判断 args.quality_gate 即可。
    if args.activate:
        args.quality_gate = True

    # 质量报告是门禁的输入，因此激活或显式门禁都不能同时跳过报告。
    if args.activate and args.skip_quality_report:
        parser.error("--activate requires quality report and quality gate; remove --skip-quality-report.")
    if args.quality_gate and args.skip_quality_report:
        parser.error("--quality-gate requires quality report generation; remove --skip-quality-report.")

    # 跨版本增量依赖基准版本中的 Milvus chunk。reset 会删除旧 Collection，
    # force 会让所有文件重新计算，二者都和“复用未变化 chunk”相矛盾。
    if args.incremental_from and args.reset_collections:
        parser.error("--incremental-from cannot be used with --reset-collections because old vectors must be copied.")
    if args.incremental_from and args.force:
        parser.error("--incremental-from cannot be used with --force; force means rebuild all documents.")

    # 增量构建必须知道“写到哪个新版本”。不能让脚本在没有目标版本时
    # 一边推断 active、一边把 active 当成自己的基准，避免版本关系自引用。
    if args.incremental_from and not (args.new_version or args.kb_version):
        parser.error("--incremental-from requires --new-version or an explicit --kb-version target.")


def quality_thresholds_from_args(args: argparse.Namespace) -> IngestionQualityThresholds:
    """从命令行参数构造入库质量门禁阈值。

    这里不执行任何质量判断，只负责把 argparse Namespace 中的离散参数
    组装成一个不可变阈值对象。后续 `evaluate_report_against_gate()` 只依赖
    这个对象，从而把“参数解析”和“门禁判定”分开。

    调用顺序：命令行入口 -> quality_thresholds_from_args()。
    """
    # 每个命令行阈值都显式映射到同名字段，避免把 Namespace 直接传入质量模块，
    # 也避免质量模块依赖脚本层的参数命名。
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
    )


def _should_create_new_version(args: argparse.Namespace, version_store) -> bool:
    """判断这次重建是否应该创建新的目标版本。

    三种输入状态：
      1. `--new-version`：调用方明确要求创建新的 STAGED 版本。
      2. 未传 `--kb-version` 且当前没有 active 候选：首次初始化，必须创建版本。
      3. 其他情况：由 `ensure_version()` 复用显式版本或当前 active 版本。

    注意：这个函数只返回布尔决策，不负责生成版本号；版本号生成和写库由
    `KnowledgeBaseVersionStore.ensure_version()` 完成。
    """
    # active_version_candidate() 读取 MySQL active 指针；这里使用 bool 只判断是否存在，
    # 不把具体版本号泄露到本函数的决策逻辑中。
    return args.new_version or (not args.kb_version and not bool(version_store.active_version_candidate()))


def _resolve_incremental_base_version(version_store, requested_base: str | None, target_kb_version: str) -> str | None:
    """把 `--incremental-from` 解析成实际基准版本号。

    `--incremental-from active` 不是一个真实版本号，而是一个动态别名，
    需要在当前任务开始时读取 MySQL active 指针。显式版本号则交给
    `resolve_active_version(requested)` 做存在性校验，避免拼写错误被静默降级。

    返回的版本号会被第 4 步写入目标版本 stats，并传给文档入库阶段。
    FAQ 不使用这个基准，因为 FAQ 采用快照式重建。
    """
    if not requested_base:
        # 没有增量参数时返回 None；文档入口会把它解释为普通全量/同版本增量。
        return None
    requested_base = requested_base.strip()
    if requested_base.lower() == "active":
        # active 是运行时指针，不能在参数解析阶段提前固定，必须读取任务启动时的值。
        incremental_base_kb_version = version_store.resolve_active_version()
    else:
        # 显式版本号必须存在于当前场景版本表中，否则直接让上层报错。
        incremental_base_kb_version = version_store.resolve_active_version(requested_base)
    if incremental_base_kb_version == target_kb_version:
        # 目标版本不能继承自己，否则 valid_from/valid_to 的继承关系没有意义。
        raise ValueError("--incremental-from must point to a different base version than the target kb_version.")
    return incremental_base_kb_version


def main() -> None:
    """按固定发布流程构建一个完整知识库版本。

    流程：解析场景 → 重置Collection(可选) → 创建目标版本 → 解析增量基准 →
         FAQ入库 → 文档入库 → 质量门禁 → 激活版本 → 输出摘要

    设计原则：
      - 前置阶段一次性准备 `scenario`、`data_scope`、`version` 和 `version_store`。
      - 下游入口优先复用这些对象，避免重复解析和重复查询控制面。
      - 入库函数只负责写数据和统计，是否上线由本函数的第七步统一决定。

    调用顺序：命令行入口 -> main()。
    """
    # ── 前置校验：还没有任何外部副作用 ──
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    # 第一步：解析场景配置并初始化 MySQL schema。
    # scenario 决定默认数据路径、FAQ/文档 Collection 和 source 白名单；
    # schema bootstrap 保证后续版本、manifest、chunk 有效期索引有可写的控制面表。
    scenario = resolve_scenario(args.scenario)
    bootstrap_mysql_schema()
    # 同一次构建只解析一次 DataScope，FAQ、文档和质量报告共享同一隔离边界。
    # 这也是离线链路中权限 metadata 的单一来源。
    data_scope = resolve_data_scope(
        tenant_id=args.tenant_id,
        dataset_id=args.dataset_id,
        visibility=args.visibility,
        user_roles=args.allowed_role,
    )

    # 第二步：按需重置 Milvus Collection（schema 变更时使用）。
    # 只有显式传入 --reset-collections 才连接 Milvus 并删除集合，
    # 默认构建不会碰已有 Collection，避免误删线上或历史版本数据。
    if args.reset_collections:
        ensure_milvus_database()
        client = MilvusClient(**langchain_connection_args())
        for collection_name in (scenario.faq_collection, scenario.doc_collection):
            if client.has_collection(collection_name):
                # FAQ 和文档集合一起重置，确保新 schema 不会出现一边已迁移、一边仍是旧结构。
                client.drop_collection(collection_name)
                print(f"Dropped Milvus collection for schema reset: {collection_name}")

    # 第三步：创建或复用目标知识库版本（STAGED 状态，不影响线上查询）。
    # version_store 和 version 会传给后续阶段，避免每个入口重复创建 Store 或 ensure_version()。
    version_store = get_kb_version_store(scenario.scenario_id)
    create_new = _should_create_new_version(args, version_store)
    version = version_store.ensure_version(
        args.kb_version,
        create_new=create_new,
        description=args.description,
        created_by="rebuild_kb_version",
    )
    kb_version = version.kb_version

    # 第四步：解析增量构建基准版本。
    # 除了版本号，同时在这里读取 version_seq；文档入口需要它写入旧 chunk 的 valid_to_seq，
    # 如果每个 source 都重新查一次，会增加数据库访问，也容易让同一任务看到不一致的基准状态。
    incremental_base_kb_version = None
    incremental_base_version_seq = 0
    if args.incremental_from:
        try:
            incremental_base_kb_version = _resolve_incremental_base_version(
                version_store,
                args.incremental_from,
                kb_version,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if incremental_base_kb_version:
            base_version = version_store.get(incremental_base_kb_version)
            if base_version is None:
                parser.error(f"增量基准版本不存在：{incremental_base_kb_version}")
            incremental_base_version_seq = base_version.version_seq
        # 只记录一次目标版本的增量关系，后续每个 source 只读取这个已确定的值。
        version_store.record_incremental_base(kb_version, incremental_base_kb_version)

    # 第五步：FAQ 入库（FAQ 不做增量，每次全量重建）。
    # FAQ 与文档共享目标 version，但使用独立 Collection 和 snapshot 版本策略。
    faq_count = 0
    if not args.skip_faq:
        faq_count = ingest_faq_csv(
            args.faq_csv or scenario.faq_csv_path,
            allowed_roles=args.allowed_role,
            scenario=scenario,
            data_scope=data_scope,
            target_version=version,
            version_store=version_store,
        )

    # 第六步：文档入库（支持增量：按文件 fingerprint 判断是否需要重新处理）。
    # 一个 source 对应一个 <source>_data 目录；每个目录入口复用同一套场景、数据域、
    # 目标版本和基准 version_seq，但各自记录本 source 的 manifest 和统计。
    doc_chunks = 0
    if not args.skip_docs:
        root = Path(args.data_dir or scenario.data_root)
        for source in scenario.valid_sources:
            source_dir = root / f"{source}_data"
            if not source_dir.exists():
                # 缺少某个可选 source 目录不是错误，跳过后继续处理其他 source。
                continue
            doc_chunks += ingest_directory(
                str(source_dir),
                source=source,
                allowed_roles=args.allowed_role,
                force=args.force,
                incremental_base_kb_version=incremental_base_kb_version,
                scenario=scenario,
                data_scope=data_scope,
                target_version=version,
                version_store=version_store,
                incremental_base_version_seq=incremental_base_version_seq,
            )

    # 第七步：质量报告 + 质量门禁 + 激活版本。
    # 质量报告会重新试解析候选资料，但只在内存中检查，不会再次写入 Milvus；
    # 实际写入数量通过 actual_ingest 附加，避免把“试运行统计”误认为“真实写入统计”。
    report_path = ""
    activated = False
    if not args.skip_quality_report:
        report = build_ingestion_quality_report(
            data_dir=args.data_dir or scenario.data_root,
            faq_csv=args.faq_csv or scenario.faq_csv_path,
            kb_version=kb_version,
            allowed_roles=args.allowed_role,
            scenario=scenario,
            data_scope=data_scope,
        )
        report["actual_ingest"] = {
            "faq_records_written": faq_count,
            "doc_chunks_written": doc_chunks,
            "activated": False,
        }
        if args.quality_gate:
            # 门禁只消费质量报告和阈值，不直接扫描文件、不修改版本状态。
            gate_result = evaluate_report_against_gate(report, quality_thresholds_from_args(args))
            report["quality_gate"] = gate_result
            if not gate_result["ok"]:
                # 失败报告必须先保存，再退出；目标版本保持 STAGED，active 指针不变。
                report_path = save_ingestion_quality_report(report)
                print(
                    "Ingestion quality gate failed; knowledge base version was not activated: "
                    f"{kb_version}, quality_report={report_path}"
                )
                sys.exit(1)
        if args.activate:
            # 只有报告生成成功且门禁未失败，才允许切换 MySQL active 指针。
            # activate_version() 内部负责旧 active 降级、流水审计和缓存 epoch 推进。
            version_store.activate_version(kb_version)
            activated = True
            report["actual_ingest"]["activated"] = True
        # 无论是否激活，只要生成了报告，就保存最终状态快照。
        report_path = save_ingestion_quality_report(report)

    # 第八步：输出构建摘要。
    # 摘要只反映本次任务结果；详细文件失败、重复 FAQ、冲突和风险要查看质量报告。
    print(
        "Rebuilt knowledge base version: "
        f"{kb_version}, faq_records={faq_count}, doc_chunks={doc_chunks}, "
        f"activated={activated}, incremental_base={incremental_base_kb_version or 'none'}, "
        f"quality_report={report_path or 'skipped'}"
    )


if __name__ == "__main__":
    main()
