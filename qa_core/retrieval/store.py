"""Milvus 混合检索适配器，统一封装 FAQ 和文档集合。

QAService 不需要关心 Milvus 连接、BM25 配置、数据隔离表达式和 CrossEncoder 重排细节，
只调用这里提供的 search/search_many 即可。

依赖分层：
- qa_core.retrieval.filters：构造 Milvus 过滤表达式。
- qa_core.retrieval.ranking：候选去重、查询清洗和重排。
- qa_core.retrieval.models：embedding 和 reranker 模型提供器。
- qa_core.governance.data_scope：租户、数据集、角色的数据隔离。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Literal

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import DataType, FunctionType
from pymilvus.exceptions import MilvusException

from qa_core.governance.data_scope import DataScope
from qa_core.config.logging_config import get_logger
from qa_core.retrieval.milvus_compat import (
    bm25_function,
    ensure_milvus_database,
    hybrid_index_params,
    hybrid_search_params,
    langchain_connection_args,
)
from qa_core.retrieval.filters import build_source_expr
from qa_core.retrieval.models import get_embeddings, get_reranker
from qa_core.retrieval.ranking import merge_hits_by_document, normalize_queries, rerank_hits, sort_hits_by_score
from qa_core.retrieval.results import RetrievalHit, RetrievalResult
from qa_core.config.settings import get_settings


logger = get_logger(__name__)
HYBRID_RANKER_KWARGS = {
    "ranker_type": "weighted",
    "ranker_params": {"weights": [0.55, 0.45]},
}


def _truthy(value) -> bool:
    """兼容 PyMilvus schema 中 bool / str 两种布尔表达，统一返回 True/False。（★★ 理解）

    PyMilvus 在 schema.fields 返回时，部分版本将布尔字段的 enable_analyzer 等属性
    以字符串形式返回（如 "true" / "false"），部分版本以 Python bool 形式返回。
    此函数统一处理两种类型，避免 validate_hybrid_schema() 中条件判断出错。

    参数：
        value: PyMilvus schema 字段属性值（bool 或 str）。

    返回：
        bool: 统一转换后的布尔值。

    调用顺序：检索准备或检索执行 -> _truthy()。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        # 字符串形式支持 "true"/"1"/"yes" 三种正例
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _index_params_by_field(collection: Any) -> dict[str, dict[str, object]]:
    """读取 collection 当前索引参数，按字段名返回。

    旧 collection 即使在 Milvus() 初始化时传入新的 index_params，也不会自动替换已有
    索引。因此启动时必须检查真实 collection 的索引类型，避免代码声明 HNSW、实际仍在
    使用 AUTOINDEX。
    """
    return {
        str(index.field_name): dict(index.params or {})
        for index in list(getattr(collection, "indexes", []) or [])
        if getattr(index, "field_name", None)
    }


def _elapsed_ms(started: float) -> float:
    """将 time.perf_counter() 起点转换为毫秒耗时，供各 search 方法统一复用。（★★ 理解）

    参数：
        started: time.perf_counter() 起始时间戳。

    返回：
        float: 经过的毫秒数（保留浮点精度）。

    调用顺序：检索准备或检索执行 -> search() / search_many() -> _elapsed_ms()。
    """
    return (time.perf_counter() - started) * 1000


def _to_hits(raw_hits: list[tuple[Document, float]]) -> list[RetrievalHit]:
    """将 langchain-milvus 原始返回的 (Document, score) 元组列表转换为内部统一的 RetrievalHit。（★★ 理解）

    隔离上层对 langchain-milvus 返回格式的依赖。如果 langchain-milvus 后续版本变更
    返回格式，只需修改此一处转换逻辑。

    参数：
        raw_hits: langchain-milvus 返回的 (Document, score) 列表。

    返回：
        list[RetrievalHit]: 转换后的内部命中列表。

    调用顺序：检索准备或检索执行 -> search() -> _to_hits()。
    """
    return [RetrievalHit(document=doc, score=float(score or 0.0)) for doc, score in raw_hits]


def _escape_milvus_string(value: str) -> str:
    """转义 Milvus boolean expr 中的字符串值，防止引号破坏表达式结构导致语法错误或注入。（★★ 理解）

    调用顺序：检索准备或检索执行 -> _escape_milvus_string()。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _pk_in_expr(ids: list[str]) -> str:
    """根据主键列表构造 Milvus pk in 表达式。

    调用顺序：检索准备或检索执行 -> _pk_in_expr()。
    """
    escaped_ids = [_escape_milvus_string(item) for item in ids]
    return "pk in [" + ", ".join(f'"{item}"' for item in escaped_ids) + "]"


def _row_dynamic_metadata(row: dict[str, Any], fixed_fields: set[str]) -> dict[str, Any]:
    """从 Milvus 查询结果中提取动态 metadata。

    调用顺序：检索准备或检索执行 -> _row_dynamic_metadata()。
    """
    dynamic_meta = row.get("$meta") if isinstance(row.get("$meta"), dict) else {}
    metadata = {
        key: value
        for key, value in row.items()
        if key not in fixed_fields and not str(key).startswith("$")
    }
    metadata.update(dynamic_meta)
    return metadata



class MilvusHybridStore:
    """单个混合检索集合的 LangChain Milvus 封装。

    它封装单个 Milvus collection，可以是 FAQ 集合，也可以是文档集合。集合支持稠密向量
    + Milvus BM25 稀疏向量的混合检索，并在首次访问时才创建连接。

    字段说明：
      - collection_name：目标 Milvus collection 名称.

    调用顺序：检索准备或检索执行 -> MilvusHybridStore。
    """

    def __init__(self, collection_name: str) -> None:
        """保存集合配置，延迟创建 Milvus 连接以使模块导入保持轻量。（★★ 理解）

        构造阶段只保存集合名和全局配置，不立即连接 Milvus；真正连接发生在首次访问 store 属性时。

        参数：
            collection_name: 目标 Milvus collection 名称.

        调用顺序：检索准备或检索执行 -> MilvusHybridStore.__init__()。
        """
        # 保存应用级检索配置；此处只保存信息，不连接 Milvus，避免创建无用连接对象。
        self.settings = get_settings()
        # collection_name 是 FAQ/文档数据隔离的边界，不能在后续请求中随意混用。
        self.collection_name = collection_name
        # None 表示尚未初始化；第一次访问 store 属性时才真正创建 LangChain wrapper。
        self._store: Milvus | None = None

    @property
    def store(self):
        """为当前集合懒加载 LangChain Milvus 存储对象。（★★ 理解）

        首次访问时创建并缓存 LangChain Milvus wrapper。创建前会确保数据库存在，
        并构造 langchain-milvus 连接参数。

        执行流程：
          1. 未缓存时确认数据库存在，构造连接参数。
          2. 创建 Milvus wrapper，配置 BGE embedding、Milvus BM25、向量字段、索引参数、文本字段和主键。
          3. collection_name 决定查询目标，connection_args 只承载 Milvus URI / database。
          4. 缓存并返回 Milvus wrapper。

        返回：
            当前集合对应的 LangChain Milvus wrapper.
        """
        if self._store is None:
            # 只有首次访问才执行初始化；之后直接复用当前集合的 wrapper。
            # 这正是 factory 层 lru_cache 与 store 层实例缓存共同降低连接开销的地方。
            # 确保 Milvus 数据库存在（当前项目使用默认数据库）
            ensure_milvus_database()
            # 统一生成 URI/database 参数，避免 FAQ 和文档集合各自维护连接配置。
            connection_args = langchain_connection_args()
            self._store = Milvus(
                # 获取已缓存的 BGE 向量模型，用于稠密向量检索
                embedding_function=get_embeddings(),
                # 获取 Milvus 2.5+ 服务端 BM25 内置函数，用于稀疏向量关键词检索
                builtin_function=bm25_function(),
                collection_name=self.collection_name,
                connection_args=connection_args,
                vector_field=["dense", "sparse"],
                index_params=hybrid_index_params(),
                search_params=hybrid_search_params(),
                text_field="text",
                primary_field="pk",
                auto_id=False,
                enable_dynamic_field=True,
                consistency_level="Session",
                drop_old=False,
            )
            # wrapper 创建后立即检查真实 collection schema，尽早发现旧集合契约不兼容。
            self.validate_hybrid_schema()
        # 预热阶段和在线查询阶段都返回同一个 wrapper，避免重复创建连接和模型绑定。
        return self._store

    def validate_hybrid_schema(self) -> None:
        """校验 collection 是否符合当前 Dense + BM25 Sparse Hybrid 索引契约。

        `langchain-milvus` 对 sparse 字段发起的是原始 query 文本，必须依赖
        Milvus collection schema 中的 BM25 Function 把文本转换成 sparse vector。
        如果线上复用了旧 collection（只有 sparse 字段、没有 BM25 Function，或 text
        字段没有 analyzer），Milvus 会收到空 sparse search request，最终报
        `nq [0] is invalid`。同样，HNSW 配置只在 collection 首次创建时生效；已有
        AUTOINDEX collection 不会因为本次传入 HNSW 参数而自动重建。这些情况都不是
        检索时可降级的小错误，必须重建 collection。

        调用顺序：检索准备或检索执行 -> MilvusHybridStore.validate_hybrid_schema()。
        """
        collection = self.store.col
        if collection is None:
            return

        schema = collection.schema
        fields = {field.name: field for field in schema.fields}
        problems: list[str] = []

        if "text" not in fields:
            problems.append("缺少 text 字段")
        else:
            text_field = fields["text"]
            if text_field.dtype != DataType.VARCHAR:
                problems.append(f"text 字段类型应为 VARCHAR，实际为 {text_field.dtype}")
            if not _truthy(text_field.params.get("enable_analyzer")):
                problems.append("text 字段未启用 analyzer，BM25 无法分析中文 query")

        if "dense" not in fields:
            problems.append("缺少 dense 向量字段")
        elif fields["dense"].dtype != DataType.FLOAT_VECTOR:
            problems.append(f"dense 字段类型应为 FLOAT_VECTOR，实际为 {fields['dense'].dtype}")

        if "sparse" not in fields:
            problems.append("缺少 sparse 向量字段")
        else:
            sparse_field = fields["sparse"]
            if sparse_field.dtype != DataType.SPARSE_FLOAT_VECTOR:
                problems.append(f"sparse 字段类型应为 SPARSE_FLOAT_VECTOR，实际为 {sparse_field.dtype}")
            is_function_output = _truthy(getattr(sparse_field, "is_function_output", False)) or _truthy(
                sparse_field.params.get("is_function_output")
            )
            if not is_function_output:
                problems.append("sparse 字段不是 BM25 Function 输出字段")

        functions = list(getattr(schema, "functions", []) or [])
        has_bm25 = any(
            getattr(func, "type", None) == FunctionType.BM25
            and list(getattr(func, "input_field_names", []) or []) == ["text"]
            and list(getattr(func, "output_field_names", []) or []) == ["sparse"]
            for func in functions
        )
        if not has_bm25:
            problems.append("缺少 text -> sparse 的 BM25 Function")

        current_indexes = _index_params_by_field(collection)
        for field_name, expected in zip(("dense", "sparse"), hybrid_index_params()):
            current = current_indexes.get(field_name)
            if current is None:
                problems.append(f"缺少 {field_name} 索引")
                continue
            expected_index_type = str(expected["index_type"])
            expected_metric_type = str(expected["metric_type"])
            if str(current.get("index_type")) != expected_index_type:
                problems.append(
                    f"{field_name} 索引类型应为 {expected_index_type}，实际为 {current.get('index_type')}"
                )
            if str(current.get("metric_type")) != expected_metric_type:
                problems.append(
                    f"{field_name} 索引 metric 应为 {expected_metric_type}，实际为 {current.get('metric_type')}"
                )

        if problems:
            details = "；".join(problems)
            raise RuntimeError(
                f"Milvus collection `{self.collection_name}` 不符合当前 Hybrid Search schema/index：{details}。"
                "这通常是复用了旧 collection 或旧版本入库脚本创建的集合。请删除该场景 FAQ/Doc "
                "collection 后重新构建知识库版本，例如："
                "`docker compose run --rm api python scripts/rebuild_kb_version.py "
                "--scenario <scenario_id> --new-version --force --reset-collections "
                "--quality-gate --activate`。"
            )

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        """把文档写入 Milvus 集合，服务端自动生成稠密+稀疏向量。

        写入时 LangChain Milvus 会生成稠密向量，Milvus 服务端 BM25 会生成稀疏向量。

        这里是正式入库链路的物理写入边界：上游已经完成 loader、metadata
        标准化、chunk 切分和 ID 生成，本方法只负责把这些已治理的 Document
        交给 Milvus。`ids` 与 chunk 列表按位置对应，保证 manifest 记录的 ID
        可以回查到实际向量行。

        参数：
            documents: 要写入的 LangChain Document 列表。
            ids: 可选主键 ID 列表。

        返回：
            写入文档的 ID 列表；documents 为空时返回空列表.

        调用顺序：检索准备或检索执行 -> MilvusHybridStore.add_documents()。
        """
        if not documents:
            return []
        # 将文档批量写入 Milvus 集合
        return self.store.add_documents(documents=documents, ids=ids)

    def expire_documents_for_version(
        self,
        ids: list[str],
        *,
        valid_to_seq: int,
        batch_size: int = 200,
    ) -> int:
        """让基准版本中的旧 chunk 从目标版本开始失效。

        ids 由调用方从本次增量基准版本 manifest 传入，不扫描所有历史版本。
        valid_to_seq 是失效起点：active_seq 等于该值时，旧 chunk 已不可见。

        调用顺序：rebuild_kb_version.py -> _expire_base_record_for_target()
        -> MilvusHybridStore.expire_documents_for_version()。
        """
        if not ids:
            return 0
        if valid_to_seq <= 0:
            raise ValueError("valid_to_seq must be a positive version sequence")

        def expire_metadata(metadata: dict[str, Any]) -> bool:
            """将旧 chunk 标记为从目标版本开始失效。

            设置 valid_to_seq 为目标版本序号，使该 chunk 对目标版本不可见。

            参数：
                metadata: chunk 的 Milvus metadata 字典（原地修改）。

            返回：
                始终返回 True，表示该 chunk 需要被更新。

            调用顺序：expire_documents_for_version() -> expire_metadata()。
            """
            # 只改变可见性窗口右边界，不触碰正文和向量；旧版本仍可通过更小
            # active_seq 命中这条物理记录。
            metadata["valid_to_seq"] = int(valid_to_seq)
            return True

        # 通过统一的 metadata 更新函数按主键批处理，避免重新 embedding，
        # 也避免把基线 chunk 物理复制成一份目标版本副本。
        return self._update_document_metadata(
            ids,
            batch_size=batch_size,
            action="引用式增量失效旧 chunk",
            update_metadata=expire_metadata,
        )

    def _update_document_metadata(
        self,
        ids: list[str],
        *,
        batch_size: int,
        action: str,
        update_metadata: Callable[[dict[str, Any]], bool],
    ) -> int:
        """按原主键重写已有 chunk 的 metadata，供版本有效期维护复用。

        Milvus 当前路径不能只更新动态 metadata 的单个字段，所以这里统一处理：
        查询旧行 -> 保留 text/dense/原 metadata -> 改 metadata -> 用原 pk 写回。

        该函数故意不重新计算 embedding：版本失效只是治理状态变化，不是内容变化。
        重新 embedding 会增加成本，还可能让同一个 chunk 在不同版本出现无意义的
        向量差异。
        """
        collection = self.store.col
        # 固定字段：不能被 update_metadata 修改的内置字段
        fixed_fields = {"pk", "text", "dense", "sparse"}
        updated = 0
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            # 查询旧行：获取 pk、text、dense 和全部动态 metadata 字段。
            # 需要保留 dense，因为更新 metadata 后仍要用原向量写回。
            rows = collection.query(expr=_pk_in_expr(batch), output_fields=["pk", "text", "dense", "*"])
            # 构建 pk → row 快速查找表，保证按 batch 顺序处理
            rows_by_pk = {str(row.get("pk")): row for row in rows}
            # 缺失 ID 为硬错误：manifest 和 Milvus 必须一致；静默跳过会造成
            # MySQL 认为已失效、Milvus 却仍可见的版本治理裂缝。
            missing = [item for item in batch if item not in rows_by_pk]
            if missing:
                raise RuntimeError(f"{action} 失败：集合 {self.collection_name} 缺少旧 chunk：{missing[:5]}")
            insert_rows: list[dict] = []
            for source_id in batch:
                source_row = rows_by_pk[source_id]
                # 保留原始 text 和 dense 向量不变，只修改 metadata
                text = str(source_row.get("text") or "")
                dense = source_row.get("dense")
                if dense is None:
                    raise RuntimeError(
                        f"{action} 失败：集合 {self.collection_name} 的旧 chunk 缺少 dense 向量：{source_id}"
                    )
                # 提取动态 metadata（排除固定字段），交给回调函数只修改治理字段。
                metadata = _row_dynamic_metadata(source_row, fixed_fields)
                # update_metadata 返回 True 时表示该行需要更新，加入批量写回列表。
                if update_metadata(metadata):
                    insert_rows.append({"pk": source_id, "text": text, "dense": dense, **metadata})
            if insert_rows:
                # 优先使用 upsert（Milvus 2.4+）原地覆盖，保留原主键和向量。
                if hasattr(collection, "upsert"):
                    collection.upsert(insert_rows)
                else:
                    # 旧版 Milvus 不支持 upsert：先 delete 再 insert
                    # 注意：delete + insert 会重建向量索引，比 upsert 慢
                    if not self.delete_ids([str(row["pk"]) for row in insert_rows]):
                        raise RuntimeError(f"{action} 失败：集合 {self.collection_name} 删除旧行失败")
                    collection.insert(insert_rows)
                updated += len(insert_rows)
        return updated

    def delete_ids_for_kb_version(self, ids: list[str], kb_version: str, *, batch_size: int = 200) -> bool:
        """只删除 metadata.kb_version 属于目标版本的行，避免误删被引用的基线 chunk。

        同一个 chunk ID 可能被目标版本 manifest 引用，也可能物理上属于基线
        版本。重建目标版本文件时只能删除目标版本自己写入的行，不能把仍被
        其他版本使用的基线行一并删除，因此这里先按主键读取 metadata，再按
        `kb_version` 精确筛选。

        调用顺序：检索准备或检索执行 -> MilvusHybridStore.delete_ids_for_kb_version()。
        """
        if not ids:
            return True
        owned_ids: list[str] = []
        collection = self.store.col
        fixed_fields = {"pk", "text", "dense", "sparse"}
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            # 不能直接 delete batch：其中可能包含目标 manifest 引用的基线行。
            rows = collection.query(expr=_pk_in_expr(batch), output_fields=["pk", "*"])
            for row in rows:
                metadata = _row_dynamic_metadata(row, fixed_fields)
                if str(metadata.get("kb_version") or "") == kb_version:
                    owned_ids.append(str(row.get("pk")))
        return self.delete_ids(owned_ids)

    def delete_ids(self, ids: list[str]) -> bool:
        """按主键删除文档，供增量重建使用（先删旧 chunk 再写新 chunk）。

        增量重建时，文件变化后需要先删除旧 chunk，再写入新 chunk。

        参数：
            ids: 待删除主键列表。

        返回：
            删除成功返回 True；ids 为空也视为成功.

        调用顺序：检索准备或检索执行 -> MilvusHybridStore.delete_ids()。
        """
        if not ids:
            return True
        try:
            # 删除动作只用于目标版本旧 chunk 的重建清理；跨版本基线 chunk 使用
            # valid_to_seq 收口，不走这里的物理删除路径。
            return bool(self.store.delete(ids=ids))
        except MilvusException as exc:
            if "collection not found" in str(exc).lower():
                logger.warning(
                    "跳过按 ID 删除：集合 %s 尚不存在，无需清理。", self.collection_name
                )
                return True
            raise

    def delete_by_expr(self, expr: str) -> bool:
        """按 Milvus 布尔表达式批量删除文档。

        适合按 source、kb_version 或数据域批量清理，不需要逐个 chunk_id 删除。

        参数：
            expr: Milvus boolean expr，例如 ``source == "hr"``。

        返回：
            删除成功返回 True；expr 为空也视为成功.

        调用顺序：检索准备或检索执行 -> MilvusHybridStore.delete_by_expr()。
        """
        if not expr:
            return True
        try:
            # 按布尔表达式批量删除文档
            return bool(self.store.delete(expr=expr))
        except MilvusException as exc:
            if "collection not found" in str(exc).lower():
                logger.warning(
                    "跳过按表达式删除：集合 %s 尚不存在，无需清理。", self.collection_name
                )
                return True
            raise

    def search(
        self,
        query: str,
        *,
        k: int,
        source_filter: str | None,
        kb_version: str | None = None,
        data_scope: DataScope | None = None,
        scenario_id: str | None = None,
        source_type: Literal["faq", "doc"],
        rerank: bool = True,
    ) -> RetrievalResult:
        """执行一次 Milvus 混合检索并按需重排，使用 weighted ranker 融合稠密+稀疏召回。（★★★ 核心）

        Dense+Sparse 融合是核心技术：稠密向量捕捉语义相似，稀疏 BM25 捕捉关键词精确匹配，二者互补。

        执行流程：
          1. 开始计时。
          2. 根据 source_filter、kb_version 和 data_scope 构造 Milvus expr。
          3. 调用 similarity_search_with_score，使用 weighted ranker 融合 dense/sparse。
          4. 将原始结果转换成 RetrievalHit。
          5. rerank=True 时调用 CrossEncoder 二阶段重排。
          6. 包装成 RetrievalResult 返回。

        参数：
            query: 用户检索问题。
            k: 从 Milvus 初始召回的数量。
            source_filter: 业务分类过滤项。
            kb_version: 知识库版本过滤项。
            data_scope: 租户、数据集、可见级别和角色过滤。
            source_type: faq 或 doc，用于标记检索来源类型。
            rerank: 是否启用 CrossEncoder 重排。

        返回：
            RetrievalResult，包含命中列表、查询文本、来源类型和耗时。
        """
        started = time.perf_counter()
        clean_query = query.strip()
        if not clean_query or k <= 0:
            return RetrievalResult(
                query=clean_query,
                source_type=source_type,
                elapsed_ms=_elapsed_ms(started),
            )
        # 将 source、kb_version、tenant_id、dataset_id、visibility、allowed_roles 等合并为 Milvus 布尔表达式
        expr = build_source_expr(
            source_filter,
            kb_version,
            data_scope,
            scenario_id=scenario_id,
            source_type=source_type,
        )
        # 执行 Milvus 稠密+稀疏混合检索，使用 weighted ranker 融合两路召回结果
        raw_hits = self._similarity_search_with_score(clean_query, k=k, expr=expr)

        # 转成内部 RetrievalHit 格式，隔离上层对 langchain-milvus 的依赖
        hits = _to_hits(raw_hits)
        if rerank and hits:
            # 二阶段 CrossEncoder 重排精度远高于向量相似度（约+10~15%），但计算成本高，仅按需启用
            hits = self._rerank(clean_query, hits)
        return RetrievalResult(
            hits=hits,
            query=clean_query,
            source_type=source_type,
            elapsed_ms=_elapsed_ms(started),
        )

    def _similarity_search_with_score(self, query: str, *, k: int, expr: str) -> list[tuple[Document, float]]:
        """执行 Milvus 混合检索；集合 schema 异常时明确提示重建。

        调用顺序：检索准备或检索执行 -> MilvusHybridStore._similarity_search_with_score()。
        """
        try:
            return self.store.similarity_search_with_score(query, k=k, expr=expr, **HYBRID_RANKER_KWARGS)
        except MilvusException as exc:
            message = str(exc)
            is_empty_query_vector_error = "nq [0] is invalid" in message or (
                "number of search vector" in message and "got 0" in message
            )
            if not is_empty_query_vector_error:
                raise
            raise RuntimeError(
                f"Milvus Hybrid Search 收到空 query-vector 请求（nq=0），collection={self.collection_name!r}。"
                "根因通常是 collection schema 没有正确的 BM25 Function，导致 sparse 字段不能把 query "
                "文本转换成稀疏向量。请使用 --reset-collections 删除旧集合并重新入库，不能用 dense-only "
                "降级替代 Hybrid Search。"
            ) from exc

    def search_many(
        self,
        queries: list[str],
        *,
        k: int,
        source_filter: str | None,
        kb_version: str | None = None,
        data_scope: DataScope | None = None,
        scenario_id: str | None = None,
        source_type: Literal["faq", "doc"],
        rerank: bool = True,
    ) -> RetrievalResult:
        """搜索多个查询变体并合并重复 chunk 命中，减少 CrossEncoder 重排次数。（★★ 理解）

        多变体搜索的核心业务价值：用户同一个问题有多种表述，变体可提升召回率；合并后统一重排避免 N 倍计算成本。

        执行流程：
          1. 开始计时。
          2. 清洗并去重查询变体。
          3. 每个变体先以 rerank=False 检索，避免每个变体单独重排。
          4. 按 chunk_id/faq_id 合并重复命中，同一 chunk 保留最高分。
          5. 合并结果按分数排序；启用 rerank 时先限制候选量，再用原问题统一重排。
          6. 截断为 k 条并包装 RetrievalResult。

        参数：
            queries: 查询变体列表，第一条应为用户原问题。
            k: 合并和重排后返回的数量。
            source_filter: 业务分类过滤项。
            kb_version: 知识库版本过滤项。
            data_scope: 数据隔离过滤。
            source_type: faq 或 doc。
            rerank: 是否启用 CrossEncoder 重排。

        返回：
            RetrievalResult，包含去重、重排后的命中列表和耗时。
        """
        started = time.perf_counter()
        merged: dict[str, RetrievalHit] = {}
        # 清洗查询变体列表：去空白、去空串、按顺序去重
        searched_queries = normalize_queries(queries)
        for clean_query in searched_queries:
            # 单变体先不做重排：同一批候选被多个变体重复召回时，合并后再统一重排比分别重排节省 N-1 倍计算
            result = self.search(
                clean_query,
                k=k,
                source_filter=source_filter,
                kb_version=kb_version,
                data_scope=data_scope,
                scenario_id=scenario_id,
                source_type=source_type,
                rerank=False,
            )
            # 按稳定 key（chunk_id/faq_id）合并同一文档的多次命中，只保留分数更高的那次 —— 防止同一个 chunk 在最终上下文中重复出现
            merge_hits_by_document(merged, result.hits)
        # 按分数从高到低排序候选结果
        hits = sort_hits_by_score(merged.values())
        if rerank and hits:
            hits = hits[:self._rerank_candidate_limit(searched_queries)]
        if rerank and hits:
            # 用原始问题（首个变体）统一做 CrossEncoder 相关性打分，避免每个变体单独重排造成 N 倍开销
            hits = self._rerank(searched_queries[0], hits)
        return RetrievalResult(
            hits=hits[:k],
            query=" | ".join(searched_queries),
            source_type=source_type,
            elapsed_ms=_elapsed_ms(started),
        )

    def _rerank_candidate_limit(self, searched_queries: list[str]) -> int:
        """计算多变体合并后进入 CrossEncoder 的候选上限。

        调用顺序：检索准备或检索执行 -> MilvusHybridStore._rerank_candidate_limit()。
        """
        return max(self.settings.rerank_top_n * max(len(searched_queries), 1), self.settings.rerank_top_n)

    def _rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """使用 CrossEncoder 对候选结果二阶段排序，比向量相似度更精准。（★★ 理解）

        CrossEncoder 是检索链的精度瓶颈：一对一问+答打分，效果最好但最慢，所以只在候选数减少后才调用。

        参数：
            query: 用于相关性打分的原始问题。
            hits: 待重排候选列表。

        返回：
            按 rerank_top_n 限制后的重排结果。

        调用顺序：检索准备或检索执行 -> MilvusHybridStore._rerank()。
        """
        return rerank_hits(
            query,
            hits,
            reranker=get_reranker(),
            top_n=self.settings.rerank_top_n,
        )



