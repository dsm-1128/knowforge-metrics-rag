"""检索层包：封装 Milvus 混合检索（稠密向量 + BM25 稀疏）的完整调用链。

包含模块：
- store.py：MilvusHybridStore，单个集合的混合检索适配器，统一封装 FAQ 和文档集合。
- factory.py：检索集合工厂与启动预热，管理 store 缓存和 collection 懒加载。
- models.py：检索链路模型加载器（BGE embedding、CrossEncoder reranker）。
- milvus_compat.py：Milvus 2.5+ BM25 内置函数与连接参数工具。
- filters.py：Milvus 布尔过滤表达式构造（source_filter、kb_version、DataScope）。
- ranking.py：检索候选合并去重、排序和 CrossEncoder 重排。
- results.py：检索结果对象（RetrievalHit / RetrievalResult）。
- strategy.py：检索计划构建（动态 top_k、阈值、规则补丁）。

调用方典型用法：from qa_core.retrieval.factory import get_faq_store, get_doc_store
"""

