"""知识库治理包。

提供知识库多版本管理、数据域隔离和 chunk 有效期追踪的治理能力。包含：
- kb_versions.py：知识库版本状态机，支持 STAGED -> ACTIVE -> ARCHIVED 生命周期。
- kb_version_models.py：版本数据模型和 JSON 序列化工具。
- chunk_versions.py：Chunk 有效期索引，支持引用式增量版本的可见性过滤。
- data_scope.py：数据域隔离模型，统一携带 tenant_id/dataset_id/visibility/角色信息。

请从具体子模块导入，例如 `from qa_core.governance.kb_versions import get_kb_version_store`。
"""