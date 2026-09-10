-- Runtime MySQL schema for KnowForge RAG Platform.

CREATE TABLE IF NOT EXISTS {{KB_VERSIONS_TABLE}} (
    scenario_id VARCHAR(128) NOT NULL,
    kb_version VARCHAR(191) NOT NULL,
    version_seq BIGINT NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL,
    description TEXT NULL,
    created_at VARCHAR(40) NOT NULL,
    activated_at VARCHAR(40) NULL,
    archived_at VARCHAR(40) NULL,
    doc_collection VARCHAR(191) NOT NULL,
    faq_collection VARCHAR(191) NOT NULL,
    embedding_model_version VARCHAR(191) NOT NULL,
    reranker_model_version VARCHAR(191) NOT NULL,
    chunk_schema_version VARCHAR(191) NOT NULL,
    created_by VARCHAR(128) NOT NULL,
    sources_json LONGTEXT NULL,
    stats_json LONGTEXT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (scenario_id, kb_version),
    INDEX idx_kb_versions_status (scenario_id, status),
    INDEX idx_kb_versions_seq (scenario_id, version_seq),
    INDEX idx_kb_versions_created_at (scenario_id, created_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{KB_ACTIVE_TABLE}} (
    scenario_id VARCHAR(128) PRIMARY KEY,
    active_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    previous_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{KB_ACTIVATION_TABLE}} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(128) NOT NULL,
    from_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    to_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    from_version_seq BIGINT NOT NULL DEFAULT 0,
    to_version_seq BIGINT NOT NULL DEFAULT 0,
    action VARCHAR(32) NOT NULL,
    reason TEXT NULL,
    activated_by VARCHAR(128) NOT NULL DEFAULT 'system',
    created_at VARCHAR(40) NOT NULL,
    INDEX idx_kb_activation_scenario_created (scenario_id, created_at),
    INDEX idx_kb_activation_to_version (scenario_id, to_kb_version)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CACHE_NAMESPACE_TABLE}} (
    scenario_id VARCHAR(128) NOT NULL,
    tenant_id VARCHAR(128) NOT NULL DEFAULT 'default',
    dataset_id VARCHAR(128) NOT NULL DEFAULT 'default',
    cache_epoch BIGINT NOT NULL DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (scenario_id, tenant_id, dataset_id),
    INDEX idx_cache_namespace_updated_at (updated_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{KB_CHUNK_VERSIONS_TABLE}} (
    scenario_id VARCHAR(128) NOT NULL,
    chunk_id VARCHAR(191) NOT NULL,
    source VARCHAR(128) NOT NULL,
    kb_version VARCHAR(191) NOT NULL DEFAULT '',
    file_path TEXT NULL,
    valid_from_seq BIGINT NOT NULL DEFAULT 0,
    valid_to_seq BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (scenario_id, chunk_id),
    INDEX idx_chunk_version_source (scenario_id, source),
    INDEX idx_chunk_version_kb_version (scenario_id, kb_version),
    INDEX idx_chunk_version_validity (scenario_id, valid_from_seq, valid_to_seq)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{INDEX_MANIFEST_TABLE}} (
    manifest_key VARCHAR(64) PRIMARY KEY,
    scenario_id VARCHAR(128) NOT NULL,
    source VARCHAR(128) NOT NULL,
    path TEXT NOT NULL,
    fingerprint VARCHAR(191) NOT NULL,
    chunk_ids_json LONGTEXT NOT NULL,
    kb_version VARCHAR(191) NOT NULL,
    embedding_model_version VARCHAR(191) NOT NULL,
    chunk_schema_version VARCHAR(191) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    row_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_manifest_scenario_version (scenario_id, kb_version),
    INDEX idx_manifest_source (scenario_id, source),
    INDEX idx_manifest_updated_at (scenario_id, updated_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{FEEDBACK_TABLE}} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(191) NULL,
    scenario_id VARCHAR(128) NULL,
    tenant_id VARCHAR(128) NULL,
    dataset_id VARCHAR(128) NULL,
    question TEXT NOT NULL,
    answer MEDIUMTEXT NOT NULL,
    rating VARCHAR(32) NOT NULL,
    comment TEXT NULL,
    sources_json LONGTEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_scenario_id (scenario_id),
    INDEX idx_tenant_dataset (tenant_id, dataset_id),
    INDEX idx_rating (rating)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CHAT_SUMMARY_TABLE}} (
    session_id VARCHAR(191) PRIMARY KEY,
    summary TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CHAT_MESSAGES_TABLE}} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(191) NOT NULL,
    message LONGTEXT NOT NULL,
    INDEX idx_chat_messages_session_id (session_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
