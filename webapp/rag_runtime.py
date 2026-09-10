"""独立项目的真实 RAG 适配层，无模拟问答或父目录导入。"""
import logging

logger = logging.getLogger(__name__)


class RagRuntime:
    def __init__(self):
        self.ready = False
        self.detail = "知识库未就绪，请配置模型服务并完成入库后重启"
        self.service = None
        self.history = None

    def initialize(self):
        try:
            from qa_core.config.preflight import validate_runtime_environment, validate_active_kb_versions
            from qa_core.storage.bootstrap import bootstrap_mysql_schema
            from qa_core.intent.decision import warmup_intent_decision_gateway
            from qa_core.retrieval.factory import start_retrieval_warmup_background, wait_for_retrieval_warmup
            from qa_core.application.factory import get_qa_service
            from qa_core.memory.history import get_history_store

            from qa_core.observability.langsmith_adapter import configure_langsmith_environment
            configure_langsmith_environment()
            validate_runtime_environment()
            bootstrap_mysql_schema()
            validate_active_kb_versions("metrics_dictionary")
            warmup_intent_decision_gateway()
            start_retrieval_warmup_background()
            wait_for_retrieval_warmup()
            self.service = get_qa_service()
            self.history = get_history_store()
            self.ready = True
            self.detail = "检索已就绪；模型供应商状态以实际问答结果为准"
        except Exception as exc:
            logger.warning("RAG initialization incomplete (%s)", type(exc).__name__)
            self.detail = "知识库未就绪：请检查 MySQL/Milvus、模型路径、API Key 及已激活版本，然后重启"

    def stream(self, **kwargs):
        if not self.ready:
            raise RuntimeError(self.detail)
        yield from self.service.stream_query(scenario_id="metrics_dictionary", **kwargs)

    def messages(self, session_id):
        if self.history is None:
            from qa_core.memory.history import get_history_store
            history = get_history_store()
        else:
            history = self.history
        return [{"role": "user" if msg.type == "human" else "assistant", "content": str(msg.content)}
                for msg in history.get_messages(session_id)]

    def clear(self, session_id):
        if self.history is None:
            from qa_core.memory.history import get_history_store
            get_history_store().clear(session_id)
        else:
            self.history.clear(session_id)
