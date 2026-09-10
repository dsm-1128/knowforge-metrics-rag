def test_rag_service_imports():
    from qa_core.pipeline.rag import stream_query
    from qa_core.application.factory import get_qa_service
    assert callable(stream_query)
    assert callable(get_qa_service)
