from qa_core.prompts.metrics_dictionary import format_direct_answer, METRICS_DICTIONARY_RULES


def test_direct_answer_keeps_values_and_explains_terms():
    answer = format_direct_answer("演示口径v1：分母为0返回NULL；按user_id去重，使用Asia/Shanghai自然日。", has_sources=True)
    assert "NULL（空值）" in answer
    assert "user_id（用户标识）" in answer
    assert "Asia/Shanghai（中国标准时间）" in answer
    assert "分母为0" in answer
    assert all(title in answer for title in ["## 已确认", "## 需确认", "## 建议"])


def test_unsupported_answer_is_not_marked_confirmed():
    answer = format_direct_answer("信息不足，无法确认GMV。", has_sources=False)
    assert "## 已确认" not in answer
    assert "## 需确认" in answer
    assert "GMV（商品交易总额）" in answer


def test_unknown_identifiers_and_existing_annotations_stay_intact():
    answer = format_direct_answer("metric_catalog.csv：GMV（商品交易总额），foo_id。", has_sources=True)
    assert "metric_catalog.csv" in answer and "foo_id" in answer
    assert answer.count("商品交易总额") == 1


def test_prompt_requires_chinese_and_grounded_sections():
    assert "简体中文" in METRICS_DICTIONARY_RULES
    assert "## 已确认" in METRICS_DICTIONARY_RULES
    assert "不要为填满三栏编造内容" in METRICS_DICTIONARY_RULES
