"""User query normalization before intent and retrieval decisions."""

from __future__ import annotations

import re


GREETING_PREFIX = re.compile(
    r"^\s*(?:你好|您好|hi|hello|哈喽|在吗|在不在|有人吗)[啊呀呢嘛么]*(?:[\s,，:：;；!！。.?？]+)",
    re.IGNORECASE,
)
POLITE_PREFIX = re.compile(
    r"^\s*(?:"
    r"请问一下|请问|"
    r"我想问一下|想问一下|咨询一下|问一下|"
    r"麻烦(?:你|您)?(?:帮我)?(?:看下|看一下|查下|查一下|确认下|确认一下|处理下|处理一下)|"
    r"帮我(?:看下|看一下|查下|查一下|确认下|确认一下)|"
    r"帮忙(?:看下|看一下|查下|查一下|确认下|确认一下)"
    r")[\s,，:：;；!！。.?？]*",
    re.IGNORECASE,
)
LEADING_SEPARATORS = " \t\r\n,，:：;；!！。.?？"


def normalize_user_query(query: str) -> str:
    """Return the business-effective query while preserving pure greetings.

    用户常把礼貌开场和真实问题写在一起，例如“你好，请问新人入职流程有哪些？”。
    入口决策和检索应使用后半段业务问题；如果剥离后为空，说明它本身就是纯问候，
    此时返回原文，交给问候规则直接处理。

    调用顺序：QAService/RAG 管线 -> normalize_user_query()。
    """

    raw_query = (query or "").strip()
    if not raw_query:
        return ""

    normalized = re.sub(r"\s+", " ", raw_query).strip()
    for _ in range(4):
        next_query = GREETING_PREFIX.sub("", normalized, count=1)
        next_query = POLITE_PREFIX.sub("", next_query, count=1)
        next_query = next_query.lstrip(LEADING_SEPARATORS).strip()
        if next_query == normalized:
            break
        normalized = next_query

    return normalized or raw_query
