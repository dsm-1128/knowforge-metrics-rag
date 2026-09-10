"""每次问答均验证账号和会话归属，再绑定服务端授权范围。"""
import asyncio
import hmac
import logging
from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import Field, ValidationError

from webapp.routes import StrictModel
from webapp.security import COOKIE_NAME

router = APIRouter()
logger = logging.getLogger(__name__)


class Query(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=64)
    source_filter: Literal["metrics", "dictionary", "versioning", "lineage"] | None = None


def next_item(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def claim_session(state, query, user):
    with state.session_lock:
        if state.auth_store.get_session(query.session_id, user["id"], user["tenant_id"]) is None:
            return "会话不存在或无权访问"
        if query.session_id in state.active_sessions or len(state.active_sessions) >= 4:
            return "正在处理问答，请稍后再试"
        state.active_sessions.add(query.session_id)
    return None


def release_session(state, session_id):
    with state.session_lock:
        state.active_sessions.discard(session_id)


async def ws_user(ws):
    store = ws.app.state.auth_store
    if store is None:
        return None
    user = await asyncio.to_thread(store.authenticate, ws.cookies.get(COOKIE_NAME, ""))
    if not user or not hmac.compare_digest(ws.query_params.get("csrf_token", ""), user["csrf_token"]):
        return None
    return user


@router.websocket("/api/stream")
async def stream(ws: WebSocket):
    if ws.headers.get("origin") not in ws.app.state.allowed_origins or not await ws_user(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    state = ws.app.state
    try:
        while True:
            raw = await ws.receive_text()
            user = await ws_user(ws)
            if not user:
                await ws.close(code=1008)
                return
            try:
                if len(raw) > 20000:
                    raise ValueError("request too large")
                query = Query.model_validate_json(raw)
                if not query.query.strip():
                    raise ValueError("empty query")
            except (ValidationError, ValueError):
                await ws.send_json({"type":"error","message":"问题参数无效；身份与权限字段不能由客户端传入"})
                continue
            owned = await asyncio.to_thread(state.auth_store.get_session, query.session_id, user["id"], user["tenant_id"])
            if owned is None:
                await ws.send_json({"type":"error","message":"会话不存在或无权访问"})
                continue
            if not state.query_limiter.allow(str(user["id"])):
                await ws.send_json({"type":"error","message":"请求过于频繁，请稍后再试"})
                continue
            if not state.runtime.ready:
                await ws.send_json({"type":"error","message":state.runtime.detail})
                continue
            claim_error = await asyncio.to_thread(claim_session, state, query, user)
            if claim_error:
                await ws.send_json({"type":"error","message":claim_error})
                continue
            iterator = None
            try:
                iterator = state.runtime.stream(query=query.query.strip(), source_filter=query.source_filter,
                    session_id=query.session_id, tenant_id=user["tenant_id"], dataset_id=user["dataset_id"],
                    visibility=user["visibility"], user_roles=["public", user["role"]])
                while True:
                    exists, event = await asyncio.to_thread(next_item, iterator)
                    if not exists:
                        break
                    if not await ws_user(ws):
                        await ws.close(code=1008)
                        return
                    # 用户界面无需接收内部路径、检索原始 metadata 或 Trace 细节。
                    if event.get("type") == "end":
                        event = {key:value for key,value in event.items() if key in {"type","session_id","answer","hit_type","is_complete","answer_confidence","sources","trace_id","processing_time"}}
                        event["sources"] = [{key:source[key] for key in ("citation","content","score","source_type") if key in source} for source in event.get("sources",[])]
                    elif event.get("type") == "error":
                        event = {"type":"error","message":"问答执行失败，请检查模型服务或联系管理员"}
                    elif event.get("type") == "start":
                        event = {"type":"start","session_id":query.session_id}
                    await ws.send_json(event)
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.warning("Question failed (%s)", type(exc).__name__)
                await ws.send_json({"type":"error","message":"问答服务暂不可用，请稍后再试"})
            finally:
                await asyncio.to_thread(release_session, state, query.session_id)
                if iterator is not None and hasattr(iterator, "close"):
                    await asyncio.to_thread(iterator.close)
    except WebSocketDisconnect:
        return
