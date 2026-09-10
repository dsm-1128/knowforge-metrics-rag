"""受账号约束的业务 HTTP 接口，旧项目公开路由不在新应用挂载。"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from webapp.security import COOKIE_NAME, current_user, get_store, public_user, require_admin, require_origin, write_user

router = APIRouter(prefix="/api")
CATEGORIES = [
    {"id":"metrics","name":"指标口径","description":"业务定义、计算公式、统计粒度、时间范围与去重规则。","questions":["GMV的计算口径是什么？","支付转化率的分子和分母如何定义？","净支付金额与GMV有什么区别？"]},
    {"id":"dictionary","name":"数据字典","description":"表与字段释义、数据类型、枚举、空值及关联约束。","questions":["dwd_trade_order表的pay_amount字段含义是什么？","order_status枚举值有哪些？"]},
    {"id":"versioning","name":"版本与变更","description":"按业务日期核对生效口径，理解版本变化与冲突依据。","questions":["指标版本变更如何审批和生效？","知识库版本就是指标口径版本吗？"]},
    {"id":"lineage","name":"数据血缘","description":"查询已登记的来源表、字段依赖和下游影响。","questions":["数据血缘如何定位上游表？","pay_amount变更会影响哪些指标？"]},
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(StrictModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class SessionInput(StrictModel):
    title: str = Field(default="新会话", min_length=1, max_length=120)


class UserInput(StrictModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=256)
    role: Literal["viewer", "knowledge_admin", "tenant_admin"] = "viewer"
    dataset_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    visibility: Literal["public", "internal", "private"] = "public"


class UserState(StrictModel):
    enabled: bool


@router.post("/auth/login")
def login(payload: Login, request: Request, response: Response):
    require_origin(request)
    key = request.client.host if request.client else "unknown"
    if not request.app.state.login_limiter.allow(key):
        raise HTTPException(429, "登录尝试过多，请稍后重试")
    store = get_store(request)
    try:
        token, csrf, user = store.login(payload.username, payload.password)
    except ValueError:
        raise HTTPException(401, "账号或密码错误") from None
    old = request.cookies.get(COOKIE_NAME)
    if old:
        store.logout(old)
    response.set_cookie(COOKIE_NAME, token, httponly=True, secure=request.app.state.cookie_secure,
                        samesite="strict", max_age=28800, path="/")
    return {"user":public_user(user), "csrf_token":csrf}


@router.get("/auth/me")
def me(user=Depends(current_user)):
    return {"user":public_user(user), "csrf_token":user["csrf_token"]}


@router.post("/auth/logout")
def logout(request: Request, response: Response, user=Depends(write_user)):
    get_store(request).logout(request.cookies.get(COOKIE_NAME, ""))
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok":True}


@router.get("/categories")
def categories(user=Depends(current_user)):
    return {"categories":CATEGORIES}


@router.get("/status")
def status(request: Request, user=Depends(current_user)):
    runtime = request.app.state.runtime
    return {"rag_ready":runtime.ready, "detail":runtime.detail}


@router.get("/sessions")
def sessions(request: Request, user=Depends(current_user)):
    return {"sessions":get_store(request).list_sessions(user["id"], user["tenant_id"])}


@router.post("/sessions")
def create_session(payload: SessionInput, request: Request, user=Depends(write_user)):
    return get_store(request).create_session(user["id"], user["tenant_id"], payload.title)


def owned_session(request, user, session_id):
    session = get_store(request).get_session(session_id, user["id"], user["tenant_id"])
    if session is None:
        raise HTTPException(404, "会话不存在")
    return session


@router.get("/sessions/{session_id}/messages")
def messages(session_id: str, request: Request, user=Depends(current_user)):
    owned_session(request, user, session_id)
    try:
        return {"messages":request.app.state.runtime.messages(session_id)}
    except Exception:
        raise HTTPException(503, "历史存储未就绪，请先运行数据库初始化") from None


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request, user=Depends(write_user)):
    with request.app.state.session_lock:
        owned_session(request, user, session_id)
        if session_id in request.app.state.active_sessions:
            raise HTTPException(409, "会话正在生成回答，请稍后删除")
        try:
            request.app.state.runtime.clear(session_id)
        except Exception:
            raise HTTPException(503, "历史存储暂不可用，未删除会话") from None
        get_store(request).delete_session(session_id, user["id"], user["tenant_id"])
        return {"ok":True}



@router.get("/admin/users")
def users(request: Request, user=Depends(current_user)):
    require_admin(user)
    return {"users":get_store(request).list_users(user["tenant_id"])}


@router.post("/admin/users")
def create_user(payload: UserInput, request: Request, user=Depends(write_user)):
    require_admin(user)
    try:
        created = get_store(request).create_user(tenant_id=user["tenant_id"], **payload.model_dump())
    except (ValueError, IntegrityError):
        raise HTTPException(400, "账号已存在或账号信息不符合要求") from None
    return {"user":created}


@router.patch("/admin/users/{user_id}")
def update_user(user_id: int, payload: UserState, request: Request, user=Depends(write_user)):
    require_admin(user)
    if user_id == user["id"]:
        raise HTTPException(400, "不能停用当前登录账号")
    changed = get_store(request).set_enabled(user["tenant_id"], user_id, payload.enabled)
    if not changed:
        raise HTTPException(404, "账号不存在")
    return {"ok":True}
