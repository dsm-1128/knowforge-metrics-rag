"""账号验证、同源写请求与登录限流。"""
import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

COOKIE_NAME = "kf_session"


class RateLimiter:
    def __init__(self, limit=10, window=60):
        self.limit = limit
        self.window = window
        self.buckets = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self.lock:
            for old in list(self.buckets):
                if not self.buckets[old] or now - self.buckets[old][-1] >= self.window:
                    del self.buckets[old]
            bucket = self.buckets[key]
            while bucket and now - bucket[0] >= self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


def require_origin(request):
    if request.headers.get("origin") not in request.app.state.allowed_origins:
        raise HTTPException(403, "请求来源不受信任，请从平台页面操作")


def get_store(request):
    if request.app.state.auth_store is None:
        raise HTTPException(503, "账号数据库尚未就绪，请检查配置并运行初始化命令")
    return request.app.state.auth_store


def current_user(request: Request):
    store = get_store(request)
    user = store.authenticate(request.cookies.get(COOKIE_NAME, ""))
    if user is None:
        raise HTTPException(401, "请先登录或重新登录")
    return user


def write_user(request: Request):
    require_origin(request)
    user = current_user(request)
    if not hmac.compare_digest(request.headers.get("x-csrf-token", ""), user["csrf_token"]):
        raise HTTPException(403, "操作校验失败，请刷新页面后重试")
    return user


def require_admin(user):
    if user["role"] != "tenant_admin":
        raise HTTPException(403, "需要租户管理员权限")


def public_user(user):
    return {key: user[key] for key in ("id", "username", "display_name", "tenant_id", "role", "dataset_id", "visibility", "enabled")}
