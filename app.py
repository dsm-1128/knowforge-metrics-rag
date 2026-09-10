"""KnowForge 单场景应用；从当前项目独立启动，无旧 API 旁路。"""
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine

from qa_core.config.settings import get_settings
from webapp.auth_store import AuthStore
from webapp.rag_runtime import RagRuntime
from webapp.routes import router as routes
from webapp.security import RateLimiter
from webapp.stream import router as stream_routes

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def create_app(*, store=None, runtime=None, allowed_origins=None):
    supplied_store = store is not None
    supplied_runtime = runtime is not None

    @asynccontextmanager
    async def lifespan(application):
        if not supplied_store:
            try:
                settings = get_settings()
                engine = create_engine(settings.mysql_sync_uri, pool_pre_ping=True,
                                       connect_args={"connect_timeout":3})
                auth_store = AuthStore(engine)
                # Schema 创建由 init_project.py 显式完成；此处只检查表可读。
                await asyncio.to_thread(auth_store.list_users, "__startup_check__")
                application.state.auth_store = auth_store
            except Exception as exc:
                logger.warning("Account database unavailable (%s); run scripts/init_project.py", type(exc).__name__)
        task = None
        if not supplied_runtime and application.state.auth_store is not None:
            task = asyncio.create_task(asyncio.to_thread(application.state.runtime.initialize))
        yield
        if task and not task.done():
            task.cancel()

    application = FastAPI(title="KnowForge · 企业指标口径与数据字典", lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    application.state.auth_store = store
    application.state.runtime = runtime or RagRuntime()
    application.state.allowed_origins = allowed_origins or {item.strip() for item in get_settings().app_origins.split(",") if item.strip()}
    application.state.cookie_secure = get_settings().cookie_secure
    application.state.login_limiter = RateLimiter(10, 60)
    application.state.query_limiter = RateLimiter(30, 60)
    application.state.active_sessions = set()
    application.state.session_lock = threading.Lock()

    @application.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response

    @application.get("/")
    @application.get("/login")
    def index():
        return FileResponse(ROOT / "static/index.html")

    @application.get("/health")
    def health():
        return {"status":"running", "account_ready":application.state.auth_store is not None,
                "rag_ready":application.state.runtime.ready}

    @application.exception_handler(Exception)
    async def unhandled(request, exc):
        logger.warning("Request failed (%s)", type(exc).__name__)
        return JSONResponse({"detail":"服务暂不可用，请联系管理员检查配置"}, status_code=503)

    application.include_router(routes)
    application.include_router(stream_routes)
    (ROOT / "static").mkdir(exist_ok=True)
    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return application


app = create_app()
