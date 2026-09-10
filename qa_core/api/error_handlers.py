"""API 层统一异常处理。

HTTP 路由里不再为每个服务调用重复编写 `try/except Exception`。统一处理的好处是：
- 路由函数只保留正常业务流程，可读性更高；
- `ValueError` 这类可预期的参数/业务校验错误统一返回 400；
- 未预期异常统一记录堆栈，但响应给前端时不暴露内部异常细节。

设计决策：
- 每个错误类型有独立的 raise_* 函数，语义清晰且便于全局搜索引用。
- WebSocket 不走 FastAPI 的 HTTP 异常响应机制，所以 WebSocket 端点仍需要自己发送
  `type=error` 事件。
- register_api_exception_handlers() 在 app 启动时注册，将 ValueError 转为 400、
  未预期 Exception 转为 500（日志记录完整堆栈，响应只返回通用文案）。
"""

from __future__ import annotations

from typing import NoReturn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from qa_core.config.logging_config import get_logger


logger = get_logger(__name__)


def raise_http_error(status_code: int, detail: str) -> NoReturn:
    """抛出带统一响应结构的 HTTPException。

    参数：
        status_code: HTTP 状态码（400/401/404/429/500）。
        detail: 错误描述文本。

    返回：
        NoReturn（函数总是抛出异常，不会正常返回）。

    调用顺序：FastAPI 路由层 -> raise_http_error()。
    """
    raise HTTPException(status_code=status_code, detail=detail)


def raise_bad_request(detail: str) -> NoReturn:
    """请求参数或业务条件不满足，返回 HTTP 400。

    参数：
        detail: 错误描述文本。

    调用顺序：FastAPI 路由层 -> raise_bad_request()。
    """
    raise_http_error(400, detail)


def raise_unauthorized(detail: str) -> NoReturn:
    """认证失败，返回 HTTP 401。

    参数：
        detail: 错误描述文本。

    调用顺序：FastAPI 路由层 -> raise_unauthorized()。
    """
    raise_http_error(401, detail)


def raise_not_found(detail: str) -> NoReturn:
    """目标资源不存在，返回 HTTP 404。

    参数：
        detail: 错误描述文本。

    调用顺序：FastAPI 路由层 -> raise_not_found()。
    """
    raise_http_error(404, detail)


def raise_too_many_requests(detail: str) -> NoReturn:
    """请求过于频繁，返回 HTTP 429。

    参数：
        detail: 错误描述文本。

    调用顺序：FastAPI 路由层 -> raise_too_many_requests()。
    """
    raise_http_error(429, detail)


def raise_server_error(detail: str) -> NoReturn:
    """服务端配置或运行状态不满足要求，返回 HTTP 500。

    参数：
        detail: 错误描述文本。

    调用顺序：FastAPI 路由层 -> raise_server_error()。
    """
    raise_http_error(500, detail)


def register_api_exception_handlers(app: FastAPI) -> None:
    """注册 HTTP API 的全局异常转换规则。（★★★ 核心）

    在 FastAPI app 启动时调用，注册两个异常处理器：
    1. ValueError -> 400 Bad Request：用于业务参数或白名单校验失败。
    2. Exception -> 500 Internal Server Error：未预期异常记录日志，前端只收通用文案。
       原因：不暴露内部异常细节（如堆栈、SQL、文件路径）给前端。

    参数：
        app: FastAPI 应用实例。

    调用顺序：app.py 启动入口 -> register_api_exception_handlers()。
    """

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        """业务参数或白名单校验失败，统一返回 HTTP 400。

        参数：
            _: FastAPI Request 对象（未使用）。
            exc: 捕获到的 ValueError 异常。

        返回：
            包含 error detail 的 JSONResponse（status_code=400）。

        调用顺序：FastAPI 路由层 -> value_error_handler()。
        """
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """未预期异常统一记录日志，前端只收到稳定的 500 文案。

        原因：不暴露内部异常细节（如堆栈、SQL、文件路径）给前端，
        同时确保运维人员可以通过日志完整排查。

        参数：
            request: FastAPI Request 对象（用于记录请求方法和路径）。
            exc: 捕获到的 Exception。

        返回：
            包含通用错误文案的 JSONResponse（status_code=500）。

        调用顺序：FastAPI 路由层 -> unexpected_error_handler()。
        """
        logger.exception("Unhandled API error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "服务内部错误，请查看后端日志"})
