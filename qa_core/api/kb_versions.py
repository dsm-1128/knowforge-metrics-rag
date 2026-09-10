"""知识库版本管理路由。

只管理本地版本清单，不直接删除 Milvus 数据。包含以下路由：
- GET /api/kb_versions: 查看版本清单和当前生效版本。
- POST /api/kb_versions/{kb_version}/activate: 激活指定版本（需 admin 令牌）。
- POST /api/kb_versions/{kb_version}/archive: 归档非 active 版本（需 admin 令牌）。

设计原则：
- 版本激活/归档操作需 admin 令牌认证（require_admin_token）。
- 归档版本不能直接激活，需先反归档（通过 DBA 操作）。
- 新版本发布必须通过脚本（rebuild_kb_version.py），管理接口只用于回滚已发布版本。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from qa_core.api.dependencies import require_admin_token
from qa_core.api.error_handlers import raise_bad_request, raise_not_found
from qa_core.governance.kb_versions import get_kb_version_store
router = APIRouter()


class ActivateVersionRequest(BaseModel):
    """版本激活请求的可选审计元数据。

    HTTP 路由 POST /api/kb_versions/{kb_version}/activate
    的请求体模型，记录激活原因和操作者。

    参数：
        reason: 激活原因描述，默认空字符串。
        activated_by: 操作者标识，默认 "admin"。

    调用顺序：FastAPI 路由层 -> ActivateVersionRequest。
    """

    reason: str = ""
    activated_by: str = "admin"


@router.get("/api/kb_versions")
def list_kb_versions(scenario_id: str | None = None):
    """返回知识库版本清单和当前生效版本。

    HTTP 路由 GET /api/kb_versions

    参数：
        scenario_id: 可选，按场景 ID 过滤。

    返回：
        版本管理视图 payload，包含 active 指针、版本列表和激活历史。

    调用顺序：FastAPI 路由层 -> list_kb_versions()。
    """
    return get_kb_version_store(scenario_id).as_payload()

@router.post("/api/kb_versions/{kb_version}/activate")
def activate_kb_version(
    kb_version: str,
    scenario_id: str | None = None,
    payload: ActivateVersionRequest | None = None,
    _: None = Depends(require_admin_token),
):
    """把已发布过的历史版本切回 active。

    HTTP 路由 POST /api/kb_versions/{kb_version}/activate
    需要 admin 令牌认证。

    限制：
    - 归档版本不能直接激活。
    - 新版本发布必须通过 scripts/rebuild_kb_version.py --quality-gate --activate。
    - 管理接口只用于回滚已发布版本。

    参数：
        kb_version: 要激活的版本号（路径参数）。
        scenario_id: 可选，场景 ID。
        payload: 可选的激活审计元数据（操作原因和操作者）。

    返回：
        包含 success 状态和激活后版本对象的字典。

    异常：
        HTTP 404: 版本不存在。
        HTTP 400: 归档版本、新版本或其他非法操作。

    调用顺序：FastAPI 路由层 -> activate_kb_version()。
    """
    request_payload = payload or ActivateVersionRequest()
    store = get_kb_version_store(scenario_id)
    record = store.get(kb_version)
    if record is None:
        raise_not_found(f"知识库版本不存在：{kb_version}")
    if record.status == "ARCHIVED":
        raise_bad_request("归档版本不能直接激活")
    if not record.activated_at:
        raise_bad_request("新版本发布必须通过 scripts/rebuild_kb_version.py --quality-gate --activate；管理接口只用于回滚已发布版本")
    try:
        version = store.activate_version(
            kb_version,
            reason=request_payload.reason or "admin_rollback",
            activated_by=request_payload.activated_by or "admin",
        )
        return {"status": "success", "version": version.as_dict()}
    except ValueError as exc:
        raise_not_found(str(exc))

@router.post("/api/kb_versions/{kb_version}/archive")
def archive_kb_version(kb_version: str, scenario_id: str | None = None, _: None = Depends(require_admin_token)):
    """归档一个非 active 知识库版本。

    HTTP 路由 POST /api/kb_versions/{kb_version}/archive
    需要 admin 令牌认证。

    限制：
    - 不能归档当前 active 版本。
    - 归档只改变控制面状态，不删除 Milvus 数据。

    参数：
        kb_version: 要归档的版本号（路径参数）。
        scenario_id: 可选，场景 ID。

    返回：
        包含 success 状态和归档后版本对象的字典。

    异常：
        HTTP 400: 版本不存在或不能归档 active 版本。

    调用顺序：FastAPI 路由层 -> archive_kb_version()。
    """
    try:
        version = get_kb_version_store(scenario_id).archive_version(kb_version)
        return {"status": "success", "version": version.as_dict()}
    except ValueError as exc:
        raise_bad_request(str(exc))
