"""轻量管理后台 API 路由。

设计原则：本地 admin API 只承担项目治理视图，LangSmith 作为可选 Trace 旁路：
  - 服务状态和场景配置概览。
  - 本地质量报告（入库质量、门禁报告、性能报告）。
  - 知识库版本管理信息。
  - 企业治理报告。
  - 低质量用户反馈（bad cases）。

不在此处重新实现完整追踪存储、标注队列或评估面板；本地质量闭环使用报告、eval_sets 和 Gate。
所有路由均通过 require_admin_token 依赖注入进行管理令牌认证。

路由列表：
- GET /api/admin/status                     — 管理后台概览状态
- GET /api/admin/cache/status               — 缓存运行状态
- GET /api/admin/intent_model               — 意图模型治理状态
- GET /api/admin/llm                        — LLM 运行状态
- POST /api/admin/cache/invalidate          — 推进缓存 epoch 失效
- GET /api/admin/langsmith                  — LangSmith 集成状态
- GET /api/admin/ingestion_reports          — 入库质量报告列表
- GET /api/admin/ingestion_report_detail    — 入库质量报告详情
- GET /api/admin/kb_version_compare         — 版本比较报告指针
- GET /api/admin/gate_reports               — 质量门禁报告指针
- GET /api/admin/performance_reports        — 性能报告指针
- GET /api/admin/enterprise_governance      — 企业治理报告指针
- GET /api/admin/report_detail              — 验证报告详情
- GET /api/admin/bad_feedback               — 低质量用户反馈列表
"""

from __future__ import annotations
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from qa_core.api.dependencies import require_admin_token
from qa_core.api.error_handlers import raise_not_found
from qa_core.cache.manager import get_cache_manager
from qa_core.common import read_json_dict
from qa_core.config.settings import PROJECT_ROOT
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.intent.governance import latest_intent_model_report
from qa_core.llm.client import llm_runtime_status, refresh_llm_runtime_status
from qa_core.memory.feedback import get_feedback_store
from qa_core.observability.langsmith_adapter import langsmith_status
from qa_core.quality.ingestion import list_ingestion_reports
from qa_core.scenarios.registry import get_scenario_registry

router = APIRouter()
# 验证报告存放目录
REPORT_DIR = PROJECT_ROOT / "reports" / "verification"
# 入库质量报告存放目录
INGESTION_REPORT_DIR = PROJECT_ROOT / "reports" / "ingestion"


class CacheInvalidateRequest(BaseModel):
    """缓存失效请求体。

    HTTP 路由 POST /api/admin/cache/invalidate 的请求体。

    参数：
        scenario_id: 要使缓存失效的业务场景 ID。

    调用顺序：FastAPI 路由层 -> CacheInvalidateRequest。
    """

    scenario_id: str


def _latest_file_summary(pattern: str) -> dict[str, Any]:
    """获取指定 glob 模式中最新文件的摘要信息。

    在 REPORT_DIR 下搜索匹配最新修改的文件，返回文件路径和更新时间。
    如果没有任何文件匹配，返回 available=False。

    参数:
        pattern: glob 匹配模式，例如 "*gate*_latest.json"。

    返回:
        包含 available（是否存在）、file（相对路径）和 updated_at（修改时间）的字典。

    调用顺序：管理后台 API 路由 -> _latest_file_summary()。
    """
    candidates = sorted(REPORT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {"available": False, "file": None}
    path = candidates[0]
    return {
        "available": True,
        "file": str(path.relative_to(PROJECT_ROOT)),
        "updated_at": path.stat().st_mtime,
    }


def _resolve_project_file(raw_path: str, *, root: Any = PROJECT_ROOT) -> Any:
    """解析并验证项目内的文件路径。（★★ 理解）

    安全校验：确保解析后的路径在项目根目录内，且指向一个真实存在的文件。
    防止路径穿越攻击（Path Traversal）。

    参数:
        raw_path: 原始文件路径字符串（相对或绝对）。
        root: 根目录，用于路径安全性校验。默认为 PROJECT_ROOT。

    返回:
        解析后的 Path 对象。

    异常:
        HTTPException(404): 如果文件不在项目根目录内或文件不存在。

    调用顺序：管理后台 API 路由 -> _resolve_project_file()。
    """
    candidate = (root / raw_path).resolve() if not str(raw_path).startswith(str(root)) else PROJECT_ROOT.__class__(raw_path).resolve()
    root_path = root.resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError:
        raise_not_found("报告文件不存在")
    if not candidate.exists() or not candidate.is_file():
        raise_not_found("报告文件不存在")
    return candidate


@router.get("/api/admin/status")
def admin_status(_: None = Depends(require_admin_token)):
    """获取管理后台概览状态。

    HTTP 路由 GET /api/admin/status
    需要 admin 令牌认证。

    返回当前已注册的场景列表、每个场景的活跃知识库版本号，
    以及缓存、意图模型、LLM、LangSmith 集成状态。

    返回:
        包含 status、scenarios、active_kb_versions、cache、intent_model、llm、langsmith 的字典。

    调用顺序：FastAPI 路由层 -> admin_status()。
    """
    registry = get_scenario_registry()
    scenario_ids = [scenario.scenario_id for scenario in registry.list_scenarios()]
    # 收集每个场景的活跃 KB 版本
    active_versions: dict[str, str] = {}
    for scenario_id in scenario_ids:
        versions = get_kb_version_store(scenario_id).list_versions()
        active = next((item for item in versions if item.status == "ACTIVE"), None)
        active_versions[scenario_id] = active.kb_version if active else ""
    return {
        "status": "ok",
        "scenarios": scenario_ids,
        "active_kb_versions": active_versions,
        "cache": get_cache_manager().status(scenario_id=None),
        "intent_model": latest_intent_model_report(),
        "llm": llm_runtime_status(),
        "langsmith": langsmith_status(),
    }


@router.get("/api/admin/cache/status")
def admin_cache_status(scenario_id: str | None = None, _: None = Depends(require_admin_token)):
    """返回缓存运行状态。

    HTTP 路由 GET /api/admin/cache/status
    需要 admin 令牌认证。

    参数：
        scenario_id: 可选，按场景 ID 过滤。

    返回：
        缓存管理器返回的状态字典，包含 L1/L2 缓存命中率和容量。

    调用顺序：FastAPI 路由层 -> admin_cache_status()。
    """
    return get_cache_manager().status(scenario_id=scenario_id)


@router.get("/api/admin/intent_model")
def admin_intent_model(_: None = Depends(require_admin_token)):
    """返回意图识别模型治理状态。

    HTTP 路由 GET /api/admin/intent_model
    需要 admin 令牌认证。

    返回：
        最近一期的意图模型评估报告。

    调用顺序：FastAPI 路由层 -> admin_intent_model()。
    """
    return latest_intent_model_report()


@router.get("/api/admin/llm")
def admin_llm(refresh: bool = False, _: None = Depends(require_admin_token)):
    """返回 LLM 运行状态；refresh=true 时主动重新探测一次。

    HTTP 路由 GET /api/admin/llm
    需要 admin 令牌认证。

    参数：
        refresh: 是否强制重新探测 LLM 状态。

    返回：
        LLM 运行时状态字典（包含可用性、延迟等）。

    调用顺序：FastAPI 路由层 -> admin_llm()。
    """
    if refresh:
        return refresh_llm_runtime_status()
    return llm_runtime_status()


@router.post("/api/admin/cache/invalidate")
def admin_cache_invalidate(payload: CacheInvalidateRequest, _: None = Depends(require_admin_token)):
    """推进场景缓存 epoch，使旧 Redis key 立即失效。

    HTTP 路由 POST /api/admin/cache/invalidate
    需要 admin 令牌认证。

    参数：
        payload: CacheInvalidateRequest，包含要失效的场景 ID。

    返回：
        包含 status、scenario_id 和 affected_namespaces 的字典。

    调用顺序：FastAPI 路由层 -> admin_cache_invalidate()。
    """
    affected = get_cache_manager().invalidate_scenario(payload.scenario_id)
    return {
        "status": "success",
        "scenario_id": payload.scenario_id,
        "affected_namespaces": affected,
    }


@router.get("/api/admin/langsmith")
def admin_langsmith(_: None = Depends(require_admin_token)):
    """返回 LangSmith 配置和项目链接提示。

    HTTP 路由 GET /api/admin/langsmith
    需要 admin 令牌认证。

    用于在管理后台前端展示 LangSmith 集成卡片，
    方便开发者快速跳转到 LangSmith 项目页面。

    返回:
        LangSmith 适配器返回的状态字典。

    调用顺序：FastAPI 路由层 -> admin_langsmith()。
    """
    return langsmith_status()


@router.get("/api/admin/ingestion_reports")
def admin_ingestion_reports(limit: int = 20, scenario_id: str | None = None, _: None = Depends(require_admin_token)):
    """获取最近的本地入库质量报告列表。

    HTTP 路由 GET /api/admin/ingestion_reports
    需要 admin 令牌认证。

    参数:
        limit: 最大返回条数，默认 20。
        scenario_id: 可选，按场景 ID 过滤。

    返回:
        包含报告列表的字典。

    调用顺序：FastAPI 路由层 -> admin_ingestion_reports()。
    """
    return {"reports": list_ingestion_reports(scenario_id=scenario_id, limit=limit)}


@router.get("/api/admin/ingestion_report_detail")
def admin_ingestion_report_detail(path: str, _: None = Depends(require_admin_token)):
    """获取单条入库质量报告的详细内容。

    HTTP 路由 GET /api/admin/ingestion_report_detail
    需要 admin 令牌认证，且只允许访问 INGESTION_REPORT_DIR 内的文件。

    参数:
        path: 报告文件的路径（需在 INGESTION_REPORT_DIR 内）。

    返回:
        包含文件路径、文件名和完整 payload 的字典。

    调用顺序：FastAPI 路由层 -> admin_ingestion_report_detail()。
    """
    # 安全解析并校验路径：确保文件在 INGESTION_REPORT_DIR 内，防止路径穿越
    report_path = _resolve_project_file(path, root=INGESTION_REPORT_DIR)
    payload = read_json_dict(report_path)
    if not payload:
        raise_not_found("入库质量报告不存在或内容为空")
    return {
        "path": str(report_path.relative_to(PROJECT_ROOT)),
        "file_name": report_path.name,
        "payload": payload,
    }


@router.get("/api/admin/kb_version_compare")
def admin_kb_version_compare(_: None = Depends(require_admin_token)):
    """返回知识库版本比较报告的指针。

    HTTP 路由 GET /api/admin/kb_version_compare
    需要 admin 令牌认证。

    仅返回报告文件的元数据指针（可用性、路径、更新时间），
    不返回报告内容本身。具体内容通过 /api/admin/report_detail 获取。

    返回:
        包含报告类型和文件摘要信息的字典。

    调用顺序：FastAPI 路由层 -> admin_kb_version_compare()。
    """
    return {
        "report_type": "kb_version_compare",
        "comparison": _latest_file_summary("kb_version_compare*_latest.json"),
        "langsmith": langsmith_status(),
    }


@router.get("/api/admin/gate_reports")
def admin_gate_reports(_: None = Depends(require_admin_token)):
    """返回质量门禁报告的指针列表。

    HTTP 路由 GET /api/admin/gate_reports
    需要 admin 令牌认证。

    返回:
        包含门禁报告文件摘要的字典。

    调用顺序：FastAPI 路由层 -> admin_gate_reports()。
    """
    return {"reports": [_latest_file_summary("*gate*_latest.json")], "langsmith": langsmith_status()}


@router.get("/api/admin/performance_reports")
def admin_performance_reports(_: None = Depends(require_admin_token)):
    """返回性能报告的指针列表。

    HTTP 路由 GET /api/admin/performance_reports
    需要 admin 令牌认证。

    返回:
        包含性能报告文件摘要的字典。

    调用顺序：FastAPI 路由层 -> admin_performance_reports()。
    """
    return {"reports": [_latest_file_summary("*performance*_latest.json")], "langsmith": langsmith_status()}


@router.get("/api/admin/enterprise_governance")
def admin_enterprise_governance(_: None = Depends(require_admin_token)):
    """返回企业治理报告的指针列表。

    HTTP 路由 GET /api/admin/enterprise_governance
    需要 admin 令牌认证。

    包含数据真实性报告和覆盖准备度报告的可用性信息。
    用于管理后台的治理概览页面。

    返回:
        包含各治理报告文件摘要的字典。

    调用顺序：FastAPI 路由层 -> admin_enterprise_governance()。
    """
    return {
        "report_type": "enterprise_governance",
        "data_realism": _latest_file_summary("enterprise_data_realism_latest.json"),
        "overlay_readiness": _latest_file_summary("enterprise_overlay_readiness_latest.json"),
        "langsmith": langsmith_status(),
    }


@router.get("/api/admin/report_detail")
def admin_report_detail(path: str, _: None = Depends(require_admin_token)):
    """获取单条验证报告的详细内容。

    HTTP 路由 GET /api/admin/report_detail
    需要 admin 令牌认证，且只允许访问 REPORT_DIR 内的文件。

    参数:
        path: 报告文件的路径（需在 REPORT_DIR 内）。

    返回:
        包含文件路径、文件名和完整 payload 的字典。

    调用顺序：FastAPI 路由层 -> admin_report_detail()。
    """
    report_path = _resolve_project_file(path, root=REPORT_DIR)
    payload = read_json_dict(report_path)
    if not payload:
        raise_not_found("验证报告不存在或内容为空")
    return {
        "path": str(report_path.relative_to(PROJECT_ROOT)),
        "file_name": report_path.name,
        "payload": payload,
    }


@router.get("/api/admin/bad_feedback")
def admin_bad_feedback(
    limit: int = 50,
    scenario_id: str | None = None,
    rating: str = "not_useful",
    _: None = Depends(require_admin_token),
):
    """获取低质量用户反馈（bad cases）列表。

    HTTP 路由 GET /api/admin/bad_feedback
    需要 admin 令牌认证。

    用于人工审查和模型改进的反馈数据收集。

    参数:
        limit: 最大返回条数，默认 50。
        scenario_id: 可选，按场景 ID 过滤反馈。
        rating: 按评分过滤，默认 "not_useful"（不有用）。

    返回:
        包含反馈条目、计数和过滤条件的字典。

    调用顺序：FastAPI 路由层 -> admin_bad_feedback()。
    """
    rows = get_feedback_store().list_bad_feedback(limit=limit, scenario_id=scenario_id, rating=rating)
    return {
        "items": rows,
        "count": len(rows),
        "rating": rating,
        "scenario_id": scenario_id,
    }
