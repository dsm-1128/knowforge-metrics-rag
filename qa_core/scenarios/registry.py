"""多业务场景注册与解析。

当前项目的 RAG 主链路是统一的，但每个业务场景有独立的 Milvus 集合、source 白名单、
资料目录和 FAQ 文件。新增或调整场景时，优先改 `scenario.toml`，
不要把业务分类硬编码进 Python 主链路。
"""

from __future__ import annotations

import re

import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa_core.config.settings import PROJECT_ROOT, get_settings


REQUIRED_SCENARIO_FIELDS = (
    "scenario_id",
    "display_name",
    "industry",
    "assistant_name",
    "business_domain",
    "support_contact",
    "valid_sources",
    "faq_collection",
    "doc_collection",
)


def _resolve_project_path(value: str | None, default: Path) -> str:
    """把场景配置中的相对路径解析为项目内绝对路径。

    参数：
        value: 配置文件中的路径字符串，可以是相对路径、绝对路径或 None。
        default: value 为空时使用的默认路径。

    返回：
        绝对路径字符串。

    调用顺序：场景解析入口 -> _resolve_project_path()。
    """
    if not value:
        return str(default)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


@dataclass(frozen=True)
class ScenarioDefinition:
    """单个业务场景的运行时配置。

    QAService 会根据它选择 FAQ/文档集合、source 白名单、Prompt 场景变量、
    数据目录。

    调用顺序：场景解析入口 -> ScenarioDefinition。
    """

    scenario_id: str
    display_name: str
    industry: str
    description: str
    assistant_name: str
    business_domain: str
    support_contact: str
    valid_sources: list[str]
    faq_collection: str
    doc_collection: str
    data_root: str
    faq_csv_path: str
    source_labels: dict[str, str] = field(default_factory=dict)
    source_patterns: dict[str, str] = field(default_factory=dict)
    sample_questions: list[str] = field(default_factory=list)
    resume_project_name: str = ""
    resume_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, base_dir: Path | None = None) -> "ScenarioDefinition":
        """从 TOML/JSON 风格字典构建 ScenarioDefinition。（★★★ 核心）

        执行流程：
          1. 校验必填字段是否齐全。
          2. 提取并清洗 scenario_id（支持 scenario_id/id 两种 key 名）。
          3. 计算场景基础目录，默认是 PROJECT_ROOT/scenarios/<scenario_id>。
          4. 解析 data_root、faq_csv 等路径（支持相对/绝对/None 三种格式）。
          5. 校验 valid_sources 非空（空分类白名单会导致检索范围异常）。
          6. 组装不可变 dataclass 实例。

        参数：
            payload: 从 scenario.toml 或 API 请求体读取的字典。
            base_dir: 路径解析基础目录；为空时根据 scenario_id 推导。

        返回：
            新的 ScenarioDefinition 实例。

        异常：
            ValueError: 必填字段缺失、scenario_id 为空或 valid_sources 为空。
        """
        # ── 步骤 1：校验必填字段 ──
        # 只要有一条缺失就整体拒绝，避免下游遇到半初始化配置导致难以排查的运行时错误
        missing = [field_name for field_name in REQUIRED_SCENARIO_FIELDS if not payload.get(field_name)]
        if missing:
            raise ValueError(f"场景配置缺少必填字段：{missing}")
        # ── 步骤 2：提取 scenario_id ──
        # 支持两种 key 名（scenario_id / id），兼容不同 TOML 写法
        scenario_id = str(payload.get("scenario_id") or payload.get("id") or "").strip()
        if not scenario_id:
            raise ValueError("scenario_id 不能为空")
        # ── 步骤 3：计算场景基础目录 ──
        # 如果调用方未指定 base_dir，则默认使用 PROJECT_ROOT/scenarios/<scenario_id>
        root = base_dir or PROJECT_ROOT / "scenarios" / scenario_id
        # ── 步骤 4：解析路径（相对/绝对/None 三种格式） ──
        # 相对路径按项目根目录解析，确保开发和容器环境路径一致
        # 原因：Docker 容器内 PROJECT_ROOT=/app，本地开发 PROJECT_ROOT=/home/user/project
        # 使用绝对路径解析保证两种环境的路径指向各自正确的目录
        data_root = _resolve_project_path(payload.get("data_root"), root / "data")
        faq_csv_path = _resolve_project_path(payload.get("faq_csv_path"), root / "faq.csv")
        # ── 步骤 5：强转 valid_sources 并校验非空 ──
        # 强转字符串保证类型安全，避免 TOML 中的整数/浮点数意外传入
        valid_sources = [str(item) for item in payload.get("valid_sources", [])]
        if not valid_sources:
            raise ValueError(f"场景 {scenario_id} 的 valid_sources 不能为空")
        return cls(
            scenario_id=scenario_id,
            display_name=str(payload["display_name"]),
            industry=str(payload["industry"]),
            description=str(payload.get("description") or ""),
            assistant_name=str(payload["assistant_name"]),
            business_domain=str(payload["business_domain"]),
            support_contact=str(payload["support_contact"]),
            valid_sources=valid_sources,
            faq_collection=str(payload["faq_collection"]),
            doc_collection=str(payload["doc_collection"]),
            data_root=data_root,
            faq_csv_path=faq_csv_path,
            source_labels={str(k): str(v) for k, v in dict(payload.get("source_labels", {})).items()},
            source_patterns={str(k): str(v) for k, v in dict(payload.get("source_patterns", {})).items()},
            sample_questions=[str(item) for item in payload.get("sample_questions", [])],
            resume_project_name=str(payload.get("resume_project_name") or payload.get("display_name") or scenario_id),
            resume_keywords=[str(item) for item in payload.get("resume_keywords", [])],
        )

    def compiled_source_patterns(self) -> dict[str, re.Pattern[str]]:
        """按 valid_sources 顺序编译 source_patterns 正则。

        无效正则会被跳过。source_patterns 只来自场景配置，是 source 自动推断和边界检测的依据。

        返回：
            按 valid_sources 顺序排列的 {source_name: compiled_pattern} 字典。

        调用顺序：场景解析入口 -> ScenarioDefinition.compiled_source_patterns()。
        """
        patterns: dict[str, re.Pattern[str]] = {}
        for source in self.valid_sources:
            # 只编译当前 source 有显式定义 pattern 的条目，跳过未配置的 source
            pattern = self.source_patterns.get(source)
            if not pattern:
                continue
            try:
                # 按 valid_sources 顺序编译，保证返回字典的 key 遍历顺序与场景配置一致
                patterns[source] = re.compile(pattern, re.IGNORECASE)
            except re.error:
                # 单个 pattern 编译失败仅跳过，不影响其他 source 的正常编译
                continue
        return patterns

    def label_for_source(self, source: str) -> str:
        """返回 source 的前端展示名称。

        如果场景配置中没有 source_labels，则直接返回 source key。

        参数：
            source: source key。

        返回：
            前端展示名称。

        调用顺序：场景解析入口 -> ScenarioDefinition.label_for_source()。
        """
        return self.source_labels.get(source, source)

    def source_options(self) -> list[dict[str, str]]:
        """返回前端分类下拉框需要的 source 选项。

        返回：
            形如 {"value": source_key, "label": display_label} 的列表。

        调用顺序：场景解析入口 -> ScenarioDefinition.source_options()。
        """
        return [{"value": source, "label": self.label_for_source(source)} for source in self.valid_sources]

    def as_dict(self, *, include_internal: bool = False) -> dict[str, Any]:
        """转换为 API 可序列化的场景字典。

        默认只返回前端需要展示的信息；当 include_internal=True 时，额外返回集合名、
        路径和 valid_sources 等内部字段。

        参数：
            include_internal: 是否包含内部配置字段。

        返回：
            可直接 JSON 序列化的字典。

        调用顺序：场景解析入口 -> ScenarioDefinition.as_dict()。
        """
        payload: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "display_name": self.display_name,
            "industry": self.industry,
            "description": self.description,
            "assistant_name": self.assistant_name,
            "business_domain": self.business_domain,
            "support_contact": self.support_contact,
            "source_options": self.source_options(),
            "sample_questions": self.sample_questions,
            "resume_project_name": self.resume_project_name,
            "resume_keywords": self.resume_keywords,
        }
        if include_internal:
            payload.update(
                {
                    "valid_sources": self.valid_sources,
                    "faq_collection": self.faq_collection,
                    "doc_collection": self.doc_collection,
                    "data_root": self.data_root,
                    "faq_csv_path": self.faq_csv_path,
                }
            )
        return payload


class ScenarioRegistry:
    """从场景配置目录加载并解析业务场景。

    每个场景由 `scenarios/<scenario_id>/scenario.toml` 定义。
    使用 TOML 而非 Python 配置类，是为了把业务场景的声明（集合名、source 列表、数据路径）
    与检索逻辑完全分离——非开发人员只需编辑 TOML 文件即可新增或调整场景，无需修改 Python 代码。

    调用顺序：场景解析入口 -> ScenarioRegistry。
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """初始化场景注册表。

        参数：
            config_dir: 场景配置目录；为空时使用 settings.scenario_config_dir。

        调用顺序：场景解析入口 -> ScenarioRegistry.__init__()。
        """
        settings = get_settings()
        self.config_dir = Path(config_dir or settings.scenario_config_dir)
        if not self.config_dir.is_absolute():
            self.config_dir = PROJECT_ROOT / self.config_dir
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self) -> dict[str, ScenarioDefinition]:
        """扫描配置目录并加载所有 scenario.toml。

        执行流程：
          1. 校验配置目录存在。
          2. 按排序顺序查找 */scenario.toml。
          3. 逐个解析 TOML，并转换成 ScenarioDefinition。
          4. 按 scenario_id 建立索引；没有加载到任何场景时抛错。

        返回：
            {scenario_id: ScenarioDefinition} 字典。

        异常：
            RuntimeError: 配置目录不存在或没有任何有效场景配置。
        """
        loaded: dict[str, ScenarioDefinition] = {}
        if not self.config_dir.exists():
            raise RuntimeError(f"场景配置目录不存在：{self.config_dir}")
        # 按目录名的字母序遍历子目录，保证多场景加载顺序稳定且可预测
        for config_path in sorted(self.config_dir.glob("*/scenario.toml")):
            # 每个 TOML 文件代表一个独立业务场景，解析结果以 scenario_id 为 key 建立索引
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
            scenario = ScenarioDefinition.from_mapping(payload, base_dir=config_path.parent)
            loaded[scenario.scenario_id] = scenario
        # 没有任何有效场景文件时不应该静默返回空字典，直接抛错让启动流程感知到配置缺失
        if loaded:
            return loaded
        raise RuntimeError(f"场景配置目录中没有有效 scenario.toml：{self.config_dir}")

    def list_scenarios(self) -> list[ScenarioDefinition]:
        """返回按 scenario_id 排序的全部可用场景。

        返回：
            ScenarioDefinition 列表。

        调用顺序：场景解析入口 -> ScenarioRegistry.list_scenarios()。
        """
        return [self.scenarios[key] for key in sorted(self.scenarios)]

    def resolve(self, scenario_id: str | None = None) -> ScenarioDefinition:
        """解析当前请求应该使用的业务场景。

        执行流程（命中即返回）：
          1. 请求显式传入 scenario_id 且存在时，优先使用它。
          2. 否则使用 ACTIVE_SCENARIO_ID 配置。
          3. 仍未命中时，回退到按 scenario_id 排序后的第一个场景。

        参数：
            scenario_id: 请求级场景覆盖值，例如前端下拉框传入的场景 ID。

        返回：
            解析后的 ScenarioDefinition。

        这个优先级会被离线入库和在线检索共同使用：脚本显式传入的
        `--scenario` 最高；未传时使用环境配置；单场景部署仍可通过排序后的
        第一个场景启动。解析结果决定 collection、source 白名单和默认路径，
        所以后续阶段不应再自行推断这些值。
        """
        # 命中即返回的优先级策略：请求级 > 全局配置级 > 默认第一个场景
        # 这样前端传入了 scenario_id 时可以精确覆盖，同时未指定时也不至于崩溃
        requested = (scenario_id or "").strip()
        if requested and requested in self.scenarios:
            return self.scenarios[requested]
        configured = get_settings().active_scenario_id.strip()
        if configured and configured in self.scenarios:
            return self.scenarios[configured]
        # 兜底行为：按排序后的第一个场景作为当前场景，保证单场景部署也正常工作
        return self.list_scenarios()[0]

    def as_payload(self) -> dict[str, Any]:
        """返回前端场景选择器需要的结构化数据。

        返回：
            包含 active_scenario_id 和场景配置列表的字典。

        调用顺序：场景解析入口 -> ScenarioRegistry.as_payload()。
        """
        active = self.resolve()
        return {
            "active_scenario_id": active.scenario_id,
            "scenarios": [scenario.as_dict(include_internal=True) for scenario in self.list_scenarios()],
        }


def get_scenario_registry() -> ScenarioRegistry:
    """返回新的 ScenarioRegistry 实例。

    有意不做缓存：每次调用都重新从磁盘读取 scenario.toml。这样做牺牲了少量 I/O 性能，
    换取了本地开发调试时修改 TOML 配置后即时生效的能力（无需重启服务）。
    对于 TOML 解析这种微秒级操作，在请求生命周期内没有性能压力。

    返回：
        新的 ScenarioRegistry。

    调用顺序：场景解析入口 -> get_scenario_registry()。
    """
    return ScenarioRegistry()


def resolve_scenario(scenario_id: str | None = None) -> ScenarioDefinition:
    """便捷函数：解析当前请求使用的业务场景。

    参数：
        scenario_id: 可选场景覆盖值。

    返回：
        解析后的 ScenarioDefinition。

    这是离线入口和多个下游服务共同使用的场景解析边界。调用方拿到结果后，
    应把同一个对象继续传给 FAQ、文档和质量报告函数，避免一轮构建中因为配置
    文件被修改或默认场景变化而产生不同的 collection/source 解释。

    调用顺序：场景解析入口 -> resolve_scenario()。
    """
    return get_scenario_registry().resolve(scenario_id)

