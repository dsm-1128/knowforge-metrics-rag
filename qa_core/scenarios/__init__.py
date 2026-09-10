"""业务场景包：多业务场景的注册、解析和边界检测。

包含模块：
- registry.py：ScenarioRegistry 从 scenario.toml 加载并解析业务场景配置；
  ScenarioDefinition 定义单个场景的元数据（集合名、source 白名单、Prompt 变量等）。
- boundary.py：跨场景和跨 source 边界检测，判断当前问题是否明显属于另一个场景或分类。

调用方典型用法：from qa_core.scenarios.registry import resolve_scenario
"""

