"""基础设施存储层包。

提供 MySQL 运行期 Schema 管理和启动引导能力。包含：
- mysql_schema.py：MySQL schema SQL runner，渲染表名占位符并执行建表语句。
- bootstrap.py：运行期 MySQL schema 初始化，在 API 启动和入库脚本运行时显式调用。

设计原则：
- 业务 Store 不在方法内部夹带 DDL 副作用。建表统一通过 bootstrap 在启动期完成。
- 表名通过安全标识符转义注入 SQL 模板，防止 SQL 注入。
"""