"""显式创建新项目业务表与首个管理员；不会修改原项目数据库。"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from qa_core.config.settings import get_settings
from qa_core.storage.bootstrap import bootstrap_mysql_schema
from webapp.auth_store import AuthStore


def main():
    parser = argparse.ArgumentParser(description="初始化账号表、RAG元数据表及管理员")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--display-name", default="平台管理员")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--visibility", choices=["public","internal","private"], default="public")
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    if settings.mysql_database != "knowforge_metrics":
        parser.error("请使用独立数据库 knowforge_metrics，避免误操作原项目")
    store = AuthStore(create_engine(settings.mysql_sync_uri, pool_pre_ping=True))
    store.initialize()
    bootstrap_mysql_schema()
    if args.schema_only:
        print("独立项目数据库表初始化完成")
        return
    if any(row["username"] == args.username.strip().lower() for row in store.list_users(args.tenant)):
        print("管理员已存在，未修改密码")
        return
    password = getpass.getpass("请设置管理员密码（至少12个字符）：")
    if password != getpass.getpass("请再次输入密码："):
        parser.error("两次密码不一致")
    try:
        store.create_user(args.username, password, args.display_name, args.tenant,
                          "tenant_admin", args.dataset, args.visibility)
    except ValueError as exc:
        parser.error(str(exc))
    print("初始化完成；管理员账号：" + args.username)


if __name__ == "__main__":
    main()
