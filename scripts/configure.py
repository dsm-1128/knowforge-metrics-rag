"""首次生成本地配置；不覆盖已有 .env，不读取原项目密钥。"""
import argparse
from pathlib import Path
import secrets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, help="已有三个模型目录的父目录")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        print(".env 已存在，未覆盖；请直接编辑此文件")
        return
    content = (root / ".env.example").read_text(encoding="utf-8")
    content = content.replace("MYSQL_PASSWORD=CHANGE_ME", "MYSQL_PASSWORD=" + secrets.token_urlsafe(24))
    content = content.replace("ADMIN_API_TOKEN=CHANGE_ME", "ADMIN_API_TOKEN=" + secrets.token_urlsafe(32))
    if args.models_dir:
        model_root = args.models_dir.resolve()
        for name in ["bge-m3", "bge-reranker-large", "bert_intent_classifier_v1"]:
            if not (model_root / name).is_dir():
                parser.error(f"缺少模型目录：{model_root / name}")
            content = content.replace("./models/" + name, (model_root / name).as_posix())
    target.write_text(content, encoding="utf-8")
    print("已生成独立 .env；请填写模型 API Key，再启动基础服务并初始化账号")


if __name__ == "__main__":
    main()
