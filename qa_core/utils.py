"""入库和检索共用的小型确定性工具函数，用于生成 doc_id、chunk_id、faq_id 等 ID。

设计决策：
- 所有函数是确定性的纯函数：相同输入总是产生相同输出。
- stable_hash 使用 SHA-256 而非 MD5，避免潜在的安全审查问题（尽管用于 ID 生成
  而非安全场景）。
- file_fingerprint 基于文件路径、修改时间和大小，而非文件内容哈希。原因是内容哈希
  在入库前需要先读取整个文件，在大文件场景下开销过高。修改时间 + 大小的组合已经
  能覆盖 99% 的变化检测场景。

调用顺序：入库阶段或检索阶段 -> utils。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def stable_hash(*parts: object) -> str:
    """根据任意值创建稳定的 SHA-256 标识，用于 Milvus 主键和清单 key。

    输入为多个可变参数，以 "||" 拼接。None 值统一转为空字符串。
    输出为完整的 SHA-256 十六进制字符串（64 字符）。需要更短摘要时调用方自行截取。

    参数：
        parts: 任意数量的可哈希值（str、int、Path 等）。

    返回：
        完整的 SHA-256 十六进制字符串（64 字符）。

    调用顺序：key_digest() / file_fingerprint() -> stable_hash()。
    """
    # 原因："||" 分隔符保证边界清晰，"ab"+"cd" 不会与 "a"+"bcd" 混淆
    raw = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """根据路径、修改时间和大小生成本地文件指纹，用于增量入库的变化判断。

    指纹基于文件路径（resolve 后的绝对路径）、修改时间（st_mtime_ns）和文件大小
    （st_size）。不基于文件内容的 SHA-256 哈希，因为内容哈希需要先读取整个文件，
    大文件场景下开销过高。

    参数：
        path: 文件路径。

    返回：
        稳定的文件指纹字符串（64 字符 SHA-256）。

    调用顺序：入库阶段 -> file_fingerprint()。
    """
    p = Path(path)
    stat = p.stat()
    # 原因：使用 resolve() 后的绝对路径，避免符号链接或相对路径导致同一文件的指纹不同
    return stable_hash(str(p.resolve()), stat.st_mtime_ns, stat.st_size)


def normalize_source_from_path(path: str | Path) -> str:
    """根据 `<source>_data` 目录名去掉 _data 后缀，得到来源名。

    场景数据目录命名规范：每个 source 的数据放在 <source>_data/ 子目录下。
    此函数从目录路径中提取 source 名称。

    示例："/data/hr_data/" -> "hr"

    参数：
        path: 数据目录路径。

    返回：
        去掉 _data 后缀的来源名；路径不包含 _data 时返回 "default"。

    调用顺序：入库阶段 -> normalize_source_from_path()。
    """
    name = os.path.basename(str(path)).replace("_data", "")
    return name or "default"
