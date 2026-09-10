"""表格资料转换为 LangChain Document。

真实企业资料里，很多关键知识不是自然段，而是表格：材料清单、验收项、付款节点、
评分表、赔付字段、单证字段。普通段落切分会破坏行列关系，所以表格需要先转换成
带表头、行号和单元格键值的结构化文本，再进入统一的 normalize/split/Milvus 入库链路。
"""

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from qa_core.indexing.ocr_review import is_reviewed_ocr_text

TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}
# 载体形式词：只有出现在文件名时才视为扫描件/图片件风险。
# 这些词在正文里常作为业务名词出现（如"补充声明扫描件""营业执照扫描件"），
# 不代表文件本身是扫描件，放在正文匹配会误判，导致干净的结构化资料被门禁拦下。
OCR_CARRIER_RE = re.compile(r"(扫描件|扫描版|图片PDF|图片 PDF)", re.IGNORECASE)
# 噪声特征词：出现在正文即视为 OCR 噪声风险（实际识别错误的元描述）。
OCR_NOISE_RE = re.compile(r"(OCR|断行|错字|O 和 0|噪声)", re.IGNORECASE)

def is_table_file(path: Path) -> bool:
    """判断文件是否属于表格型资料。

    调用顺序：入库脚本或索引服务 -> is_table_file()。
    """
    return path.suffix.lower() in TABLE_SUFFIXES

def looks_like_ocr_risk(path: Path, text: str) -> bool:
    """识别疑似 OCR 或扫描件风险。
    该函数只做风险识别，不执行 OCR。扫描件进入默认主链路前必须先人工复核或走独立
    OCR 清洗流程，避免把错字、断行和金额识别错误直接写进 active 知识库。

    区分两类信号避免误判：
    - 载体形式词（扫描件/图片PDF）只在文件名命中才算风险，正文里多是业务名词。
    - 噪声特征词（OCR/断行/错字等）在正文命中即算风险。

    调用顺序：入库脚本或索引服务 -> looks_like_ocr_risk()。
    """
    if is_reviewed_ocr_text(text):
        # 已复核的 OCR 文本不再重复判断：人工复核过的内容即使有"扫描件"字样也是正常业务上下文
        return False
    if OCR_CARRIER_RE.search(path.name):
        # 文件名包含"扫描件""图片PDF"等载体形式词，说明该文件本身就是扫描件，有 OCR 风险
        return True
    # 只检查正文前 2000 字符的噪声特征，避免全文扫描大文件影响首次加载性能；
    # 噪声词在正文中出现说明该文本可能是 OCR 产物（包含识别错误的元描述）
    return bool(OCR_NOISE_RE.search(text[:2000]))

def looks_like_table_text(path: Path, text: str) -> bool:
    """识别普通文本中是否包含表格特征。
    Markdown 表格仍可由普通 Markdown loader 处理；这里额外给质量报告打标，方便
    后续判断是否需要把某份文档拆成独立表格资料。

    注意这里不按"清单、台账、表格"这类普通业务词直接判定。企业制度里经常写
    "问题清单、图纸台账、修改表格"，这些是自然语言，不代表文件真的具有行列结构。
    如果误报，质量报告会把正常 Markdown 当成表格资料，反而增加复核噪声。

    调用顺序：入库脚本或索引服务 -> looks_like_table_text()。
    """
    sample = text[:4000]
    if is_table_file(path):
        return True
    markdown_table_lines = [line for line in sample.splitlines() if line.count("|") >= 2]
    if len(markdown_table_lines) >= 2:
        return True
    comma_like_rows = [line for line in sample.splitlines() if line.count(",") >= 2]
    return len(comma_like_rows) >= 2

def load_table_file(path: Path) -> list[Document]:
    """把 CSV/Excel 表格转换为保留行列语义的 Document 列表。

    CSV 默认按整表读取；Excel 会逐个 sheet 读取。每一行生成一个 Document，并在正文中
    同时写入表头、sheet、行号和"列名：值"。这样检索"付款节点""验收状态""材料缺失"
    这类问题时，行列关系不会被普通文本切分打散。

    调用顺序：入库脚本或索引服务 -> load_table_file()。
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frames = [("csv", pd.read_csv(path, encoding="utf-8-sig"))]
    elif suffix == ".xlsx":
        frames = list(pd.read_excel(path, sheet_name=None, engine="openpyxl").items())
    elif suffix == ".xls":
        frames = list(pd.read_excel(path, sheet_name=None, engine="xlrd").items())
    else:
        raise ValueError(f"不支持的表格文件类型：{path}")

    documents: list[Document] = []
    for sheet_name, frame in frames:
        normalized = _normalize_frame(frame)
        headers = [str(column) for column in normalized.columns]
        table_id = _table_id(path, str(sheet_name))
        # 逐行转换成 Document：每行附带表头和 sheet 信息，确保检索时能还原行列上下文，
        # 避免普通段落切分破坏"某列某值"的精确匹配关系
        for row_number, row in enumerate(normalized.to_dict(orient="records"), start=1):
            cells = {str(key): _cell_text(value) for key, value in row.items()}
            # 跳过全空行：表格中可能有合并单元格或格式遗留的空行，不生成 Document 以减少无效 chunk
            if not any(cells.values()):
                continue
            # 格式化正文：列出每列的"列名：值"，便于语义检索时直接匹配到具体的列值
            cell_lines = [f"- {key}：{value}" for key, value in cells.items() if value]
            content = "\n".join(
                [
                    f"表格文件：{path.name}",
                    f"工作表：{sheet_name}",
                    f"表头：{' / '.join(headers)}",
                    f"行号：{row_number}",
                    "单元格：",
                    *cell_lines,
                ]
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "content_type": "table_row",
                        "table_id": table_id,
                        "sheet_name": str(sheet_name),
                        "row_number": row_number,
                        "row_count": len(normalized),
                        "column_count": len(headers),
                        "table_headers": " | ".join(headers),
                    },
                )
            )
    return documents

def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """清理表格空行空列，并把缺失表头补成稳定列名。

    调用顺序：入库脚本或索引服务 -> _normalize_frame()。
    """
    # dropna(how="all") 删除全空行，dropna(axis=1, how="all") 删除全空列，
    # fillna("") 将 NaN 替换为空字符串，避免后续拼接时出现 "nan" 字符串
    data = frame.dropna(how="all").dropna(axis=1, how="all").fillna("")
    columns: list[str] = []
    for index, column in enumerate(data.columns, start=1):
        name = str(column).strip()
        if not name or name.lower().startswith("unnamed:"):
            # pandas 自动给无表头列命名为 "Unnamed: N"，此处替换为 "列N"，
            # 使检索结果中的列名对中文用户更友好
            name = f"列{index}"
        columns.append(name)
    data.columns = columns
    return data

def _cell_text(value: object) -> str:
    """把单元格值转换成适合检索的短文本。

    调用顺序：入库脚本或索引服务 -> _cell_text()。
    """
    text = str(value).strip()
    # Excel 导出时整数列可能带 ".0" 后缀（如 1200.0），去掉末尾的 ".0" 保持数值原有格式，
    # 避免检索 "1200" 时因格式差异匹配不到
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text

def _table_id(path: Path, sheet_name: str) -> str:
    """生成稳定表格标识，便于质量报告和来源回查。

    调用顺序：入库脚本或索引服务 -> _table_id()。
    """
    return f"{path.stem}:{sheet_name}"
