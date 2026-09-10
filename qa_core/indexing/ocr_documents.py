"""扫描件和图片资料的离线 OCR 清洗工具。

OCR 不进入在线问答主链路，也不默认写入 active 知识库。它只负责把扫描 PDF、图片 PDF
或图片附件转换成待复核 Markdown，并生成质量报告。人工确认后，再把清洗后的 Markdown
放入对应场景的 `data/<source>_data` 目录并重建知识库版本。
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import fitz
from paddleocr import PaddleOCR
OCR_SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

@dataclass(frozen=True)
class OCRLine:
    """OCR 识别出的单行文本和置信度。

    调用顺序：入库脚本或索引服务 -> OCRLine。
    """
    text: str
    confidence: float

@dataclass(frozen=True)
class OCRPage:
    """单页 OCR 结果。

    调用顺序：入库脚本或索引服务 -> OCRPage。
    """
    page_index: int
    text: str
    line_count: int
    avg_confidence: float

@dataclass(frozen=True)
class OCRDocumentResult:
    """单个文件的 OCR 清洗结果。

    调用顺序：入库脚本或索引服务 -> OCRDocumentResult。
    """
    source_path: str
    output_path: str
    page_count: int
    line_count: int
    avg_confidence: float
    ready_for_review: bool
    pages: list[dict[str, Any]]
    def as_dict(self) -> dict[str, Any]:
        """返回可写入 JSON 报告的结构。

        调用顺序：入库脚本或索引服务 -> OCRDocumentResult.as_dict()。
        """
        return asdict(self)

def is_ocr_supported_file(path: Path) -> bool:
    """判断文件是否适合交给离线 OCR 脚本处理。

    调用顺序：入库脚本或索引服务 -> is_ocr_supported_file()。
    """
    return path.suffix.lower() in OCR_SUPPORTED_SUFFIXES

def create_ocr_engine(lang: str = "ch", use_angle_cls: bool = True) -> PaddleOCR:
    """创建 PaddleOCR 引擎。
    PaddleOCR 是成熟中文 OCR 开源方案，适合本地扫描件清洗。这里固定在离线脚本中使用，
    不接入在线问答请求，避免 OCR 的模型加载、图像渲染和识别耗时影响用户提问。

    调用顺序：入库脚本或索引服务 -> create_ocr_engine()。
    """
    return PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=False, show_log=False)

def render_pdf_pages(path: Path, image_dir: Path, *, dpi: int = 180, max_pages: int = 0) -> list[Path]:
    """把 PDF 每页渲染成临时 PNG，供 OCR 引擎识别。
    PyMuPDF 只负责离线渲染页面，不参与普通 PDF 文本层解析。普通文本层 PDF 仍由
    LangChain `PyPDFLoader` 进入默认入库链路；只有扫描件或图片 PDF 才走该函数。

    调用顺序：入库脚本或索引服务 -> render_pdf_pages()。
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(path)
    page_paths: list[Path] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    page_limit = min(len(document), max_pages) if max_pages > 0 else len(document)
    for page_index in range(page_limit):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = image_dir / f"{path.stem}_page_{page_index + 1}.png"
        pixmap.save(str(image_path))
        page_paths.append(image_path)
    document.close()
    return page_paths

def recognize_file(
    path: Path,
    *,
    output_dir: Path,
    engine: PaddleOCR,
    min_confidence: float = 0.78,
    dpi: int = 180,
    max_pages: int = 0,
) -> OCRDocumentResult:
    """识别一个文件并输出待复核 Markdown。
    输出文件不是直接入库文件，而是"清洗候选稿"。原因是 OCR 可能把金额、日期、合同号、
    责任主体识别错，必须先人工复核，再进入正式知识库版本。

    调用顺序：入库脚本或索引服务 -> recognize_file()。
    """
    if not is_ocr_supported_file(path):
        raise ValueError(f"不支持的 OCR 文件类型：{path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "_rendered_pages"
    # PDF 需要先渲染成图片再识别；非 PDF（图片文件）直接送入 OCR 引擎
    image_paths = render_pdf_pages(path, image_dir, dpi=dpi, max_pages=max_pages) if path.suffix.lower() == ".pdf" else [path]

    pages: list[OCRPage] = []
    for page_index, image_path in enumerate(image_paths, start=1):
        lines = recognize_image(image_path, engine)
        page_text = "\n".join(line.text for line in lines)
        # 加权平均置信度：按行数加权，避免单行高置信度拉高多行页面的整体评分
        avg_confidence = round(sum(line.confidence for line in lines) / max(len(lines), 1), 4)
        pages.append(OCRPage(page_index=page_index, text=page_text, line_count=len(lines), avg_confidence=avg_confidence))

    total_lines = sum(page.line_count for page in pages)
    # 全局平均置信度：以每页行数为权重，精确反映整体识别质量
    avg_confidence = round(
        sum(page.avg_confidence * page.line_count for page in pages) / max(total_lines, 1),
        4,
    )
    # ready_for_review 阈值判断：必须有识别结果且置信度 >= min_confidence 才标记为"待复核"
    # 低于阈值的文件应重新扫描或人工整理后再进入复核流程
    ready_for_review = total_lines > 0 and avg_confidence >= min_confidence
    output_path = output_dir / f"{path.stem}_ocr.md"
    output_path.write_text(build_ocr_markdown(path, pages, avg_confidence, ready_for_review), encoding="utf-8")
    return OCRDocumentResult(
        source_path=str(path),
        output_path=str(output_path),
        page_count=len(pages),
        line_count=total_lines,
        avg_confidence=avg_confidence,
        ready_for_review=ready_for_review,
        pages=[asdict(page) for page in pages],
    )

def recognize_image(path: Path, engine: PaddleOCR) -> list[OCRLine]:
    """识别单张图片，并按阅读顺序返回文本行。

    调用顺序：入库脚本或索引服务 -> recognize_image()。
    """
    raw_result = engine.ocr(str(path), cls=True)
    lines: list[OCRLine] = []
    for item in flatten_ocr_result(raw_result):
        text = str(item[0]).strip()
        confidence = float(item[1])
        if text:
            lines.append(OCRLine(text=text, confidence=round(confidence, 4)))
    return lines

def flatten_ocr_result(raw_result: Any) -> list[tuple[str, float]]:
    """把 PaddleOCR 不同返回层级压平成 `(文本, 置信度)`。
    不同版本对单图、多页输入的嵌套层级略有差异。这里不做兼容降级，只处理 PaddleOCR
    官方结果结构里的文本与置信度提取，让报告格式稳定。

    调用顺序：入库脚本或索引服务 -> flatten_ocr_result()。
    """
    result: list[tuple[str, float]] = []
    if not raw_result:
        # OCR 引擎未返回任何结果时直接返回空列表，避免后续索引错误
        return result
    # 单图检测：PaddleOCR 对单张图片返回 [[[bbox, (text, conf)], ...]] 嵌套结构，
    # 此处自动展平一层，保证后续按行遍历时的格式统一
    page_items = raw_result[0] if len(raw_result) == 1 and isinstance(raw_result[0], list) else raw_result
    for line in page_items or []:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        payload = line[1]
        # PaddleOCR 标准返回格式：line[1] = (text_string, confidence_float)
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            result.append((str(payload[0]), float(payload[1])))
    return result

def build_ocr_markdown(path: Path, pages: list[OCRPage], avg_confidence: float, ready_for_review: bool) -> str:
    """构造待人工复核的 OCR Markdown 文本。

    调用顺序：入库脚本或索引服务 -> build_ocr_markdown()。
    """
    status = "待人工复核" if ready_for_review else "置信度不足，需重新扫描或人工整理"
    lines = [
        f"# OCR 清洗候选稿：{path.name}",
        "",
        f"- 原始文件：{path}",
        f"- OCR 平均置信度：{avg_confidence}",
        f"- 复核状态：{status}",
        "",
        "> 说明：本文件由离线 OCR 生成，不能直接视为已确认业务资料。人工复核后，才可放入场景资料目录并重建知识库版本。",
        "",
    ]
    for page in pages:
        lines.extend(
            [
                f"## 第 {page.page_index} 页",
                "",
                f"- 行数：{page.line_count}",
                f"- 页平均置信度：{page.avg_confidence}",
                "",
                # 无识别内容的页显示占位提示，避免复核人员误认为文件读取失败
                page.text or "（本页未识别到文本）",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"
