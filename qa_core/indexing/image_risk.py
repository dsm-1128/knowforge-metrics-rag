"""图片与图文混排资料的风险检测——入库质量报告用。

在线 RAG 链路不能假设嵌入的图片已被索引。本模块在离线入库质量检查时
识别含图片的文件，使运维人员能将扫描件和关键业务截图分流到 OCR 复核
流程，避免未经清洗的图片直接进入 active 知识库。

设计决策：
- 三种风险等级：block（拦截——必须 OCR 复核后才能入库）、
  review（告警——图文混排，文本层已入但图片需要关注）、
  None（无风险——文件不含图片）。
- 对于 PDF/DOCX/PPTX 等容器格式，同时检测嵌入图片数量和文本层字符数。
  文本层不足 120 字符视为"主要依赖图片"，标记为 block 级别。
- 独立图片文件（.png/.jpg/.bmp 等）直接拦截，不允许进入主链路。

依赖分层：
- fitz（PyMuPDF）：PDF 嵌入图片数量统计。
- python-docx：DOCX 嵌入图片数量统计。
- python-pptx：PPTX 嵌入图片数量统计。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz
from docx import Document as DocxDocument
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


IMAGE_FILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_CONTAINER_SUFFIXES = {".pdf", ".docx", ".pptx"}
IMAGE_RISK_SUFFIXES = IMAGE_FILE_SUFFIXES | IMAGE_CONTAINER_SUFFIXES
BLOCKING_TEXT_CHAR_THRESHOLD = 120


@dataclass(frozen=True)
class ImageRisk:
    """A visible image-ingestion risk attached to a candidate source file."""

    path: str
    suffix: str
    severity: str
    image_count: int
    text_char_count: int
    reason: str
    requires_ocr_review: bool

    def as_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的诊断数据。

        调用顺序：测试或业务入口 -> ImageRisk.as_dict()。
        """
        return asdict(self)


def analyze_image_risk(path: Path, extracted_text: str = "") -> ImageRisk | None:
    """检测含图片的文件是否需要 OCR 复核或运维关注。

    三种返回结果：
      - None：文件不含图片或无风险。
      - severity="block"：必须 OCR 复核后方可入库（独立图片或文本层不足的文件）。
      - severity="review"：图文混排，文本层已入但需要关注图片承载的信息。

    参数：
        path: 待检测文件路径。
        extracted_text: 已提取的文本内容（可选），为空时只做后缀判断。

    返回：
        ImageRisk 对象或 None。

    调用顺序：入库质量门禁流程 -> analyze_image_risk()。
    """

    # ── 步骤 1：跳过非图片相关后缀（如 .md/.txt/.csv 等） ──
    suffix = path.suffix.lower()
    if suffix not in IMAGE_RISK_SUFFIXES:
        return None

    # ── 步骤 2：统计文本层有效字符数（去空白后） ──
    text_char_count = _visible_text_length(extracted_text)
    # ── 步骤 3：独立图片文件 → 直接拦截 ──
    # 原因：独立图片没有文本层，不能直接进入知识库，必须经过 OCR 转换为可检索的文本
    if suffix in IMAGE_FILE_SUFFIXES:
        return ImageRisk(
            path=str(path),
            suffix=suffix,
            severity="block",
            image_count=1,
            text_char_count=0,
            reason="独立图片不能直接进入知识库；必须先离线 OCR，人工复核后再提升为 Markdown 资料。",
            requires_ocr_review=True,
        )

    # ── 步骤 4：容器格式（PDF/DOCX/PPTX）统计嵌入图片数量 ──
    image_count = _embedded_image_count(path)
    if image_count <= 0:
        return None

    # ── 步骤 5：文本层不足 → 拦截 ──
    # 原因：BLOCKING_TEXT_CHAR_THRESHOLD（120）是经验值——少于 120 有效字符的文件
    # 几乎完全依赖图片承载信息，文本层不具备独立检索价值
    if text_char_count < BLOCKING_TEXT_CHAR_THRESHOLD:
        return ImageRisk(
            path=str(path),
            suffix=suffix,
            severity="block",
            image_count=image_count,
            text_char_count=text_char_count,
            reason="文件主要依赖图片或扫描页，文本层不足，必须先走 OCR 复核流程。",
            requires_ocr_review=True,
        )

    # ── 步骤 6：文本层充足但有嵌入图片 → 告警 ──
    return ImageRisk(
        path=str(path),
        suffix=suffix,
        severity="review",
        image_count=image_count,
        text_char_count=text_char_count,
        reason="文件包含图片；当前入库只保证文本层进入知识库，若图片承载业务信息需单独 OCR/复核。",
        requires_ocr_review=False,
    )


def _visible_text_length(text: str) -> int:
    """计算可见文本字符数（去空白后）。

    用于判断文件中实际可读文本的体量。去掉所有空白字符后统计有效字数，
    避免空格/换行/缩进等非语义字符影响判断。

    参数：
        text: 原始文本字符串。

    返回：
        去空白后的字符数。

    调用顺序：风险检测流程 -> _visible_text_length()。
    """
    # 原因：使用 split() 去空白比逐个字符过滤更高效，且对中文文档同样适用
    return len("".join(str(text or "").split()))


def _embedded_image_count(path: Path) -> int:
    """统计容器格式文件（PDF/DOCX/PPTX）中嵌入的图片数量。

    参数：
        path: 文件路径。

    返回：
        嵌入图片的总数。文件类型不支持时返回 0。

    调用顺序：风险检测流程 -> _embedded_image_count()。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_image_count(path)
    if suffix == ".docx":
        return _docx_image_count(path)
    if suffix == ".pptx":
        return _pptx_image_count(path)
    return 0


def _pdf_image_count(path: Path) -> int:
    """统计 PDF 文件中的嵌入图片数量。

    使用 PyMuPDF（fitz）的 get_images() API 统计所有页面的图片对象数。
    包括内嵌图片和外部引用图片。

    参数：
        path: PDF 文件路径。

    返回：
        PDF 中嵌入图片的总数。

    调用顺序：风险检测流程 -> _pdf_image_count()。
    """
    with fitz.open(str(path)) as document:
        return sum(len(page.get_images(full=True)) for page in document)


def _docx_image_count(path: Path) -> int:
    """统计 DOCX 文件中的嵌入图片数量。

    同时检查两种途径：inline_shapes（内联形状）和 package relations（包关系）。
    取两者最大值而非之和，以避免重复计数。

    参数：
        path: DOCX 文件路径。

    返回：
        DOCX 中嵌入图片的估计数量。

    调用顺序：风险检测流程 -> _docx_image_count()。
    """
    document = DocxDocument(str(path))
    # inline_shapes 统计文档正文中的内联图片
    inline_count = len(document.inline_shapes)
    # relation_count 通过包关系类型识别嵌入图片（包括页眉页脚和文本框中的图片）
    relation_count = sum(1 for relation in document.part.rels.values() if "image" in relation.reltype)
    # 取两者最大值：有些图片只在 inline_shapes 中出现，有些只在 relations 中出现
    return max(inline_count, relation_count)


def _pptx_image_count(path: Path) -> int:
    """统计 PPTX 文件中的嵌入图片数量。

    递归遍历每一页幻灯片中的所有形状，统计类型为 PICTURE 的形状数量。
    嵌套形状（如组合图形中的图片）也会被递归统计。

    参数：
        path: PPTX 文件路径。

    返回：
        PPTX 中嵌入图片的总数。

    调用顺序：风险检测流程 -> _pptx_image_count()。
    """
    presentation = Presentation(str(path))
    return sum(_count_shape_images(slide.shapes) for slide in presentation.slides)


def _count_shape_images(shapes: Any) -> int:
    """递归统计形状集合中的图片数量。

    参数：
        shapes: python-pptx 的 Shape 对象集合（slide.shapes）。

    返回：
        该集合中 PICTURE 类型的形状总数。

    调用顺序：_pptx_image_count() -> _count_shape_images()。
    """
    count = 0
    for shape in shapes:
        # ── 直接形状：检查是否为图片类型 ──
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            count += 1
        # ── 嵌套形状：递归处理组合图形中的子形状 ──
        nested = getattr(shape, "shapes", None)
        if nested is not None:
            count += _count_shape_images(nested)
    return count
