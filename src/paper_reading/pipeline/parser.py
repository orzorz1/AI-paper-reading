"""PDF 解析模块。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

from ..errors import DependencyMissingError, PipelineExecutionError
from ..models import BoundingBox, PageImageBlock, PageTextBlock, PaperMetadata, ParsedPage, ParsedPaper
from ..utils import ensure_dir, normalize_whitespace

ABSTRACT_PATTERN = re.compile(
    r"(?is)\babstract\b[\s:：-]*?(.*?)(?:\n\s*(?:keywords|index terms|1\.?\s+introduction|introduction)\b|\Z)"
)
ARXIV_PATTERN = re.compile(r"\barxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
NON_TITLE_PREFIX_PATTERN = re.compile(
    r"^(?:abstract|figure|fig\.?|table|keywords?|index terms|introduction)\b",
    re.IGNORECASE,
)
AFFILIATION_HINT_PATTERN = re.compile(
    r"\b(?:university|institute|school|laboratory|lab|department|academy|college|casia|baai)\b",
    re.IGNORECASE,
)


def extract_title_from_page_dict(page_dict: dict[str, Any], page_height: float, page_width: float | None = None) -> str:
    """根据首页文本块推断标题。"""
    candidates: list[dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        if float(bbox[1]) > page_height * 0.35:
            continue
        line_texts: list[str] = []
        sizes: list[float] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                if not text:
                    continue
                line_texts.append(text)
                sizes.append(float(span.get("size", 0.0)))
        if not line_texts or not sizes:
            continue
        text = normalize_whitespace(" ".join(line_texts))
        if not text:
            continue
        candidates.append(
            {
                "text": text,
                "font_size": max(sizes),
                "y0": float(bbox[1]),
                "y1": float(bbox[3]),
                "bbox": bbox,
            }
        )

    if not candidates:
        return ""

    eligible_candidates = [
        candidate
        for candidate in candidates
        if not _is_non_title_block(candidate["text"], candidate["bbox"], page_height, page_width)
    ]
    ranked_candidates = eligible_candidates or candidates
    ranked_candidates.sort(key=lambda item: (-_score_title_candidate(item), item["y0"]))

    best_candidate = ranked_candidates[0]
    return normalize_whitespace(best_candidate["text"])


def extract_abstract_from_text(text: str) -> str:
    """从首页或全文文本中提取 abstract 段落。"""
    match = ABSTRACT_PATTERN.search(text)
    if not match:
        return ""
    abstract = normalize_whitespace(match.group(1))
    abstract = re.sub(r"^(abstract[\s:：-]*)", "", abstract, flags=re.IGNORECASE)
    return abstract


def build_abstract_fallback(paragraphs: Iterable[str], limit: int = 1200) -> str:
    """当论文没有标准摘要时，拼接前几段正文作为退路。"""
    parts: list[str] = []
    current = 0
    for paragraph in paragraphs:
        normalized = normalize_whitespace(paragraph)
        if not normalized:
            continue
        parts.append(normalized)
        current += len(normalized)
        if current >= limit:
            break
    return "\n\n".join(parts)


class PDFParser:
    """用 PyMuPDF 提取文本、图片块和页面渲染图。"""

    def __init__(self, *, render_dpi: int = 150) -> None:
        self._render_dpi = render_dpi

    def parse(
        self,
        pdf_path: Path,
        artifacts_dir: Path,
        *,
        override_title: Optional[str] = None,
        override_abstract: Optional[str] = None,
    ) -> ParsedPaper:
        try:
            import fitz
        except ImportError as exc:
            raise DependencyMissingError("缺少 PyMuPDF，请先安装依赖后再运行。") from exc

        page_images_dir = ensure_dir(artifacts_dir / "page_images")

        try:
            document = fitz.open(str(pdf_path))
        except Exception as exc:
            raise PipelineExecutionError(f"无法打开 PDF：{pdf_path}") from exc

        pages: list[ParsedPage] = []
        full_text_parts: list[str] = []
        first_page_dict: Optional[dict[str, Any]] = None
        first_page_text = ""
        fallback_paragraphs: list[str] = []

        for page_index in range(len(document)):
            page = document.load_page(page_index)
            page_dict = page.get_text("dict")
            page_text = normalize_whitespace(page.get_text("text"))
            if page_index == 0:
                first_page_dict = page_dict
                first_page_text = page_text
            full_text_parts.append(page_text)
            fallback_paragraphs.extend([part for part in page_text.split("\n\n") if part.strip()])

            text_blocks: list[PageTextBlock] = []
            image_blocks: list[PageImageBlock] = []
            for block in page_dict.get("blocks", []):
                block_type = block.get("type")
                bbox = BoundingBox.from_sequence(block.get("bbox", [0, 0, 0, 0]))
                if block_type == 0:
                    block_text, font_size = _extract_block_text_and_font(block)
                    if block_text:
                        text_blocks.append(
                            PageTextBlock(
                                page=page_index + 1,
                                bbox=bbox,
                                text=block_text,
                                block_type="text",
                                font_size=font_size,
                            )
                        )
                elif block_type == 1:
                    image_blocks.append(
                        PageImageBlock(
                            page=page_index + 1,
                            bbox=bbox,
                            width=float(block.get("width", 0) or 0),
                            height=float(block.get("height", 0) or 0),
                        )
                    )

            pixmap = page.get_pixmap(dpi=self._render_dpi, alpha=False)
            image_path = page_images_dir / f"page_{page_index + 1:03d}.png"
            pixmap.save(str(image_path))

            pages.append(
                ParsedPage(
                    page=page_index + 1,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    text=page_text,
                    rendered_image_path=str(image_path),
                    text_blocks=text_blocks,
                    image_blocks=image_blocks,
                )
            )

        full_text = "\n\n".join(part for part in full_text_parts if part)
        title = override_title or extract_title_from_page_dict(
            first_page_dict or {},
            pages[0].height if pages else 0,
            pages[0].width if pages else None,
        )
        abstract = override_abstract or extract_abstract_from_text(first_page_text) or extract_abstract_from_text(full_text)
        fallback_text = build_abstract_fallback(fallback_paragraphs)
        metadata = PaperMetadata(
            title=title or pdf_path.stem,
            abstract=abstract or fallback_text,
            source_pdf=str(pdf_path),
            abstract_fallback_text=fallback_text,
        )
        return ParsedPaper(metadata=metadata, pages=pages, full_text=full_text)


def _extract_block_text_and_font(block: dict[str, Any]) -> tuple[str, Optional[float]]:
    texts: list[str] = []
    font_sizes: list[float] = []
    for line in block.get("lines", []):
        line_parts: list[str] = []
        for span in line.get("spans", []):
            text = str(span.get("text", "")).strip()
            if text:
                line_parts.append(text)
                font_sizes.append(float(span.get("size", 0.0)))
        if line_parts:
            texts.append(" ".join(line_parts))
    block_text = normalize_whitespace("\n".join(texts))
    if not block_text:
        return "", None
    if not font_sizes:
        return block_text, None
    return block_text, sum(font_sizes) / len(font_sizes)


def _is_non_title_block(
    text: str,
    bbox: list[float],
    page_height: float,
    page_width: float | None,
) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized or len(normalized) < 8:
        return True

    lower_text = normalized.lower()
    if "@" in normalized or "www." in lower_text or "http" in lower_text:
        return True
    if ARXIV_PATTERN.search(normalized):
        return True
    if NON_TITLE_PREFIX_PATTERN.match(normalized):
        return True
    if AFFILIATION_HINT_PATTERN.search(normalized):
        return True
    if "equal contribution" in lower_text or "corresponding author" in lower_text:
        return True
    if _looks_like_margin_metadata_block(bbox, page_height, page_width):
        return True
    return False


def _looks_like_margin_metadata_block(
    bbox: list[float],
    page_height: float,
    page_width: float | None,
) -> bool:
    """过滤掉页边竖排的 arXiv 头信息，避免其字号过大时抢走标题。"""
    if page_width is None or page_width <= 0 or page_height <= 0:
        return False
    width = max(float(bbox[2]) - float(bbox[0]), 0.0)
    height = max(float(bbox[3]) - float(bbox[1]), 0.0)
    if width <= 0 or height <= 0:
        return False
    return width <= page_width * 0.12 and height >= page_height * 0.2 and height >= width * 2.0


def _score_title_candidate(candidate: dict[str, Any]) -> float:
    """标题打分优先考虑字号、位置和文本形态，而不是只看最大字号。"""
    text = str(candidate["text"])
    font_size = float(candidate["font_size"])
    y0 = float(candidate["y0"])

    score = font_size * 10.0
    score += min(len(text), 140) * 0.12
    score -= y0 * 0.04

    word_count = len(text.split())
    if 4 <= word_count <= 24:
        score += 8.0
    if _looks_like_author_line(text):
        score -= 25.0
    return score


def _looks_like_author_line(text: str) -> bool:
    words = [word for word in re.split(r"\s+", text) if word]
    capitalized_words = sum(1 for word in words if word[:1].isupper())
    has_author_markers = bool(re.search(r"[\d*†]", text))
    return capitalized_words >= 4 and has_author_markers
