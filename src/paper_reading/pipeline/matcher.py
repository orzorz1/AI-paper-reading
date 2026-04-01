"""图表候选构建与匹配。"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..errors import DependencyMissingError
from ..models import (
    BoundingBox,
    FigureCandidate,
    FigureMatchOption,
    LayoutRegion,
    PageImageBlock,
    PageTextBlock,
    ParsedPaper,
)
from ..utils import ensure_dir

FIGURE_ID_PATTERN = re.compile(r"\b(fig(?:ure)?|table)\.?\s*([0-9]+[a-z]?)", re.IGNORECASE)
CAPTION_DIRECT_HEADER_PATTERN = re.compile(
    r"^\s*(?:fig(?:ure)?|table)\.?\s*[0-9]+[a-z]?\b",
    re.IGNORECASE,
)
SUBFIGURE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:(?:\([a-z]\)|\[[a-z]\]|[a-z]\))\s*(?:[A-Za-z0-9][A-Za-z0-9-]{0,20}\s*)?){1,8}$",
    re.IGNORECASE,
)


def normalize_figure_id(text: str) -> str | None:
    """把各种写法的图表编号统一成 Fig1 / Table1。"""
    match = FIGURE_ID_PATTERN.search(text)
    if not match:
        return None
    prefix = "Table" if match.group(1).lower().startswith("table") else "Fig"
    return f"{prefix}{match.group(2).upper()}"


class FigureMatcher:
    """综合 caption、版面框和原始图片块来构造候选图。"""

    def __init__(self, *, crop_dpi: int = 220) -> None:
        self._crop_dpi = crop_dpi

    def build_candidates(
        self,
        parsed_paper: ParsedPaper,
        layout_by_page: dict[int, list[LayoutRegion]],
        figures_dir: Path,
    ) -> list[FigureCandidate]:
        try:
            import fitz
        except ImportError as exc:
            raise DependencyMissingError("缺少 PyMuPDF，无法裁剪候选图片。") from exc

        ensure_dir(figures_dir)
        document = fitz.open(parsed_paper.metadata.source_pdf)
        grouped: dict[str, list[FigureMatchOption]] = defaultdict(list)
        pages_by_number = {page.page: page for page in parsed_paper.pages}

        for page in parsed_paper.pages:
            layout_regions = layout_by_page.get(page.page, [])
            caption_blocks = _collect_caption_blocks(page.text_blocks, layout_regions)
            figure_regions = [region for region in layout_regions if region.label in {"figure", "table"}]
            for caption_block in caption_blocks:
                normalized_id = normalize_figure_id(caption_block.text)
                if normalized_id is None:
                    continue
                options = self._build_options(caption_block, figure_regions, page.image_blocks)
                if not options:
                    continue
                grouped[normalized_id].extend(options)

        candidates: list[FigureCandidate] = []
        for normalized_id, options in grouped.items():
            options.sort(key=lambda item: item.score, reverse=True)
            best = options[0]
            page_number = best.page
            page_meta = pages_by_number[page_number]
            page = document.load_page(page_number - 1)
            page_width = page_meta.width
            page_height = page_meta.height
            crop_bbox = best.candidate_bbox.with_padding(8.0, max_width=page_width, max_height=page_height)
            image_path = figures_dir / f"{normalized_id}.png"
            # 这里直接使用 PDF 坐标裁图。PDF 与渲染图坐标系都是左上角原点，但单位不同，
            # fitz 的 clip 接受 PDF 坐标，所以不能直接拿页面 PNG 上的像素坐标来裁。
            pixmap = page.get_pixmap(clip=fitz.Rect(*crop_bbox.as_tuple()), dpi=self._crop_dpi, alpha=False)
            pixmap.save(str(image_path))
            candidates.append(
                FigureCandidate(
                    normalized_id=normalized_id,
                    page=page_number,
                    caption_text=best.caption_text,
                    caption_bbox=best.caption_bbox,
                    figure_bbox=best.candidate_bbox,
                    image_path=str(image_path),
                    match_score=best.score,
                    source=best.source,
                    match_options=options,
                )
            )
        candidates.sort(key=lambda item: (item.page, item.normalized_id))
        return candidates

    def _build_options(
        self,
        caption_block: PageTextBlock,
        figure_regions: list[LayoutRegion],
        image_blocks: list[PageImageBlock],
    ) -> list[FigureMatchOption]:
        options: list[FigureMatchOption] = []
        normalized_id = normalize_figure_id(caption_block.text)
        if normalized_id is None:
            return options

        if figure_regions:
            for region in figure_regions:
                score = _score_match(caption_block.bbox, region.bbox, image_blocks)
                options.append(
                    FigureMatchOption(
                        normalized_id=normalized_id,
                        page=caption_block.page,
                        caption_text=caption_block.text,
                        caption_bbox=caption_block.bbox,
                        candidate_bbox=region.bbox,
                        score=score,
                        source="layout+rule",
                    )
                )
        else:
            for image_block in image_blocks:
                score = _score_match(caption_block.bbox, image_block.bbox, image_blocks)
                options.append(
                    FigureMatchOption(
                        normalized_id=normalized_id,
                        page=caption_block.page,
                        caption_text=caption_block.text,
                        caption_bbox=caption_block.bbox,
                        candidate_bbox=image_block.bbox,
                        score=score,
                        source="rule_only",
                    )
                )
        return options


def _collect_caption_blocks(text_blocks: list[PageTextBlock], layout_regions: list[LayoutRegion]) -> list[PageTextBlock]:
    captions: list[PageTextBlock] = []
    caption_regions = [region for region in layout_regions if region.label == "caption"]
    seen: set[tuple[int, int, int, int, str]] = set()

    for block in text_blocks:
        matched = _looks_like_caption_header(block.text)
        if not matched and caption_regions and normalize_figure_id(block.text) is not None:
            matched = any(block.bbox.iou(region.bbox) > 0.1 or _contains_center(region.bbox, block.bbox) for region in caption_regions)
        if not matched:
            continue
        signature = (
            round(block.bbox.x0),
            round(block.bbox.y0),
            round(block.bbox.x1),
            round(block.bbox.y1),
            block.text,
        )
        if signature in seen:
            continue
        seen.add(signature)
        captions.append(block)
    return captions


def _contains_center(container: BoundingBox, item: BoundingBox) -> bool:
    center_x = (item.x0 + item.x1) / 2.0
    center_y = (item.y0 + item.y1) / 2.0
    return container.x0 <= center_x <= container.x1 and container.y0 <= center_y <= container.y1


def _looks_like_caption_header(text: str) -> bool:
    normalized = text.replace("\n", " ").strip()
    if CAPTION_DIRECT_HEADER_PATTERN.search(normalized):
        return True
    match = FIGURE_ID_PATTERN.search(normalized)
    if match is None or match.start() > 80:
        return False
    prefix = normalized[: match.start()].strip()
    return bool(prefix and SUBFIGURE_PREFIX_PATTERN.match(prefix))


def _score_match(caption_bbox: BoundingBox, candidate_bbox: BoundingBox, image_blocks: list[PageImageBlock]) -> float:
    """综合多个启发式分数来判断 caption 与图的匹配度。"""
    vertical_gap = caption_bbox.vertical_distance(candidate_bbox)
    vertical_score = max(0.0, 1.0 - min(vertical_gap, 300.0) / 300.0)
    horizontal_score = min(1.0, caption_bbox.horizontal_overlap_ratio(candidate_bbox))
    # 多数论文 caption 在图下方，这里给予“图在上 caption 在下”更高分。
    relative_bonus = 0.2 if candidate_bbox.y0 <= caption_bbox.y0 else 0.05
    best_iou = max((candidate_bbox.iou(image_block.bbox) for image_block in image_blocks), default=0.0)
    return round(vertical_score * 0.45 + horizontal_score * 0.30 + relative_bonus + best_iou * 0.05, 4)
