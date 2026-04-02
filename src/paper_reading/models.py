"""全局数据模型定义。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

ContentFocus = Literal["method", "experiment"]
OutputLength = Literal["short", "medium", "long"]


class BoundingBox(BaseModel):
    """PDF 或图像坐标系下的矩形框。"""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float, float, float]) -> "BoundingBox":
        return cls(x0=float(values[0]), y0=float(values[1]), x1=float(values[2]), y1=float(values[3]))

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        inter_x0 = max(self.x0, other.x0)
        inter_y0 = max(self.y0, other.y0)
        inter_x1 = min(self.x1, other.x1)
        inter_y1 = min(self.y1, other.y1)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h
        if inter_area <= 0 or self.area <= 0 or other.area <= 0:
            return 0.0
        return inter_area / (self.area + other.area - inter_area)

    def vertical_distance(self, other: "BoundingBox") -> float:
        if self.y1 < other.y0:
            return other.y0 - self.y1
        if other.y1 < self.y0:
            return self.y0 - other.y1
        return 0.0

    def horizontal_overlap_ratio(self, other: "BoundingBox") -> float:
        overlap = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        base = min(self.width, other.width)
        if base <= 0:
            return 0.0
        return overlap / base

    def with_padding(self, padding: float, max_width: Optional[float] = None, max_height: Optional[float] = None) -> "BoundingBox":
        x0 = max(0.0, self.x0 - padding)
        y0 = max(0.0, self.y0 - padding)
        x1 = self.x1 + padding
        y1 = self.y1 + padding
        if max_width is not None:
            x1 = min(max_width, x1)
        if max_height is not None:
            y1 = min(max_height, y1)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class PaperMetadata(BaseModel):
    """论文基础信息。"""

    title: str
    abstract: str
    source_pdf: str
    abstract_fallback_text: str = ""


class PageTextBlock(BaseModel):
    """PDF 文本块。"""

    page: int
    bbox: BoundingBox
    text: str
    block_type: str = "text"
    font_size: Optional[float] = None


class PageImageBlock(BaseModel):
    """PDF 内原始图片块。"""

    page: int
    bbox: BoundingBox
    width: Optional[float] = None
    height: Optional[float] = None


class ParsedPage(BaseModel):
    """解析后的单页结果。"""

    page: int
    width: float
    height: float
    text: str
    rendered_image_path: str
    text_blocks: list[PageTextBlock] = Field(default_factory=list)
    image_blocks: list[PageImageBlock] = Field(default_factory=list)


class ParsedPaper(BaseModel):
    """解析后的整篇论文。"""

    metadata: PaperMetadata
    pages: list[ParsedPage]
    full_text: str


class LayoutRegion(BaseModel):
    """版面识别区域。"""

    page: int
    bbox: BoundingBox
    label: Literal["figure", "table", "caption", "text"]
    score: float


class FigureMatchOption(BaseModel):
    """同一个图编号可能命中的候选区域。"""

    normalized_id: str
    page: int
    caption_text: str
    caption_bbox: BoundingBox
    candidate_bbox: BoundingBox
    score: float
    source: Literal["layout+rule", "rule_only"]


class FigureCandidate(BaseModel):
    """最终的图表候选。"""

    normalized_id: str
    page: int
    caption_text: str
    caption_bbox: BoundingBox
    figure_bbox: BoundingBox
    image_path: str
    match_score: float
    source: Literal["layout+rule", "rule_only"]
    match_options: list[FigureMatchOption] = Field(default_factory=list)


class SelectedFigure(BaseModel):
    """模型挑出的关键图。"""

    normalized_id: str
    reason: str
    importance_rank: int


class StoryOutline(BaseModel):
    """整体阅读结构。"""

    title_translation: str = ""
    one_sentence_summary: str
    motivation: str
    method_core: str
    result_summary: str = ""
    figure_roles: dict[str, str]
    takeaways: list[str] = Field(default_factory=list)


class FigureExplanation(BaseModel):
    """单图解释。"""

    normalized_id: str
    title: str
    what_it_shows: str
    how_to_read: str
    why_it_matters: str
    reader_takeaway: str = ""


class DocumentSection(BaseModel):
    """最终文档章节。"""

    title: str
    paragraphs: list[str]
    image_path: Optional[str] = None
    image_after_paragraph: int = 0
    caption: Optional[str] = None
    figure_id: Optional[str] = None


class FinalDocument(BaseModel):
    """最终可阅读文档。"""

    title: str
    title_translation: str = ""
    one_sentence_summary: str
    sections: list[DocumentSection]
    takeaways: list[str] = Field(default_factory=list)
    asset_paths: list[str]


class BuildResult(BaseModel):
    """构建完成后的输出摘要。"""

    markdown_path: str
    pdf_path: str = ""
    output_dir: str
    artifact_dir: str
    selected_count: int
    warnings: list[str] = Field(default_factory=list)


class BuildOptions(BaseModel):
    """CLI 传入的运行参数。"""

    pdf_path: Path
    title: Optional[str] = None
    abstract: Optional[str] = None
    output_dir: Optional[Path] = None
    max_figures: Optional[int] = None
    lang: str = "zh-CN"
    content_focus: ContentFocus = "method"
    output_length: OutputLength = "medium"
