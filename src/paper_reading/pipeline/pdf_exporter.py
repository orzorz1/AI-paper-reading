"""Markdown 转 PDF 导出。"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from ..errors import DependencyMissingError, PipelineExecutionError


class MarkdownPdfExporter:
    """把最终 Markdown 渲染成可阅读的 PDF。"""

    def export(self, markdown_path: Path, output_path: Path) -> Path:
        """读取 Markdown 并导出 PDF。"""
        (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            RLImage,
            PageBreak,
            KeepTogether,
            A4,
            colors,
            getSampleStyleSheet,
            ParagraphStyle,
            pdfmetrics,
            UnicodeCIDFont,
        ) = _import_reportlab()
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName="STSong-Light",
            fontSize=10.5,
            leading=16,
            spaceAfter=10,
            wordWrap="CJK",
            splitLongWords=False,
        )
        title_style = ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName="STSong-Light",
            fontSize=19,
            leading=24,
            spaceAfter=10,
            textColor=colors.HexColor("#111111"),
        )
        subtitle_style = ParagraphStyle(
            "SubtitleCN",
            parent=body_style,
            fontName="STSong-Light",
            fontSize=13,
            leading=18,
            spaceAfter=12,
            textColor=colors.HexColor("#475569"),
            wordWrap="CJK",
            splitLongWords=False,
        )
        section_style = ParagraphStyle(
            "SectionCN",
            parent=styles["Heading2"],
            fontName="STSong-Light",
            fontSize=15.5,
            leading=22,
            spaceBefore=16,
            spaceAfter=12,
            textColor=colors.HexColor("#0f172a"),
            backColor=colors.HexColor("#e8f0fe"),
            borderWidth=0.8,
            borderColor=colors.HexColor("#93c5fd"),
            borderPadding=7,
            wordWrap="CJK",
            splitLongWords=False,
        )
        quote_style = ParagraphStyle(
            "QuoteCN",
            parent=body_style,
            fontName="STSong-Light",
            textColor=colors.HexColor("#374151"),
            backColor=colors.HexColor("#f3f4f6"),
            borderPadding=8,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=12,
        )
        list_style = ParagraphStyle(
            "ListCN",
            parent=body_style,
            fontName="STSong-Light",
            leftIndent=18,
            firstLineIndent=0,
            bulletIndent=4,
            spaceBefore=1,
            spaceAfter=4,
            wordWrap="CJK",
            splitLongWords=False,
        )

        markdown_text = markdown_path.read_text(encoding="utf-8")
        blocks = _parse_markdown_blocks(markdown_text)
        story = []
        base_dir = markdown_path.parent
        page_width, page_height = A4
        max_image_width = page_width - 96
        max_image_height = page_height * 0.42
        seen_h1 = False
        consumed_subtitle = False

        for block_type, payload in blocks:
            if block_type == "h1":
                story.append(Paragraph(_format_inline(payload), title_style))
                story.append(Spacer(1, 6))
                seen_h1 = True
                continue
            if block_type == "paragraph" and seen_h1 and not consumed_subtitle:
                story.append(_keep_short_block_together(Paragraph(_format_inline(payload), subtitle_style), Spacer(1, 0), KeepTogether))
                consumed_subtitle = True
                continue
            if block_type == "h2":
                story.append(_keep_short_block_together(Paragraph(_format_inline(payload), section_style), Spacer(1, 4), KeepTogether))
                continue
            if block_type == "quote":
                story.append(_keep_short_block_together(Paragraph(_format_inline(payload), quote_style), Spacer(1, 0), KeepTogether))
                continue
            if block_type == "list_item":
                story.append(Paragraph(_format_inline(payload), list_style, bulletText="•"))
                continue
            if block_type == "image":
                image_path = (base_dir / payload).resolve()
                if not image_path.exists():
                    story.append(Paragraph(_format_inline(f"图片缺失：{payload}"), body_style))
                    continue
                image = RLImage(str(image_path))
                image._restrictSize(max_image_width, max_image_height)
                story.append(image)
                story.append(Spacer(1, 10))
                continue
            if block_type == "pagebreak":
                story.append(PageBreak())
                continue
            story.append(_keep_short_block_together(Paragraph(_format_inline(payload), body_style), Spacer(1, 0), KeepTogether))

        try:
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                leftMargin=48,
                rightMargin=48,
                topMargin=48,
                bottomMargin=48,
                title=blocks[0][1] if blocks and blocks[0][0] == "h1" else "paper_readable",
            )
            doc.build(story)
        except Exception as exc:
            raise PipelineExecutionError(f"PDF 导出失败：{exc}") from exc
        return output_path


def _import_reportlab() -> tuple:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise DependencyMissingError("缺少 reportlab，请先安装 PDF 导出依赖。") from exc

    return (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        RLImage,
        PageBreak,
        KeepTogether,
        A4,
        colors,
        getSampleStyleSheet,
        ParagraphStyle,
        pdfmetrics,
        UnicodeCIDFont,
    )


def _parse_markdown_blocks(markdown_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    paragraph_lines: list[str] = []
    quote_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            blocks.append(("paragraph", " ".join(line.strip() for line in paragraph_lines if line.strip())))
            paragraph_lines.clear()

    def flush_quote() -> None:
        if quote_lines:
            blocks.append(("quote", " ".join(line.strip() for line in quote_lines if line.strip())))
            quote_lines.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        image_match = re.match(r"^!\[[^\]]*\]\((.+?)\)$", stripped)
        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered_match = re.match(r"^\d+\.\s+(.+)$", stripped)

        if not stripped:
            flush_paragraph()
            flush_quote()
            continue
        if stripped == "---":
            flush_paragraph()
            flush_quote()
            blocks.append(("pagebreak", ""))
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            flush_quote()
            blocks.append(("h1", stripped[2:].strip()))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_quote()
            blocks.append(("h2", stripped[3:].strip()))
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            quote_lines.append(stripped[2:].strip())
            continue
        if image_match:
            flush_paragraph()
            flush_quote()
            blocks.append(("image", image_match.group(1).strip()))
            continue
        if bullet_match or ordered_match:
            flush_paragraph()
            flush_quote()
            blocks.append(("list_item", (bullet_match or ordered_match).group(1).strip()))
            continue

        flush_quote()
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_quote()
    return blocks


def _format_inline(text: str) -> str:
    escaped = escape(_prepare_text_for_pdf(text))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font face=\"Courier\">\1</font>", escaped)
    return escaped.replace("\n", "<br/>")


def _prepare_text_for_pdf(text: str) -> str:
    """尽量减少中英混排时的异常断行。

    ReportLab 在中文段落里遇到短英文、数字和中文量词挨在一起时，
    有时会把 `391个`、`7万` 这类片段拆开。这里用 word joiner 把
    “ASCII/数字 token 与紧邻的中文字符”轻量绑定，避免阅读体验很差。
    """
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])", "\u2060", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z][A-Za-z0-9.-]{1,})", "\u2060", text)
    return text


def _keep_short_block_together(paragraph, spacer, keep_together):
    return keep_together([paragraph, spacer])
