"""Markdown 转 PDF 导出。"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from ..errors import DependencyMissingError, PipelineExecutionError


class MarkdownPdfExporter:
    """把最终 Markdown 渲染成可阅读的 PDF。"""

    def export(self, markdown_path: Path, output_path: Path) -> Path:
        """读取 Markdown 并优先通过 TeX 引擎导出 PDF。"""
        markdown_text = markdown_path.read_text(encoding="utf-8")
        tex_engine = _find_tex_engine()
        if tex_engine is None:
            return _export_with_reportlab(markdown_path, output_path, markdown_text)
        engine_kind, engine_path = tex_engine

        title = _extract_document_title(markdown_text) or output_path.stem
        tex_text = render_markdown_to_tex(markdown_text, title=title, base_dir=markdown_path.parent)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="paper-reading-tex-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            tex_path = temp_dir / "paper_readable.tex"
            tex_path.write_text(tex_text, encoding="utf-8")
            command = _build_tex_command(engine_kind, engine_path, tex_path=tex_path, output_dir=temp_dir)
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                raise DependencyMissingError(f"无法执行 TeX 引擎 {engine_path}：{exc}") from exc

            if result.returncode != 0:
                log_excerpt = _tail_text((result.stdout or "") + "\n" + (result.stderr or ""))
                raise PipelineExecutionError(f"PDF 导出失败：TeX 编译出错。\n{log_excerpt}")

            generated_pdf = temp_dir / "paper_readable.pdf"
            if not generated_pdf.exists():
                raise PipelineExecutionError("PDF 导出失败：TeX 编译完成后没有生成 PDF 文件。")
            shutil.copyfile(generated_pdf, output_path)
        return output_path


def render_markdown_to_tex(markdown_text: str, *, title: str, base_dir: Path) -> str:
    blocks = _parse_markdown_blocks(markdown_text)
    body = _render_blocks_to_tex(blocks, base_dir=base_dir)
    return _wrap_tex_document(title=title, body=body)


def _find_tex_engine() -> tuple[str, str] | None:
    for candidate, kind in (("tectonic", "tectonic"), ("xelatex", "latex"), ("lualatex", "latex")):
        executable = shutil.which(candidate)
        if executable:
            return kind, executable
    return None


def _build_tex_command(engine_kind: str, engine_path: str, *, tex_path: Path, output_dir: Path) -> list[str]:
    if engine_kind == "tectonic":
        return [
            engine_path,
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(output_dir),
            str(tex_path),
        ]
    return [
        engine_path,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(output_dir),
        str(tex_path),
    ]


def _wrap_tex_document(*, title: str, body: str) -> str:
    return (
        "\\documentclass[11pt]{ctexart}\n"
        "\\usepackage[a4paper,margin=2.2cm]{geometry}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{float}\n"
        "\\usepackage{amsmath,amssymb,bm}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{setspace}\n"
        "\\setlength{\\parindent}{0pt}\n"
        "\\setlength{\\parskip}{0.65em}\n"
        "\\setstretch{1.25}\n"
        "\\pagestyle{plain}\n"
        "\\begin{document}\n"
        f"\\hypersetup{{pdftitle={{{_escape_tex_text(title)}}}}}\n"
        f"{body}\n"
        "\\end{document}\n"
    )


def _render_blocks_to_tex(blocks: list[tuple[str, str]], *, base_dir: Path) -> str:
    parts: list[str] = []
    seen_h1 = False
    consumed_subtitle = False
    index = 0

    while index < len(blocks):
        block_type, payload = blocks[index]
        if block_type == "h1":
            parts.append(_render_title_block(payload))
            seen_h1 = True
            index += 1
            continue
        if block_type == "paragraph" and seen_h1 and not consumed_subtitle:
            parts.append(_render_subtitle_block(payload))
            consumed_subtitle = True
            index += 1
            continue
        if block_type == "h2":
            parts.append(f"\\section*{{{_render_inline_tex(payload)}}}")
            index += 1
            continue
        if block_type == "h3":
            parts.append(f"\\subsection*{{{_render_inline_tex(payload)}}}")
            index += 1
            continue
        if block_type == "quote":
            parts.append("\\begin{quote}\\small")
            parts.append(_render_inline_tex(payload))
            parts.append("\\end{quote}")
            index += 1
            continue
        if block_type == "pagebreak":
            parts.append("\\newpage")
            index += 1
            continue
        if block_type == "image":
            parts.append(_render_image_block(payload, base_dir=base_dir))
            index += 1
            continue
        if block_type == "list_item":
            items: list[str] = []
            while index < len(blocks) and blocks[index][0] == "list_item":
                items.append(blocks[index][1])
                index += 1
            parts.append(_render_list_block(items))
            continue
        parts.append(_render_paragraph_block(payload))
        index += 1

    return "\n\n".join(part for part in parts if part.strip())


def _render_title_block(text: str) -> str:
    return "\\begin{center}\n{\\LARGE\\bfseries " + _render_inline_tex(text) + "}\\\\[0.8em]\n\\end{center}"


def _render_subtitle_block(text: str) -> str:
    return "\\begin{center}\n{\\large\\color{gray} " + _render_inline_tex(text) + "}\\\\[0.5em]\n\\end{center}"


def _render_paragraph_block(text: str) -> str:
    return _render_inline_tex(text) + "\n\\par"


def _render_list_block(items: list[str]) -> str:
    rendered = ["\\begin{itemize}[leftmargin=1.4em,itemsep=0.3em]"]
    for item in items:
        rendered.append(f"\\item {_render_inline_tex(item)}")
    rendered.append("\\end{itemize}")
    return "\n".join(rendered)


def _render_image_block(path_text: str, *, base_dir: Path) -> str:
    image_path = (base_dir / path_text).resolve()
    if not image_path.exists():
        return "\\fbox{\\parbox{0.9\\linewidth}{图片缺失：" + _escape_tex_text(path_text) + "}}"
    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.92\\linewidth]{{\\detokenize{{{image_path.as_posix()}}}}}\n"
        "\\end{figure}"
    )


def _render_inline_tex(text: str) -> str:
    parts: list[str] = []
    last_end = 0
    for match in _MATH_PATTERN.finditer(text):
        if match.start() > last_end:
            parts.append(_render_text_segment(text[last_end : match.start()]))
        parts.append(match.group(0))
        last_end = match.end()
    if last_end < len(text):
        parts.append(_render_text_segment(text[last_end:]))
    return "".join(parts)


def _render_text_segment(text: str) -> str:
    if not text:
        return ""

    parts: list[str] = []
    last_end = 0
    for match in re.finditer(r"`([^`]+)`|\*\*(.+?)\*\*", text):
        if match.start() > last_end:
            parts.append(_escape_tex_text(text[last_end : match.start()]))
        code_content, bold_content = match.groups()
        if code_content is not None:
            parts.append(r"\texttt{" + _escape_tex_text(code_content) + "}")
        elif bold_content is not None:
            parts.append(r"\textbf{" + _render_text_segment(bold_content) + "}")
        last_end = match.end()
    if last_end < len(text):
        parts.append(_escape_tex_text(text[last_end:]))
    return "".join(parts)


def _escape_tex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "^": r"\^{}",
        "~": r"\textasciitilde{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    escaped = escaped.replace("\u2060", "")
    return escaped


def _extract_document_title(markdown_text: str) -> str:
    for block_type, payload in _parse_markdown_blocks(markdown_text):
        if block_type == "h1":
            return payload
    return ""


def _tail_text(text: str, *, max_chars: int = 1200) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


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
        if stripped.startswith("### "):
            flush_paragraph()
            flush_quote()
            blocks.append(("h3", stripped[4:].strip()))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_quote()
            blocks.append(("h2", stripped[3:].strip()))
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            flush_quote()
            blocks.append(("h1", stripped[2:].strip()))
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


_MATH_PATTERN = re.compile(
    r"\\\(.+?\\\)|\\\[.+?\\\]|\$\$.+?\$\$|(?<!\$)\$(.+?)(?<!\$)\$",
    re.DOTALL,
)

_CN_FONT = "CNFont"
_CN_FONT_BOLD = "CNFont-Bold"


def _export_with_reportlab(markdown_path: Path, output_path: Path, markdown_text: str) -> Path:
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
        TTFont,
    ) = _import_reportlab()
    font_name = _register_chinese_fonts(pdfmetrics, TTFont)

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        spaceAfter=10,
        wordWrap="CJK",
        splitLongWords=False,
    )
    title_style = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=19,
        leading=24,
        spaceAfter=10,
        textColor=colors.HexColor("#111111"),
    )
    subtitle_style = ParagraphStyle(
        "SubtitleCN",
        parent=body_style,
        fontName=font_name,
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
        fontName=font_name,
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
    subsection_style = ParagraphStyle(
        "SubsectionCN",
        parent=styles["Heading3"],
        fontName=font_name,
        fontSize=12.5,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
        textColor=colors.HexColor("#1e293b"),
        wordWrap="CJK",
        splitLongWords=False,
    )
    quote_style = ParagraphStyle(
        "QuoteCN",
        parent=body_style,
        fontName=font_name,
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
        fontName=font_name,
        leftIndent=18,
        firstLineIndent=0,
        bulletIndent=4,
        spaceBefore=1,
        spaceAfter=4,
        wordWrap="CJK",
        splitLongWords=False,
    )

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
        if block_type == "h3":
            story.append(_keep_short_block_together(Paragraph(_format_inline(payload), subsection_style), Spacer(1, 2), KeepTogether))
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
        from reportlab.pdfbase.ttfonts import TTFont
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
        TTFont,
    )


def _register_chinese_fonts(pdfmetrics, TTFont) -> str:
    system = platform.system()
    candidates: list[tuple[str, int, int | None]] = []
    if system == "Darwin":
        songti = "/System/Library/Fonts/Supplemental/Songti.ttc"
        if Path(songti).exists():
            candidates.append((songti, 6, 1))
    if system == "Linux":
        for p in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        ]:
            if Path(p).exists():
                candidates.append((p, 0, None))
    if system == "Windows":
        simsun = "C:/Windows/Fonts/simsun.ttc"
        if Path(simsun).exists():
            candidates.append((simsun, 0, None))

    for font_path, regular_idx, bold_idx in candidates:
        try:
            pdfmetrics.registerFont(TTFont(_CN_FONT, font_path, subfontIndex=regular_idx))
            if bold_idx is not None:
                pdfmetrics.registerFont(TTFont(_CN_FONT_BOLD, font_path, subfontIndex=bold_idx))
                pdfmetrics.registerFontFamily(_CN_FONT, normal=_CN_FONT, bold=_CN_FONT_BOLD)
            return _CN_FONT
        except Exception:
            continue

    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def _format_inline(text: str) -> str:
    escaped = escape(_prepare_text_for_pdf(_normalize_latex_fragments(text)))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", escaped)
    return escaped.replace("\n", "<br/>")


def _prepare_text_for_pdf(text: str) -> str:
    text = re.sub("[\u2060\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])", "\u2060", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z][A-Za-z0-9.-]{1,})", "\u2060", text)
    return text


def _normalize_latex_fragments(text: str) -> str:
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"[{_latex_to_plain_text(m.group(1))}]", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: _latex_to_plain_text(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: f"[{_latex_to_plain_text(m.group(1))}]", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$(.+?)(?<!\$)\$", lambda m: _latex_to_plain_text(m.group(1)), text, flags=re.DOTALL)
    return text


def _latex_to_plain_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    for command, replacement in {
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ϵ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\rho": "ρ",
        r"\sigma": "σ",
        r"\tau": "τ",
        r"\phi": "φ",
        r"\omega": "ω",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\cdot": "·",
        r"\times": "×",
    }.items():
        normalized = normalized.replace(command, replacement)
    normalized = re.sub(r"\\(?:mathbf|mathrm|mathit|boldsymbol)\s*\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\(?:mathbf|mathrm|mathit|boldsymbol)\s+([A-Za-z])", r"\1", normalized)
    while True:
        updated = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", normalized)
        if updated == normalized:
            break
        normalized = updated
    normalized = _replace_integrals(normalized)
    normalized = re.sub(r"([A-Za-z0-9])\^\{([^{}]+)\}", r"\1^(\2)", normalized)
    normalized = re.sub(r"([A-Za-z0-9])_\\?\{([^{}]+)\}", r"\1_(\2)", normalized)
    normalized = re.sub(r"([A-Za-z0-9])\^([A-Za-z0-9+\-]+)", r"\1^(\2)", normalized)
    normalized = re.sub(r"([A-Za-z0-9])_([A-Za-z0-9+\-]+)", r"\1_(\2)", normalized)
    normalized = normalized.replace(r"\int", "∫")
    normalized = re.sub(r"\\left|\\right", "", normalized)
    normalized = re.sub(r"\\,", " ", normalized)
    normalized = re.sub(r"\\!", "", normalized)
    normalized = re.sub(r"\\;", " ", normalized)
    normalized = re.sub(r"\\:", " ", normalized)
    normalized = re.sub(r"\\([A-Za-z]+)", r"\1", normalized)
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _replace_integrals(text: str) -> str:
    marker = r"\int_"
    start = text.find(marker)
    if start == -1:
        return text
    parts: list[str] = []
    cursor = 0
    while start != -1:
        parts.append(text[cursor:start])
        lower, after_lower = _read_latex_bound(text, start + len(marker))
        if after_lower >= len(text) or text[after_lower] != "^":
            parts.append(marker)
            cursor = start + len(marker)
            start = text.find(marker, cursor)
            continue
        upper, after_upper = _read_latex_bound(text, after_lower + 1)
        parts.append(f"∫[{_normalize_bound_text(lower)}, {_normalize_bound_text(upper)}]")
        cursor = after_upper
        start = text.find(marker, cursor)
    parts.append(text[cursor:])
    return "".join(parts)


def _read_latex_bound(text: str, start: int) -> tuple[str, int]:
    if start >= len(text):
        return "", start
    if text[start] == "{":
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1 : index], index + 1
        return text[start + 1 :], len(text)
    end = start
    while end < len(text) and not text[end].isspace() and text[end] not in "_^":
        end += 1
    return text[start:end], end


def _normalize_bound_text(text: str) -> str:
    normalized = re.sub(r"\\(?:mathbf|mathrm|mathit|boldsymbol)\s*\{([^{}]+)\}", r"\1", text)
    normalized = re.sub(r"\\(?:mathbf|mathrm|mathit|boldsymbol)\s+([A-Za-z])", r"\1", normalized)
    normalized = re.sub(r"([A-Za-z0-9])\^\{([^{}]+)\}", r"\1^(\2)", normalized)
    normalized = re.sub(r"([A-Za-z0-9])_\\?\{([^{}]+)\}", r"\1_(\2)", normalized)
    normalized = re.sub(r"([A-Za-z0-9])\^([A-Za-z0-9+\-]+)", r"\1^(\2)", normalized)
    normalized = re.sub(r"([A-Za-z0-9])_([A-Za-z0-9+\-]+)", r"\1_(\2)", normalized)
    return normalized.replace("{", "").replace("}", "").strip()


def _keep_short_block_together(paragraph, spacer, keep_together):
    return keep_together([paragraph, spacer])
