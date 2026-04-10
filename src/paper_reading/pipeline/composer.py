"""Markdown 输出模块。"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from ..llm import OpenAICompatibleClient
from ..models import ContentFocus, DocumentSection, FigureCandidate, FigureExplanation, FinalDocument, OutputLength, ReportStructure, StoryOutline, WritingStyle
from ..utils import relative_posix_path, truncate_text


class MarkdownComposer:
    """把结构化结果拼成最终 Markdown。"""

    def __init__(
        self,
        client: Optional[OpenAICompatibleClient] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._client = client
        self._model_name = model_name

    def compose(
        self,
        *,
        title: str,
        outline: StoryOutline,
        report_structure: ReportStructure | None = None,
        ordered_candidates: list[FigureCandidate],
        explanations: list[FigureExplanation],
        output_dir: Path,
    ) -> FinalDocument:
        report_structure = report_structure or ReportStructure()
        explanation_by_id = {item.normalized_id: item for item in explanations}
        sections = self._build_dynamic_sections(
            outline=outline,
            report_structure=report_structure,
            ordered_candidates=ordered_candidates,
            explanation_by_id=explanation_by_id,
            output_dir=output_dir,
        )

        asset_paths = [section.image_path for section in sections if section.image_path]
        return FinalDocument(
            title=title,
            title_translation=outline.title_translation.strip(),
            one_sentence_summary=_polish_text(
                _compress_paragraph(outline.one_sentence_summary, max_sentences=2, max_chars=120)
            ),
            sections=sections,
            takeaways=[],
            asset_paths=asset_paths,
        )

    def generate_markdown(
        self,
        *,
        title: str,
        abstract: str,
        paper_context: str,
        outline: StoryOutline,
        report_structure: ReportStructure | None = None,
        selected_figures: list[dict[str, str | int]],
        ordered_candidates: list[FigureCandidate],
        explanations: list[FigureExplanation],
        output_dir: Path,
        content_focus: ContentFocus = "method",
        output_length: OutputLength = "medium",
        writing_style: WritingStyle = "professional",
    ) -> str:
        """优先交给 LLM 直接写 Markdown；缺少 LLM 时回退到规则渲染。"""
        report_structure = report_structure or ReportStructure()
        final_document = self.compose(
            title=title,
            outline=outline,
            report_structure=report_structure,
            ordered_candidates=ordered_candidates,
            explanations=explanations,
            output_dir=output_dir,
        )
        if self._client is None or not self._model_name:
            markdown = self.render_markdown(final_document)
            return _deduplicate_acronym_annotations(markdown)

        explanation_by_id = {item.normalized_id: item for item in explanations}
        selected_order = [item["normalized_id"] for item in selected_figures]
        figure_context: list[dict[str, str | int]] = []
        for candidate in ordered_candidates:
            explanation = explanation_by_id[candidate.normalized_id]
            figure_context.append(
                {
                    "id": candidate.normalized_id,
                    "page": candidate.page,
                    "role": outline.figure_roles.get(candidate.normalized_id, "support"),
                    "image_path": relative_posix_path(Path(candidate.image_path), output_dir),
                    "reason": next(
                        (str(item["reason"]) for item in selected_figures if item["normalized_id"] == candidate.normalized_id),
                        "",
                    ),
                    "caption": candidate.caption_text,
                    "title": explanation.title,
                    "what_it_shows": explanation.what_it_shows,
                    "how_to_read": explanation.how_to_read,
                    "why_it_matters": explanation.why_it_matters,
                }
            )

        system_prompt = (
            "你要直接撰写一份可阅读的中文 Markdown，用来帮助读者在 1 到 2 分钟内读懂一篇 AI 论文。"
            "你会拿到论文标题、摘要、较长上下文、结构化提炼结果，以及每张已选图片的路径和解释。"
            "请把这些材料整合成自然、连贯、信息密度高的 Markdown，并服从给定的内容偏好和篇幅要求。"
            "只能依据给定材料写作，不能补充文章里没有明确出现的事实、贡献层级或历史判断。"
        )
        focus_instruction = {
            "method": "正文整体偏向方法，优先把方法主线讲清楚。",
            "experiment": "正文整体偏向实验，优先把评测任务、实验设计和结果含义讲清楚。",
        }[content_focus]
        length_instruction = {
            "short": "短篇输出：尽量压到 1 张图和 2 到 3 个小节，整体非常凝练。",
            "medium": "中篇输出：保持当前默认长度和信息密度。",
            "long": "长篇输出：允许更充实，通常写 4 到 6 个小节。无论偏向方法还是偏向实验，都要同时覆盖方法和实验。",
        }[output_length]
        formula_instruction = {
            "short": "除非特别必要，否则不要主动引入公式。",
            "medium": "如果论文里有对理解方法、结论或核心论证确实重要的公式，可以保留关键公式，并配一句简短解释；不要写成公式堆砌。",
            "long": "如果论文里存在对理解主线重要的公式，可以保留关键公式，并在正文里配一句通俗解释。",
        }[output_length]
        style_instruction = {
            "professional": "标题和正文都要更书面、正式、专业，少用口语化表达。",
            "colloquial": "标题和正文都要更平实、更好懂，像在给读者讲明白，但仍要保持文档质量。",
        }[writing_style]
        structure_instruction = {
            ("method", "short"): "小节布局建议：导读摘要 -> 问题与任务 -> 方法主线。实验只在确实关键时用 1 段带过。",
            ("method", "medium"): "小节布局建议：研究背景与任务定义 -> 方法设计与核心机制 -> 可选的实验结果。",
            ("method", "long"): "小节布局建议：研究背景 -> 方法设计 -> 关键模块/数据 -> 实验结果与分析。方法部分篇幅必须明显多于实验。",
            ("experiment", "short"): "小节布局建议：导读摘要 -> 研究背景与任务定义 -> 方法主线 -> 结果结论。即使偏向实验，也要先把动机和方法讲清楚。",
            ("experiment", "medium"): "小节布局建议：研究背景与任务定义 -> 方法主线 -> 评测设定与实验结果。实验部分要更突出，但顺序上仍然先方法后实验。",
            ("experiment", "long"): "小节布局建议：研究背景 -> 方法主线 -> 评测设定 -> 实验结果 -> 结果解读。实验部分篇幅必须明显多于方法，但顺序上必须先讲方法再讲实验。",
        }[(content_focus, output_length)]
        user_prompt = (
            "写作要求：\n"
            "- 最终输出必须是 Markdown 正文，不要输出 JSON，不要加解释前言\n"
            "- 只能输出一份完整文档，从一级标题开始，到最后一个小节结束；不要在中途重新从标题开始再写一遍\n"
            "- 一级标题只能出现 1 次，中文题目只能出现 1 次，`> 导读摘要：...` 只能出现 1 次\n"
            "- 不要在 `# 英文标题` 之前额外输出任何文字\n"
            "- 写完最后一个小节后就直接结束，不要追加第二版内容，不要重复任何已经写过的小节\n"
            "- 开头必须依次包含：一级标题（英文原题）、一行中文题目、一个 `> 导读摘要：...` 引导块\n"
            "- 中文题目单独占一行，不要写成 `**中文题目：** xxx`\n"
            "- 图片必须使用给定的相对路径，格式严格写成 `![FigX](path)`\n"
            "- 每张图片前后都必须各留一个空行，确保 Markdown 可以把图片正确解析成独立块，不要把图片紧贴上一段或下一段文字\n"
            "- 对给定的所有已选图片，都要在正文中自然使用一次，不要漏掉\n"
            "- 不要把内容写成“关键图1/2/3”这种机械结构\n"
            "- 可以自由决定 3 到 5 个正式小节标题，但要自然、书面，不要太口语化\n"
            "- 不要引用原始 caption，不要捏造图片路径\n"
            "- 对不太常见、但理解论文必须知道的缩写，例如 RES、MRES，第一次出现时可写成“RES（中文）”；后面直接写缩写\n"
            "- 对常见缩写，例如 CNN、LLM、SOTA，不要专门解释\n"
            "- 不要写英文全称，正文只保留必要中文括注\n"
            "- 不要套用固定模板句，尽量让语气自然，直接成文\n\n"
            "- 不要擅自写“首次提出”“首次建立”“首次实现”“开创性”等表述，除非给定材料里明确这样说\n"
            "- 如果文章只是给出方法、量表、实验或讨论，就按证据如实表述，不要上升成更强的历史判断\n\n"
            "- 输出前请自检：标题没有重复，小节没有重复，文档没有第二次从头开始\n\n"
            f"内容偏好：{content_focus}\n"
            f"篇幅：{output_length}\n"
            f"风格：{writing_style}\n"
            f"{focus_instruction}\n"
            f"{length_instruction}\n"
            f"{structure_instruction}\n"
            f"{formula_instruction}\n"
            f"{style_instruction}\n\n"
            f"论文标题：{title}\n\n"
            f"论文摘要：{abstract}\n\n"
            f"论文上下文：{truncate_text(paper_context, 32000)}\n\n"
            f"报告结构建议：\n{json.dumps(report_structure.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n\n"
            f"结构化提炼结果：\n{json.dumps(outline.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n\n"
            f"已选图片顺序：{selected_order}\n\n"
            f"已选图片与解释：\n{json.dumps(figure_context, ensure_ascii=False, indent=2)}\n"
        )
        markdown = self._client.complete_text(
            model=self._model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
        )
        markdown = _ensure_markdown_header(markdown.strip() + "\n", final_document)
        markdown = _ensure_all_selected_figures_present(markdown.strip() + "\n", figure_context)
        markdown = _normalize_markdown_image_spacing(markdown)
        markdown = _deduplicate_full_document_repetition(markdown)
        return _deduplicate_acronym_annotations(_normalize_markdown_header(markdown))

    def render_markdown(self, document: FinalDocument) -> str:
        """把 FinalDocument 转成 Markdown 字符串。"""
        lines = [f"# {document.title}", ""]
        if document.title_translation:
            lines.extend([document.title_translation, ""])
        lines.extend([f"> 导读摘要：{document.one_sentence_summary}", ""])
        for section in document.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            for index, paragraph in enumerate(section.paragraphs, start=1):
                if paragraph:
                    lines.append(_polish_text(paragraph))
                    lines.append("")
                if section.image_path and section.image_after_paragraph == index:
                    alt = section.figure_id or section.title
                    lines.append(f"![{alt}]({section.image_path})")
                    lines.append("")
        markdown = "\n".join(lines).strip() + "\n"
        markdown = _normalize_markdown_image_spacing(markdown)
        markdown = _deduplicate_full_document_repetition(markdown)
        return _deduplicate_acronym_annotations(_normalize_markdown_header(markdown))

    def _group_figures_by_role(
        self,
        ordered_candidates: list[FigureCandidate],
        figure_roles: dict[str, str],
    ) -> dict[str, FigureCandidate]:
        buckets: dict[str, FigureCandidate] = {}
        for candidate in ordered_candidates:
            role = _normalize_role(figure_roles.get(candidate.normalized_id, "support"))
            if role not in buckets:
                buckets[role] = candidate
        if "problem" not in buckets and ordered_candidates:
            buckets["problem"] = ordered_candidates[0]
        if "method" not in buckets and ordered_candidates:
            buckets["method"] = ordered_candidates[min(1, len(ordered_candidates) - 1)]
        return buckets

    def _build_dynamic_sections(
        self,
        *,
        outline: StoryOutline,
        report_structure: ReportStructure,
        ordered_candidates: list[FigureCandidate],
        explanation_by_id: dict[str, FigureExplanation],
        output_dir: Path,
    ) -> list[DocumentSection]:
        role_buckets = self._group_figures_by_role(ordered_candidates, outline.figure_roles)
        role_priority_by_section = {
            section.key: _infer_section_roles(section.key, section.title)
            for section in outline.sections
        }
        used_ids: set[str] = set()
        sections: list[DocumentSection] = []

        for section in outline.sections:
            paragraphs = [_compress_paragraph(section.content, max_sentences=4, max_chars=360)]
            candidate = _pick_candidate_for_section(
                ordered_candidates=ordered_candidates,
                role_buckets=role_buckets,
                section_roles=role_priority_by_section.get(section.key, ["support"]),
                used_ids=used_ids,
            )
            image_path = None
            figure_id = None
            image_after = 0
            if candidate is not None:
                used_ids.add(candidate.normalized_id)
                explanation = explanation_by_id[candidate.normalized_id]
                image_path = relative_posix_path(Path(candidate.image_path), output_dir)
                figure_id = candidate.normalized_id
                image_after = len(paragraphs)
                follow_up = [
                    _compress_paragraph(
                        _soften_figure_reference(explanation.what_it_shows),
                        max_sentences=2,
                        max_chars=220,
                    ),
                    _compress_paragraph(explanation.how_to_read, max_sentences=2, max_chars=180),
                    _compress_paragraph(explanation.why_it_matters, max_sentences=2, max_chars=160),
                ]
                paragraphs = _append_nonredundant_paragraphs(paragraphs, [item for item in follow_up if item])
            paragraphs = [item for item in paragraphs if item]
            if paragraphs:
                sections.append(
                    DocumentSection(
                        title=section.title.strip() or _title_from_key(section.key, report_structure),
                        paragraphs=paragraphs,
                        image_path=image_path,
                        image_after_paragraph=image_after,
                        figure_id=figure_id,
                    )
                )

        sections.extend(
            self._build_additional_figure_sections(
                ordered_candidates=ordered_candidates,
                explanation_by_id=explanation_by_id,
                output_dir=output_dir,
                used_ids=used_ids,
            )
        )
        return sections

    def _build_additional_figure_sections(
        self,
        *,
        ordered_candidates: list[FigureCandidate],
        explanation_by_id: dict[str, FigureExplanation],
        output_dir: Path,
        used_ids: set[str],
    ) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        for candidate in ordered_candidates:
            if candidate.normalized_id in used_ids:
                continue
            explanation = explanation_by_id[candidate.normalized_id]
            paragraphs = [
                _compress_paragraph(_soften_figure_reference(explanation.what_it_shows), max_sentences=2, max_chars=220),
            ]
            how_to_read = _compress_paragraph(explanation.how_to_read, max_sentences=2, max_chars=180)
            if how_to_read:
                paragraphs.append(how_to_read)
            why_it_matters = _compress_paragraph(explanation.why_it_matters, max_sentences=2, max_chars=160)
            if why_it_matters:
                paragraphs.append(why_it_matters)
            paragraphs = _append_nonredundant_paragraphs([], [item for item in paragraphs if item])
            sections.append(
                DocumentSection(
                    title=explanation.title.strip() or candidate.normalized_id,
                    paragraphs=[item for item in paragraphs if item],
                    image_path=relative_posix_path(Path(candidate.image_path), output_dir),
                    image_after_paragraph=1,
                    figure_id=candidate.normalized_id,
                )
            )
        return sections


def _normalize_role(role_text: str) -> str:
    role = role_text.strip().lower()
    if role in {"problem", "method", "result", "support"}:
        return role
    if any(token in role for token in ("问题", "动机", "任务", "设定")):
        return "problem"
    if any(token in role for token in ("方法", "模型", "流程", "框架", "架构")):
        return "method"
    if any(token in role for token in ("结果", "实验", "性能", "比较", "benchmark", "ablation")):
        return "result"
    return "support"


def _infer_section_roles(section_key: str, section_title: str) -> list[str]:
    text = f"{section_key} {section_title}".lower()
    if any(token in text for token in ("motivation", "background", "problem", "任务", "背景", "问题", "研究议题")):
        return ["problem", "support", "method", "result"]
    if any(token in text for token in ("method", "analysis", "framework", "mechanism", "路径", "方法", "机制", "分析")):
        return ["method", "support", "problem", "result"]
    if any(token in text for token in ("result", "findings", "discussion", "experiment", "evaluation", "结论", "发现", "实验", "评测", "讨论")):
        return ["result", "support", "method", "problem"]
    return ["support", "problem", "method", "result"]


def _pick_candidate_for_section(
    *,
    ordered_candidates: list[FigureCandidate],
    role_buckets: dict[str, FigureCandidate],
    section_roles: list[str],
    used_ids: set[str],
) -> FigureCandidate | None:
    for role in section_roles:
        candidate = role_buckets.get(role)
        if candidate is not None and candidate.normalized_id not in used_ids:
            return candidate
    for candidate in ordered_candidates:
        if candidate.normalized_id not in used_ids:
            return candidate
    return None


def _title_from_key(section_key: str, report_structure: ReportStructure) -> str:
    for section in report_structure.sections:
        if section.key == section_key:
            return section.title
    return section_key


def _compress_paragraph(text: str, *, max_sentences: int, max_chars: int) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])", normalized) if part.strip()]
    if sentences:
        collected: list[str] = []
        current_length = 0
        for sentence in sentences[:max_sentences]:
            if current_length + len(sentence) > max_chars and collected:
                break
            collected.append(sentence)
            current_length += len(sentence)
        if collected:
            return "".join(collected)
    return truncate_text(normalized, max_chars)


def _append_nonredundant_paragraphs(existing: list[str], candidates: list[str]) -> list[str]:
    merged = [item for item in existing if item]
    for candidate in candidates:
        if not candidate:
            continue
        if any(_is_redundant_paragraph(candidate, paragraph) for paragraph in merged):
            continue
        merged.append(candidate)
    return merged


def _is_redundant_paragraph(candidate: str, existing: str) -> bool:
    left = _normalize_similarity_text(candidate)
    right = _normalize_similarity_text(existing)
    if not left or not right:
        return False
    if len(left) >= 20 and left in right:
        return True
    if len(right) >= 20 and right in left:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.68


def _normalize_similarity_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def _polish_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    # 模型有时会生成“RES（英文全称，中文）”，这里统一压成“RES（中文）”。
    normalized = re.sub(
        r"([A-Z][A-Z0-9-]{1,})（[^（）]{2,80}[，,]\s*([\u4e00-\u9fffA-Za-z0-9、\-]{2,40})）",
        r"\1（\2）",
        normalized,
    )
    return normalized


def _deduplicate_acronym_annotations(text: str) -> str:
    seen_acronyms: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        acronym = match.group(1)
        if acronym in seen_acronyms:
            return acronym
        seen_acronyms.add(acronym)
        return match.group(0)

    return re.sub(
        r"([A-Z][A-Z0-9-]{1,})（[\u4e00-\u9fffA-Za-z0-9、\-]{2,40}）",
        replace,
        text,
    )


def _ensure_markdown_header(markdown: str, document: FinalDocument) -> str:
    header_parts: list[str] = []
    if not re.search(r"(?m)^#\s+", markdown):
        header_parts.extend([f"# {document.title}", ""])
    has_plain_title_translation = bool(
        re.search(rf"(?m)^{re.escape(document.title_translation)}\s*$", markdown)
    ) if document.title_translation else False
    has_labeled_title_translation = (
        document.title_translation and f"**中文题目：** {document.title_translation}" in markdown
    )
    if document.title_translation and not has_plain_title_translation and not has_labeled_title_translation:
        header_parts.extend([document.title_translation, ""])
    if document.one_sentence_summary and "> 导读摘要：" not in markdown:
        header_parts.extend([f"> 导读摘要：{document.one_sentence_summary}", ""])
    if not header_parts:
        return markdown
    return "\n".join(header_parts).rstrip() + "\n\n" + markdown.lstrip()


def _normalize_markdown_header(markdown: str) -> str:
    lines = markdown.splitlines()
    normalized_lines: list[str] = []
    seen_plain_translation: set[str] = set()

    for line in lines:
        match = re.fullmatch(r"\*\*中文题目：\*\*\s*(.+)", line.strip())
        if match:
            translation = match.group(1).strip()
            if translation and translation not in seen_plain_translation:
                normalized_lines.append(translation)
                seen_plain_translation.add(translation)
            continue
        stripped = line.strip()
        if stripped and stripped in seen_plain_translation:
            continue
        normalized_lines.append(line)

    # 如果模型在标题前多吐出一行独立中文题目，这里直接丢掉标题前的孤立前言。
    first_h1_index = next((index for index, line in enumerate(normalized_lines) if line.startswith("# ")), None)
    if first_h1_index not in (None, 0):
        prelude = [line.strip() for line in normalized_lines[:first_h1_index] if line.strip()]
        if prelude and all(not line.startswith("#") for line in prelude):
            normalized_lines = normalized_lines[first_h1_index:]

    return "\n".join(normalized_lines).strip() + "\n"


def _normalize_markdown_image_spacing(markdown: str) -> str:
    """保证 Markdown 图片前后各有一个空行，避免被解析成普通段落。"""
    lines = markdown.splitlines()
    normalized_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_image_line = bool(re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", stripped))
        if is_image_line:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            normalized_lines.append(stripped)
            normalized_lines.append("")
            continue
        normalized_lines.append(line)

    compact_lines: list[str] = []
    blank_run = 0
    for line in normalized_lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                compact_lines.append(line)
            continue
        blank_run = 0
        compact_lines.append(line)

    return "\n".join(compact_lines).strip() + "\n"


def _deduplicate_full_document_repetition(markdown: str) -> str:
    """有些模型会把整篇 Markdown 原样重复一遍，这里做一次保守去重。"""
    normalized = markdown.strip()
    if not normalized:
        return markdown

    lines = normalized.splitlines()
    header_indexes = [index for index, line in enumerate(lines) if line.startswith("# ")]
    if len(header_indexes) >= 2:
        first_header = lines[header_indexes[0]].strip()
        canonical_first = "\n".join(lines[header_indexes[0] :]).strip()
        for index in header_indexes[1:]:
            if lines[index].strip() != first_header:
                continue
            first_part = "\n".join(lines[:index]).strip()
            second_part = "\n".join(lines[index:]).strip()
            if first_part == second_part:
                return first_part + "\n"
            if second_part == canonical_first:
                return canonical_first + "\n"
            if first_part.endswith(second_part):
                return first_part + "\n"

    if len(normalized) % 2 == 0:
        half = len(normalized) // 2
        if normalized[:half] == normalized[half:]:
            return normalized[:half].rstrip() + "\n"

    return markdown


def _soften_figure_reference(text: str) -> str:
    normalized = _polish_text(text)
    normalized = re.sub(r"^(?:如图所示[，,]?|从图中可以看到[，,]?|可以看到[，,]?|这张图(?:中)?|下图|图中|该图)\s*", "", normalized)
    normalized = re.sub(r"^(?:展示(?:了)?|说明(?:了)?|对比(?:了)?|给出(?:了)?|呈现(?:了)?|体现(?:了)?)", "", normalized)
    normalized = normalized.lstrip("：:，,。 ")
    return normalized


def _ensure_all_selected_figures_present(markdown: str, figure_context: list[dict[str, str | int]]) -> str:
    missing_items = [item for item in figure_context if str(item["image_path"]) not in markdown]
    if not missing_items:
        return markdown

    extra_lines = ["", "## 补充图示", ""]
    for item in missing_items:
        title = str(item.get("title") or item["id"])
        what_it_shows = _compress_paragraph(_soften_figure_reference(str(item.get("what_it_shows", ""))), max_sentences=2, max_chars=220)
        why_it_matters = _compress_paragraph(str(item.get("why_it_matters", "")), max_sentences=2, max_chars=160)
        extra_lines.append(f"### {title}")
        extra_lines.append("")
        extra_lines.append(f"![{item['id']}]({item['image_path']})")
        extra_lines.append("")
        if what_it_shows:
            extra_lines.append(what_it_shows)
            extra_lines.append("")
        if why_it_matters:
            extra_lines.append(why_it_matters)
            extra_lines.append("")
    return markdown.rstrip() + "\n" + "\n".join(extra_lines).rstrip() + "\n"
