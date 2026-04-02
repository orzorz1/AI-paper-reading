"""主流水线编排。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AppConfig, OpenAISettings
from ..errors import PipelineExecutionError, SelectionValidationError
from ..llm import OpenAICompatibleClient
from ..logging_utils import get_logger
from ..models import BuildOptions, BuildResult, ContentFocus, FigureCandidate, FigureExplanation, LayoutRegion, OutputLength, ParsedPaper
from ..utils import dump_json, dump_model_json, dump_models_json, ensure_dir, truncate_text
from .composer import MarkdownComposer
from .explainer import FigureExplainer
from .layout import LayoutDetector
from .matcher import FigureMatcher
from .parser import PDFParser
from .planner import StoryPlanner
from .pdf_exporter import MarkdownPdfExporter
from .selector import FigureSelector


class PaperReadingOrchestrator:
    """串起所有子模块，形成完整构建流程。"""

    def __init__(
        self,
        config: AppConfig,
        *,
        parser: Optional[PDFParser] = None,
        layout_detector: Optional[LayoutDetector] = None,
        matcher: Optional[FigureMatcher] = None,
        selector: Optional[FigureSelector] = None,
        planner: Optional[StoryPlanner] = None,
        explainer: Optional[FigureExplainer] = None,
        composer: Optional[MarkdownComposer] = None,
        pdf_exporter: Optional[MarkdownPdfExporter] = None,
    ) -> None:
        self._config = config
        self._logger = get_logger()
        self._parser = parser or PDFParser(render_dpi=config.layout.render_dpi)
        self._layout_detector = layout_detector or LayoutDetector(
            model_path=config.layout.model_path,
            confidence=config.layout.confidence,
            device=config.layout.device,
            predict_imgsz=config.layout.predict_imgsz,
        )
        self._matcher = matcher or FigureMatcher(crop_dpi=config.layout.crop_dpi)
        client = None
        if selector is None or planner is None or explainer is None:
            client = OpenAICompatibleClient(config.openai)
        self._selector = selector or FigureSelector(client, config.openai.text_model)
        self._planner = planner or StoryPlanner(client, config.openai.text_model)
        self._explainer = explainer or FigureExplainer(client, config.openai.vision_model)
        self._composer = composer or MarkdownComposer(client, config.openai.text_model)
        self._pdf_exporter = pdf_exporter or MarkdownPdfExporter()

    def with_openai_settings(self, openai: OpenAISettings) -> "PaperReadingOrchestrator":
        """共享版面解析、PDF 解析等非 LLM 组件，仅替换大模型接口与相关调用链。"""
        merged_config = self._config.model_copy(update={"openai": openai})
        client = OpenAICompatibleClient(openai)
        return PaperReadingOrchestrator(
            merged_config,
            parser=self._parser,
            layout_detector=self._layout_detector,
            matcher=self._matcher,
            selector=FigureSelector(client, openai.text_model),
            planner=StoryPlanner(client, openai.text_model),
            explainer=FigureExplainer(client, openai.vision_model),
            composer=MarkdownComposer(client, openai.text_model),
            pdf_exporter=self._pdf_exporter,
        )

    def build(self, options: BuildOptions) -> BuildResult:
        """构建最终 Markdown 和所有中间产物。"""
        pdf_path = options.pdf_path
        if not pdf_path.exists():
            raise PipelineExecutionError(f"PDF 不存在：{pdf_path}")

        explicit_output_dir = options.output_dir
        output_dir = ensure_dir(explicit_output_dir or self._default_output_dir(pdf_path))
        artifacts_dir = ensure_dir(output_dir / "artifacts")
        resolved_max_figures = _resolve_max_figures(options.output_length, options.max_figures)

        warnings: list[str] = []
        self._logger.info("开始处理论文：%s", pdf_path)

        parsed_paper = self._parser.parse(
            pdf_path,
            artifacts_dir,
            override_title=options.title,
            override_abstract=options.abstract,
        )
        figures_dir = ensure_dir(artifacts_dir / "figures")
        layout_dir = ensure_dir(artifacts_dir / "layout")
        self._logger.info("输出目录：%s", output_dir)
        dump_json(
            artifacts_dir / "run_config.json",
            {
                "pdf_path": str(pdf_path),
                "title": options.title,
                "abstract": options.abstract,
                "output_dir": str(output_dir),
                "max_figures": resolved_max_figures,
                "lang": options.lang,
                "content_focus": options.content_focus,
                "output_length": options.output_length,
                "config": self._config.model_dump(mode="json"),
            },
        )
        self._logger.info("PDF 解析完成：共 %d 页，标题=%s", len(parsed_paper.pages), parsed_paper.metadata.title)
        dump_model_json(artifacts_dir / "metadata.json", parsed_paper.metadata)
        dump_model_json(artifacts_dir / "parsed_paper.json", parsed_paper)
        dump_models_json(
            artifacts_dir / "paper_blocks.json",
            [block for page in parsed_paper.pages for block in page.text_blocks],
        )

        layout_by_page: dict[int, list[LayoutRegion]] = {}
        self._logger.info("开始执行版面识别...")
        for page in parsed_paper.pages:
            try:
                regions = self._layout_detector.detect(
                    Path(page.rendered_image_path),
                    page.page,
                    page_width=page.width,
                    page_height=page.height,
                )
            except Exception as exc:
                regions = []
                warnings.append(f"第 {page.page} 页版面识别失败，已回退到规则匹配：{exc}")
                self._logger.warning("第 %d 页版面识别失败，已回退：%s", page.page, exc)
            layout_by_page[page.page] = regions
            dump_models_json(layout_dir / f"page_{page.page:03d}.json", regions)
            self._logger.info("第 %d 页识别到 %d 个版面区域", page.page, len(regions))

        candidates = self._matcher.build_candidates(parsed_paper, layout_by_page, figures_dir)
        if not candidates:
            raise PipelineExecutionError("没有识别到任何图表候选，无法继续生成结果。")
        body_page_limits = _build_body_page_limits(parsed_paper)
        candidates = _filter_candidates_to_body(candidates, body_page_limits)
        if not candidates:
            raise PipelineExecutionError("正文部分没有识别到任何图表候选，无法继续生成结果。")
        self._logger.info("图表候选生成完成：共 %d 个候选", len(candidates))
        dump_models_json(artifacts_dir / "figure_candidates.json", candidates)

        figure_candidates = [item for item in candidates if item.normalized_id.startswith("Fig")]
        table_candidates = [item for item in candidates if item.normalized_id.startswith("Table")]
        if figure_candidates:
            self._logger.info("正文图片候选共 %d 个", len(figure_candidates))
        else:
            warnings.append("正文中未识别到图片候选。")
            self._logger.warning("正文中未识别到图片候选。")
        if options.content_focus == "experiment":
            self._logger.info("实验模式下正文表格候选共 %d 个", len(table_candidates))

        self._logger.info("开始选择关键图...")
        body_text = _build_body_text(parsed_paper)
        paper_context = _build_paper_context(parsed_paper, body_text)
        selected_figures = self._selector.select(
            title=parsed_paper.metadata.title,
            abstract=parsed_paper.metadata.abstract,
            full_text=body_text,
            candidates=figure_candidates or candidates,
            max_figures=resolved_max_figures,
            content_focus=options.content_focus,
            output_length=options.output_length,
            candidate_kind="figure",
        )
        selected_figures = _enforce_selection_strategy(
            selected=selected_figures,
            candidates=figure_candidates or candidates,
            content_focus=options.content_focus,
            output_length=options.output_length,
            max_figures=resolved_max_figures,
        )
        selected_tables = []
        if options.content_focus == "experiment" and table_candidates:
            self._logger.info("开始筛选实验相关表格...")
            selected_tables = self._selector.select(
                title=parsed_paper.metadata.title,
                abstract=parsed_paper.metadata.abstract,
                full_text=body_text,
                candidates=table_candidates,
                max_figures=len(table_candidates),
                content_focus=options.content_focus,
                output_length=options.output_length,
                candidate_kind="table",
            )
        selected = _reindex_selected(selected_figures + selected_tables)
        candidate_map = {item.normalized_id: item for item in candidates}
        for item in selected:
            if item.normalized_id not in candidate_map:
                raise SelectionValidationError(f"模型选出了不存在的图编号：{item.normalized_id}")
        self._logger.info("关键图选择完成：选中 %d 张", len(selected))
        dump_models_json(artifacts_dir / "selected_figures.json", selected)

        self._logger.info("开始生成整体阅读结构...")
        outline = self._planner.plan(
            title=parsed_paper.metadata.title,
            abstract=parsed_paper.metadata.abstract,
            full_text=body_text,
            paper_context=paper_context,
            selected_figures=selected,
            candidates_by_id=candidate_map,
            content_focus=options.content_focus,
            output_length=options.output_length,
        )
        dump_model_json(artifacts_dir / "story_outline.json", outline)

        self._logger.info("开始逐图解释...")
        explanations: list[FigureExplanation] = []
        ordered_candidates: list[FigureCandidate] = []
        for item in sorted(selected, key=lambda selected_item: selected_item.importance_rank):
            candidate = candidate_map[item.normalized_id]
            ordered_candidates.append(candidate)
            page_text = _get_figure_context(parsed_paper, candidate.page)
            self._logger.info("解释 %s（第 %d 页）", candidate.normalized_id, candidate.page)
            explanation = self._explainer.explain(
                candidate=candidate,
                figure_role=outline.figure_roles.get(candidate.normalized_id, item.reason),
                paper_title=parsed_paper.metadata.title,
                abstract=parsed_paper.metadata.abstract,
                paper_context=paper_context,
                surrounding_text=page_text,
                content_focus=options.content_focus,
                output_length=options.output_length,
            )
            explanations.append(explanation)
        dump_models_json(artifacts_dir / "figure_explanations.json", explanations)

        self._logger.info("开始生成 Markdown...")
        final_document = self._composer.compose(
            title=parsed_paper.metadata.title,
            outline=outline,
            ordered_candidates=ordered_candidates,
            explanations=explanations,
            output_dir=output_dir,
        )
        dump_model_json(artifacts_dir / "final_document.json", final_document)

        markdown_path = output_dir / "paper_readable.md"
        markdown_text = self._composer.generate_markdown(
            title=parsed_paper.metadata.title,
            abstract=parsed_paper.metadata.abstract,
            paper_context=paper_context,
            outline=outline,
            selected_figures=[
                {
                    "normalized_id": item.normalized_id,
                    "reason": item.reason,
                    "importance_rank": item.importance_rank,
                }
                for item in selected
            ],
            ordered_candidates=ordered_candidates,
            explanations=explanations,
            output_dir=output_dir,
            content_focus=options.content_focus,
            output_length=options.output_length,
        )
        markdown_path.write_text(markdown_text, encoding="utf-8")
        self._logger.info("Markdown 生成完成：%s", markdown_path)
        pdf_path = output_dir / "paper_readable.pdf"
        self._logger.info("开始导出 PDF...")
        self._pdf_exporter.export(markdown_path, pdf_path)
        self._logger.info("PDF 导出完成：%s", pdf_path)

        return BuildResult(
            markdown_path=str(markdown_path),
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
            artifact_dir=str(artifacts_dir),
            selected_count=len(ordered_candidates),
            warnings=warnings,
        )

    def _default_output_dir(self, pdf_path: Path) -> Path:
        return self._deduplicate_output_dir(pdf_path.parent / self._build_output_dir_name(pdf_path))

    def _build_output_dir_name(self, pdf_path: Path) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        base_name = pdf_path.stem.strip() or "paper"
        return f"{base_name}-{timestamp}"

    def _deduplicate_output_dir(self, candidate: Path) -> Path:
        if not candidate.exists():
            return candidate
        for index in range(2, 1000):
            next_candidate = candidate.with_name(f"{candidate.name}-{index}")
            if not next_candidate.exists():
                return next_candidate
        raise PipelineExecutionError(f"输出目录冲突过多，无法创建目录：{candidate.parent}")


def _get_page_text(parsed_paper: ParsedPaper, page_number: int) -> str:
    for page in parsed_paper.pages:
        if page.page == page_number:
            return page.text
    return parsed_paper.full_text


def _resolve_max_figures(output_length: OutputLength, explicit_max_figures: int | None) -> int:
    if explicit_max_figures is not None:
        return explicit_max_figures
    if output_length == "short":
        return 1
    if output_length == "long":
        return 5
    return 3


def _enforce_selection_strategy(
    *,
    selected: list,
    candidates: list[FigureCandidate],
    content_focus: ContentFocus,
    output_length: OutputLength,
    max_figures: int,
) -> list:
    candidate_map = {item.normalized_id: item for item in candidates}
    selected_by_id = {item.normalized_id: item for item in selected if item.normalized_id in candidate_map}
    ranked_selected = sorted(selected_by_id.values(), key=lambda item: item.importance_rank)
    selected_tables = [item for item in ranked_selected if item.normalized_id.startswith("Table")]
    selected_figures = [item for item in ranked_selected if item.normalized_id.startswith("Fig")]

    if output_length == "short":
        preferred_roles = ["result", "support", "method", "problem"] if content_focus == "experiment" else ["method", "problem", "support", "result"]
        chosen = _pick_first_by_roles(selected_figures, candidate_map, preferred_roles) or (selected_figures[0] if selected_figures else None)
        return _reindex_selected(([chosen] if chosen else []) + selected_tables)

    if output_length == "medium":
        if content_focus == "method":
            preferred_roles = ["method", "problem", "support", "result"]
        else:
            preferred_roles = ["result", "support", "method", "problem"]
        reordered = _reorder_selected_by_roles(selected_figures, candidate_map, preferred_roles)
        return _reindex_selected(reordered[:max_figures] + selected_tables)

    figure_candidates = [item for item in candidates if item.normalized_id.startswith("Fig")]
    reordered = _build_long_form_selection(selected_figures, candidate_map, figure_candidates, max_figures)
    return _reindex_selected(reordered[:max_figures] + selected_tables)


def _build_long_form_selection(
    selected: list,
    candidate_map: dict[str, FigureCandidate],
    all_candidates: list[FigureCandidate],
    max_figures: int,
) -> list:
    chosen: list = []
    method_item = _pick_first_by_roles(selected, candidate_map, ["method", "problem"])
    result_item = _pick_first_by_roles(selected, candidate_map, ["result"])

    if method_item is None:
        method_item = _build_supplemental_selected(all_candidates, selected, ["method", "problem"], "按长篇策略补充方法图")
    if result_item is None:
        result_item = _build_supplemental_selected(all_candidates, selected, ["result"], "按长篇策略补充实验图")

    for item in [method_item, result_item]:
        if item is not None and item.normalized_id not in {picked.normalized_id for picked in chosen}:
            chosen.append(item)

    reordered = _reorder_selected_by_roles(selected, candidate_map, ["problem", "method", "support", "result"])
    for item in reordered:
        if item.normalized_id not in {picked.normalized_id for picked in chosen}:
            chosen.append(item)
        if len(chosen) >= max_figures:
            break

    while len(chosen) < max_figures:
        supplemental = _build_supplemental_selected(all_candidates, chosen, ["support", "problem", "method", "result"], "按长篇策略补充关键图")
        if supplemental is None or supplemental.normalized_id in {picked.normalized_id for picked in chosen}:
            break
        chosen.append(supplemental)
    return chosen


def _reorder_selected_by_roles(selected: list, candidate_map: dict[str, FigureCandidate], role_order: list[str]) -> list:
    buckets: dict[str, list] = {role: [] for role in role_order}
    other_items: list = []
    for item in selected:
        role = _guess_figure_role(candidate_map[item.normalized_id])
        if role in buckets:
            buckets[role].append(item)
        else:
            other_items.append(item)
    ordered: list = []
    for role in role_order:
        ordered.extend(sorted(buckets[role], key=lambda current: current.importance_rank))
    ordered.extend(sorted(other_items, key=lambda current: current.importance_rank))
    return ordered


def _pick_first_by_roles(selected: list, candidate_map: dict[str, FigureCandidate], roles: list[str]):
    for role in roles:
        for item in selected:
            if _guess_figure_role(candidate_map[item.normalized_id]) == role:
                return item
    return None


def _build_supplemental_selected(
    candidates: list[FigureCandidate],
    existing: list,
    roles: list[str],
    reason: str,
):
    existing_ids = {item.normalized_id for item in existing}
    role_set = set(roles)
    for candidate in candidates:
        if candidate.normalized_id in existing_ids:
            continue
        if _guess_figure_role(candidate) in role_set:
            from ..models import SelectedFigure

            return SelectedFigure(normalized_id=candidate.normalized_id, reason=reason, importance_rank=999)
    return None


def _reindex_selected(selected: list) -> list:
    from ..models import SelectedFigure

    normalized: list[SelectedFigure] = []
    for index, item in enumerate([current for current in selected if current is not None], start=1):
        normalized.append(
            SelectedFigure(
                normalized_id=item.normalized_id,
                reason=item.reason,
                importance_rank=index,
            )
        )
    return normalized


def _guess_figure_role(candidate: FigureCandidate) -> str:
    text = f"{candidate.normalized_id} {candidate.caption_text}".lower()
    if any(token in text for token in ("result", "results", "experiment", "evaluation", "comparison", "benchmark", "ablation", "performance", "qualitative", "quantitative")):
        return "result"
    if any(token in text for token in ("overview", "pipeline", "framework", "architecture", "method", "approach", "model", "workflow")):
        return "method"
    if any(token in text for token in ("task", "motivation", "example", "scenario", "problem", "setting")):
        return "problem"
    return "support"


def _build_body_text(parsed_paper: ParsedPaper) -> str:
    parts: list[str] = []
    for page in parsed_paper.pages:
        page_text = _strip_back_matter_from_page(page.text)
        if page_text:
            parts.append(f"[第{page.page}页]\n{page_text}")
        if _page_starts_back_matter(page.text):
            break
    return "\n\n".join(part for part in parts if part.strip())


def _build_paper_context(parsed_paper: ParsedPaper, body_text: str) -> str:
    section_context = _build_section_context(body_text)
    heading_parts = _collect_heading_snippets(parsed_paper)
    parts: list[str] = []
    if body_text.strip():
        parts.append("[全文正文（已去除参考文献和附录）]\n" + truncate_text(body_text, 32000))
    if section_context.strip():
        parts.append("[重点章节摘录]\n" + section_context)
    if heading_parts:
        parts.append("[章节标题]\n" + "\n".join(heading_parts))
    return "\n\n".join(part for part in parts if part.strip())


def _get_figure_context(parsed_paper: ParsedPaper, page_number: int) -> str:
    start_page = max(1, page_number - 1)
    end_page = min(len(parsed_paper.pages), page_number + 1)
    parts: list[str] = []
    for page in parsed_paper.pages:
        if start_page <= page.page <= end_page:
            page_text = _strip_back_matter_from_page(page.text)
            if page_text:
                parts.append(f"[第{page.page}页]\n{page_text}")
        if _page_starts_back_matter(page.text):
            break
    return truncate_text("\n\n".join(parts), 12000)


def _collect_heading_snippets(parsed_paper: ParsedPaper) -> list[str]:
    headings: list[str] = []
    for page in parsed_paper.pages:
        for block in page.text_blocks:
            text = block.text.replace("\n", " ").strip()
            if not text or len(text) > 80:
                continue
            if block.font_size is not None and block.font_size >= 11.0:
                headings.append(f"第{page.page}页 {text}")
            elif text[:2].isdigit() and "." in text[:5]:
                headings.append(f"第{page.page}页 {text}")
        if len(headings) >= 20:
            break
    return headings[:20]


BACK_MATTER_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(references?|bibliography|appendix|appendices|supplementary(?: material)?|acknowledg(?:e)?ments?)\s*$"
)
SECTION_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*)?\.?\s*(introduction|related work|background|preliminaries|method(?:ology)?|approach|framework|model|architecture|data engine|dataset|experiments?|evaluation|results?|ablation|analysis|discussion|conclusion[s]?)\s*$"
)


def _page_starts_back_matter(page_text: str) -> bool:
    return BACK_MATTER_HEADING_PATTERN.search(page_text) is not None


def _strip_back_matter_from_page(page_text: str) -> str:
    match = BACK_MATTER_HEADING_PATTERN.search(page_text)
    if match is None:
        return page_text.strip()
    return page_text[: match.start()].strip()


def _build_section_context(body_text: str) -> str:
    sections = _extract_section_snippets(body_text)
    parts: list[str] = []
    for label, content in sections:
        if content:
            parts.append(f"[{label}]\n{content}")
    return "\n\n".join(parts)


def _build_body_page_limits(parsed_paper: ParsedPaper) -> dict[int, float]:
    limits: dict[int, float] = {}
    for page in parsed_paper.pages:
        cutoff_y = page.height
        for block in page.text_blocks:
            if BACK_MATTER_HEADING_PATTERN.search(block.text.strip()):
                cutoff_y = min(cutoff_y, block.bbox.y0)
        limits[page.page] = cutoff_y
        if _page_starts_back_matter(page.text):
            break
    return limits


def _filter_candidates_to_body(candidates: list[FigureCandidate], body_page_limits: dict[int, float]) -> list[FigureCandidate]:
    filtered: list[FigureCandidate] = []
    for candidate in candidates:
        cutoff_y = body_page_limits.get(candidate.page)
        if cutoff_y is None:
            continue
        if candidate.caption_bbox.y0 >= cutoff_y or candidate.figure_bbox.y0 >= cutoff_y:
            continue
        filtered.append(candidate)
    return filtered


def _extract_section_snippets(body_text: str) -> list[tuple[str, str]]:
    normalized = body_text.strip()
    if not normalized:
        return []

    heading_matches = list(SECTION_HEADING_PATTERN.finditer(normalized))
    sections: list[tuple[str, str]] = []
    section_specs = [
        ("Introduction", ("introduction", "background", "preliminaries")),
        ("Method", ("method", "methodology", "approach", "framework", "model", "architecture", "data engine", "dataset")),
        ("Experiment", ("experiment", "evaluation", "result", "ablation", "analysis")),
    ]

    for label, keywords in section_specs:
        snippet = _extract_section_by_keywords(normalized, heading_matches, keywords)
        if snippet:
            sections.append((label, truncate_text(snippet, 9000)))
    return sections


def _extract_section_by_keywords(
    text: str,
    heading_matches: list[re.Match[str]],
    keywords: tuple[str, ...],
) -> str:
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    for index, match in enumerate(heading_matches):
        heading = match.group(1).lower()
        if not any(keyword in heading for keyword in lowered_keywords):
            continue
        start = match.start()
        end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(text)
        return text[start:end].strip()

    lower_text = text.lower()
    for keyword in lowered_keywords:
        position = lower_text.find(keyword)
        if position == -1:
            continue
        return text[position : position + 9000].strip()
    return ""
