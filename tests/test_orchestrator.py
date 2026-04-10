from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from paper_reading.config import AppConfig, LayoutSettings, OpenAISettings, RuntimeSettings
from paper_reading.models import (
    BoundingBox,
    BuildOptions,
    FigureCandidate,
    FigureExplanation,
    LayoutRegion,
    PageImageBlock,
    PageTextBlock,
    PaperMetadata,
    ParsedPage,
    ParsedPaper,
    ReportSectionBlueprint,
    ReportStructure,
    SelectedFigure,
    StorySection,
    StoryOutline,
)
from paper_reading.pipeline.composer import MarkdownComposer
from paper_reading.pipeline.orchestrator import (
    PaperReadingOrchestrator,
    _build_body_text,
    _build_body_page_limits,
    _enforce_selection_strategy,
    _filter_candidates_to_body,
    _build_paper_context,
    _resolve_max_figures,
)


class _FakeParser:
    last_max_pages: int | None = None

    def parse(self, pdf_path: Path, artifacts_dir: Path, **_: object) -> ParsedPaper:
        self.__class__.last_max_pages = _.get("max_pages")
        page_dir = artifacts_dir / "page_images"
        page_dir.mkdir(parents=True, exist_ok=True)
        page_image_path = page_dir / "page_001.png"
        Image.new("RGB", (400, 400), color="white").save(page_image_path)
        return ParsedPaper(
            metadata=PaperMetadata(title="测试论文", abstract="摘要内容", source_pdf=str(pdf_path), abstract_fallback_text=""),
            pages=[
                ParsedPage(
                    page=1,
                    width=400,
                    height=400,
                    text="Figure 1 describes the full pipeline.",
                    rendered_image_path=str(page_image_path),
                    text_blocks=[
                        PageTextBlock(
                            page=1,
                            bbox=BoundingBox(x0=20, y0=320, x1=380, y1=360),
                            text="Figure 1: Full pipeline.",
                        )
                    ],
                    image_blocks=[PageImageBlock(page=1, bbox=BoundingBox(x0=30, y0=40, x1=370, y1=300))],
                )
            ],
            full_text="Figure 1 describes the full pipeline. The method is simple.",
        )


class _FakeLayoutDetector:
    def detect(self, page_image: Path, page_number: int, **_: object) -> list[LayoutRegion]:
        self._last_image = page_image
        return [
            LayoutRegion(page=page_number, bbox=BoundingBox(x0=30, y0=40, x1=370, y1=300), label="figure", score=0.95),
            LayoutRegion(page=page_number, bbox=BoundingBox(x0=20, y0=320, x1=380, y1=360), label="caption", score=0.92),
        ]


class _FakeMatcher:
    def build_candidates(self, parsed_paper: ParsedPaper, layout_by_page: dict[int, list[LayoutRegion]], figures_dir: Path) -> list[FigureCandidate]:
        image_path = figures_dir / "Fig1.png"
        Image.new("RGB", (200, 120), color="white").save(image_path)
        return [
            FigureCandidate(
                normalized_id="Fig1",
                page=1,
                caption_text="Figure 1: Full pipeline.",
                caption_bbox=BoundingBox(x0=20, y0=320, x1=380, y1=360),
                figure_bbox=BoundingBox(x0=30, y0=40, x1=370, y1=300),
                image_path=str(image_path),
                match_score=0.95,
                source="layout+rule",
                match_options=[],
            )
        ]


class _FakeSelector:
    def select(self, **_: object) -> list[SelectedFigure]:
        return [SelectedFigure(normalized_id="Fig1", reason="最适合讲清方法主流程", importance_rank=1)]


class _FakePlanner:
    def plan(self, **_: object) -> StoryOutline:
        return StoryOutline(
            title_translation="测试论文中文译名",
            one_sentence_summary="这是一段更完整的导读摘要，用来帮助读者快速理解 RES（Referring Expression Segmentation，指称表达分割）论文。",
            sections=[
                StorySection(
                    key="background",
                    title="研究议题与问题提出",
                    content="论文想解决输入和输出之间的映射问题，并扩展到 MRES（Multi-Granularity Referring Expression Segmentation，多粒度指称表达分割）场景。",
                ),
                StorySection(
                    key="analysis",
                    title="分析路径与核心机制",
                    content="方法核心是一个分阶段流程。",
                ),
            ],
            figure_roles={"Fig1": "method"},
        )


class _FakeReportStructurePlanner:
    def plan(self, **_: object) -> ReportStructure:
        return ReportStructure(
            sections=[
                ReportSectionBlueprint(
                    key="background",
                    title="研究议题与问题提出",
                    guidance="交代研究议题、背景和问题。",
                ),
                ReportSectionBlueprint(
                    key="analysis",
                    title="分析路径与核心机制",
                    guidance="解释分析路径和核心机制。",
                ),
                ReportSectionBlueprint(
                    key="discussion",
                    title="主要发现与讨论",
                    guidance="概述主要发现和讨论。",
                ),
            ],
            structure_rationale="这篇文章更适合用更中性的章节标题。",
        )


class _FakeExplainer:
    def explain(self, **_: object) -> FigureExplanation:
        return FigureExplanation(
            normalized_id="Fig1",
            title="整体流程图",
            what_it_shows="这张图展示了完整方法。",
            how_to_read="按从左到右的顺序阅读。",
            why_it_matters="它能最快建立读者对方法的整体认识。",
        )


class _FakePdfExporter:
    def export(self, markdown_path: Path, output_path: Path) -> Path:
        output_path.write_bytes(b"%PDF-1.4 fake")
        return output_path


class OrchestratorTests(unittest.TestCase):
    def test_resolve_max_figures_uses_length_defaults(self) -> None:
        self.assertEqual(_resolve_max_figures("short", None), 1)
        self.assertEqual(_resolve_max_figures("medium", None), 3)
        self.assertEqual(_resolve_max_figures("long", None), 5)
        self.assertEqual(_resolve_max_figures("long", 4), 4)

    def test_enforce_selection_strategy_prefers_result_for_short_experiment(self) -> None:
        candidates = [
            FigureCandidate(
                normalized_id="Fig1",
                page=1,
                caption_text="Figure 1: Overview pipeline.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Fig1.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Fig2",
                page=2,
                caption_text="Figure 2: Experimental results on benchmark.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Fig2.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
        ]
        selected = [
            SelectedFigure(normalized_id="Fig1", reason="方法主线", importance_rank=1),
            SelectedFigure(normalized_id="Fig2", reason="实验结果", importance_rank=2),
        ]

        enforced = _enforce_selection_strategy(
            selected=selected,
            candidates=candidates,
            content_focus="experiment",
            output_length="short",
            max_figures=1,
        )

        self.assertEqual(len(enforced), 1)
        self.assertEqual(enforced[0].normalized_id, "Fig2")

    def test_enforce_selection_strategy_supplements_method_and_result_for_long(self) -> None:
        candidates = [
            FigureCandidate(
                normalized_id="Fig1",
                page=1,
                caption_text="Figure 1: Overview pipeline.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Fig1.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Fig2",
                page=2,
                caption_text="Figure 2: Experimental comparison on benchmark.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Fig2.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
        ]
        selected = [SelectedFigure(normalized_id="Fig1", reason="方法主线", importance_rank=1)]

        enforced = _enforce_selection_strategy(
            selected=selected,
            candidates=candidates,
            content_focus="method",
            output_length="long",
            max_figures=5,
        )

        self.assertEqual([item.normalized_id for item in enforced[:2]], ["Fig1", "Fig2"])

    def test_enforce_selection_strategy_keeps_tables_without_counting_against_figures(self) -> None:
        candidates = [
            FigureCandidate(
                normalized_id="Fig1",
                page=1,
                caption_text="Figure 1: Overview pipeline.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Fig1.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Fig2",
                page=2,
                caption_text="Figure 2: Experimental comparison on benchmark.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Fig2.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Table1",
                page=3,
                caption_text="Table 1: Main benchmark results.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Table1.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Table2",
                page=4,
                caption_text="Table 2: Ablation results.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="Table2.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
        ]
        selected = [
            SelectedFigure(normalized_id="Fig1", reason="方法主线", importance_rank=1),
            SelectedFigure(normalized_id="Fig2", reason="实验结果", importance_rank=2),
            SelectedFigure(normalized_id="Table1", reason="主结果表", importance_rank=3),
            SelectedFigure(normalized_id="Table2", reason="消融表", importance_rank=4),
        ]

        enforced = _enforce_selection_strategy(
            selected=selected,
            candidates=candidates,
            content_focus="experiment",
            output_length="short",
            max_figures=1,
        )

        self.assertEqual([item.normalized_id for item in enforced], ["Fig2", "Table1", "Table2"])

    def test_build_body_text_excludes_references_and_appendix(self) -> None:
        parsed_paper = ParsedPaper(
            metadata=PaperMetadata(title="测试论文", abstract="摘要", source_pdf="demo.pdf"),
            pages=[
                ParsedPage(page=1, width=400, height=400, text="1. Introduction\nIntro text.", rendered_image_path="page1.png"),
                ParsedPage(page=2, width=400, height=400, text="3. Method\nMethod text.\nReferences\n[1] Paper", rendered_image_path="page2.png"),
                ParsedPage(page=3, width=400, height=400, text="Appendix\nMore content", rendered_image_path="page3.png"),
            ],
            full_text="",
        )

        body_text = _build_body_text(parsed_paper)
        self.assertIn("Intro text.", body_text)
        self.assertIn("Method text.", body_text)
        self.assertNotIn("References", body_text)
        self.assertNotIn("Appendix", body_text)
        self.assertNotIn("[1] Paper", body_text)

    def test_build_paper_context_contains_section_snippets(self) -> None:
        parsed_paper = ParsedPaper(
            metadata=PaperMetadata(title="测试论文", abstract="摘要", source_pdf="demo.pdf"),
            pages=[
                ParsedPage(
                    page=1,
                    width=400,
                    height=400,
                    text="1. Introduction\nIntro text.\n2. Method\nMethod text.\n3. Experiments\nExperiment text.",
                    rendered_image_path="page1.png",
                    text_blocks=[],
                )
            ],
            full_text="",
        )

        body_text = _build_body_text(parsed_paper)
        paper_context = _build_paper_context(parsed_paper, body_text)
        self.assertIn("[全文正文（已去除参考文献和附录）]", paper_context)
        self.assertIn("[重点章节摘录]", paper_context)
        self.assertIn("[Introduction]", paper_context)
        self.assertIn("[Method]", paper_context)
        self.assertIn("[Experiment]", paper_context)

    def test_build_body_page_limits_stops_at_references(self) -> None:
        parsed_paper = ParsedPaper(
            metadata=PaperMetadata(title="测试论文", abstract="摘要", source_pdf="demo.pdf"),
            pages=[
                ParsedPage(
                    page=1,
                    width=400,
                    height=400,
                    text="正文内容",
                    rendered_image_path="page1.png",
                    text_blocks=[
                        PageTextBlock(page=1, bbox=BoundingBox(x0=0, y0=100, x1=100, y1=130), text="正文内容"),
                    ],
                ),
                ParsedPage(
                    page=2,
                    width=400,
                    height=400,
                    text="Method\n...\nReferences\n[1] Paper",
                    rendered_image_path="page2.png",
                    text_blocks=[
                        PageTextBlock(page=2, bbox=BoundingBox(x0=0, y0=120, x1=100, y1=150), text="Method"),
                        PageTextBlock(page=2, bbox=BoundingBox(x0=0, y0=300, x1=100, y1=330), text="References"),
                    ],
                ),
                ParsedPage(
                    page=3,
                    width=400,
                    height=400,
                    text="Appendix",
                    rendered_image_path="page3.png",
                    text_blocks=[
                        PageTextBlock(page=3, bbox=BoundingBox(x0=0, y0=80, x1=100, y1=110), text="Appendix"),
                    ],
                ),
            ],
            full_text="",
        )

        limits = _build_body_page_limits(parsed_paper)

        self.assertEqual(limits[1], 400)
        self.assertEqual(limits[2], 300)
        self.assertNotIn(3, limits)

    def test_filter_candidates_to_body_removes_reference_and_appendix_candidates(self) -> None:
        limits = {1: 400, 2: 300}
        candidates = [
            FigureCandidate(
                normalized_id="Fig1",
                page=1,
                caption_text="Figure 1: Main figure.",
                caption_bbox=BoundingBox(x0=0, y0=200, x1=1, y1=220),
                figure_bbox=BoundingBox(x0=0, y0=100, x1=1, y1=180),
                image_path="Fig1.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Table1",
                page=2,
                caption_text="Table 1: Reference table.",
                caption_bbox=BoundingBox(x0=0, y0=320, x1=1, y1=340),
                figure_bbox=BoundingBox(x0=0, y0=260, x1=1, y1=310),
                image_path="Table1.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Fig3",
                page=3,
                caption_text="Figure 3: Appendix figure.",
                caption_bbox=BoundingBox(x0=0, y0=120, x1=1, y1=140),
                figure_bbox=BoundingBox(x0=0, y0=60, x1=1, y1=110),
                image_path="Fig3.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
        ]

        filtered = _filter_candidates_to_body(candidates, limits)

        self.assertEqual([item.normalized_id for item in filtered], ["Fig1"])

    def test_build_uses_pdf_directory_and_stem_for_default_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 demo")
            layout_detector = _FakeLayoutDetector()

            orchestrator = PaperReadingOrchestrator(
                AppConfig(
                    openai=OpenAISettings(api_key="test-key"),
                    layout=LayoutSettings(model_path="fake-model.pt"),
                    runtime=RuntimeSettings(output_root=str(temp_root / "unused-output-root")),
                ),
                parser=_FakeParser(),
                layout_detector=layout_detector,
                matcher=_FakeMatcher(),
                selector=_FakeSelector(),
                structure_planner=_FakeReportStructurePlanner(),
                planner=_FakePlanner(),
                explainer=_FakeExplainer(),
                composer=MarkdownComposer(),
                pdf_exporter=_FakePdfExporter(),
            )

            result = orchestrator.build(BuildOptions(pdf_path=pdf_path))

            output_dir = Path(result.output_dir)
            self.assertEqual(output_dir.parent, pdf_path.parent)
            self.assertRegex(output_dir.name, r"^demo-\d{12}(?:-\d+)?$")
            self.assertTrue((output_dir / "artifacts" / "selected_figures.json").exists())
            self.assertTrue(Path(layout_detector._last_image).resolve().is_relative_to(output_dir.resolve()))
            self.assertEqual(_FakeParser.last_max_pages, 40)

    def test_build_generates_markdown_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 demo")
            output_dir = temp_root / "output"

            orchestrator = PaperReadingOrchestrator(
                AppConfig(
                    openai=OpenAISettings(api_key="test-key"),
                    layout=LayoutSettings(model_path="fake-model.pt"),
                    runtime=RuntimeSettings(output_root=str(output_dir)),
                ),
                parser=_FakeParser(),
                layout_detector=_FakeLayoutDetector(),
                matcher=_FakeMatcher(),
                selector=_FakeSelector(),
                structure_planner=_FakeReportStructurePlanner(),
                planner=_FakePlanner(),
                explainer=_FakeExplainer(),
                composer=MarkdownComposer(),
                pdf_exporter=_FakePdfExporter(),
            )

            result = orchestrator.build(BuildOptions(pdf_path=pdf_path, output_dir=output_dir))

            markdown_path = Path(result.markdown_path)
            pdf_path = Path(result.pdf_path)
            self.assertTrue(markdown_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertTrue((output_dir / "artifacts" / "selected_figures.json").exists())
            self.assertTrue((output_dir / "artifacts" / "report_structure.json").exists())
            self.assertTrue((output_dir / "assets" / "Fig1.png").exists())
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("\n测试论文中文译名\n", markdown)
            self.assertNotIn("**中文题目：**", markdown)
            self.assertNotIn("## 核心术语与缩写", markdown)
            self.assertIn("RES（指称表达分割）", markdown)
            self.assertIn("## 分析路径与核心机制", markdown)
            self.assertIn("![Fig1](assets/Fig1.png)", markdown)
            self.assertNotIn("![Fig1](artifacts/figures/Fig1.png)", markdown)
            self.assertNotIn("原始 caption", markdown)

    def test_build_passes_custom_max_pages_to_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 demo")
            output_dir = temp_root / "output"

            orchestrator = PaperReadingOrchestrator(
                AppConfig(
                    openai=OpenAISettings(api_key="test-key"),
                    layout=LayoutSettings(model_path="fake-model.pt"),
                    runtime=RuntimeSettings(output_root=str(output_dir)),
                ),
                parser=_FakeParser(),
                layout_detector=_FakeLayoutDetector(),
                matcher=_FakeMatcher(),
                selector=_FakeSelector(),
                structure_planner=_FakeReportStructurePlanner(),
                planner=_FakePlanner(),
                explainer=_FakeExplainer(),
                composer=MarkdownComposer(),
                pdf_exporter=_FakePdfExporter(),
            )

            orchestrator.build(BuildOptions(pdf_path=pdf_path, output_dir=output_dir, max_pages=12))

            self.assertEqual(_FakeParser.last_max_pages, 12)


if __name__ == "__main__":
    unittest.main()
