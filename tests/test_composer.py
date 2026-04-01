from __future__ import annotations

import unittest
from pathlib import Path

from paper_reading.models import BoundingBox, FigureCandidate, FigureExplanation, StoryOutline
from paper_reading.pipeline.composer import MarkdownComposer


class _FakeClient:
    def complete_text(self, **_: object) -> str:
        return (
            "## 研究背景与任务定义\n\n"
            "已有方法主要关注 RES（指称表达分割）。\n\n"
            "![Fig1](artifacts/figures/Fig1.png)\n"
        )


class ComposerTests(unittest.TestCase):
    def test_render_markdown_contains_required_sections(self) -> None:
        composer = MarkdownComposer()
        output_dir = Path("/tmp/demo-output")
        outline = StoryOutline(
            title_translation="测试论文的中文译名",
            one_sentence_summary="这是一段更完整的导读摘要，用来帮助读者快速理解 RES（Referring Expression Segmentation，指称表达分割）论文主线。",
            motivation="动机说明，介绍研究背景和任务定义，并说明 MRES（Multi-Granularity Referring Expression Segmentation，多粒度指称表达分割）的目标。随后继续说明 RES 在细粒度场景下的局限。",
            method_core="方法说明，介绍整体框架与关键机制。该方法继续沿用 RES 的基本设定。",
            result_summary="结果说明",
            figure_roles={"Fig1": "method"},
        )
        candidate = FigureCandidate(
            normalized_id="Fig1",
            page=1,
            caption_text="Figure 1: Pipeline.",
            caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            image_path="/tmp/demo-output/artifacts/figures/Fig1.png",
            match_score=0.9,
            source="layout+rule",
            match_options=[],
        )
        explanation = FigureExplanation(
            normalized_id="Fig1",
            title="整体流程图",
            what_it_shows="这张图说明模型的整体流程。",
            how_to_read="从左到右看输入、编码器和输出。",
            why_it_matters="它把方法主线集中展示出来。",
            reader_takeaway="核心是把视觉特征和文本特征对齐。",
        )
        document = composer.compose(
            title="测试论文",
            outline=outline,
            ordered_candidates=[candidate],
            explanations=[explanation],
            output_dir=output_dir,
        )
        markdown = composer.render_markdown(document)
        self.assertIn("# 测试论文", markdown)
        self.assertIn("\n测试论文的中文译名\n", markdown)
        self.assertNotIn("**中文题目：**", markdown)
        self.assertIn("> 导读摘要：", markdown)
        self.assertNotIn("## 核心术语与缩写", markdown)
        self.assertIn("RES（指称表达分割）", markdown)
        self.assertIn("MRES（多粒度指称表达分割）", markdown)
        self.assertEqual(markdown.count("RES（指称表达分割）"), 1)
        self.assertIn("RES 在细粒度场景下的局限", markdown)
        self.assertIn("模型的整体流程。", markdown)
        self.assertNotIn("配合下图", markdown)
        self.assertNotIn("结合下图看", markdown)
        self.assertNotIn("如图所示", markdown)
        self.assertIn("## 研究背景与任务定义", markdown)
        self.assertIn("## 方法设计与核心机制", markdown)
        self.assertIn("![Fig1](artifacts/figures/Fig1.png)", markdown)
        self.assertNotIn("原始 caption", markdown)
        self.assertNotIn("读者应记住什么", markdown)

    def test_generate_markdown_fills_missing_selected_figures(self) -> None:
        composer = MarkdownComposer(_FakeClient(), "demo-model")
        output_dir = Path("/tmp/demo-output")
        outline = StoryOutline(
            title_translation="测试论文的中文译名",
            one_sentence_summary="导读摘要。",
            motivation="背景说明。",
            method_core="方法说明。",
            result_summary="",
            figure_roles={"Fig1": "problem", "Fig3": "support"},
        )
        candidates = [
            FigureCandidate(
                normalized_id="Fig1",
                page=1,
                caption_text="Figure 1: Pipeline.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="/tmp/demo-output/artifacts/figures/Fig1.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
            FigureCandidate(
                normalized_id="Fig3",
                page=2,
                caption_text="Figure 3: Data engine.",
                caption_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                figure_bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                image_path="/tmp/demo-output/artifacts/figures/Fig3.png",
                match_score=0.9,
                source="layout+rule",
                match_options=[],
            ),
        ]
        explanations = [
            FigureExplanation(
                normalized_id="Fig1",
                title="整体流程图",
                what_it_shows="模型的整体流程。",
                how_to_read="按从左到右的顺序阅读。",
                why_it_matters="它能最快建立读者对方法的整体认识。",
            ),
            FigureExplanation(
                normalized_id="Fig3",
                title="数据构建流程",
                what_it_shows="展示数据引擎如何构建多粒度标注。",
                how_to_read="按 a 到 c 的顺序阅读。",
                why_it_matters="它说明数据集是如何扩展出来的。",
            ),
        ]

        markdown = composer.generate_markdown(
            title="测试论文",
            abstract="摘要内容",
            paper_context="上下文内容",
            outline=outline,
            selected_figures=[
                {"normalized_id": "Fig1", "reason": "问题定义", "importance_rank": 1},
                {"normalized_id": "Fig3", "reason": "数据构建", "importance_rank": 2},
            ],
            ordered_candidates=candidates,
            explanations=explanations,
            output_dir=output_dir,
        )

        self.assertIn("# 测试论文", markdown)
        self.assertIn("\n测试论文的中文译名\n", markdown)
        self.assertNotIn("**中文题目：**", markdown)
        self.assertIn("> 导读摘要：导读摘要。", markdown)
        self.assertIn("![Fig1](artifacts/figures/Fig1.png)", markdown)
        self.assertIn("![Fig3](artifacts/figures/Fig3.png)", markdown)
        self.assertIn("## 补充图示", markdown)


if __name__ == "__main__":
    unittest.main()
