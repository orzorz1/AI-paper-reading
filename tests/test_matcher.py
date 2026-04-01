from __future__ import annotations

import unittest

from paper_reading.models import BoundingBox, LayoutRegion, PageImageBlock, PageTextBlock
from paper_reading.pipeline.matcher import _collect_caption_blocks, _score_match, normalize_figure_id


class MatcherTests(unittest.TestCase):
    def test_normalize_figure_id(self) -> None:
        self.assertEqual(normalize_figure_id("Fig. 1 Overview"), "Fig1")
        self.assertEqual(normalize_figure_id("Figure 2 shows results"), "Fig2")
        self.assertEqual(normalize_figure_id("TABLE 3 ablation"), "Table3")

    def test_collect_caption_blocks_uses_layout_regions(self) -> None:
        blocks = [
            PageTextBlock(page=1, bbox=BoundingBox(x0=0, y0=90, x1=100, y1=120), text="Overview of pipeline"),
            PageTextBlock(page=1, bbox=BoundingBox(x0=0, y0=130, x1=200, y1=160), text="Figure 1: Overall pipeline."),
        ]
        regions = [
            LayoutRegion(page=1, bbox=BoundingBox(x0=0, y0=125, x1=220, y1=165), label="caption", score=0.9),
        ]
        captions = _collect_caption_blocks(blocks, regions)
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].text, "Figure 1: Overall pipeline.")

    def test_collect_caption_blocks_ignores_body_reference(self) -> None:
        blocks = [
            PageTextBlock(
                page=1,
                bbox=BoundingBox(x0=0, y0=80, x1=220, y1=120),
                text="As shown in Fig. 1, our method improves the baseline.",
            ),
            PageTextBlock(
                page=1,
                bbox=BoundingBox(x0=0, y0=130, x1=260, y1=170),
                text="Figure 1. Overall pipeline.",
            ),
        ]
        regions = [
            LayoutRegion(page=1, bbox=BoundingBox(x0=0, y0=125, x1=280, y1=175), label="caption", score=0.9),
        ]
        captions = _collect_caption_blocks(blocks, regions)
        self.assertEqual([item.text for item in captions], ["Figure 1. Overall pipeline."])

    def test_collect_caption_blocks_supports_subfigure_prefix(self) -> None:
        blocks = [
            PageTextBlock(
                page=1,
                bbox=BoundingBox(x0=0, y0=130, x1=260, y1=180),
                text="(a) Input (b) Output Figure 5. Qualitative comparison.",
            ),
        ]
        captions = _collect_caption_blocks(blocks, [])
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].text, "(a) Input (b) Output Figure 5. Qualitative comparison.")

    def test_score_prefers_nearby_figure(self) -> None:
        caption = BoundingBox(x0=100, y0=300, x1=400, y1=340)
        near_figure = BoundingBox(x0=80, y0=120, x1=420, y1=290)
        far_figure = BoundingBox(x0=80, y0=20, x1=420, y1=90)
        image_blocks = [PageImageBlock(page=1, bbox=near_figure)]
        near_score = _score_match(caption, near_figure, image_blocks)
        far_score = _score_match(caption, far_figure, image_blocks)
        self.assertGreater(near_score, far_score)


if __name__ == "__main__":
    unittest.main()
