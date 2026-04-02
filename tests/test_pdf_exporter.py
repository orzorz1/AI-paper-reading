from __future__ import annotations

import unittest

from paper_reading.pipeline.pdf_exporter import _parse_markdown_blocks, _prepare_text_for_pdf


class PdfExporterTests(unittest.TestCase):
    def test_parse_markdown_blocks_keeps_list_items_separate(self) -> None:
        markdown = (
            "# 标题\n\n"
            "中文标题\n\n"
            "## 实验结果\n\n"
            "- 在任务层面，提出 MRES。\n"
            "- 在评测层面，构建 RefCOCOm。\n"
            "- 在模型层面，提出 UniRES。\n"
        )

        blocks = _parse_markdown_blocks(markdown)

        self.assertEqual(
            blocks,
            [
                ("h1", "标题"),
                ("paragraph", "中文标题"),
                ("h2", "实验结果"),
                ("list_item", "在任务层面，提出 MRES。"),
                ("list_item", "在评测层面，构建 RefCOCOm。"),
                ("list_item", "在模型层面，提出 UniRES。"),
            ],
        )

    def test_parse_markdown_blocks_recognizes_h3(self) -> None:
        markdown = "# 主标题\n\n## 二级\n\n### 三级小节\n\n正文。\n"

        blocks = _parse_markdown_blocks(markdown)

        self.assertEqual(
            blocks,
            [
                ("h1", "主标题"),
                ("h2", "二级"),
                ("h3", "三级小节"),
                ("paragraph", "正文。"),
            ],
        )

    def test_prepare_text_for_pdf_binds_ascii_and_cjk_boundaries(self) -> None:
        text = "覆盖80个物体类别和391个部件类别，并支持MRES-32M训练。"

        prepared = _prepare_text_for_pdf(text)

        self.assertIn("80\u2060个", prepared)
        self.assertIn("391\u2060个", prepared)
        self.assertIn("支持\u2060MRES-32M", prepared)


if __name__ == "__main__":
    unittest.main()
