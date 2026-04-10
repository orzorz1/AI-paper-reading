from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_reading.pipeline.pdf_exporter import (
    MarkdownPdfExporter,
    _build_tex_command,
    _find_tex_engine,
    _parse_markdown_blocks,
    render_markdown_to_tex,
)


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

    def test_render_markdown_to_tex_preserves_math_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            image_path = temp_dir / "figures" / "Fig1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake")

            markdown = (
                "# High-order paper\n\n"
                "中文标题\n\n"
                "## 方法\n\n"
                "核心公式是 \\[\\frac{d\\mathbf U}{dt}=\\mathbf R(t,\\mathbf U)\\]。\n\n"
                "![Fig1](figures/Fig1.png)\n"
            )

            rendered = render_markdown_to_tex(markdown, title="High-order paper", base_dir=temp_dir)

            self.assertIn("\\documentclass[11pt]{ctexart}", rendered)
            self.assertIn("{\\LARGE\\bfseries High-order paper}", rendered)
            self.assertIn("{\\large\\color{gray} 中文标题}", rendered)
            self.assertIn("\\section*{方法}", rendered)
            self.assertIn("\\[\\frac{d\\mathbf U}{dt}=\\mathbf R(t,\\mathbf U)\\]", rendered)
            self.assertIn(
                f"\\includegraphics[width=0.92\\linewidth]{{\\detokenize{{{image_path.resolve().as_posix()}}}}}",
                rendered,
            )

    def test_find_tex_engine_prefers_tectonic(self) -> None:
        with patch(
            "paper_reading.pipeline.pdf_exporter.shutil.which",
            side_effect=lambda name: {
                "tectonic": "/usr/local/bin/tectonic",
                "xelatex": "/Library/TeX/texbin/xelatex",
                "lualatex": None,
            }.get(name),
        ):
            engine = _find_tex_engine()

        self.assertEqual(engine, ("tectonic", "/usr/local/bin/tectonic"))

    def test_build_tex_command_supports_tectonic(self) -> None:
        command = _build_tex_command(
            "tectonic",
            "/usr/local/bin/tectonic",
            tex_path=Path("/tmp/demo.tex"),
            output_dir=Path("/tmp/out"),
        )

        self.assertEqual(
            command,
            [
                "/usr/local/bin/tectonic",
                "--keep-logs",
                "--keep-intermediates",
                "--outdir",
                "/tmp/out",
                "/tmp/demo.tex",
            ],
        )

    def test_export_falls_back_when_tex_engine_missing(self) -> None:
        exporter = MarkdownPdfExporter()
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            markdown_path = temp_dir / "paper_readable.md"
            markdown_path.write_text("# 标题\n\n正文。\n", encoding="utf-8")
            pdf_path = temp_dir / "paper_readable.pdf"

            with patch("paper_reading.pipeline.pdf_exporter.shutil.which", return_value=None):
                with patch("paper_reading.pipeline.pdf_exporter._export_with_reportlab", return_value=pdf_path) as mocked:
                    result = exporter.export(markdown_path, pdf_path)

        self.assertEqual(result, pdf_path)
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
