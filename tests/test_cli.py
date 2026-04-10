from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from paper_reading.cli import app
from paper_reading.config import AppConfig, LayoutSettings, OpenAISettings, RuntimeSettings
from paper_reading.models import BuildResult


class _FakeOrchestrator:
    built_paths: list[Path] = []
    built_options: list[object] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.__class__.built_paths = []
        self.__class__.built_options = []

    def build(self, options) -> BuildResult:
        self.__class__.built_paths.append(options.pdf_path)
        self.__class__.built_options.append(options)
        output_dir = options.pdf_path.parent / f"{options.pdf_path.stem}-202604012200"
        artifact_dir = output_dir / "artifacts"
        return BuildResult(
            markdown_path=str(output_dir / "paper_readable.md"),
            pdf_path=str(output_dir / "paper_readable.pdf"),
            output_dir=str(output_dir),
            artifact_dir=str(artifact_dir),
            selected_count=3,
            warnings=[],
        )


class CLITests(unittest.TestCase):
    def test_help_shows_build_and_batch_subcommands(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "paper_reading.cli", "--help"],
            env={"PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("build", result.stdout)
        self.assertIn("batch", result.stdout)

    def test_batch_only_processes_top_level_pdfs(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "a.pdf").write_bytes(b"%PDF-1.4 a")
            (temp_root / "b.PDF").write_bytes(b"%PDF-1.4 b")
            nested_dir = temp_root / "nested"
            nested_dir.mkdir()
            (nested_dir / "c.pdf").write_bytes(b"%PDF-1.4 c")
            (temp_root / "note.txt").write_text("ignore", encoding="utf-8")

            with patch("paper_reading.cli.load_app_config", return_value=AppConfig(
                openai=OpenAISettings(api_key="test-key"),
                layout=LayoutSettings(model_path="fake-model.pt"),
                runtime=RuntimeSettings(),
            )), patch("paper_reading.cli.PaperReadingOrchestrator", _FakeOrchestrator):
                result = runner.invoke(app, ["batch", str(temp_root)])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("共发现 2 个 PDF", result.stdout)
        built_names = sorted(path.name for path in _FakeOrchestrator.built_paths)
        self.assertEqual(built_names, ["a.pdf", "b.PDF"])
        self.assertTrue(all(item.content_focus == "method" for item in _FakeOrchestrator.built_options))
        self.assertTrue(all(item.output_length == "medium" for item in _FakeOrchestrator.built_options))

    def test_build_accepts_focus_and_length_options(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 demo")

            with patch("paper_reading.cli.load_app_config", return_value=AppConfig(
                openai=OpenAISettings(api_key="test-key"),
                layout=LayoutSettings(model_path="fake-model.pt"),
                runtime=RuntimeSettings(),
            )), patch("paper_reading.cli.PaperReadingOrchestrator", _FakeOrchestrator):
                result = runner.invoke(app, ["build", str(pdf_path), "--focus", "experiment", "--length", "long"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertEqual(_FakeOrchestrator.built_options[0].content_focus, "experiment")
        self.assertEqual(_FakeOrchestrator.built_options[0].output_length, "long")
        self.assertEqual(_FakeOrchestrator.built_options[0].max_pages, 40)
        self.assertEqual(_FakeOrchestrator.built_options[0].writing_style, "professional")

    def test_build_accepts_style_option(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 demo")

            with patch("paper_reading.cli.load_app_config", return_value=AppConfig(
                openai=OpenAISettings(api_key="test-key"),
                layout=LayoutSettings(model_path="fake-model.pt"),
                runtime=RuntimeSettings(),
            )), patch("paper_reading.cli.PaperReadingOrchestrator", _FakeOrchestrator):
                result = runner.invoke(app, ["build", str(pdf_path), "--style", "colloquial"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertEqual(_FakeOrchestrator.built_options[0].writing_style, "colloquial")

    def test_build_accepts_max_pages_option(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 demo")

            with patch("paper_reading.cli.load_app_config", return_value=AppConfig(
                openai=OpenAISettings(api_key="test-key"),
                layout=LayoutSettings(model_path="fake-model.pt"),
                runtime=RuntimeSettings(),
            )), patch("paper_reading.cli.PaperReadingOrchestrator", _FakeOrchestrator):
                result = runner.invoke(app, ["build", str(pdf_path), "--max-pages", "12"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertEqual(_FakeOrchestrator.built_options[0].max_pages, 12)


if __name__ == "__main__":
    unittest.main()
