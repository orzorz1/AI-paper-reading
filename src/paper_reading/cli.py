"""命令行入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import load_app_config
from .errors import PaperReadingError
from .logging_utils import configure_logging
from .models import BuildOptions, BuildResult
from .pipeline.orchestrator import PaperReadingOrchestrator

app = typer.Typer(help="AI 论文读图理解 CLI 工具")


@app.callback()
def main() -> None:
    """命令行主入口。"""


def _validate_lang(lang: str) -> None:
    if lang != "zh-CN":
        raise typer.BadParameter("V1 目前只支持 zh-CN 输出。")


def _validate_content_focus(content_focus: str) -> None:
    if content_focus not in {"method", "experiment"}:
        raise typer.BadParameter("内容偏好只支持 method 或 experiment。")


def _validate_output_length(output_length: str) -> None:
    if output_length not in {"short", "medium", "long"}:
        raise typer.BadParameter("篇幅只支持 short、medium 或 long。")


def _validate_writing_style(writing_style: str) -> None:
    if writing_style not in {"professional", "colloquial"}:
        raise typer.BadParameter("风格只支持 professional 或 colloquial。")


def _print_build_result(result: BuildResult) -> None:
    typer.secho(f"Markdown 已生成：{result.markdown_path}", fg=typer.colors.GREEN)
    if result.pdf_path:
        typer.secho(f"PDF 已生成：{result.pdf_path}", fg=typer.colors.GREEN)
    typer.echo(f"中间产物目录：{result.artifact_dir}")
    if result.warnings:
        typer.secho("运行告警：", fg=typer.colors.YELLOW)
        for warning in result.warnings:
            typer.echo(f"- {warning}")


def _build_single(
    *,
    orchestrator: PaperReadingOrchestrator,
    pdf_path: Path,
    title: Optional[str],
    abstract: Optional[str],
    output_dir: Optional[Path],
    max_figures: Optional[int],
    max_pages: int,
    lang: str,
    content_focus: str,
    output_length: str,
    writing_style: str,
) -> BuildResult:
    options = BuildOptions(
        pdf_path=pdf_path,
        title=title,
        abstract=abstract,
        output_dir=output_dir,
        max_figures=max_figures,
        max_pages=max_pages,
        lang=lang,
        content_focus=content_focus,
        output_length=output_length,
        writing_style=writing_style,
    )
    return orchestrator.build(options)


@app.command()
def build(
    pdf_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="论文 PDF 路径"),
    title: Optional[str] = typer.Option(None, "--title", help="可选，手动覆盖标题"),
    abstract: Optional[str] = typer.Option(None, "--abstract", help="可选，手动覆盖摘要"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", file_okay=False, help="输出目录"),
    max_figures: Optional[int] = typer.Option(None, "--max-figures", min=1, max=10, help="可选，手动覆盖自动选图数量"),
    max_pages: int = typer.Option(40, "--max-pages", min=1, help="最多处理前 N 页，默认 40"),
    lang: str = typer.Option("zh-CN", "--lang", help="输出语言，V1 只支持 zh-CN"),
    content_focus: str = typer.Option("method", "--focus", help="内容偏好：method 或 experiment"),
    output_length: str = typer.Option("medium", "--length", help="篇幅：short、medium、long"),
    writing_style: str = typer.Option("professional", "--style", help="风格：professional 或 colloquial"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, dir_okay=False, help="可选 YAML 配置文件"),
    verbose: bool = typer.Option(False, "--verbose", help="输出更详细的调试日志"),
) -> None:
    """构建单篇论文阅读版 Markdown 和 PDF。"""
    try:
        _validate_lang(lang)
        _validate_content_focus(content_focus)
        _validate_output_length(output_length)
        _validate_writing_style(writing_style)
        configure_logging(verbose=verbose)
        app_config = load_app_config(config)
        orchestrator = PaperReadingOrchestrator(app_config)
        result = _build_single(
            orchestrator=orchestrator,
            pdf_path=pdf_path,
            title=title,
            abstract=abstract,
            output_dir=output_dir,
            max_figures=max_figures,
            max_pages=max_pages,
            lang=lang,
            content_focus=content_focus,
            output_length=output_length,
            writing_style=writing_style,
        )
    except PaperReadingError as exc:
        typer.secho(f"构建失败：{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _print_build_result(result)


@app.command()
def batch(
    folder_path: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="包含 PDF 的目录路径"),
    max_figures: Optional[int] = typer.Option(None, "--max-figures", min=1, max=10, help="可选，手动覆盖自动选图数量"),
    max_pages: int = typer.Option(40, "--max-pages", min=1, help="每篇最多处理前 N 页，默认 40"),
    lang: str = typer.Option("zh-CN", "--lang", help="输出语言，V1 只支持 zh-CN"),
    content_focus: str = typer.Option("method", "--focus", help="内容偏好：method 或 experiment"),
    output_length: str = typer.Option("medium", "--length", help="篇幅：short、medium、long"),
    writing_style: str = typer.Option("professional", "--style", help="风格：professional 或 colloquial"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, dir_okay=False, help="可选 YAML 配置文件"),
    verbose: bool = typer.Option(False, "--verbose", help="输出更详细的调试日志"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="遇到失败时立即停止批量处理"),
) -> None:
    """批量处理某个目录下这一层的所有 PDF。"""
    _validate_lang(lang)
    _validate_content_focus(content_focus)
    _validate_output_length(output_length)
    _validate_writing_style(writing_style)
    configure_logging(verbose=verbose)
    app_config = load_app_config(config)
    orchestrator = PaperReadingOrchestrator(app_config)

    pdf_paths = sorted(
        [path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda item: item.name.lower(),
    )
    if not pdf_paths:
        typer.secho(f"未在目录中找到 PDF：{folder_path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"共发现 {len(pdf_paths)} 个 PDF，开始批量处理。")
    successes: list[tuple[Path, BuildResult]] = []
    failures: list[tuple[Path, str]] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        typer.secho(f"[{index}/{len(pdf_paths)}] 处理：{pdf_path.name}", fg=typer.colors.CYAN)
        try:
            result = _build_single(
                orchestrator=orchestrator,
                pdf_path=pdf_path,
                title=None,
                abstract=None,
                output_dir=None,
                max_figures=max_figures,
                max_pages=max_pages,
                lang=lang,
                content_focus=content_focus,
                output_length=output_length,
                writing_style=writing_style,
            )
        except PaperReadingError as exc:
            failures.append((pdf_path, str(exc)))
            typer.secho(f"失败：{pdf_path.name} -> {exc}", fg=typer.colors.RED, err=True)
            if fail_fast:
                raise typer.Exit(code=1) from exc
            continue

        successes.append((pdf_path, result))
        typer.secho(f"完成：{pdf_path.name}", fg=typer.colors.GREEN)
        typer.echo(f"  Markdown: {result.markdown_path}")
        if result.pdf_path:
            typer.echo(f"  PDF: {result.pdf_path}")

    typer.echo("")
    typer.secho(
        f"批量处理完成：成功 {len(successes)}，失败 {len(failures)}",
        fg=typer.colors.GREEN if not failures else typer.colors.YELLOW,
    )
    if failures:
        typer.secho("失败列表：", fg=typer.colors.YELLOW)
        for pdf_path, message in failures:
            typer.echo(f"- {pdf_path.name}: {message}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
