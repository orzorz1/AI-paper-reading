"""带静态前端的本地 Web 服务。"""

import argparse
import os
import re
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ..config import load_app_config
from ..errors import PaperReadingError
from ..logging_utils import configure_logging
from ..models import BuildOptions, BuildResult
from ..pipeline.orchestrator import PaperReadingOrchestrator
from ..utils import ensure_dir

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# 全局互斥：同一时间只允许一个 /api/build 任务执行，避免共享编排器/版面模型并发问题。
_BUILD_LOCK = threading.Lock()
_BUSY_MESSAGE = "已有任务正在生成，请稍后再试。"


def _web_output_root() -> Path:
    raw = os.getenv("WEB_OUTPUT_ROOT", "./web_output").strip() or "./web_output"
    return Path(raw).expanduser().resolve()


def _allocate_run_dir(base: Path, original_filename: str) -> Path:
    stem = Path(original_filename).stem.strip() or "paper"
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip()[:120] or "paper"
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    candidate = base / f"{safe_stem}-{timestamp}"
    if not candidate.exists():
        return ensure_dir(candidate)
    for index in range(2, 1000):
        next_candidate = base / f"{safe_stem}-{timestamp}-{index}"
        if not next_candidate.exists():
            return ensure_dir(next_candidate)
    raise HTTPException(status_code=500, detail="输出目录冲突过多，请稍后重试。")


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _validate_focus(focus: str) -> None:
    if focus not in {"method", "experiment"}:
        raise ValueError("侧重只支持 method 或 experiment")


def _validate_length(length: str) -> None:
    if length not in {"short", "medium", "long"}:
        raise ValueError("篇幅只支持 short、medium、long")


def _truthy_form(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "on", "yes")


def _orchestrator_for_request(
    app: Any,
    *,
    api_custom: bool,
    openai_base_url: str,
    openai_api_key: str,
    openai_text_model: str,
    openai_vision_model: str,
) -> PaperReadingOrchestrator:
    base_config = app.state.config
    orchestrator: PaperReadingOrchestrator = app.state.orchestrator
    if not api_custom:
        return orchestrator
    bu = (openai_base_url or "").strip()
    key = (openai_api_key or "").strip()
    text_m = (openai_text_model or "").strip()
    vision_m = (openai_vision_model or "").strip()
    if not bu or not key or not text_m or not vision_m:
        raise HTTPException(
            status_code=400,
            detail="已开启自定义 API：请填写 Base URL、API Key、文本模型与视觉模型。",
        )
    if not bu.startswith(("http://", "https://")):
        bu = "https://" + bu
    custom_openai = base_config.openai.model_copy(
        update={
            "base_url": bu.rstrip("/"),
            "api_key": key,
            "text_model": text_m,
            "vision_model": vision_m,
        }
    )
    return orchestrator.with_openai_settings(custom_openai)


def _run_build(
    *,
    pdf_path: Path,
    output_dir: Path,
    content_focus: str,
    output_length: str,
    orchestrator: PaperReadingOrchestrator,
) -> BuildResult:
    options = BuildOptions(
        pdf_path=pdf_path,
        title=None,
        abstract=None,
        output_dir=output_dir,
        max_figures=None,
        lang="zh-CN",
        content_focus=content_focus,  # type: ignore[arg-type]
        output_length=output_length,  # type: ignore[arg-type]
    )
    return orchestrator.build(options)


@asynccontextmanager
async def _lifespan(app: Any):
    configure_logging(verbose=False)
    app.state.config = load_app_config(None)
    app.state.orchestrator = PaperReadingOrchestrator(app.state.config)
    yield


def create_app() -> Any:
    app = FastAPI(title="paper-reading web", lifespan=_lifespan)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/build")
    async def build_pdf(
        file: UploadFile = File(...),
        focus: str = Form("method"),
        length: str = Form("medium"),
        api_custom: str = Form("false"),
        openai_base_url: str = Form(""),
        openai_api_key: str = Form(""),
        openai_text_model: str = Form(""),
        openai_vision_model: str = Form(""),
    ) -> FileResponse:
        acquired = _BUILD_LOCK.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail=_BUSY_MESSAGE)

        try:
            return await _build_pdf_impl(
                app,
                file=file,
                focus=focus,
                length=length,
                api_custom=api_custom,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                openai_text_model=openai_text_model,
                openai_vision_model=openai_vision_model,
            )
        finally:
            _BUILD_LOCK.release()

    async def _build_pdf_impl(
        app: Any,
        *,
        file: UploadFile,
        focus: str,
        length: str,
        api_custom: str,
        openai_base_url: str,
        openai_api_key: str,
        openai_text_model: str,
        openai_vision_model: str,
    ) -> FileResponse:
        use_custom_api = _truthy_form(api_custom)
        if use_custom_api:
            try:
                _validate_focus(focus)
                _validate_length(length)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            focus = "method"
            length = "medium"

        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="请上传一个 PDF 文件")

        raw = await file.read()
        if len(raw) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件过大（上限 10MB）")
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="文件为空")

        root = _web_output_root()
        ensure_dir(root)
        run_dir = _allocate_run_dir(root, file.filename)
        input_pdf = run_dir / "input.pdf"
        output_dir = run_dir
        input_pdf.write_bytes(raw)

        orchestrator = _orchestrator_for_request(
            app,
            api_custom=use_custom_api,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            openai_text_model=openai_text_model,
            openai_vision_model=openai_vision_model,
        )

        def _build_sync() -> BuildResult:
            return _run_build(
                pdf_path=input_pdf,
                output_dir=output_dir,
                content_focus=focus,
                output_length=length,
                orchestrator=orchestrator,
            )

        try:
            result = await run_in_threadpool(_build_sync)
        except PaperReadingError as exc:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not result.pdf_path:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="未生成 PDF")

        pdf_path = Path(result.pdf_path)
        if not pdf_path.is_file():
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="PDF 文件缺失")

        return FileResponse(
            path=str(pdf_path),
            filename="paper_readable.pdf",
            media_type="application/pdf",
        )

    static = _static_dir()
    if not static.is_dir():
        raise RuntimeError(f"静态资源目录不存在：{static}")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static / "index.html")

    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="启动 paper-reading Web 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="端口")
    args = parser.parse_args()
    uvicorn.run(
        "paper_reading.web.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        factory=False,
    )
