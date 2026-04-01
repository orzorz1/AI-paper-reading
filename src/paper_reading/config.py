"""应用配置读取。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from .errors import ConfigurationError


class OpenAISettings(BaseModel):
    """OpenAI 兼容接口配置。"""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    text_model: str = "gpt-4.1-mini"
    vision_model: str = "gpt-4.1-mini"
    timeout_seconds: float = 120.0
    max_retries: int = 5
    retry_backoff_seconds: float = 1.5


class LayoutSettings(BaseModel):
    """版面识别模型配置。"""

    model_path: str = ""
    confidence: float = 0.1
    device: str = "cpu"
    predict_imgsz: int = 1024
    render_dpi: int = 150
    crop_dpi: int = 220


class RuntimeSettings(BaseModel):
    """运行时配置。"""

    default_lang: str = "zh-CN"
    output_root: str = "./output"
    max_figures: int = 3


class AppConfig(BaseModel):
    """应用总配置。"""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    layout: LayoutSettings = Field(default_factory=LayoutSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


def _load_dotenv(dotenv_path: Path) -> None:
    """从 .env 文件读取环境变量。

    这里只实现项目需要的最小能力，避免为了一个简单 CLI 再引入额外依赖。
    已经存在于进程环境里的变量不覆盖，保证命令行显式导出的值优先级更高。
    """
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_yaml_config(config_path: Optional[Path]) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise ConfigurationError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ConfigurationError("配置文件根节点必须是对象。")
    return content


def _override_from_env(data: dict[str, Any]) -> dict[str, Any]:
    openai = dict(data.get("openai", {}))
    layout = dict(data.get("layout", {}))
    runtime = dict(data.get("runtime", {}))

    openai["api_key"] = os.getenv("OPENAI_API_KEY", openai.get("api_key", ""))
    openai["base_url"] = os.getenv("OPENAI_BASE_URL", openai.get("base_url", "https://api.openai.com/v1"))
    openai["text_model"] = os.getenv("TEXT_MODEL", openai.get("text_model", "gpt-4.1-mini"))
    openai["vision_model"] = os.getenv("VISION_MODEL", openai.get("vision_model", "gpt-4.1-mini"))
    openai["max_retries"] = int(os.getenv("OPENAI_MAX_RETRIES", openai.get("max_retries", 5)))
    openai["retry_backoff_seconds"] = float(
        os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", openai.get("retry_backoff_seconds", 1.5))
    )

    layout["model_path"] = os.getenv("LAYOUT_MODEL_PATH", layout.get("model_path", ""))
    layout["device"] = os.getenv("LAYOUT_DEVICE", layout.get("device", "cpu"))
    layout["confidence"] = float(os.getenv("LAYOUT_CONFIDENCE", layout.get("confidence", 0.1)))
    layout["predict_imgsz"] = int(os.getenv("LAYOUT_PREDICT_IMGSZ", layout.get("predict_imgsz", 1024)))

    runtime["output_root"] = os.getenv("OUTPUT_ROOT", runtime.get("output_root", "./output"))
    runtime["default_lang"] = os.getenv("DEFAULT_LANG", runtime.get("default_lang", "zh-CN"))

    return {"openai": openai, "layout": layout, "runtime": runtime}


def load_app_config(config_path: Optional[Path] = None) -> AppConfig:
    """从 YAML 和环境变量中读取配置。"""
    _load_dotenv(Path.cwd() / ".env")
    base = _load_yaml_config(config_path)
    merged = _override_from_env(base)
    return AppConfig.model_validate(merged)
