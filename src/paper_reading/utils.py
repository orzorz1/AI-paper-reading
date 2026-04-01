"""通用工具函数。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回原路径。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify_filename(value: str) -> str:
    """把任意字符串转成适合文件名的形式。"""
    value = value.strip().lower()
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "paper"


def dump_json(path: Path, data: Any) -> None:
    """统一输出 UTF-8 JSON。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dump_model_json(path: Path, model: BaseModel) -> None:
    """输出单个 Pydantic 模型。"""
    dump_json(path, model.model_dump(mode="json"))


def dump_models_json(path: Path, models: list[BaseModel]) -> None:
    """输出多个 Pydantic 模型。"""
    dump_json(path, [item.model_dump(mode="json") for item in models])


def relative_posix_path(target: Path, start: Path) -> str:
    """生成稳定的相对路径，供 Markdown 引用。"""
    return target.relative_to(start).as_posix() if target.is_relative_to(start) else target.as_posix()


def truncate_text(text: str, limit: int) -> str:
    """截断长文本，避免 prompt 过长。"""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_whitespace(text: str) -> str:
    """清理多余空白，保留单个空格和换行。"""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
