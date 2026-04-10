"""OpenAI 兼容接口封装。"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel

from .config import OpenAISettings
from .errors import ConfigurationError, LLMResponseError, LLMServiceError
from .logging_utils import get_logger

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class OpenAICompatibleClient:
    """统一的 OpenAI 兼容接口客户端。"""

    def __init__(self, settings: OpenAISettings) -> None:
        if not settings.api_key:
            raise ConfigurationError("缺少 OPENAI_API_KEY，无法调用大模型。")
        self._settings = settings
        self._logger = get_logger()
        self._client = httpx.Client(
            base_url=settings.base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=settings.timeout_seconds,
        )

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[ResponseModelT],
        temperature: float = 0.2,
    ) -> ResponseModelT:
        """调用纯文本模型，并解析结构化 JSON。"""
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return self._complete_structured(
            model=model,
            path="chat/completions",
            payload=payload,
            response_model=response_model,
        )

    def complete_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """调用纯文本模型，返回原始文本。"""
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = self._post_with_retry("chat/completions", payload=payload, model=model)
        return _extract_message_text(response.json()).strip()

    def complete_vision_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[Path],
        response_model: Type[ResponseModelT],
        temperature: float = 0.2,
    ) -> ResponseModelT:
        """调用支持图像输入的模型，并解析结构化 JSON。"""
        user_content = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            user_content.append({"type": "image_url", "image_url": {"url": _to_data_uri(image_path)}})
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        return self._complete_structured(
            model=model,
            path="chat/completions",
            payload=payload,
            response_model=response_model,
        )

    def _complete_structured(
        self,
        *,
        model: str,
        path: str,
        payload: dict,
        response_model: Type[ResponseModelT],
    ) -> ResponseModelT:
        max_parse_attempts = 3
        last_error: LLMResponseError | None = None

        for attempt in range(1, max_parse_attempts + 1):
            response = self._post_with_retry(path, payload=payload, model=model)
            content = _extract_message_text(response.json())
            try:
                return _parse_json_content(content, response_model)
            except LLMResponseError as exc:
                last_error = exc
                if attempt >= max_parse_attempts:
                    break
                self._logger.warning(
                    "模型结构化输出解析失败，准备第 %d/%d 次重新生成，模型=%s",
                    attempt + 1,
                    max_parse_attempts,
                    model,
                )

        raise last_error or LLMResponseError("模型返回无法解析为 JSON。")

    def _post_with_retry(self, path: str, *, payload: dict, model: str) -> httpx.Response:
        """对模型接口做有限重试，优先处理网关错误、超时和连接失败。"""
        max_attempts = max(1, self._settings.max_retries + 1)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.post(path, json=payload)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                if _should_retry_status(status_code) and attempt < max_attempts:
                    self._logger.warning(
                        "模型请求失败（HTTP %s），准备第 %d/%d 次重试，模型=%s",
                        status_code,
                        attempt + 1,
                        max_attempts,
                        model,
                    )
                    time.sleep(self._settings.retry_backoff_seconds * attempt)
                    continue
                raise LLMServiceError(_format_http_error(exc, model)) from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < max_attempts:
                    self._logger.warning(
                        "模型请求异常（%s），准备第 %d/%d 次重试，模型=%s",
                        exc.__class__.__name__,
                        attempt + 1,
                        max_attempts,
                        model,
                    )
                    time.sleep(self._settings.retry_backoff_seconds * attempt)
                    continue
                raise LLMServiceError(_format_request_error(exc, model)) from exc

        raise LLMServiceError(f"模型请求失败：{last_error}") from last_error


def _extract_message_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMResponseError("模型未返回任何候选结果。")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts).strip()
    raise LLMResponseError("模型返回内容格式无法识别。")


def _parse_json_content(content: str, response_model: Type[ResponseModelT]) -> ResponseModelT:
    try:
        return response_model.model_validate_json(content)
    except Exception:
        try:
            json_text = _extract_json_snippet(content)
            return response_model.model_validate_json(json_text)
        except Exception as exc:
            raise LLMResponseError(f"模型返回无法解析为 JSON：{content}") from exc


def _extract_json_snippet(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    raise LLMResponseError("未在模型输出中找到 JSON 片段。")


def _to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def _should_retry_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _format_http_error(exc: httpx.HTTPStatusError, model: str) -> str:
    response_text = exc.response.text.strip()
    response_text = response_text[:300] + "..." if len(response_text) > 300 else response_text
    if response_text:
        return (
            f"模型服务调用失败：HTTP {exc.response.status_code}，模型={model}，"
            f"地址={exc.request.url}，响应={response_text}"
        )
    return f"模型服务调用失败：HTTP {exc.response.status_code}，模型={model}，地址={exc.request.url}"


def _format_request_error(exc: httpx.RequestError, model: str) -> str:
    return f"模型服务调用失败：{exc.__class__.__name__}，模型={model}，地址={exc.request.url}"
