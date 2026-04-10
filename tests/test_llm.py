from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx
from pydantic import BaseModel

from paper_reading.config import OpenAISettings
from paper_reading.errors import LLMResponseError, LLMServiceError
from paper_reading.llm import OpenAICompatibleClient


class _SimpleResponse(BaseModel):
    value: str


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, path: str, json: dict):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class LLMClientTests(unittest.TestCase):
    def test_complete_json_retries_on_http_502_then_succeeds(self) -> None:
        settings = OpenAISettings(api_key="test-key", max_retries=2, retry_backoff_seconds=0.0)
        client = OpenAICompatibleClient(settings)

        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        failure = httpx.Response(502, request=request, text="bad gateway")
        success = httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
        )
        client._client = _FakeClient([httpx.HTTPStatusError("bad gateway", request=request, response=failure), success])

        with patch("paper_reading.llm.time.sleep", return_value=None):
            result = client.complete_json(
                model="demo-model",
                system_prompt="system",
                user_prompt="user",
                response_model=_SimpleResponse,
            )

        self.assertEqual(result.value, "ok")
        self.assertEqual(client._client.calls, 2)

    def test_complete_json_raises_clean_error_after_retries(self) -> None:
        settings = OpenAISettings(api_key="test-key", max_retries=1, retry_backoff_seconds=0.0)
        client = OpenAICompatibleClient(settings)

        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(502, request=request, text="bad gateway")
        error = httpx.HTTPStatusError("bad gateway", request=request, response=response)
        client._client = _FakeClient([error, error])

        with patch("paper_reading.llm.time.sleep", return_value=None):
            with self.assertRaises(LLMServiceError) as ctx:
                client.complete_json(
                    model="demo-model",
                    system_prompt="system",
                    user_prompt="user",
                    response_model=_SimpleResponse,
                )

        self.assertIn("HTTP 502", str(ctx.exception))
        self.assertIn("demo-model", str(ctx.exception))

    def test_complete_json_regenerates_when_json_parse_fails(self) -> None:
        settings = OpenAISettings(api_key="test-key", max_retries=0, retry_backoff_seconds=0.0)
        client = OpenAICompatibleClient(settings)

        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        malformed = httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"value":"bad"'}}]},
        )
        success = httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
        )
        client._client = _FakeClient([malformed, success])

        result = client.complete_json(
            model="demo-model",
            system_prompt="system",
            user_prompt="user",
            response_model=_SimpleResponse,
        )

        self.assertEqual(result.value, "ok")
        self.assertEqual(client._client.calls, 2)

    def test_complete_json_raises_after_three_parse_failures(self) -> None:
        settings = OpenAISettings(api_key="test-key", max_retries=0, retry_backoff_seconds=0.0)
        client = OpenAICompatibleClient(settings)

        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        malformed = httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "not json at all"}}]},
        )
        client._client = _FakeClient([malformed, malformed, malformed])

        with self.assertRaises(LLMResponseError):
            client.complete_json(
                model="demo-model",
                system_prompt="system",
                user_prompt="user",
                response_model=_SimpleResponse,
            )

        self.assertEqual(client._client.calls, 3)


if __name__ == "__main__":
    unittest.main()
