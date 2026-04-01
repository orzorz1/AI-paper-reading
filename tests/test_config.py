from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_reading.config import load_app_config


class ConfigTests(unittest.TestCase):
    def test_load_app_config_reads_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "OPENAI_BASE_URL=https://example.com/v1",
                        "TEXT_MODEL=test-text-model",
                        "VISION_MODEL=test-vision-model",
                        "OPENAI_MAX_RETRIES=5",
                        "OPENAI_RETRY_BACKOFF_SECONDS=2.0",
                        "LAYOUT_MODEL_PATH=/tmp/layout.pt",
                        "LAYOUT_DEVICE=cpu",
                        "LAYOUT_CONFIDENCE=0.2",
                        "LAYOUT_PREDICT_IMGSZ=1280",
                        "OUTPUT_ROOT=./demo-output",
                        "DEFAULT_LANG=zh-CN",
                    ]
                ),
                encoding="utf-8",
            )

            env_keys = [
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "TEXT_MODEL",
                "VISION_MODEL",
                "OPENAI_MAX_RETRIES",
                "OPENAI_RETRY_BACKOFF_SECONDS",
                "LAYOUT_MODEL_PATH",
                "LAYOUT_DEVICE",
                "LAYOUT_CONFIDENCE",
                "LAYOUT_PREDICT_IMGSZ",
                "OUTPUT_ROOT",
                "DEFAULT_LANG",
            ]
            cleaned_env = {key: value for key, value in os.environ.items() if key not in env_keys}
            with patch.dict(os.environ, cleaned_env, clear=True):
                with patch("paper_reading.config.Path.cwd", return_value=root):
                    config = load_app_config()

            self.assertEqual(config.openai.api_key, "test-key")
            self.assertEqual(config.openai.base_url, "https://example.com/v1")
            self.assertEqual(config.openai.text_model, "test-text-model")
            self.assertEqual(config.openai.vision_model, "test-vision-model")
            self.assertEqual(config.openai.max_retries, 5)
            self.assertEqual(config.openai.retry_backoff_seconds, 2.0)
            self.assertEqual(config.layout.model_path, "/tmp/layout.pt")
            self.assertEqual(config.layout.confidence, 0.2)
            self.assertEqual(config.layout.predict_imgsz, 1280)
            self.assertEqual(config.runtime.output_root, "./demo-output")


if __name__ == "__main__":
    unittest.main()
