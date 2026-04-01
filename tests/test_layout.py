from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_reading.errors import ConfigurationError
from paper_reading.pipeline.layout import _load_doclayout_model, _looks_like_huggingface_repo


class _FakeYOLOv10:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @classmethod
    def from_pretrained(cls, repo_id: str):
        instance = cls(repo_id)
        instance.repo_id = repo_id
        instance.is_pretrained = True
        return instance


class LayoutTests(unittest.TestCase):
    def test_looks_like_huggingface_repo(self) -> None:
        self.assertTrue(_looks_like_huggingface_repo("juliozhao/DocLayout-YOLO-DocStructBench"))
        self.assertFalse(_looks_like_huggingface_repo("/tmp/model.pt"))
        self.assertFalse(_looks_like_huggingface_repo("doclayout_yolo_docstructbench_imgsz1024.pt"))

    def test_load_doclayout_model_with_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"fake")
            model = _load_doclayout_model(_FakeYOLOv10, str(model_path))
            self.assertEqual(model.model_path, str(model_path))
            self.assertFalse(hasattr(model, "is_pretrained"))

    def test_load_doclayout_model_with_hf_repo(self) -> None:
        model = _load_doclayout_model(_FakeYOLOv10, "juliozhao/DocLayout-YOLO-DocStructBench")
        self.assertEqual(model.repo_id, "juliozhao/DocLayout-YOLO-DocStructBench")
        self.assertTrue(model.is_pretrained)

    def test_load_doclayout_model_rejects_invalid_source(self) -> None:
        with self.assertRaises(ConfigurationError):
            _load_doclayout_model(_FakeYOLOv10, "not-a-path-and-not-a-repo")


if __name__ == "__main__":
    unittest.main()
