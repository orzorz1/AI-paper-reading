from __future__ import annotations

import unittest

from paper_reading.pipeline.parser import (
    build_abstract_fallback,
    extract_abstract_from_text,
    extract_title_from_page_dict,
    extract_title_from_pdf_metadata,
)


class ParserHelperTests(unittest.TestCase):
    def test_extract_title_from_first_page_dict(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [50, 40, 500, 80],
                    "lines": [{"spans": [{"text": "Vision", "size": 24}, {"text": "Transformer", "size": 24}]}],
                },
                {
                    "type": 0,
                    "bbox": [50, 120, 500, 160],
                    "lines": [{"spans": [{"text": "Abstract", "size": 12}]}],
                },
            ]
        }
        self.assertEqual(extract_title_from_page_dict(page_dict, 800), "Vision Transformer")

    def test_extract_title_ignores_arxiv_margin_metadata(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [100, 100, 500, 140],
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Unveiling Parts Beyond Objects: Towards Finer-Granularity Referring Expression Segmentation",
                                    "size": 14.3,
                                }
                            ]
                        }
                    ],
                },
                {
                    "type": 0,
                    "bbox": [12, 200, 38, 560],
                    "lines": [{"spans": [{"text": "arXiv:2312.08007v2 [cs.CV] 21 Mar 2024", "size": 20.0}]}],
                },
                {
                    "type": 0,
                    "bbox": [88, 160, 500, 190],
                    "lines": [{"spans": [{"text": "Wenxuan Wang 1 Tongtian Yue 2 Jing Liu 3", "size": 9.1}]}],
                },
            ]
        }
        self.assertEqual(
            extract_title_from_page_dict(page_dict, 800, 560),
            "Unveiling Parts Beyond Objects: Towards Finer-Granularity Referring Expression Segmentation",
        )

    def test_extract_abstract_from_text(self) -> None:
        text = "Abstract\nThis paper studies image understanding.\n\n1 Introduction\nBody starts here."
        self.assertEqual(extract_abstract_from_text(text), "This paper studies image understanding.")

    def test_extract_title_from_pdf_metadata(self) -> None:
        metadata = {"title": "Attention Is All You Need"}
        self.assertEqual(extract_title_from_pdf_metadata(metadata), "Attention Is All You Need")

    def test_extract_title_from_pdf_metadata_ignores_invalid_values(self) -> None:
        self.assertEqual(extract_title_from_pdf_metadata({"title": "Untitled"}), "")
        self.assertEqual(extract_title_from_pdf_metadata({"title": "arXiv:2312.08007"}), "")
        self.assertEqual(extract_title_from_pdf_metadata({"title": "https://example.com/paper"}), "")

    def test_build_abstract_fallback(self) -> None:
        paragraphs = ["", "First paragraph.", "Second paragraph.", "Third paragraph."]
        fallback = build_abstract_fallback(paragraphs, limit=30)
        self.assertIn("First paragraph.", fallback)
        self.assertIn("Second paragraph.", fallback)


if __name__ == "__main__":
    unittest.main()
