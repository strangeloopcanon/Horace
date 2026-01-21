from __future__ import annotations

import unittest

from tools.studio.text_normalize import normalize_for_studio


class TestTextNormalize(unittest.TestCase):
    def test_prose_dewrap_preserves_paragraphs(self) -> None:
        raw = "One line wraps here\nand continues.\n\nNew para.\n"
        norm, meta = normalize_for_studio(raw, doc_type="prose", enabled=True)
        self.assertIn("\n\n", norm)
        self.assertNotIn("here\nand", norm)
        self.assertIn("wraps here and continues.", norm)
        self.assertTrue(meta["enabled"])
        self.assertGreaterEqual(int(meta["replaced_single_newlines"]), 2)

    def test_prose_joins_hyphen_breaks(self) -> None:
        raw = "This is exam-\nple text."
        norm, meta = normalize_for_studio(raw, doc_type="prose", enabled=True)
        self.assertIn("example", norm)
        self.assertNotIn("exam-\nple", norm)
        self.assertGreaterEqual(int(meta["joined_hyphen_breaks"]), 1)

    def test_poem_does_not_dewrap(self) -> None:
        raw = "First line\nSecond line\n"
        norm, meta = normalize_for_studio(raw, doc_type="poem", enabled=True)
        self.assertIn("First line\nSecond line", norm)
        self.assertEqual(int(meta["replaced_single_newlines"]), 0)

    def test_disabled_still_normalizes_crlf(self) -> None:
        raw = "A\r\nB\r\n"
        norm, meta = normalize_for_studio(raw, doc_type="prose", enabled=False)
        self.assertEqual(norm, "A\nB\n")
        self.assertFalse(meta["enabled"])

    def test_strips_gutenberg_boilerplate_markers(self) -> None:
        raw = (
            "Header\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Body line one.\nBody line two.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Footer\n"
        )
        norm, meta = normalize_for_studio(raw, doc_type="prose", enabled=True)
        self.assertIn("Body line one.", norm)
        self.assertNotIn("Header", norm)
        self.assertTrue(bool(meta.get("stripped_gutenberg_boilerplate")))
