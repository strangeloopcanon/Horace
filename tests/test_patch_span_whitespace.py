from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.meaning_lock import MeaningLockReport
from tools.studio.span_patcher import patch_span


class TestPatchSpanWhitespace(unittest.TestCase):
    def test_patch_span_preserves_boundary_whitespace(self) -> None:
        text = "A.\n\n  Hello world.  \n\nB."
        start = text.index("  Hello")
        end = start + len("  Hello world.  \n\n")

        fake_analysis = {"doc_metrics": {"tokens_count": 20}}
        ok_ml = MeaningLockReport(
            ok=True,
            cosine_sim=1.0,
            length_ratio=1.0,
            edit_ratio=0.1,
            numbers_added=[],
            numbers_removed=[],
            negations_added=[],
            negations_removed=[],
            proper_nouns_added=[],
            proper_nouns_removed=[],
            reasons=[],
        )

        with patch("tools.studio.span_patcher.analyze_text", return_value=fake_analysis):
            with patch("tools.studio.span_patcher.generate_span_rewrites", return_value=["X"]):
                with patch("tools.studio.span_patcher.check_meaning_lock", return_value=ok_ml):
                    with patch(
                        "tools.studio.span_patcher._primary_score_for_text",
                        return_value={"overall_0_100": 50.0, "source": "rubric"},
                    ):
                        out = patch_span(
                            text,
                            start_char=start,
                            end_char=end,
                            doc_type="prose",
                            normalize_text=False,
                            n_candidates=1,
                        )

        self.assertIn("candidates", out)
        self.assertTrue(out["candidates"])
        patched = out["candidates"][0]["patched_text"]
        self.assertEqual(patched, "A.\n\n  X  \n\nB.")


if __name__ == "__main__":
    unittest.main()

