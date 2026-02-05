from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.meaning_lock import MeaningLockReport
from tools.studio.span_patcher import patch_span


def _ok_meaning_lock() -> MeaningLockReport:
    return MeaningLockReport(
        ok=True,
        cosine_sim=0.99,
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


class TestSpanPatcherPrimaryScoringStatus(unittest.TestCase):
    def test_primary_before_embedded_error_is_reported(self) -> None:
        text = "Alpha beta gamma delta."
        fake_analysis = {"doc_metrics": {"tokens_count": 20}}
        with patch("tools.studio.span_patcher.analyze_text", return_value=fake_analysis):
            with patch("tools.studio.span_patcher.generate_span_rewrites", return_value=["Alpha beta gamma delta revised."]):
                with patch("tools.studio.span_patcher.check_meaning_lock", return_value=_ok_meaning_lock()):
                    with patch(
                        "tools.studio.span_patcher._primary_score_for_text",
                        side_effect=[
                            {"error": "trained_scorer_failed: boom"},
                            {"overall_0_100": 55.0, "source": "rubric"},
                        ],
                    ):
                        out = patch_span(
                            text,
                            start_char=0,
                            end_char=len(text),
                            normalize_text=False,
                            n_candidates=1,
                            score_top_n=1,
                        )

        self.assertEqual(out.get("primary_before_status"), "error")
        self.assertIn("trained_scorer_failed", str(out.get("primary_before_error") or ""))
        self.assertEqual((out.get("candidates") or [])[0].get("primary_after_status"), "scored")

    def test_candidates_show_not_scored_when_top_n_limit_applies(self) -> None:
        text = "Alpha beta gamma delta."
        fake_analysis = {"doc_metrics": {"tokens_count": 20}}
        rewrites = ["Alpha beta gamma delta revised one.", "Alpha beta gamma delta revised two."]
        with patch("tools.studio.span_patcher.analyze_text", return_value=fake_analysis):
            with patch("tools.studio.span_patcher.generate_span_rewrites", return_value=rewrites):
                with patch("tools.studio.span_patcher.check_meaning_lock", return_value=_ok_meaning_lock()):
                    with patch(
                        "tools.studio.span_patcher._primary_score_for_text",
                        side_effect=[
                            {"overall_0_100": 50.0, "source": "rubric"},
                            {"overall_0_100": 52.0, "source": "rubric"},
                        ],
                    ):
                        out = patch_span(
                            text,
                            start_char=0,
                            end_char=len(text),
                            normalize_text=False,
                            n_candidates=2,
                            score_top_n=1,
                        )

        statuses = [c.get("primary_after_status") for c in (out.get("candidates") or [])]
        self.assertEqual(out.get("primary_before_status"), "scored")
        self.assertEqual(statuses.count("scored"), 1)
        self.assertEqual(statuses.count("not_scored"), 1)
        self.assertTrue(any(c.get("primary_after_reason") == "score_top_n_limit" for c in (out.get("candidates") or [])))


class TestSpanPatcherDeletionPenalty(unittest.TestCase):
    def test_ranking_penalizes_aggressive_shortening(self) -> None:
        text = "This text should stay mostly intact and meaningful."
        short = "tiny"
        longer = "This text should stay mostly intact and meaningful overall."
        fake_analysis = {"doc_metrics": {"tokens_count": 20}}
        with patch("tools.studio.span_patcher.analyze_text", return_value=fake_analysis):
            with patch("tools.studio.span_patcher.generate_span_rewrites", return_value=[short, longer]):
                with patch("tools.studio.span_patcher.check_meaning_lock", return_value=_ok_meaning_lock()):
                    with patch("tools.studio.span_patcher._droning_score", side_effect=[0.0, 0.0, 0.0]):
                        with patch(
                            "tools.studio.span_patcher._primary_score_for_text",
                            return_value={"overall_0_100": 50.0, "source": "rubric"},
                        ):
                            out = patch_span(
                                text,
                                start_char=0,
                                end_char=len(text),
                                normalize_text=False,
                                n_candidates=2,
                                score_top_n=0,
                            )

        cands = out.get("candidates") or []
        self.assertGreaterEqual(len(cands), 2)
        self.assertEqual(cands[0].get("replacement"), longer)


if __name__ == "__main__":
    unittest.main()
