from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.rewrite import rewrite_and_rerank
from tools.studio.score import ScoreReport


class TestRewriteRerankNormalization(unittest.TestCase):
    def test_rerank_scores_already_normalized_text_without_second_normalization(self) -> None:
        fake_score = ScoreReport(overall_0_100=50.0, categories={}, metrics={})
        with patch(
            "tools.studio.rewrite.normalize_for_studio",
            return_value=("normalized text", {"changed": True}),
        ):
            with patch("tools.studio.rewrite.analyze_text", return_value={"doc_metrics": {}}) as mock_analyze:
                with patch("tools.studio.rewrite._ensure_baseline", return_value=object()):
                    with patch("tools.studio.rewrite.score_text", return_value=fake_score):
                        with patch("tools.studio.rewrite.generate_rewrites", return_value=["candidate text"]):
                            rewrite_and_rerank(
                                "raw text",
                                normalize_text=True,
                                n_candidates=1,
                                keep_top=1,
                            )

        self.assertEqual(mock_analyze.call_count, 2)
        self.assertTrue(all(call.kwargs.get("normalize_text") is False for call in mock_analyze.call_args_list))

    def test_rerank_respects_normalize_text_false(self) -> None:
        fake_score = ScoreReport(overall_0_100=50.0, categories={}, metrics={})
        with patch(
            "tools.studio.rewrite.normalize_for_studio",
            side_effect=lambda text, doc_type, enabled: (text, {"changed": bool(enabled)}),
        ) as mock_norm:
            with patch("tools.studio.rewrite.analyze_text", return_value={"doc_metrics": {}}) as mock_analyze:
                with patch("tools.studio.rewrite._ensure_baseline", return_value=object()):
                    with patch("tools.studio.rewrite.score_text", return_value=fake_score):
                        with patch("tools.studio.rewrite.generate_rewrites", return_value=["candidate text"]):
                            rewrite_and_rerank(
                                "raw text",
                                normalize_text=False,
                                n_candidates=1,
                                keep_top=1,
                            )

        self.assertEqual(mock_norm.call_args.kwargs.get("enabled"), False)
        self.assertTrue(all(call.kwargs.get("normalize_text") is False for call in mock_analyze.call_args_list))


if __name__ == "__main__":
    unittest.main()
