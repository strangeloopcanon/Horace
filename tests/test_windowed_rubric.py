from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.score import MetricScore, ScoreReport
from tools.studio.windowed_rubric import windowed_rubric_for_text


class TestWindowedRubricAggregation(unittest.TestCase):
    def test_aggregate_metrics_are_weighted_across_windows(self) -> None:
        score_a = ScoreReport(
            overall_0_100=80.0,
            categories={"focus": 0.8},
            metrics={"entropy_mean": MetricScore(value=1.0, percentile=10.0, score_0_1=0.1, mode="match_baseline")},
        )
        score_b = ScoreReport(
            overall_0_100=20.0,
            categories={"focus": 0.2},
            metrics={"entropy_mean": MetricScore(value=9.0, percentile=90.0, score_0_1=0.9, mode="match_baseline")},
        )

        with patch(
            "tools.studio.windowed_rubric._select_windows",
            return_value=[(0, 10, "chunk-a"), (10, 20, "chunk-b")],
        ):
            with patch(
                "tools.studio.windowed_rubric.analyze_text",
                side_effect=[{"doc_metrics": {"tokens_count": 10}}, {"doc_metrics": {"tokens_count": 10}}],
            ):
                with patch("tools.studio.windowed_rubric.score_text", side_effect=[score_a, score_b]):
                    with patch("tools.studio.windowed_rubric._prose_weight", side_effect=[3.0, 1.0]):
                        wr = windowed_rubric_for_text(
                            "full text",
                            baseline=object(),
                            scoring_model_id="gpt2",
                            doc_type="prose",
                            backend="auto",
                            max_input_tokens=64,
                            normalize_text=False,
                            compute_cohesion=False,
                            max_windows=2,
                        )

        m = wr.aggregate.metrics["entropy_mean"]
        self.assertAlmostEqual(m.value, 3.0, places=6)  # (1*3 + 9*1) / 4
        self.assertAlmostEqual(m.percentile or 0.0, 30.0, places=6)
        self.assertAlmostEqual(m.score_0_1 or 0.0, 0.3, places=6)
        # Regression guard: this used to be copied from the worst window (value=9).
        self.assertNotAlmostEqual(m.value, 9.0, places=6)


if __name__ == "__main__":
    unittest.main()
