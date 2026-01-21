from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.studio.baselines import build_baseline_from_rows, load_baseline
from tools.studio.critique import suggest_edits
from tools.studio.score import score_text


def _rows_for_rubric(doc_type: str) -> list[dict]:
    rows: list[dict] = []
    for i in range(20):
        x = float(i)
        rows.append(
            {
                "doc_type": doc_type,
                "entropy_mean": 2.0 + 0.05 * x,
                "nucleus_w_mean": 150.0 + 2.0 * x,
                "high_surprise_rate_per_100": 8.0 + 0.2 * x,
                "ipi_mean": 6.0 + 0.1 * x,
                "cooldown_entropy_drop_3": 0.1 + 0.01 * x,
                "cohesion_delta": -0.9 + 0.03 * x,
                "spike_next_content_rate": 0.4 + 0.01 * x,
                "spike_prev_punct_rate": 0.25 - 0.005 * x,
                "content_fraction": 0.25 + 0.005 * x,
            }
        )
    return rows


class TestScoreAndCritique(unittest.TestCase):
    def test_score_text_produces_categories(self) -> None:
        rows = _rows_for_rubric("novel")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "baseline.json"
            build_baseline_from_rows("gpt2", rows, out_path=out)
            baseline = load_baseline("gpt2", path=out)

            doc_metrics = {
                "entropy_mean": 2.5,
                "nucleus_w_mean": 170.0,
                "high_surprise_rate_per_100": 11.0,
                "ipi_mean": 7.1,
                "cooldown_entropy_drop_3": 0.18,
                "cohesion_delta": -0.6,
                "spike_next_content_rate": 0.55,
                "spike_prev_punct_rate": 0.12,
                "content_fraction": 0.30,
            }
            score = score_text(doc_metrics, baseline, doc_type="prose")
            self.assertGreaterEqual(score.overall_0_100, 0.0)
            self.assertLessEqual(score.overall_0_100, 100.0)
            self.assertTrue(score.categories)
            self.assertTrue(score.metrics)

    def test_suggest_edits_has_summary(self) -> None:
        rows = _rows_for_rubric("novel")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "baseline.json"
            build_baseline_from_rows("gpt2", rows, out_path=out)
            baseline = load_baseline("gpt2", path=out)

            doc_metrics = {
                "entropy_mean": 2.9,
                "nucleus_w_mean": 190.0,
                "high_surprise_rate_per_100": 18.0,
                "ipi_mean": 4.0,
                "cooldown_entropy_drop_3": 0.05,
                "cohesion_delta": -0.1,
                "spike_next_content_rate": 0.30,
                "spike_prev_punct_rate": 0.30,
                "content_fraction": 0.18,
            }
            score = score_text(doc_metrics, baseline, doc_type="prose")
            critique = suggest_edits(
                doc_metrics=doc_metrics,
                score=score,
                spikes=[{"context": "…example…"}],
                segments={"sentences": {"burst_cv": 0.9}},
            )
            self.assertIn("summary", critique)
            self.assertIn("suggestions", critique)
            self.assertIsInstance(critique["suggestions"], list)

