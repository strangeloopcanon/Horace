from __future__ import annotations

import unittest

from tools.studio.calibrator import featurize_from_report_row


class TestCalibrator(unittest.TestCase):
    def test_tokens_frac_feature(self) -> None:
        feats = featurize_from_report_row(
            feature_names=["d:tokens_frac"],
            categories={},
            rubric_metrics={},
            doc_metrics={"tokens_count": 511},
            max_input_tokens=512,
            missing_value=0.5,
        )
        self.assertEqual(len(feats), 1)
        self.assertAlmostEqual(float(feats[0]), 1.0, places=6)

    def test_tokens_frac_missing_uses_fill(self) -> None:
        feats = featurize_from_report_row(
            feature_names=["d:tokens_frac"],
            categories={},
            rubric_metrics={},
            doc_metrics={},
            max_input_tokens=512,
            missing_value=0.42,
        )
        self.assertEqual(len(feats), 1)
        self.assertAlmostEqual(float(feats[0]), 0.42, places=6)

