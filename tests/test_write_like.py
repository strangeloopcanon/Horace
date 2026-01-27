"""Unit tests for cadence match display helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.cadence_profile import CadenceProfile
from tools.studio.write_like import extract_cadence_for_display


class TestExtractCadenceForDisplay(unittest.TestCase):
    def test_requests_token_metrics(self) -> None:
        fake_profile = CadenceProfile()
        fake_analysis = {
            "doc_metrics": {"spike_rate": 7.0},
            "token_metrics": {"surprisal": [0.1, 0.2, 0.3]},
            "series": {"surprisal": [9.9]},  # should prefer token_metrics when present
            "spikes": [],
        }
        with patch("tools.studio.write_like.extract_cadence_profile", return_value=fake_profile) as mock_profile:
            with patch("tools.studio.write_like.analyze_text", return_value=fake_analysis) as mock_analyze:
                out = extract_cadence_for_display("hello world", model_id="gpt2", backend="auto", doc_type="prose")

        self.assertTrue(mock_profile.called)
        self.assertTrue(mock_analyze.called)
        _, kwargs = mock_analyze.call_args
        self.assertTrue(bool(kwargs.get("include_token_metrics")))
        self.assertEqual(out.get("token_surprisal"), [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()

