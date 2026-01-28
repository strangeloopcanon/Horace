"""Unit tests for paragraph and windowed cadence modules."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.paragraph_cadence import (
    ParagraphCadence,
    DocumentParagraphCadence,
    extract_paragraph_cadence,
    _coefficient_of_variation,
    _momentum,
    _spike_front_loading,
)
from tools.studio.windowed_cadence import (
    WindowCadence,
    CadenceTimeline,
    _paragraph_starts,
    _select_windows,
)


class TestCoefficientOfVariation(unittest.TestCase):
    def test_empty_array(self) -> None:
        import numpy as np
        self.assertEqual(_coefficient_of_variation(np.array([])), 0.0)

    def test_single_element(self) -> None:
        import numpy as np
        self.assertEqual(_coefficient_of_variation(np.array([5.0])), 0.0)

    def test_constant_array(self) -> None:
        import numpy as np
        # All same values -> std=0 -> CV=0
        self.assertAlmostEqual(_coefficient_of_variation(np.array([3.0, 3.0, 3.0])), 0.0)

    def test_varied_array(self) -> None:
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cv = _coefficient_of_variation(arr)
        self.assertGreater(cv, 0.0)
        self.assertLess(cv, 1.0)  # For this range, CV should be moderate


class TestMomentum(unittest.TestCase):
    def test_rising_trend(self) -> None:
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = _momentum(arr)
        self.assertGreater(m, 0.5)  # Strong positive correlation

    def test_falling_trend(self) -> None:
        import numpy as np
        arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        m = _momentum(arr)
        self.assertLess(m, -0.5)  # Strong negative correlation

    def test_flat_trend(self) -> None:
        import numpy as np
        arr = np.array([3.0, 3.0, 3.0, 3.0])
        m = _momentum(arr)
        self.assertAlmostEqual(m, 0.0, places=1)


class TestSpikeFrontLoading(unittest.TestCase):
    def test_front_loaded(self) -> None:
        # Spikes at positions 1, 2, 3 in a 100-token paragraph
        loading = _spike_front_loading([1, 2, 3], 100)
        self.assertGreater(loading, 0.5)  # Front-loaded

    def test_back_loaded(self) -> None:
        # Spikes at positions 90, 95, 98 in a 100-token paragraph
        loading = _spike_front_loading([90, 95, 98], 100)
        self.assertLess(loading, -0.5)  # Back-loaded

    def test_centered(self) -> None:
        # Spikes at middle positions
        loading = _spike_front_loading([45, 50, 55], 100)
        self.assertAlmostEqual(loading, 0.0, places=1)


class TestExtractParagraphCadence(unittest.TestCase):
    def test_empty_analysis(self) -> None:
        result = extract_paragraph_cadence({})
        self.assertIsInstance(result, DocumentParagraphCadence)
        self.assertEqual(result.para_count, 0)

    def test_with_fake_analysis(self) -> None:
        fake_analysis = {
            "segments": {
                "paragraphs": {
                    "items": [
                        {"start_token": 0, "end_token": 20, "start_char": 0, "end_char": 100},
                        {"start_token": 20, "end_token": 40, "start_char": 100, "end_char": 200},
                    ],
                },
                "sentences": {
                    "items": [
                        {"start_token": 0, "end_token": 10, "start_char": 0, "end_char": 50},
                        {"start_token": 10, "end_token": 20, "start_char": 50, "end_char": 100},
                        {"start_token": 20, "end_token": 30, "start_char": 100, "end_char": 150},
                        {"start_token": 30, "end_token": 40, "start_char": 150, "end_char": 200},
                    ],
                    "mean_surprisal": [2.5, 3.0, 2.0, 2.8],
                    "token_counts": [10, 10, 10, 10],
                },
            },
            "token_metrics": {
                "surprisal": [2.0] * 40,
            },
        }
        result = extract_paragraph_cadence(fake_analysis)
        self.assertEqual(result.para_count, 2)
        self.assertEqual(len(result.paragraphs), 2)
        self.assertEqual(result.paragraphs[0].sentence_count, 2)
        self.assertEqual(result.paragraphs[1].sentence_count, 2)

    def test_pacing_variety_uses_paragraph_mean_surprisal(self) -> None:
        # Paragraph means are equal, but openings differ. pacing_variety should be 0.
        fake_analysis = {
            "segments": {
                "paragraphs": {
                    "items": [
                        {"start_token": 0, "end_token": 20, "start_char": 0, "end_char": 100},
                        {"start_token": 20, "end_token": 40, "start_char": 100, "end_char": 200},
                    ],
                },
                "sentences": {
                    "items": [
                        {"start_token": 0, "end_token": 10},
                        {"start_token": 10, "end_token": 20},
                        {"start_token": 20, "end_token": 30},
                        {"start_token": 30, "end_token": 40},
                    ],
                    "mean_surprisal": [1.0, 5.0, 3.0, 3.0],
                    "token_counts": [10, 10, 10, 10],
                },
            },
            "token_metrics": {"surprisal": [0.0] * 40},
            "series": {"threshold_surprisal": 1.0},
        }
        result = extract_paragraph_cadence(fake_analysis)
        self.assertAlmostEqual(result.paragraphs[0].mean_surprisal, 3.0, places=3)
        self.assertAlmostEqual(result.paragraphs[1].mean_surprisal, 3.0, places=3)
        self.assertAlmostEqual(result.pacing_variety, 0.0, places=6)


class TestParagraphStarts(unittest.TestCase):
    def test_no_paragraphs(self) -> None:
        starts = _paragraph_starts("Single line of text")
        self.assertEqual(starts, [0])

    def test_multiple_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        starts = _paragraph_starts(text)
        self.assertEqual(len(starts), 3)
        self.assertEqual(starts[0], 0)


class TestSelectWindows(unittest.TestCase):
    def test_short_text(self) -> None:
        text = "Short text."
        windows = _select_windows(text, window_chars=1000, max_windows=5)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][2], "Short text.")

    def test_long_text_multiple_windows(self) -> None:
        # Create text long enough for multiple windows
        text = ("This is a paragraph. " * 50 + "\n\n") * 10
        windows = _select_windows(text, window_chars=500, max_windows=5)
        self.assertGreater(len(windows), 1)
        self.assertLessEqual(len(windows), 5)


class TestWindowCadence(unittest.TestCase):
    def test_to_dict(self) -> None:
        wc = WindowCadence(
            window_index=0,
            start_char=0,
            end_char=100,
            spike_rate=7.5,
            cadence_score=65.0,
            is_worst=True,
            reasons=["Low spike rate"],
        )
        d = wc.to_dict()
        self.assertEqual(d["window_index"], 0)
        self.assertEqual(d["spike_rate"], 7.5)
        self.assertEqual(d["is_worst"], True)
        self.assertIn("Low spike rate", d["reasons"])


class TestCadenceTimeline(unittest.TestCase):
    def test_to_dict(self) -> None:
        timeline = CadenceTimeline(
            windows=[
                WindowCadence(window_index=0, start_char=0, end_char=100),
                WindowCadence(window_index=1, start_char=100, end_char=200),
            ],
            worst_window_index=0,
            best_window_index=1,
            overall_cadence_score=55.0,
            pacing_variety=0.15,
        )
        d = timeline.to_dict()
        self.assertEqual(len(d["windows"]), 2)
        self.assertEqual(d["worst_window_index"], 0)
        self.assertEqual(d["best_window_index"], 1)


if __name__ == "__main__":
    unittest.main()
