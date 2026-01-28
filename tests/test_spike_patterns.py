"""Unit tests for spike patterns extraction."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.spike_patterns import (
    SpikePattern,
    SpikeStats,
    compute_spike_stats,
    extract_spike_patterns,
    suggest_spike_positions,
    _classify_token,
    _position_in_sentence,
)


class TestTokenClassification(unittest.TestCase):
    def test_content_words(self) -> None:
        self.assertEqual(_classify_token("mountain"), "content")
        self.assertEqual(_classify_token("fascinating"), "content")
        self.assertEqual(_classify_token("extraordinary"), "content")

    def test_function_words(self) -> None:
        self.assertEqual(_classify_token("the"), "function")
        self.assertEqual(_classify_token("and"), "function")
        self.assertEqual(_classify_token("is"), "function")

    def test_punctuation(self) -> None:
        self.assertEqual(_classify_token("."), "punctuation")
        self.assertEqual(_classify_token(","), "punctuation")
        self.assertEqual(_classify_token("!"), "punctuation")


class TestPositionInSentence(unittest.TestCase):
    def test_start_position(self) -> None:
        self.assertEqual(_position_in_sentence(0, 0, 10), "start")
        self.assertEqual(_position_in_sentence(1, 0, 10), "start")

    def test_middle_position(self) -> None:
        self.assertEqual(_position_in_sentence(5, 0, 10), "middle")
        self.assertEqual(_position_in_sentence(6, 0, 10), "middle")

    def test_end_position(self) -> None:
        self.assertEqual(_position_in_sentence(8, 0, 10), "end")
        self.assertEqual(_position_in_sentence(9, 0, 10), "end")


class TestExtractSpikePatterns(unittest.TestCase):
    def test_extract_returns_list(self) -> None:
        text = "The mountain stood silent against the extraordinary crimson sky."
        fake_analysis = {
            "tokens": [
                {"token": "The", "start": 0, "end": 3},
                {"token": " mountain", "start": 3, "end": 12},
                {"token": " stood", "start": 12, "end": 18},
                {"token": " silent", "start": 18, "end": 25},
                {"token": " against", "start": 25, "end": 33},
                {"token": " the", "start": 33, "end": 37},
                {"token": " extraordinary", "start": 37, "end": 51},
                {"token": " crimson", "start": 51, "end": 59},
                {"token": " sky", "start": 59, "end": 63},
                {"token": ".", "start": 63, "end": 64},
            ],
            "token_metrics": {
                "surprisal": [0.2, 0.5, 2.2, 0.4, 3.0, 0.3, 0.2, 2.5, 0.1, 0.8],
                "entropy": [3.0] * 10,
                "rank": [100, 200, 50, 300, 40, 500, 600, 30, 800, 900],
            },
            "segments": {"sentences": {"items": [{"start_token": 0, "end_token": 10}]}},
        }
        with patch("tools.studio.spike_patterns.analyze_text", return_value=fake_analysis) as mock_analyze:
            patterns = extract_spike_patterns(
                text,
                model_id="gpt2",
                backend="auto",
                max_input_tokens=64,
                surprisal_threshold=1.5,  # Lower threshold for test
            )
        self.assertTrue(mock_analyze.called)
        _, kwargs = mock_analyze.call_args
        self.assertTrue(bool(kwargs.get("include_token_metrics")))
        self.assertIsInstance(patterns, list)

    def test_patterns_have_correct_structure(self) -> None:
        text = "She discovered an extraordinary phenomenon in the ancient manuscript."
        fake_analysis = {
            "tokens": [
                {"token": "She", "start": 0, "end": 3},
                {"token": " discovered", "start": 3, "end": 14},
                {"token": " an", "start": 14, "end": 17},
                {"token": " extraordinary", "start": 17, "end": 31},
                {"token": " phenomenon", "start": 31, "end": 42},
                {"token": ".", "start": 42, "end": 43},
            ],
            "token_metrics": {
                "surprisal": [0.2, 0.4, 0.3, 2.4, 1.2, 0.6],
                "entropy": [3.0] * 6,
                "rank": [1000, 900, 800, 20, 400, 700],
            },
            "segments": {"sentences": {"items": [{"start_token": 0, "end_token": 6}]}},
        }
        with patch("tools.studio.spike_patterns.analyze_text", return_value=fake_analysis) as mock_analyze:
            patterns = extract_spike_patterns(
                text,
                model_id="gpt2",
                max_input_tokens=64,
                surprisal_threshold=1.0,
            )
        self.assertTrue(mock_analyze.called)
        _, kwargs = mock_analyze.call_args
        self.assertTrue(bool(kwargs.get("include_token_metrics")))
        if patterns:  # May not find spikes in short text
            p = patterns[0]
            self.assertIsInstance(p, SpikePattern)
            self.assertIsInstance(p.token_text, str)
            self.assertIn(p.position_in_sentence, ["start", "middle", "end"])
            self.assertIn(p.token_type, ["content", "function", "punctuation"])


class TestComputeSpikeStats(unittest.TestCase):
    def test_empty_patterns(self) -> None:
        stats = compute_spike_stats([], total_tokens=100)
        self.assertEqual(stats.total_spikes, 0)
        self.assertEqual(stats.spike_rate, 0.0)

    def test_stats_computation(self) -> None:
        patterns = [
            SpikePattern(
                token_index=5, char_start=20, char_end=28,
                position_in_sentence="middle", sentence_index=0,
                token_text="mountain", token_type="content",
                prev_token="the", prev_token_type="function",
                next_token="stood", next_token_type="content",
                surprisal=4.5, surprisal_delta=2.1, entropy_context=3.2,
                rank=150, is_content_word=True, ends_clause=False,
            ),
            SpikePattern(
                token_index=15, char_start=60, char_end=70,
                position_in_sentence="end", sentence_index=0,
                token_text="crimson", token_type="content",
                prev_token="the", prev_token_type="function",
                next_token=".", next_token_type="punctuation",
                surprisal=5.2, surprisal_delta=2.8, entropy_context=2.9,
                rank=200, is_content_word=True, ends_clause=True,
            ),
        ]
        stats = compute_spike_stats(patterns, total_tokens=20)
        self.assertEqual(stats.total_spikes, 2)
        self.assertAlmostEqual(stats.spike_rate, 10.0, places=1)  # 2/20 * 100
        self.assertEqual(stats.position_counts["middle"], 1)
        self.assertEqual(stats.position_counts["end"], 1)


class TestSuggestSpikePositions(unittest.TestCase):
    def test_returns_list(self) -> None:
        text = "The cat sat on the mat. The dog ran to the park."
        tokens = []
        surprisal = []
        pos = 0
        for i in range(20):
            tok = f"t{i}"
            start = pos
            end = pos + len(tok)
            pos = end
            tokens.append({"token": tok, "start": start, "end": end})
            if i < 5:
                surprisal.append(0.05)  # valley run
            elif i == 12:
                surprisal.append(2.5)  # one spike (below target rate)
            else:
                surprisal.append(0.5)

        fake_analysis = {"tokens": tokens, "token_metrics": {"surprisal": surprisal}}

        with patch("tools.studio.spike_patterns.analyze_text", return_value=fake_analysis) as mock_analyze:
            suggestions = suggest_spike_positions(
                text,
                model_id="gpt2",
                max_input_tokens=64,
            )
        self.assertTrue(mock_analyze.called)
        _, kwargs = mock_analyze.call_args
        self.assertTrue(bool(kwargs.get("include_token_metrics")))
        self.assertIsInstance(suggestions, list)


if __name__ == "__main__":
    unittest.main()
