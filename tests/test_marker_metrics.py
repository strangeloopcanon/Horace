from __future__ import annotations

import unittest

from tools.studio.marker_metrics import marker_sentence_features


class TestMarkerMetrics(unittest.TestCase):
    def test_marker_sentence_features_tags_and_counts(self) -> None:
        text = "We met Alice yesterday. However, NASA said it cost 3 dollars."
        # Manually provide sentence spans (start/end char offsets in `text`).
        s0 = 0
        s1 = text.index(".") + 1
        s2 = s1 + 1  # skip space
        spans = [(s0, s1), (s2, len(text))]

        feats = marker_sentence_features(text, sent_spans=spans, max_sentences=10, max_sentence_chars=1000)
        self.assertEqual(len(feats), 2)

        first = feats[0]
        second = feats[1]

        self.assertEqual(first["numbers"], 0)
        self.assertEqual(first["proper_nouns"], 1)  # Alice (not sentence-initial)
        self.assertTrue(first["markers"]["temporal"])  # yesterday

        self.assertEqual(second["numbers"], 1)  # 3
        self.assertEqual(second["proper_nouns"], 1)  # NASA (not sentence-initial)
        self.assertTrue(second["markers"]["contrastive"])  # However
        self.assertTrue(second["markers"]["evidential"])  # said

