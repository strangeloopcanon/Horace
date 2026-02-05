from __future__ import annotations

import unittest

from tools.studio.meaning_lock import MeaningLockConfig, check_meaning_lock, extract_negations, extract_numbers, extract_proper_nouns


class TestMeaningLock(unittest.TestCase):
    def test_extract_numbers(self) -> None:
        self.assertEqual(extract_numbers("No numbers here."), [])
        self.assertEqual(extract_numbers("It was 12% bigger than 3.5 last year."), ["12%", "3.5"])

    def test_extract_negations(self) -> None:
        self.assertEqual(extract_negations("We agree."), [])
        self.assertEqual(extract_negations("We do not agree."), ["not"])
        self.assertEqual(extract_negations("We don't agree."), ["don't"])

    def test_extract_proper_nouns_ignores_sentence_initial(self) -> None:
        # "Alice" is sentence-initial in the first sentence, but still appears later.
        text = "Alice went home. Later, Alice met Bob in NYC."
        p = extract_proper_nouns(text)
        self.assertIn("Alice", p)
        self.assertIn("Bob", p)
        self.assertIn("NYC", p)

    def test_meaning_lock_blocks_new_numbers_and_proper_nouns(self) -> None:
        cfg = MeaningLockConfig(embedder_model_id="", allow_new_numbers=False, allow_new_proper_nouns=False)
        rep = check_meaning_lock("We shipped 3 features for Acme.", "We shipped 4 features for Acme.", cfg=cfg)
        self.assertFalse(rep.ok)
        self.assertIn("numbers_changed", rep.reasons)

        rep2 = check_meaning_lock("We shipped 3 features for Acme.", "We shipped 3 features for Globex.", cfg=cfg)
        self.assertFalse(rep2.ok)
        self.assertIn("proper_nouns_changed", rep2.reasons)

    def test_meaning_lock_blocks_large_edits(self) -> None:
        cfg = MeaningLockConfig(embedder_model_id="", max_edit_ratio=0.25)
        rep = check_meaning_lock("A short sentence.", "Completely different content with many new words.", cfg=cfg)
        self.assertFalse(rep.ok)
        self.assertIn("too_much_changed", rep.reasons)

    def test_meaning_lock_blocks_aggressive_shortening(self) -> None:
        cfg = MeaningLockConfig(embedder_model_id="", min_length_ratio=0.85)
        rep = check_meaning_lock("This sentence has enough detail.", "Too short.", cfg=cfg)
        self.assertFalse(rep.ok)
        self.assertIn("too_short", rep.reasons)

    def test_meaning_lock_blocks_negation_flip(self) -> None:
        cfg = MeaningLockConfig(embedder_model_id="", allow_negation_change=False)
        rep = check_meaning_lock("I do not agree.", "I agree.", cfg=cfg)
        self.assertFalse(rep.ok)
        self.assertIn("negation_changed", rep.reasons)
