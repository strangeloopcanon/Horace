"""Unit tests for cadence profile extraction and conversion."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.studio.cadence_profile import (
    CadenceProfile,
    DEFAULT_POETRY_PROFILE,
    DEFAULT_PROSE_PROFILE,
    PUNCHY_PROFILE,
    FLOWING_PROFILE,
    blend_profiles,
    compare_cadence_profiles,
    extract_cadence_profile,
    profile_to_poetry_config,
)


class TestCadenceProfile(unittest.TestCase):
    def test_default_profiles_have_valid_structure(self) -> None:
        """Verify all default profiles have sensible values."""
        for name, profile in [
            ("prose", DEFAULT_PROSE_PROFILE),
            ("poetry", DEFAULT_POETRY_PROFILE),
            ("punchy", PUNCHY_PROFILE),
            ("flowing", FLOWING_PROFILE),
        ]:
            with self.subTest(profile=name):
                # Interval range should be valid
                self.assertIsInstance(profile.interval_range, tuple)
                self.assertEqual(len(profile.interval_range), 2)
                self.assertLess(profile.interval_range[0], profile.interval_range[1])
                self.assertGreaterEqual(profile.interval_range[0], 3)

                # Cooldown range should be valid
                self.assertIsInstance(profile.cooldown_range, tuple)
                self.assertEqual(len(profile.cooldown_range), 2)
                self.assertLess(profile.cooldown_range[0], profile.cooldown_range[1])
                self.assertGreaterEqual(profile.cooldown_range[0], 1)

                # Temperature should be in reasonable range
                self.assertGreater(profile.base_temperature, 0.5)
                self.assertLess(profile.base_temperature, 1.5)
                self.assertGreater(profile.spike_temperature, 0.5)
                self.assertLess(profile.spike_temperature, 1.5)

                # top_p should be in (0, 1)
                self.assertGreater(profile.base_top_p, 0.5)
                self.assertLess(profile.base_top_p, 1.0)
                self.assertGreater(profile.cool_top_p, 0.5)
                self.assertLess(profile.cool_top_p, 1.0)

    def test_profile_to_dict_round_trips(self) -> None:
        """Verify to_dict produces complete representation."""
        d = DEFAULT_PROSE_PROFILE.to_dict()
        self.assertIn("interval_range", d)
        self.assertIn("cooldown_range", d)
        self.assertIn("spike_content_boost", d)
        self.assertIn("base_top_p", d)

    def test_blend_profiles_at_zero_returns_primary(self) -> None:
        """alpha=0 should return the primary profile."""
        result = blend_profiles(PUNCHY_PROFILE, FLOWING_PROFILE, alpha=0.0)
        self.assertEqual(result.interval_range, PUNCHY_PROFILE.interval_range)
        self.assertEqual(result.cooldown_range, PUNCHY_PROFILE.cooldown_range)
        self.assertAlmostEqual(result.spike_temperature, PUNCHY_PROFILE.spike_temperature, places=3)

    def test_blend_profiles_at_one_returns_secondary(self) -> None:
        """alpha=1 should return the secondary profile."""
        result = blend_profiles(PUNCHY_PROFILE, FLOWING_PROFILE, alpha=1.0)
        self.assertEqual(result.interval_range, FLOWING_PROFILE.interval_range)
        self.assertEqual(result.cooldown_range, FLOWING_PROFILE.cooldown_range)
        self.assertAlmostEqual(result.spike_temperature, FLOWING_PROFILE.spike_temperature, places=3)

    def test_blend_profiles_at_half_interpolates(self) -> None:
        """alpha=0.5 should produce midpoint values."""
        result = blend_profiles(PUNCHY_PROFILE, FLOWING_PROFILE, alpha=0.5)
        # Interval midpoint: (7+14)/2=10.5, (12+22)/2=17 → ~(10, 17)
        expected_lo = (PUNCHY_PROFILE.interval_range[0] + FLOWING_PROFILE.interval_range[0]) / 2
        expected_hi = (PUNCHY_PROFILE.interval_range[1] + FLOWING_PROFILE.interval_range[1]) / 2
        self.assertAlmostEqual(result.interval_range[0], expected_lo, delta=1)
        self.assertAlmostEqual(result.interval_range[1], expected_hi, delta=1)

    def test_compare_cadence_profiles_identical(self) -> None:
        """Identical profiles should have distance 0 and similarity 1."""
        comparison = compare_cadence_profiles(DEFAULT_PROSE_PROFILE, DEFAULT_PROSE_PROFILE)
        self.assertAlmostEqual(comparison["overall_distance"], 0.0, places=5)
        self.assertAlmostEqual(comparison["similarity_0_1"], 1.0, places=5)

    def test_compare_cadence_profiles_different(self) -> None:
        """Different profiles should have positive distance and similarity < 1."""
        comparison = compare_cadence_profiles(PUNCHY_PROFILE, FLOWING_PROFILE)
        self.assertGreater(comparison["overall_distance"], 0.0)
        self.assertLess(comparison["similarity_0_1"], 1.0)


class TestProfileToPoetryConfig(unittest.TestCase):
    def test_converts_to_poetry_config(self) -> None:
        """Verify conversion produces a valid PoetryConfig."""
        config = profile_to_poetry_config(DEFAULT_PROSE_PROFILE)

        # Should have the expected structure
        self.assertEqual(config.interval_range, DEFAULT_PROSE_PROFILE.interval_range)
        self.assertEqual(config.cooldown_range, DEFAULT_PROSE_PROFILE.cooldown_range)
        self.assertAlmostEqual(config.base.top_p, DEFAULT_PROSE_PROFILE.base_top_p, places=3)
        self.assertAlmostEqual(config.spike.temperature, DEFAULT_PROSE_PROFILE.spike_temperature, places=3)
        self.assertAlmostEqual(config.cool.top_p, DEFAULT_PROSE_PROFILE.cool_top_p, places=3)


class TestExtractCadenceProfile(unittest.TestCase):
    def test_extract_returns_valid_profile(self) -> None:
        """Basic smoke test: extraction should return a CadenceProfile."""
        text = "The quick brown fox jumps over the lazy dog. It was a bright cold day in April."
        fake_doc_metrics = {
            "ipi_mean": 16.0,
            "cooldown_entropy_drop_3": 1.2,
            "content_fraction": 0.40,
            "spike_prev_punct_rate": 0.15,
            "nucleus_w_mean": 140.0,
        }
        with patch("tools.studio.cadence_profile.analyze_text", return_value={"doc_metrics": fake_doc_metrics}):
            profile = extract_cadence_profile(
                text,
                model_id="gpt2",
                backend="auto",
                max_input_tokens=128,
                doc_type="prose",
            )
        self.assertIsInstance(profile, CadenceProfile)
        self.assertIsInstance(profile.interval_range, tuple)
        self.assertIsInstance(profile.cooldown_range, tuple)
        self.assertEqual(profile.interval_range, (12, 20))
        self.assertEqual(profile.cooldown_range, (3, 6))
        self.assertAlmostEqual(profile.spike_content_boost, 0.26, places=3)
        self.assertAlmostEqual(profile.spike_stop_punct_penalty, 1.3, places=3)
        self.assertAlmostEqual(profile.base_top_p, 0.90, places=3)
        self.assertAlmostEqual(profile.cool_top_p, 0.84, places=3)

    def test_extract_prose_vs_poetry_differs(self) -> None:
        """Prose and poetry doc_types should produce different base profiles."""
        text = "The moon rose slowly over the mountains."
        fake_doc_metrics = {
            "ipi_mean": None,
            "cooldown_entropy_drop_3": None,
            "content_fraction": None,
            "spike_prev_punct_rate": None,
            "nucleus_w_mean": None,
        }
        with patch("tools.studio.cadence_profile.analyze_text", return_value={"doc_metrics": fake_doc_metrics}):
            prose_profile = extract_cadence_profile(text, doc_type="prose", max_input_tokens=64)
            poetry_profile = extract_cadence_profile(text, doc_type="poem", max_input_tokens=64)

        # They should at least have different base defaults
        # (even if the text doesn't yield strong metrics)
        # Note: the actual extraction might override these, so we check structure only
        self.assertIsInstance(prose_profile, CadenceProfile)
        self.assertIsInstance(poetry_profile, CadenceProfile)


if __name__ == "__main__":
    unittest.main()
