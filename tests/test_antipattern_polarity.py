from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# studio_api is API-only and optional-fastapi; provide minimal stubs so helper tests
# run in environments that do not install API extras.
fastapi_stub = types.ModuleType("fastapi")
class _FastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def middleware(self, *args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper

    def get(self, *args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper

    def post(self, *args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper

fastapi_stub.FastAPI = _FastAPI
sys.modules["fastapi"] = fastapi_stub

pydantic_stub = types.ModuleType("pydantic")
pydantic_stub.BaseModel = object
pydantic_stub.Field = lambda *args, **kwargs: kwargs.get("default", None)
sys.modules["pydantic"] = pydantic_stub

studio_api = importlib.import_module("tools.studio_api")  # type: ignore[import-not-found]


class TestAntipatternPolarity(unittest.TestCase):
    def _report(self, directory: Path, *, positives: list[str]) -> None:
        train_report = {
            "run_meta": {
                "positive_sources": positives,
            }
        }
        (directory / "train_report.json").write_text(json.dumps(train_report), encoding="utf-8")

    def test_invert_based_on_train_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "model"
            d.mkdir()
            self._report(d, positives=["human_original"])

            score, inverted, msg = studio_api._resolve_antipattern_prob(str(d), 0.17)
            self.assertTrue(inverted)
            self.assertAlmostEqual(score, 0.83)
            self.assertIsNotNone(msg)

    def test_keep_probability_when_llm_antipattern_positive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "model"
            d.mkdir()
            self._report(d, positives=["llm_antipattern_write_like"])

            score, inverted, msg = studio_api._resolve_antipattern_prob(str(d), 0.17)
            self.assertFalse(inverted)
            self.assertEqual(score, 0.17)
            self.assertIsNone(msg)

    def test_fallback_name_heuristic_for_legacy_models(self) -> None:
        score, inverted, _msg = studio_api._resolve_antipattern_prob(
            "models/scorer_v5_antipattern_mix_plusfull_v1", 0.17
        )
        self.assertTrue(inverted)
        self.assertEqual(score, 0.83)

        score, inverted, _msg = studio_api._resolve_antipattern_prob(
            "models/scorer_v5_authenticity_v1", 0.17
        )
        self.assertFalse(inverted)
        self.assertEqual(score, 0.17)

    def test_none_probability_defaults_to_zero(self) -> None:
        score, inverted, msg = studio_api._resolve_antipattern_prob(
            "models/scorer_v5_antipattern_mix_plusfull_v1", None
        )
        self.assertEqual(score, 0.0)
        self.assertFalse(inverted)
        self.assertIsNone(msg)

    def test_scorer_warning_for_antipattern_checkpoint(self) -> None:
        self.assertIsNotNone(
            studio_api._scorer_model_warning("models/scorer_v5_antipattern_mix_plusfull_v1")
        )
        self.assertIsNotNone(
            studio_api._scorer_model_warning("models/scorer_v5_antipattern_pilot_v1")
        )

    def test_no_scorer_warning_for_authenticity_or_empty(self) -> None:
        self.assertIsNone(studio_api._scorer_model_warning("models/scorer_v5_authenticity_v1"))
        self.assertIsNone(studio_api._scorer_model_warning(""))

    def test_adaptive_penalty_starts_before_hard_threshold(self) -> None:
        out = studio_api._compute_antipattern_adjustment(
            base_score_0_100=88.0,
            anti_prob_0_1=0.80,
            threshold_0_1=0.85,
            weight=0.85,
            mode="adaptive",
        )
        self.assertEqual(out["mode"], "adaptive")
        self.assertGreater(out["penalty_0_100"], 0.0)
        self.assertLess(out["adjusted_score_0_100"], 88.0)
        self.assertAlmostEqual(float(out["hard_threshold_0_1"]), 0.85)

    def test_adaptive_is_stronger_than_legacy_midband(self) -> None:
        adaptive = studio_api._compute_antipattern_adjustment(
            base_score_0_100=88.0,
            anti_prob_0_1=0.80,
            threshold_0_1=0.85,
            weight=0.85,
            mode="adaptive",
        )
        legacy = studio_api._compute_antipattern_adjustment(
            base_score_0_100=88.0,
            anti_prob_0_1=0.80,
            threshold_0_1=0.85,
            weight=0.85,
            mode="legacy",
        )
        self.assertGreater(adaptive["penalty_0_100"], legacy["penalty_0_100"])

    def test_adaptive_cap_clamps_high_ai_probability(self) -> None:
        out = studio_api._compute_antipattern_adjustment(
            base_score_0_100=92.0,
            anti_prob_0_1=0.96,
            threshold_0_1=0.85,
            weight=0.85,
            mode="adaptive",
        )
        self.assertIsNotNone(out["authenticity_cap_0_100"])
        self.assertLess(out["adjusted_score_0_100"], 60.0)


if __name__ == "__main__":
    unittest.main()
