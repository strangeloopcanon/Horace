from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.studio.baselines import build_baseline_from_rows, load_baseline, percentile, safe_model_id


class TestBaselines(unittest.TestCase):
    def test_percentile_empirical(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(vals, 0.0), 0.0)
        self.assertEqual(percentile(vals, 1.0), 25.0)
        self.assertEqual(percentile(vals, 2.0), 50.0)
        self.assertEqual(percentile(vals, 4.0), 100.0)

    def test_build_baseline_from_rows_writes_slices(self) -> None:
        rows = [
            {"doc_type": "novel", "entropy_mean": 1.0, "nucleus_w_mean": 10.0},
            {"doc_type": "novel", "entropy_mean": 2.0, "nucleus_w_mean": 20.0},
            {"doc_type": "shortstory", "entropy_mean": 3.0, "nucleus_w_mean": 30.0},
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "baseline.json"
            build_baseline_from_rows("gpt2", rows, out_path=out)
            raw = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("doc_types", raw)
            self.assertIn("all", raw["doc_types"])
            self.assertIn("prose", raw["doc_types"])
            self.assertIn("novel", raw["doc_types"])
            # Ensure values are present and sorted
            ent_vals = raw["doc_types"]["all"]["metrics"]["entropy_mean"]["values"]
            self.assertEqual(ent_vals, sorted(ent_vals))

            b = load_baseline("gpt2", path=out)
            self.assertIn("all", b.doc_types)

    def test_safe_model_id(self) -> None:
        self.assertEqual(safe_model_id("Qwen/Qwen2.5-1.5B"), "Qwen_Qwen2.5-1.5B")

