from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.studio.build_antipattern_pairs import build_antipattern_pairs
from tools.studio.dataset_utils import iter_jsonl


class TestBuildAntipatternPairs(unittest.TestCase):
    def _write_originals(self, path: Path) -> None:
        row = {
            "sample_id": "orig1",
            "group_id": "group1",
            "source": "standardebooks_excerpt",
            "title": "Example Original",
            "url": "https://example.invalid/original",
            "author": "Virginia Woolf",
            "author_norm": "virginia woolf",
            "doc_type": "prose",
            "text": ("The room held its breath while the rain tapped the glass in patient measures. " * 16).strip(),
            "meta": {"origin": "test"},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_dry_run_multi_provider_randomization_generates_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            originals = root / "originals.jsonl"
            out_pairs = root / "pairs.jsonl"
            out_negs = root / "negs.jsonl"
            self._write_originals(originals)

            res = build_antipattern_pairs(
                originals_path=originals,
                out_pairs_path=out_pairs,
                out_negatives_path=out_negs,
                out_openai_batch_requests=None,
                openai_batch_results=(),
                out_unresolved_jobs_path=None,
                seed=42,
                provider_specs=(
                    "openai:gpt-5",
                    "google:gemini-3-flash",
                    "anthropic:claude-haiku-4.5",
                ),
                tiers=("write_like", "continue_from", "rewrite_from_memory"),
                variants_per_tier=2,
                max_originals=0,
                temperature=0.9,
                max_output_tokens=300,
                min_generated_chars=20,
                max_generated_chars=2000,
                similarity_threshold=0.99,
                run_online=False,
                dry_run=True,
            )

            pairs = list(iter_jsonl(out_pairs))
            negs = list(iter_jsonl(out_negs))
            self.assertEqual(len(pairs), 6)
            self.assertEqual(len(negs), 6)
            self.assertEqual(res["stats"]["pairs_out"], 6)
            self.assertEqual(res["stats"]["unresolved_jobs"], 0)

            neg_sources = {str(r.get("source")) for r in negs}
            self.assertIn("llm_antipattern_write_like", neg_sources)
            self.assertIn("llm_antipattern_continue_from", neg_sources)
            self.assertIn("llm_antipattern_rewrite_from_memory", neg_sources)

    def test_openai_batch_request_and_result_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            originals = root / "originals.jsonl"
            out_pairs = root / "pairs.jsonl"
            out_negs = root / "negs.jsonl"
            req_path = root / "openai_reqs.jsonl"
            res_path = root / "openai_results.jsonl"
            self._write_originals(originals)

            prep = build_antipattern_pairs(
                originals_path=originals,
                out_pairs_path=out_pairs,
                out_negatives_path=out_negs,
                out_openai_batch_requests=req_path,
                openai_batch_results=(),
                out_unresolved_jobs_path=root / "unresolved.jsonl",
                seed=7,
                provider_specs=("openai:gpt-5",),
                tiers=("write_like", "continue_from"),
                variants_per_tier=1,
                max_originals=0,
                temperature=0.9,
                max_output_tokens=240,
                min_generated_chars=20,
                max_generated_chars=2000,
                similarity_threshold=0.99,
                run_online=False,
                dry_run=False,
            )
            self.assertEqual(prep["stats"]["pairs_out"], 0)
            self.assertEqual(prep["stats"]["openai_batch_requests_out"], 2)
            req_rows = list(iter_jsonl(req_path))
            self.assertEqual(len(req_rows), 2)
            for r in req_rows:
                body = dict(r.get("body") or {})
                self.assertEqual(body.get("reasoning_effort"), "minimal")
                self.assertNotIn("temperature", body)

            with res_path.open("w", encoding="utf-8") as f:
                for idx, r in enumerate(req_rows, start=1):
                    payload = {
                        "custom_id": r["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": (
                                                f"Synthetic anti-pattern output {idx}. "
                                                "It stays coherent and intentionally stylized."
                                            )
                                        }
                                    }
                                ]
                            },
                        },
                    }
                    f.write(json.dumps(payload) + "\n")

            merged = build_antipattern_pairs(
                originals_path=originals,
                out_pairs_path=out_pairs,
                out_negatives_path=out_negs,
                out_openai_batch_requests=None,
                openai_batch_results=(res_path,),
                out_unresolved_jobs_path=None,
                seed=7,
                provider_specs=("openai:gpt-5",),
                tiers=("write_like", "continue_from"),
                variants_per_tier=1,
                max_originals=0,
                temperature=0.9,
                max_output_tokens=240,
                min_generated_chars=20,
                max_generated_chars=2000,
                similarity_threshold=0.99,
                run_online=False,
                dry_run=False,
            )

            pairs = list(iter_jsonl(out_pairs))
            negs = list(iter_jsonl(out_negs))
            self.assertEqual(len(pairs), 2)
            self.assertEqual(len(negs), 2)
            self.assertEqual(merged["stats"]["pairs_out"], 2)
            self.assertEqual(merged["stats"]["unresolved_jobs"], 0)


if __name__ == "__main__":
    unittest.main()
