from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.studio.build_antipattern_originals import build_antipattern_originals
from tools.studio.dataset_utils import iter_jsonl


class TestBuildAntipatternOriginals(unittest.TestCase):
    def test_build_originals_from_jsonl_and_text_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            in_jsonl = root / "in.jsonl"
            text_dir = root / "poem" / "emily_dickinson"
            text_dir.mkdir(parents=True, exist_ok=True)
            poem_path = text_dir / "Hope_is_the_thing_with_feathers.txt"
            poem_path.write_text(
                "Hope is the thing with feathers that perches in the soul.\n" * 20,
                encoding="utf-8",
            )

            long_text = ("This is a long literary sentence with cadence and detail. " * 40).strip()
            rows = [
                {
                    "sample_id": "s1",
                    "group_id": "g1",
                    "source": "standardebooks_excerpt",
                    "title": "Example Work",
                    "url": "https://example.invalid/work",
                    "text": long_text,
                    "meta": {"author": "Virginia Woolf"},
                },
                {
                    # Duplicate text should be deduped.
                    "sample_id": "s2",
                    "group_id": "g2",
                    "source": "gutenberg_top_excerpt",
                    "title": "Dup Work",
                    "url": "https://example.invalid/dup",
                    "text": long_text,
                    "meta": {"author": "Virginia Woolf"},
                },
                {
                    # Too short; should be skipped by min_chars.
                    "sample_id": "s3",
                    "group_id": "g3",
                    "source": "standardebooks_excerpt",
                    "title": "Short Work",
                    "url": "https://example.invalid/short",
                    "text": "tiny",
                    "meta": {"author": "Jane Austen"},
                },
            ]
            with in_jsonl.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            out_path = root / "originals.jsonl"
            res = build_antipattern_originals(
                input_jsonl=(in_jsonl,),
                input_text_dirs=(root / "poem",),
                out_path=out_path,
                seed=1337,
                max_samples=0,
                min_chars=120,
                max_chars=700,
                windows_per_text_file=1,
                normalize_text=True,
                doc_type="auto",
                skip_missing=False,
            )

            out_rows = list(iter_jsonl(out_path))
            self.assertGreaterEqual(len(out_rows), 2)
            self.assertEqual(res["stats"]["skipped_duplicate_text"], 1)
            self.assertEqual(res["stats"]["skipped_short"], 1)

            jsonl_sources = {str(r.get("source")) for r in out_rows}
            self.assertIn("standardebooks_excerpt", jsonl_sources)
            self.assertIn("poem", jsonl_sources)

            first = out_rows[0]
            self.assertIn("author_norm", first)
            self.assertIn("doc_type", first)
            self.assertTrue(str(first.get("text") or "").strip())


if __name__ == "__main__":
    unittest.main()
