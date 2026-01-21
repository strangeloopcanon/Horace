from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.studio.split_eval_set import split_eval_set


class TestSplitEvalSet(unittest.TestCase):
    def test_no_group_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            samples = base / "samples.jsonl"
            out_dir = base / "splits"

            rows = []
            # Two sources, three groups each, two samples per group.
            for src in ("a", "b"):
                for gi in range(3):
                    gid = f"{src}:g{gi}"
                    for si in range(2):
                        rows.append(
                            {
                                "sample_id": f"{src}_{gi}_{si}",
                                "group_id": gid,
                                "source": src,
                                "title": "t",
                                "url": f"https://example/{src}/{gi}",
                                "text": "hello",
                                "fetched_at_unix": 0,
                                "meta": {},
                            }
                        )
            with samples.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            split_eval_set(samples_path=samples, out_dir=out_dir, seed=1337, train_frac=0.7, val_frac=0.15)

            def load_gids(p: Path) -> set[str]:
                gids: set[str] = set()
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    gids.add(str(obj.get("group_id")))
                return gids

            train_g = load_gids(out_dir / "train.jsonl")
            val_g = load_gids(out_dir / "val.jsonl")
            test_g = load_gids(out_dir / "test.jsonl")

            self.assertTrue(train_g.isdisjoint(val_g))
            self.assertTrue(train_g.isdisjoint(test_g))
            self.assertTrue(val_g.isdisjoint(test_g))

