"""Consolidate multiple anti-pattern pair JSONLs into a single deduplicated dataset with train/val/test splits."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.studio.dataset_utils import iter_jsonl, write_jsonl


def consolidate_pairs(
    *,
    input_paths: Sequence[Path],
    out_dir: Path,
    seed: int = 1337,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, Any]:
    """Read multiple pair JONLs, deduplicate by pair_id, split by group_id."""
    seen_ids: dict[str, str] = {}
    unique_rows: List[dict] = []

    for p in input_paths:
        for row in iter_jsonl(Path(p)):
            pid = str(row.get("pair_id") or "")
            if not pid or pid in seen_ids:
                continue
            chosen = str(row.get("chosen_text") or "")
            rejected = str(row.get("rejected_text") or "")
            if not chosen.strip() or not rejected.strip():
                continue
            seen_ids[pid] = str(p)
            unique_rows.append(row)

    # Group by group_id for leakage-safe splits.
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in unique_rows:
        gid = str(row.get("group_id") or "unknown")
        groups[gid].append(row)

    group_ids = sorted(groups.keys())
    rng = random.Random(int(seed))
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = max(1, int(n_groups * float(train_frac)))
    n_val = max(1, int(n_groups * float(val_frac)))

    train_gids = set(group_ids[:n_train])
    val_gids = set(group_ids[n_train : n_train + n_val])
    test_gids = set(group_ids[n_train + n_val :])

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []

    for gid in train_gids:
        train_rows.extend(groups[gid])
    for gid in val_gids:
        val_rows.extend(groups[gid])
    for gid in test_gids:
        test_rows.extend(groups[gid])

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    stats = {
        "input_files": [str(p) for p in input_paths],
        "total_lines_read": sum(1 for _ in []),  # placeholder
        "unique_pairs": len(unique_rows),
        "duplicates_dropped": 0,
        "groups": n_groups,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "test_pairs": len(test_rows),
        "train_groups": len(train_gids),
        "val_groups": len(val_gids),
        "test_groups": len(test_gids),
        "seed": int(seed),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Consolidate anti-pattern pairs into a deduplicated train/val/test split.")
    ap.add_argument("--input", action="append", required=True, help="Input pair JSONL (repeatable)")
    ap.add_argument("--out-dir", required=True, help="Output directory for splits")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args(argv)

    stats = consolidate_pairs(
        input_paths=[Path(p) for p in args.input],
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
