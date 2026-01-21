from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.studio.dataset_utils import iter_jsonl, stable_sha1_hex, write_jsonl
from tools.studio.split_eval_set import split_eval_set


def _iter_rows(paths: Sequence[Path]) -> Iterable[dict]:
    for p in paths:
        for row in iter_jsonl(Path(p)):
            if isinstance(row, dict):
                yield row


def _dedup_key(row: Dict[str, Any]) -> str:
    sid = str(row.get("sample_id") or "").strip()
    if sid:
        return f"sample_id:{sid}"
    text = str(row.get("text") or "").strip()
    if text:
        return "text:" + stable_sha1_hex([text])
    return "row:" + stable_sha1_hex([json.dumps(row, sort_keys=True, ensure_ascii=False)])


def build_mixed_windows_corpus(
    *,
    inputs: Sequence[Path],
    out_dir: Path,
    seed: int,
    train_frac: float,
    val_frac: float,
    stratify_by_source: bool,
    max_rows: int,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    samples_path = out_dir / "samples.jsonl"
    splits_dir = out_dir / "splits"

    seen: set[str] = set()
    kept: List[dict] = []
    counts_by_input: Dict[str, int] = {}
    for row in _iter_rows(inputs):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        key = _dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(dict(row))
        src = str(row.get("source") or "unknown").strip() or "unknown"
        counts_by_input[src] = counts_by_input.get(src, 0) + 1
        if int(max_rows) > 0 and len(kept) >= int(max_rows):
            break

    if not kept:
        raise RuntimeError("No samples found in inputs")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples_path, kept)
    manifest = split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(seed),
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        stratify_by_source=bool(stratify_by_source),
    )
    return {
        "out_dir": str(out_dir),
        "samples_path": str(samples_path),
        "splits_dir": str(splits_dir),
        "n_samples": int(len(kept)),
        "by_source": {k: int(v) for k, v in sorted(counts_by_input.items(), key=lambda kv: kv[0])},
        "split_manifest_path": str(Path(manifest.get("out_dir") or str(splits_dir)) / "manifest.json"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Merge multiple samples.jsonl files into a single leakage-safe corpus with splits.")
    ap.add_argument("--out-dir", default="data/corpora/mixed_windows_v1")
    ap.add_argument("--input", action="append", default=[], help="Path to a samples.jsonl file (repeatable)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-stratify-by-source", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0, help="Stop after this many merged rows (0 = all)")
    args = ap.parse_args(argv)

    inputs = [Path(str(p)) for p in (args.input or []) if str(p).strip()]
    if not inputs:
        raise SystemExit("Provide at least one --input samples.jsonl")

    res = build_mixed_windows_corpus(
        inputs=inputs,
        out_dir=Path(str(args.out_dir)),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        stratify_by_source=not bool(args.no_stratify_by_source),
        max_rows=int(args.max_rows),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

