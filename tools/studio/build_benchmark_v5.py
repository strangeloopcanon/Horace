from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from tools.studio.build_benchmark_v4 import build_benchmark_v4
from tools.studio.dataset_utils import iter_jsonl, stable_sha1_hex, write_jsonl
from tools.studio.split_eval_set import split_eval_set


def _load_antipattern_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    for r in iter_jsonl(path):
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        sample_id = str(r.get("sample_id") or "").strip() or stable_sha1_hex([text])[:12]
        group_id = str(r.get("group_id") or "").strip() or f"antipattern:{stable_sha1_hex([sample_id])[:12]}"
        source = str(r.get("source") or "llm_antipattern").strip() or "llm_antipattern"
        rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "source": source,
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "text": text,
                "fetched_at_unix": int(r.get("fetched_at_unix") or int(time.time())),
                "meta": dict(r.get("meta") or {}) if isinstance(r.get("meta"), dict) else {},
            }
        )
    return rows


def build_benchmark_v5(
    *,
    out_dir: Path,
    seed: int,
    max_chars: int,
    top_books: int,
    top_excerpts_per_book: int,
    random_books: int,
    random_excerpts_per_book: int,
    tail_min_index: int,
    tail_max_index: int,
    corruption_kinds: Sequence[str],
    antipattern_negatives: Optional[Path],
    train_frac: float,
    val_frac: float,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    samples_path = out_dir / "samples.jsonl"
    splits_dir = out_dir / "splits"

    base_samples = build_benchmark_v4(
        seed=int(seed),
        max_chars=int(max_chars),
        top_books=int(top_books),
        top_excerpts_per_book=int(top_excerpts_per_book),
        random_books=int(random_books),
        random_excerpts_per_book=int(random_excerpts_per_book),
        tail_min_index=int(tail_min_index),
        tail_max_index=int(tail_max_index),
        corruption_kinds=tuple(corruption_kinds or []),
    )
    rows: List[dict] = [asdict(s) for s in base_samples]
    antipattern_rows: List[dict] = []
    if antipattern_negatives is not None and Path(antipattern_negatives).exists():
        antipattern_rows = _load_antipattern_rows(Path(antipattern_negatives))
        rows.extend(antipattern_rows)

    # Deduplicate by sample_id
    seen: Set[str] = set()
    deduped: List[dict] = []
    for r in rows:
        sid = str(r.get("sample_id") or "").strip()
        if not sid:
            sid = stable_sha1_hex([str(r.get("group_id") or ""), str(r.get("source") or ""), str(r.get("text") or "")])[:12]
            r["sample_id"] = sid
        if sid in seen:
            continue
        seen.add(sid)
        if not str(r.get("text") or "").strip():
            continue
        deduped.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples_path, deduped)
    split_manifest = split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(seed),
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        stratify_by_source=False,
    )

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "n_base_samples": int(len(base_samples)),
        "n_antipattern_samples": int(len(antipattern_rows)),
        "n_total_samples": int(len(deduped)),
        "sources": {},
        "split_manifest_path": str(splits_dir / "manifest.json"),
    }
    for r in deduped:
        src = str(r.get("source") or "unknown")
        stats["sources"][src] = int(stats["sources"].get(src, 0)) + 1
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps({"stats": stats, "split_manifest": split_manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "samples_path": str(samples_path),
        "splits_dir": str(splits_dir),
        "stats_path": str(stats_path),
        "stats": stats,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build benchmark v5: v4 within-domain benchmark + optional LLM anti-pattern negatives, then leakage-safe splits."
        )
    )
    ap.add_argument("--out-dir", default="data/benchmarks/studio_benchmark_v5")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-chars", type=int, default=3800)
    ap.add_argument("--top-books", type=int, default=220)
    ap.add_argument("--top-excerpts-per-book", type=int, default=2)
    ap.add_argument("--random-books", type=int, default=450)
    ap.add_argument("--random-excerpts-per-book", type=int, default=1)
    ap.add_argument("--tail-min-index", type=int, default=2500)
    ap.add_argument("--tail-max-index", type=int, default=5000)
    ap.add_argument("--corrupt", action="append", default=["shuffle_sentences_global", "shuffle_paragraphs", "repeat_sentences", "flatten"])
    ap.add_argument("--antipattern-negatives", default="", help="Optional JSONL with source=llm_antipattern_* rows")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args(argv)

    res = build_benchmark_v5(
        out_dir=Path(str(args.out_dir)),
        seed=int(args.seed),
        max_chars=int(args.max_chars),
        top_books=int(args.top_books),
        top_excerpts_per_book=int(args.top_excerpts_per_book),
        random_books=int(args.random_books),
        random_excerpts_per_book=int(args.random_excerpts_per_book),
        tail_min_index=int(args.tail_min_index),
        tail_max_index=int(args.tail_max_index),
        corruption_kinds=tuple(str(x) for x in (args.corrupt or []) if str(x).strip()),
        antipattern_negatives=Path(str(args.antipattern_negatives)) if str(args.antipattern_negatives).strip() else None,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
