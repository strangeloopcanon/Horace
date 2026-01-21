from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline_from_rows, safe_model_id
from tools.studio.build_benchmark_set import gutenberg_excerpts_from_top


@dataclass(frozen=True)
class BuildBaselineResult:
    baseline_path: Path
    rows_path: Optional[Path]
    n_rows: int
    model_id: str
    doc_type: str
    max_input_tokens: int


def build_gutenberg_prose_baseline(
    *,
    model_id: str,
    backend: str,
    doc_type: str,
    max_input_tokens: int,
    excerpt_chars: int,
    n: int,
    seed: int,
    gutenberg_books: Optional[int] = None,
    gutenberg_excerpts_per_book: Optional[int] = None,
    exclude_book_ids: Optional[Sequence[int]] = None,
    normalize_text: bool = True,
    compute_cohesion: bool = False,
    out_path: Path,
    rows_out: Optional[Path] = None,
) -> BuildBaselineResult:
    rng = random.Random(int(seed))
    target_n = max(0, int(n))
    books = int(gutenberg_books) if gutenberg_books is not None else min(target_n, 200)
    books = max(1, books)
    per_book = (
        int(gutenberg_excerpts_per_book)
        if gutenberg_excerpts_per_book is not None
        else max(1, int(math.ceil(target_n / books)))
    )
    samples = gutenberg_excerpts_from_top(
        n_books=int(books),
        excerpts_per_book=int(per_book),
        seed=int(seed),
        max_chars=int(excerpt_chars),
        exclude_book_ids=exclude_book_ids,
    )
    if target_n and len(samples) > target_n:
        rng.shuffle(samples)
        samples = samples[:target_n]
    source_urls = sorted({(str(s.title), str(s.url)) for s in samples})

    rows: List[Dict[str, Any]] = []
    for s in samples:
        res = analyze_text(
            s.text,
            model_id=str(model_id),
            doc_type=str(doc_type),
            backend=str(backend),
            max_input_tokens=int(max_input_tokens),
            normalize_text=bool(normalize_text),
            compute_cohesion=bool(compute_cohesion),
        )
        dm = res.get("doc_metrics") or {}
        if not isinstance(dm, dict) or not dm:
            continue
        # Keep metadata stringly-typed so baseline metric-key discovery doesn't
        # accidentally absorb it as a numeric distribution.
        row: Dict[str, Any] = {
            **dm,
            "source": str(s.source),
            "title": str(s.title),
            "url": str(s.url),
            "sample_id": str(s.sample_id),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No baseline rows collected (network/model failure?)")

    if rows_out is not None:
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        with rows_out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_baseline_from_rows(str(model_id), rows, out_path=out_path)
    try:
        obj = json.loads(out_path.read_text(encoding="utf-8"))
        obj["build_meta"] = {
            "source": "gutenberg_excerpts",
            "n_rows": int(len(rows)),
            "seed": int(seed),
            "excerpt_chars": int(excerpt_chars),
            "max_input_tokens": int(max_input_tokens),
            "normalize_text": bool(normalize_text),
            "compute_cohesion": bool(compute_cohesion),
            "gutenberg_books": int(books),
            "gutenberg_excerpts_per_book": int(per_book),
            "exclude_book_ids": sorted({int(x) for x in (exclude_book_ids or [])}),
            "books": [{"title": t, "url": u} for (t, u) in source_urls],
        }
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return BuildBaselineResult(
        baseline_path=out_path,
        rows_path=rows_out,
        n_rows=len(rows),
        model_id=str(model_id),
        doc_type=str(doc_type),
        max_input_tokens=int(max_input_tokens),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a prose baseline by sampling Project Gutenberg excerpts and analyzing them (network required)."
    )
    ap.add_argument("--model-id", default="gpt2")
    ap.add_argument("--baseline-tag", default=None, help="Optional name suffix for the baseline file")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "hf"])
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--excerpt-chars", type=int, default=3800)
    ap.add_argument("--n", type=int, default=200, help="Number of excerpts to sample")
    ap.add_argument("--gutenberg-books", type=int, default=200, help="Unique Gutenberg books to draw from")
    ap.add_argument("--gutenberg-excerpts-per-book", type=int, default=1, help="Excerpts per book (before truncating to --n)")
    ap.add_argument(
        "--exclude-book-ids",
        default="",
        help="Comma-separated book ids or a JSON file path of ids to exclude (to keep baselines benchmark-disjoint).",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--normalize-text", action="store_true", help="Enable prose dewrap normalization (recommended)")
    ap.add_argument("--no-normalize-text", action="store_true", help="Disable normalization (debug only)")
    ap.add_argument("--compute-cohesion", action="store_true", help="Slower; computes cohesion_delta when possible")
    ap.add_argument("--out-dir", default="data/baselines")
    ap.add_argument("--out-path", default=None, help="Explicit baseline JSON output path")
    ap.add_argument("--rows-out", default=None, help="Optional JSONL of per-excerpt doc metrics (for debugging)")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    tag = safe_model_id(str(args.model_id))
    baseline_tag = (str(args.baseline_tag).strip() if args.baseline_tag else "").strip()
    suffix = f"gutenberg_{int(args.max_input_tokens)}" + (f"_{baseline_tag}" if baseline_tag else "")

    out_dir = Path(str(args.out_dir))
    out_path = Path(str(args.out_path)) if args.out_path else (out_dir / f"{tag}_{suffix}_docs.json")
    rows_out = Path(str(args.rows_out)) if args.rows_out else None

    exclude: List[int] = []
    raw_excl = str(args.exclude_book_ids or "").strip()
    if raw_excl:
        p = Path(raw_excl)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    exclude = [int(x) for x in data]
            except Exception:
                exclude = []
        else:
            for part in raw_excl.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    exclude.append(int(part))
                except Exception:
                    continue

    res = build_gutenberg_prose_baseline(
        model_id=str(args.model_id),
        backend=str(args.backend),
        doc_type=str(args.doc_type),
        max_input_tokens=int(args.max_input_tokens),
        excerpt_chars=int(args.excerpt_chars),
        n=int(args.n),
        gutenberg_books=int(args.gutenberg_books),
        gutenberg_excerpts_per_book=int(args.gutenberg_excerpts_per_book),
        exclude_book_ids=exclude,
        seed=int(args.seed),
        normalize_text=bool(normalize_text),
        compute_cohesion=bool(args.compute_cohesion),
        out_path=out_path,
        rows_out=rows_out,
    )
    payload = {
        **asdict(res),
        "baseline_path": str(res.baseline_path),
        "rows_path": str(res.rows_path) if res.rows_path else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
