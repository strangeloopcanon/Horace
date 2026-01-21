from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tools.studio.dataset_utils import FixedSample, group_id as make_group_id, make_sample_id, sample_window, write_jsonl
from tools.studio.split_eval_set import split_eval_set
from tools.studio.standardebooks import fetch_ebook, list_ebook_paths, download_epub_bytes, extract_text_from_epub
from tools.studio.text_normalize import normalize_for_studio


def _middle_slice(text: str, *, lo_frac: float = 0.05, hi_frac: float = 0.95, min_chars: int = 25000) -> str:
    t = text or ""
    if len(t) < int(min_chars):
        return t
    lo = int(max(0, min(len(t), int(len(t) * float(lo_frac)))))
    hi = int(max(lo, min(len(t), int(len(t) * float(hi_frac)))))
    return t[lo:hi].strip() or t


def build_standardebooks_corpus(
    *,
    seed: int,
    max_chars: int,
    min_chars: int,
    max_books: int,
    excerpts_per_book: int,
    start_page: int,
    max_pages: int,
    sleep_s: float,
    normalize_text: bool,
    doc_type: str,
) -> Tuple[List[FixedSample], Dict[str, Any], List[Dict[str, Any]]]:
    rng = random.Random(int(seed))
    paths = list_ebook_paths(max_pages=int(max_pages), start_page=int(start_page))
    rng.shuffle(paths)

    want = max(0, int(max_books))
    if want <= 0:
        return [], {"error": "max_books<=0"}, []
    paths = paths[:want]

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "max_chars": int(max_chars),
        "min_chars": int(min_chars),
        "max_books": int(max_books),
        "excerpts_per_book": int(excerpts_per_book),
        "start_page": int(start_page),
        "max_pages": int(max_pages),
        "sleep_s": float(sleep_s),
        "normalize_text": bool(normalize_text),
        "doc_type": str(doc_type),
        "paths_listed": int(len(paths)),
        "books_attempted": 0,
        "books_fetched": 0,
        "books_downloaded": 0,
        "books_extracted": 0,
        "books_too_short": 0,
        "books_failed": 0,
        "excerpts_attempted": 0,
        "excerpts_kept": 0,
        "failures_by_stage": {},
        "failures_by_type": {},
    }
    failures: List[Dict[str, Any]] = []

    samples: List[FixedSample] = []
    seen: Set[str] = set()
    now = int(time.time())

    for p in paths:
        stats["books_attempted"] += 1
        eb = None
        try:
            eb = fetch_ebook(p)
            stats["books_fetched"] += 1
        except Exception as e:
            stats["books_failed"] += 1
            stats["failures_by_stage"]["fetch_ebook"] = int(stats["failures_by_stage"].get("fetch_ebook", 0)) + 1
            key = f"fetch_ebook:{type(e).__name__}"
            stats["failures_by_type"][key] = int(stats["failures_by_type"].get(key, 0)) + 1
            if len(failures) < 200:
                failures.append({"stage": "fetch_ebook", "path": str(p), "error": f"{type(e).__name__}: {e}"})
            continue

        try:
            epub = download_epub_bytes(eb)
            stats["books_downloaded"] += 1
        except Exception as e:
            stats["books_failed"] += 1
            stats["failures_by_stage"]["download_epub"] = int(stats["failures_by_stage"].get("download_epub", 0)) + 1
            key = f"download_epub:{type(e).__name__}"
            stats["failures_by_type"][key] = int(stats["failures_by_type"].get(key, 0)) + 1
            if len(failures) < 200:
                failures.append({"stage": "download_epub", "path": str(p), "url": eb.url, "error": f"{type(e).__name__}: {e}"})
            continue

        try:
            full = extract_text_from_epub(epub)
            stats["books_extracted"] += 1
        except Exception as e:
            stats["books_failed"] += 1
            stats["failures_by_stage"]["extract_epub"] = int(stats["failures_by_stage"].get("extract_epub", 0)) + 1
            key = f"extract_epub:{type(e).__name__}"
            stats["failures_by_type"][key] = int(stats["failures_by_type"].get(key, 0)) + 1
            if len(failures) < 200:
                failures.append({"stage": "extract_epub", "path": str(p), "url": eb.url, "error": f"{type(e).__name__}: {e}"})
            continue

        try:
            pause = float(sleep_s)
            if pause > 0:
                time.sleep(pause)
        except Exception:
            pass

        if normalize_text:
            full, norm_meta = normalize_for_studio(full, doc_type=str(doc_type), enabled=True)
        else:
            norm_meta = None

        # Skip tiny/failed extractions.
        if len(full) < 6000:
            stats["books_too_short"] += 1
            continue

        gid = f"gutenberg:{int(eb.gutenberg_id)}" if eb.gutenberg_id else make_group_id("standardebooks", stable=eb.url)

        sample_source = _middle_slice(full)
        per_book = max(1, int(excerpts_per_book))
        for _ in range(per_book):
            stats["excerpts_attempted"] += 1
            excerpt = sample_window(sample_source, rng=rng, max_chars=int(max_chars), min_chars=int(min_chars))
            if not excerpt:
                continue
            sid = make_sample_id("standardebooks_excerpt", eb.title, eb.url, excerpt)
            if sid in seen:
                continue
            seen.add(sid)
            samples.append(
                FixedSample(
                    sample_id=sid,
                    group_id=gid,
                    source="standardebooks_excerpt",
                    title=eb.title,
                    url=eb.url,
                    text=excerpt,
                    fetched_at_unix=now,
                    meta={
                        "license_hint": "public_domain_standardebooks",
                        "author": eb.author,
                        "standardebooks_path": eb.path,
                        "download_epub_url": eb.download_epub_url,
                        "gutenberg_id": eb.gutenberg_id,
                        "normalization": norm_meta,
                    },
                )
            )
            stats["excerpts_kept"] += 1

    return samples, stats, failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a Standard Ebooks corpus snapshot (EPUB→text windows) with leakage-safe splits."
    )
    ap.add_argument("--out-dir", default="data/corpora/standardebooks_corpus_v1")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-chars", type=int, default=3800)
    ap.add_argument("--min-chars", type=int, default=900)
    ap.add_argument("--max-books", type=int, default=240)
    ap.add_argument("--excerpts-per-book", type=int, default=2)
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--sleep-s", type=float, default=0.0, help="Optional politeness delay between ebook fetches")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--doc-type", default="prose")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out_dir))
    samples_path = out_dir / "samples.jsonl"
    splits_dir = out_dir / "splits"

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)

    samples, stats, failures = build_standardebooks_corpus(
        seed=int(args.seed),
        max_chars=int(args.max_chars),
        min_chars=int(args.min_chars),
        max_books=int(args.max_books),
        excerpts_per_book=int(args.excerpts_per_book),
        start_page=int(args.start_page),
        max_pages=int(args.max_pages),
        sleep_s=float(args.sleep_s),
        normalize_text=bool(normalize_text),
        doc_type=str(args.doc_type),
    )

    if not samples:
        raise RuntimeError(
            "No Standard Ebooks samples produced. Try increasing --max-pages/--max-books, "
            "or set HORACE_HTTP_NO_CACHE=1 if cached HTML responses are interfering."
        )

    write_jsonl(samples_path, (asdict(s) for s in samples))
    split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        stratify_by_source=False,
    )

    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        write_jsonl(out_dir / "failures.jsonl", failures)

    print(str(samples_path))
    print(f"n={len(samples)} splits={splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
