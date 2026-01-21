from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from tools.studio.build_benchmark_set import (
    _extract_gutenberg_title,
    _gutenberg_is_english,
    _gutenberg_text_urls,
    gutenberg_most_downloaded_book_ids,
)
from tools.studio.dataset_utils import (
    FixedSample,
    make_sample_id,
    sample_window,
    strip_gutenberg_boilerplate,
    write_jsonl,
)
from tools.studio.split_eval_set import split_eval_set
from tools.studio.text_corrupt import corrupt_text


def _fetch_gutenberg_book_text(book_id: int) -> Tuple[str, str, str]:
    bid = int(book_id)
    raw = None
    used_url = ""
    for url in _gutenberg_text_urls(bid):
        try:
            from tools.studio.dataset_utils import http_get_text

            raw = http_get_text(url)
            used_url = url
            break
        except Exception:
            continue
    if not raw:
        raise RuntimeError(f"Failed to fetch Gutenberg text for id={bid}")
    if not _gutenberg_is_english(raw):
        raise RuntimeError(f"Gutenberg id={bid} is not English (skipped)")
    title = _extract_gutenberg_title(raw) or f"Gutenberg {bid}"
    body = strip_gutenberg_boilerplate(raw)
    if not body or len(body) < 6000:
        raise RuntimeError(f"Gutenberg id={bid} body too short after stripping boilerplate")
    return title, used_url, body


def _sample_book_excerpts(
    *,
    book_id: int,
    title: str,
    url: str,
    body: str,
    source: str,
    rng: random.Random,
    excerpts_per_book: int,
    max_chars: int,
) -> List[FixedSample]:
    out: List[FixedSample] = []
    gid = f"gutenberg:{int(book_id)}"
    for _ in range(max(1, int(excerpts_per_book))):
        excerpt = sample_window(body, rng=rng, max_chars=int(max_chars))
        if not excerpt:
            continue
        sid = make_sample_id(source, title, url, excerpt)
        out.append(
            FixedSample(
                sample_id=sid,
                group_id=gid,
                source=str(source),
                title=str(title),
                url=str(url),
                text=excerpt,
                fetched_at_unix=int(time.time()),
                meta={"license_hint": "public_domain_gutenberg", "book_id": int(book_id)},
            )
        )
    return out


def _collect_top_book_ids(*, n_books: int, seed: int, max_ids: int = 2500) -> List[int]:
    ids = gutenberg_most_downloaded_book_ids(max_ids=int(max_ids), start_index=1, max_pages=max(40, int(max_ids // 25) + 2))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    return ids[: max(0, int(n_books))]


def _collect_tail_book_ids(
    *,
    n_books: int,
    seed: int,
    exclude: Set[int],
    tail_min_index: int = 2500,
    tail_max_index: int = 5000,
) -> List[int]:
    rng = random.Random(int(seed))
    want = max(0, int(n_books))
    if want <= 0:
        return []

    # Gutenberg's downloads search pagination is finite; we sample a "long-tail-ish" pool by
    # scraping a bounded range of rank offsets, then choosing ids uniformly from that pool.
    #
    # Default strategy: take ranks ~2500–5000 (still English-heavy, but much less "top hits"),
    # and avoid hitting pagination limits by never requesting pages beyond tail_max_index.
    lo = max(1, int(tail_min_index))
    hi = max(lo, int(tail_max_index))
    per_page = 25
    pages = int(math.ceil((hi - lo + 1) / per_page))
    pool = max(per_page, pages * per_page)
    ids = gutenberg_most_downloaded_book_ids(
        max_ids=int(pool),
        start_index=int(lo),
        max_pages=int(pages),
    )
    ids = [int(x) for x in ids if int(x) not in exclude]
    rng.shuffle(ids)
    return ids[:want]


def build_benchmark_v4(
    *,
    seed: int,
    max_chars: int,
    top_books: int,
    top_excerpts_per_book: int,
    random_books: int,
    random_excerpts_per_book: int,
    tail_min_index: int,
    tail_max_index: int,
    corruption_kinds: Sequence[str],
) -> List[FixedSample]:
    rng = random.Random(int(seed))

    top_ids = _collect_top_book_ids(n_books=int(top_books), seed=int(seed), max_ids=max(2500, int(top_books) * 10))
    top_set = set(int(x) for x in top_ids)

    random_ids = _collect_tail_book_ids(
        n_books=int(random_books),
        seed=int(seed) ^ 0xBADC0DE,
        exclude=top_set,
        tail_min_index=int(tail_min_index),
        tail_max_index=int(tail_max_index),
    )

    samples: List[FixedSample] = []

    # Positive proxy: top-download Gutenberg books (broadly canon-ish).
    for bid in top_ids:
        try:
            title, url, body = _fetch_gutenberg_book_text(int(bid))
        except Exception:
            continue
        samples.extend(
            _sample_book_excerpts(
                book_id=int(bid),
                title=title,
                url=url,
                body=body,
                source="gutenberg_top_excerpt",
                rng=rng,
                excerpts_per_book=int(top_excerpts_per_book),
                max_chars=int(max_chars),
            )
        )

    # Within-domain negatives: random Gutenberg ids (long-tail) + controlled corruptions of top excerpts.
    for bid in random_ids:
        try:
            title, url, body = _fetch_gutenberg_book_text(int(bid))
        except Exception:
            continue
        samples.extend(
            _sample_book_excerpts(
                book_id=int(bid),
                title=title,
                url=url,
                body=body,
                source="gutenberg_random_excerpt",
                rng=rng,
                excerpts_per_book=int(random_excerpts_per_book),
                max_chars=int(max_chars),
            )
        )

    # Corruptions: keep group_id at the book level so there is no leakage across splits.
    if corruption_kinds:
        kinds = [str(k).strip() for k in corruption_kinds if str(k).strip()]
        for s in list(samples):
            if s.source != "gutenberg_top_excerpt":
                continue
            for k in kinds:
                try:
                    corrupted = corrupt_text(s.text, rng=rng, kind=k)
                except Exception:
                    continue
                if not corrupted or corrupted == s.text:
                    continue
                sid = make_sample_id(f"gutenberg_corrupt_{k}", s.title, s.url, corrupted)
                samples.append(
                    FixedSample(
                        sample_id=sid,
                        group_id=str(s.group_id),
                        source=f"gutenberg_corrupt_{k}",
                        title=str(s.title),
                        url=str(s.url),
                        text=corrupted,
                        fetched_at_unix=int(time.time()),
                        meta={**(s.meta or {}), "corruption_kind": str(k)},
                    )
                )

    # Deduplicate by sample_id
    uniq: List[FixedSample] = []
    seen: Set[str] = set()
    for s in samples:
        if s.sample_id in seen:
            continue
        if not (s.text or "").strip():
            continue
        seen.add(s.sample_id)
        uniq.append(s)
    return uniq


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a within-domain Gutenberg benchmark (top vs long-tail + corruptions) with leakage-safe splits."
    )
    ap.add_argument("--out-dir", default="data/benchmarks/studio_benchmark_v4")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-chars", type=int, default=3800)

    ap.add_argument("--top-books", type=int, default=220)
    ap.add_argument("--top-excerpts-per-book", type=int, default=2)
    ap.add_argument("--random-books", type=int, default=450)
    ap.add_argument("--random-excerpts-per-book", type=int, default=1)
    ap.add_argument(
        "--tail-min-index",
        type=int,
        default=2500,
        help="Gutenberg downloads rank offset floor for long-tail books",
    )
    ap.add_argument(
        "--tail-max-index",
        type=int,
        default=5000,
        help="Gutenberg downloads rank offset ceiling for long-tail books",
    )
    ap.add_argument(
        "--tail-start-index",
        type=int,
        default=None,
        help="Deprecated alias for --tail-min-index",
    )

    ap.add_argument(
        "--corrupt",
        action="append",
        default=None,
        help="Corruption kind(s) applied to top excerpts (repeatable).",
    )

    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out_dir))
    samples_path = out_dir / "samples.jsonl"
    splits_dir = out_dir / "splits"

    samples = build_benchmark_v4(
        seed=int(args.seed),
        max_chars=int(args.max_chars),
        top_books=int(args.top_books),
        top_excerpts_per_book=int(args.top_excerpts_per_book),
        random_books=int(args.random_books),
        random_excerpts_per_book=int(args.random_excerpts_per_book),
        tail_min_index=int(args.tail_start_index) if args.tail_start_index is not None else int(args.tail_min_index),
        tail_max_index=int(args.tail_max_index),
        corruption_kinds=tuple(
            args.corrupt
            if args.corrupt is not None
            else ["shuffle_sentences_global", "shuffle_paragraphs", "repeat_sentences", "flatten"]
        ),
    )

    write_jsonl(samples_path, (asdict(s) for s in samples))

    # Important: some groups contain multiple sources (top excerpt + corruption), so we disable per-source stratification.
    split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        stratify_by_source=False,
    )

    print(str(samples_path))
    print(f"n={len(samples)} splits={splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
