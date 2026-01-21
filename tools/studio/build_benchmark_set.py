from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tools.studio.build_eval_set import (
    gibberish_controls,
    nasa_breaking_news,
    rfc_excerpts,
)
from tools.studio.dataset_utils import (
    FixedSample,
    clean_text,
    group_id as make_group_id,
    html_to_text,
    http_get_json,
    http_get_text,
    make_sample_id,
    sample_window,
    strip_gutenberg_boilerplate,
    write_jsonl,
)
from tools.studio.split_eval_set import split_eval_set


def _extract_gutenberg_title(raw_text: str) -> str:
    if not raw_text:
        return ""
    head = raw_text[:12000]
    m = re.search(r"^Title:\s*(.+?)\s*$", head, flags=re.I | re.M)
    if m:
        return clean_text(m.group(1))
    m = re.search(r"Project Gutenberg eBook of\s+(.+?),\s+by", head, flags=re.I)
    if m:
        return clean_text(m.group(1))
    return ""


def _gutenberg_text_urls(book_id: int) -> List[str]:
    bid = int(book_id)
    return [
        f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
        f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}-0.txt",
        f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
        f"https://www.gutenberg.org/files/{bid}/{bid}.txt",
    ]

def gutenberg_most_downloaded_book_ids(*, max_ids: int = 800, start_index: int = 1, max_pages: int = 80) -> List[int]:
    """Collect Gutenberg book ids by scraping the downloads-sorted search pages.

    This scales beyond the "top 100" page, and is stable enough to build a snapshot.
    """
    ids: List[int] = []
    seen: set[int] = set()
    per_page = 25
    start_index = max(1, int(start_index))
    pages = 0
    while len(ids) < int(max_ids) and pages < int(max_pages):
        pages += 1
        url = f"https://www.gutenberg.org/ebooks/search/?sort_order=downloads&start_index={start_index}"
        try:
            page = http_get_text(url)
        except Exception:
            break
        found = re.findall(r'href="/ebooks/(\d+)"', page)
        if not found:
            break
        new_any = False
        for raw in found:
            try:
                bid = int(raw)
            except Exception:
                continue
            if bid in seen:
                continue
            seen.add(bid)
            ids.append(bid)
            new_any = True
            if len(ids) >= int(max_ids):
                break
        if not new_any:
            break
        start_index += per_page
    return ids


def _gutenberg_is_english(raw_text: str) -> bool:
    head = (raw_text or "")[:15000]
    m = re.search(r"^Language:\s*(.+?)\s*$", head, flags=re.I | re.M)
    if not m:
        return True
    lang = m.group(1).strip().lower()
    return lang.startswith("english")


def gutenberg_excerpts_from_top(
    *,
    n_books: int,
    excerpts_per_book: int,
    seed: int,
    max_chars: int,
    exclude_book_ids: Optional[Sequence[int]] = None,
) -> List[FixedSample]:
    rng = random.Random(int(seed))
    exclude = {int(x) for x in (exclude_book_ids or [])}
    max_ids = max(600, int(n_books) * 8)
    max_pages = max(40, int(max_ids // 25) + 2)
    book_ids = gutenberg_most_downloaded_book_ids(max_ids=max_ids, max_pages=max_pages)
    rng.shuffle(book_ids)

    out: List[FixedSample] = []
    target_books = max(0, int(n_books))
    books_ok = 0
    min_body_chars = max(int(max_chars) * max(1, int(excerpts_per_book)), 6000)
    for bid in book_ids:
        if books_ok >= target_books:
            break
        if int(bid) in exclude:
            continue
        raw = None
        used_url = ""
        for url in _gutenberg_text_urls(int(bid)):
            try:
                raw = http_get_text(url)
                used_url = url
                break
            except Exception:
                continue
        if not raw:
            continue
        if not _gutenberg_is_english(raw):
            continue
        title = _extract_gutenberg_title(raw) or f"Gutenberg {bid}"
        body = strip_gutenberg_boilerplate(raw)
        if not body or len(body) < min_body_chars:
            continue
        books_ok += 1
        for _ in range(max(1, int(excerpts_per_book))):
            excerpt = sample_window(body, rng=rng, max_chars=int(max_chars))
            if not excerpt:
                continue
            sid = make_sample_id("gutenberg_excerpt", title, used_url, excerpt)
            out.append(
                FixedSample(
                    sample_id=sid,
                    group_id=f"gutenberg:{bid}",
                    source="gutenberg_excerpt",
                    title=title,
                    url=used_url,
                    text=excerpt,
                    fetched_at_unix=int(time.time()),
                    meta={"license_hint": "public_domain_gutenberg", "book_id": int(bid)},
                )
            )
    return out


def wikipedia_featured_window_samples(
    n: int,
    *,
    max_chars: int,
    rng: random.Random,
) -> List[FixedSample]:
    target = max(0, int(n))
    if target <= 0:
        return []

    # Build a set of high-quality encyclopedia prose with enough length to sample windows.
    min_chars = max(1600, int(max_chars) // 2)
    title_target = int(target) * 3
    titles: List[str] = []

    cmcontinue = None
    while len(titles) < title_target:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Featured articles",
            "cmsort": "timestamp",
            "cmdir": "desc",
            "cmprop": "title",
            "cmlimit": 200,
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        try:
            data = http_get_json(url)
        except Exception:
            break
        members = ((data.get("query") or {}).get("categorymembers") or [])
        for m in members:
            if not isinstance(m, dict):
                continue
            ttl = str(m.get("title") or "").strip()
            if ttl:
                titles.append(ttl)
                if len(titles) >= target:
                    break
        cmcontinue = (data.get("continue") or {}).get("cmcontinue")
        if not cmcontinue:
            break

    out: List[FixedSample] = []
    for title in titles:
        if len(out) >= target:
            break
        ttl = str(title).strip()
        if not ttl:
            continue
        try:
            html_url = "https://en.wikipedia.org/api/rest_v1/page/html/" + urllib.parse.quote(ttl, safe="")
            raw_html = http_get_text(html_url)
            text = html_to_text(raw_html)
            if not text or len(text) < int(min_chars):
                continue
            excerpt = sample_window(text, rng=rng, max_chars=int(max_chars))
            if not excerpt:
                continue
            page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(ttl.replace(" ", "_"), safe="")  # noqa: S310
            sid = make_sample_id("wikipedia_summary", ttl, page_url, excerpt)
            out.append(
                FixedSample(
                    sample_id=sid,
                    group_id=make_group_id("wikipedia_summary", stable=page_url),
                    source="wikipedia_summary",
                    title=ttl,
                    url=page_url,
                    text=excerpt,
                    fetched_at_unix=int(time.time()),
                    meta={
                        "license_hint": "cc_by_sa_wikipedia",
                        "category": "Featured articles",
                        "content_source": "rest_page_html",
                        "html_url": html_url,
                    },
                )
            )
        except Exception:
            continue
    return out


def wikinews_latest_published_window_samples(
    n: int,
    *,
    max_chars: int,
    rng: random.Random,
) -> List[FixedSample]:
    target = max(0, int(n))
    if target <= 0:
        return []

    min_chars = max(1600, int(max_chars) // 2)
    title_target = int(target) * 3

    # Pull the latest published titles, then fetch full content via REST HTML.
    titles: List[Tuple[str, str]] = []
    cmcontinue = None
    while len(titles) < title_target:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Published",
            "cmsort": "timestamp",
            "cmdir": "desc",
            "cmprop": "title|timestamp",
            "cmlimit": 200,
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        url = "https://en.wikinews.org/w/api.php?" + urllib.parse.urlencode(params)
        try:
            data = http_get_json(url)
        except Exception:
            break
        members = ((data.get("query") or {}).get("categorymembers") or [])
        for m in members:
            if not isinstance(m, dict):
                continue
            ttl = str(m.get("title") or "").strip()
            ts = str(m.get("timestamp") or "").strip()
            if ttl:
                titles.append((ttl, ts))
                if len(titles) >= title_target:
                    break
        cmcontinue = (data.get("continue") or {}).get("cmcontinue")
        if not cmcontinue:
            break

    out: List[FixedSample] = []
    for ttl, ts in titles:
        if len(out) >= target:
            break
        title = str(ttl).strip()
        if not title:
            continue
        try:
            html_url = "https://en.wikinews.org/api/rest_v1/page/html/" + urllib.parse.quote(title, safe="")
            raw_html = http_get_text(html_url)
            text = html_to_text(raw_html)
            if not text or len(text) < int(min_chars):
                continue
            excerpt = sample_window(text, rng=rng, max_chars=int(max_chars))
            if not excerpt:
                continue
            page_url = "https://en.wikinews.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="")  # noqa: S310
            sid = make_sample_id("wikinews_published", title, page_url, excerpt)
            out.append(
                FixedSample(
                    sample_id=sid,
                    group_id=make_group_id("wikinews_published", stable=page_url),
                    source="wikinews_published",
                    title=title,
                    url=page_url,
                    text=excerpt,
                    fetched_at_unix=int(time.time()),
                    meta={
                        "license_hint": "cc_by_wikinews",
                        "published_timestamp": ts,
                        "content_source": "rest_page_html",
                        "html_url": html_url,
                    },
                )
            )
        except Exception:
            continue
    return out


def build_benchmark_set(
    *,
    seed: int,
    max_chars: int,
    gutenberg_books: int,
    gutenberg_excerpts_per_book: int,
    wikipedia_featured: int,
    wikinews_n: int,
    nasa_n: int,
    rfc_n: int,
    gibberish_n: int,
) -> List[FixedSample]:
    rng = random.Random(int(seed))
    samples: List[FixedSample] = []

    samples.extend(
        gutenberg_excerpts_from_top(
            n_books=int(gutenberg_books),
            excerpts_per_book=int(gutenberg_excerpts_per_book),
            seed=int(seed),
            max_chars=int(max_chars),
        )
    )
    samples.extend(wikipedia_featured_window_samples(int(wikipedia_featured), max_chars=int(max_chars), rng=rng))
    samples.extend(wikinews_latest_published_window_samples(int(wikinews_n), max_chars=int(max_chars), rng=rng))
    samples.extend(nasa_breaking_news(int(nasa_n), max_chars=int(max_chars)))
    samples.extend(rfc_excerpts(int(rfc_n), rng=rng, max_chars=int(max_chars)))
    samples.extend(gibberish_controls(int(gibberish_n), rng=rng, max_chars=int(max_chars)))

    # Deduplicate by sample_id
    uniq: List[FixedSample] = []
    seen: set[str] = set()
    for s in samples:
        if s.sample_id in seen:
            continue
        if not (s.text or "").strip():
            continue
        seen.add(s.sample_id)
        uniq.append(s)
    return uniq


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a larger Studio benchmark set + train/val/test splits (JSONL).")
    ap.add_argument("--out-dir", default="data/benchmarks/studio_benchmark_v3")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-chars", type=int, default=3800)

    ap.add_argument("--gutenberg-books", type=int, default=200)
    ap.add_argument("--gutenberg-excerpts-per-book", type=int, default=3)
    ap.add_argument("--wikipedia-featured", type=int, default=400)
    ap.add_argument("--wikinews", type=int, default=400)
    ap.add_argument("--nasa", type=int, default=400)
    ap.add_argument("--rfc", type=int, default=150)
    ap.add_argument("--gibberish", type=int, default=400)

    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out_dir))
    samples_path = out_dir / "samples.jsonl"
    splits_dir = out_dir / "splits"

    samples = build_benchmark_set(
        seed=int(args.seed),
        max_chars=int(args.max_chars),
        gutenberg_books=int(args.gutenberg_books),
        gutenberg_excerpts_per_book=int(args.gutenberg_excerpts_per_book),
        wikipedia_featured=int(args.wikipedia_featured),
        wikinews_n=int(args.wikinews),
        nasa_n=int(args.nasa),
        rfc_n=int(args.rfc),
        gibberish_n=int(args.gibberish),
    )

    write_jsonl(samples_path, (asdict(s) for s in samples))
    split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        stratify_by_source=True,
    )

    print(str(samples_path))
    print(f"n={len(samples)} splits={splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
