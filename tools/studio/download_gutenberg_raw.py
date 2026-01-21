from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from tools.studio.author_utils import load_author_list, normalize_author_name
from tools.studio.build_benchmark_set import _extract_gutenberg_title, _gutenberg_is_english, _gutenberg_text_urls, gutenberg_most_downloaded_book_ids
from tools.studio.dataset_utils import append_jsonl, clean_text, http_get_text, strip_gutenberg_boilerplate
from tools.studio.text_normalize import normalize_for_studio


def _extract_gutenberg_author(raw_text: str) -> str:
    if not raw_text:
        return ""
    head = raw_text[:18000]
    m = re.search(r"^Author:\s*(.+?)\s*$", head, flags=re.I | re.M)
    if m:
        return clean_text(m.group(1))
    m = re.search(r"Project Gutenberg eBook of\s+.+?,\s+by\s+(.+?)\s*$", head, flags=re.I | re.M)
    if m:
        return clean_text(m.group(1))
    return ""


def _book_text_path(out_dir: Path, book_id: int) -> Path:
    return out_dir / "books" / f"pg{int(book_id)}.txt"


def download_gutenberg_raw(
    *,
    out_dir: Path,
    seed: int,
    start_index: int,
    max_pages: int,
    max_books: int,
    max_bytes: int,
    sleep_s: float,
    normalize_text: bool,
    doc_type: str,
    great_authors_path: Optional[Path],
    keep: str,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    books_dir = out_dir / "books"
    meta_path = out_dir / "meta.jsonl"
    failures_path = out_dir / "failures.jsonl"
    stats_path = out_dir / "stats.json"

    great: set[str] = set()
    if great_authors_path is not None and great_authors_path.exists():
        great = load_author_list(great_authors_path.read_text(encoding="utf-8").splitlines())

    keep_norm = str(keep or "all").strip().lower()
    if keep_norm not in ("all", "great", "other"):
        raise ValueError("--keep must be one of: all, great, other")

    ids = gutenberg_most_downloaded_book_ids(max_ids=max(1, int(max_books) * 3), start_index=int(start_index), max_pages=int(max_pages))
    ids = ids[: max(0, int(max_books))]

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "start_index": int(start_index),
        "max_pages": int(max_pages),
        "max_books": int(max_books),
        "max_bytes": int(max_bytes),
        "sleep_s": float(sleep_s),
        "normalize_text": bool(normalize_text),
        "doc_type": str(doc_type),
        "great_authors_path": str(great_authors_path) if great_authors_path else "",
        "keep": keep_norm,
        "ids_listed": int(len(ids)),
        "books_attempted": 0,
        "books_saved": 0,
        "books_skipped_existing": 0,
        "books_skipped_not_english": 0,
        "books_skipped_too_short": 0,
        "books_skipped_filter": 0,
        "failures": 0,
        "bytes_text_saved": 0,
    }

    now = int(time.time())

    for i, bid in enumerate(ids, 1):
        if int(max_bytes) > 0 and int(stats["bytes_text_saved"]) >= int(max_bytes):
            break

        stats["books_attempted"] += 1
        book_id = int(bid)
        text_path = _book_text_path(out_dir, book_id)
        if text_path.exists():
            stats["books_skipped_existing"] += 1
            stats["bytes_text_saved"] += int(text_path.stat().st_size)
            continue

        doc_stage = "init"
        raw = None
        used_url = ""
        try:
            doc_stage = "download"
            for url in _gutenberg_text_urls(book_id):
                try:
                    raw = http_get_text(url)
                    used_url = url
                    break
                except Exception:
                    continue
            if not raw:
                stats["failures"] += 1
                continue
            doc_stage = "language"
            if not _gutenberg_is_english(raw):
                stats["books_skipped_not_english"] += 1
                continue

            doc_stage = "parse_header"
            title = _extract_gutenberg_title(raw) or f"Gutenberg {book_id}"
            author = _extract_gutenberg_author(raw)
            author_norm = normalize_author_name(author)
            is_great = True if not great else (author_norm in great)
            bucket = "great_author" if is_great else "other_author"

            if keep_norm == "great" and not is_great:
                stats["books_skipped_filter"] += 1
                continue
            if keep_norm == "other" and is_great:
                stats["books_skipped_filter"] += 1
                continue

            doc_stage = "strip_boilerplate"
            body = strip_gutenberg_boilerplate(raw)
            if normalize_text:
                doc_stage = "normalize"
                body, norm_meta = normalize_for_studio(body, doc_type=str(doc_type), enabled=True)
            else:
                norm_meta = None
            if not body.strip() or len(body) < 6000:
                stats["books_skipped_too_short"] += 1
                continue

            doc_stage = "write_text"
            books_dir.mkdir(parents=True, exist_ok=True)
            text_path.write_text(body, encoding="utf-8")
            sz = int(text_path.stat().st_size)
            stats["books_saved"] += 1
            stats["bytes_text_saved"] += sz

            doc_stage = "write_meta"
            append_jsonl(
                meta_path,
                [
                    {
                        "doc_id": f"pg{book_id}",
                        "group_id": f"gutenberg:{book_id}",
                        "source": "gutenberg_raw",
                        "bucket": bucket,
                        "title": title,
                        "author": author,
                        "author_norm": author_norm,
                        "url": used_url,
                        "book_id": book_id,
                        "text_path": str(text_path.relative_to(out_dir)),
                        "chars": int(len(body)),
                        "bytes": int(sz),
                        "fetched_at_unix": now,
                        "normalization": norm_meta,
                        "license_hint": "public_domain_gutenberg",
                    }
                ],
            )
        except Exception as e:
            stats["failures"] += 1
            try:
                append_jsonl(
                    failures_path,
                    [
                        {
                            "stage": doc_stage,
                            "book_id": int(book_id),
                            "url": str(used_url),
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    ],
                )
            except Exception:
                pass
        finally:
            pause = float(sleep_s)
            if pause > 0:
                time.sleep(pause)
        if i % 50 == 0:
            print(
                f"[gutenberg_raw] attempted={stats['books_attempted']} saved={stats['books_saved']} "
                f"skipped_existing={stats['books_skipped_existing']} skipped_not_english={stats['books_skipped_not_english']} "
                f"skipped_too_short={stats['books_skipped_too_short']} skipped_filter={stats['books_skipped_filter']} "
                f"failures={stats['failures']} bytes_saved={stats['bytes_text_saved']}",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "meta_path": str(meta_path),
        "failures_path": str(failures_path),
        "stats_path": str(stats_path),
        "stats": stats,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Download Gutenberg books (plain text), normalize, and store locally.")
    ap.add_argument("--out-dir", default="data/corpora/gutenberg_raw_v1")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--start-index", type=int, default=1, help="Downloads rank offset (1 = top)")
    ap.add_argument("--max-pages", type=int, default=200, help="How many Gutenberg search pages to scrape for ids")
    ap.add_argument("--max-books", type=int, default=3000)
    ap.add_argument("--max-bytes", type=int, default=0, help="Stop after this many bytes of saved texts (0 = unlimited)")
    ap.add_argument("--sleep-s", type=float, default=0.1)
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--doc-type", default="prose")

    ap.add_argument("--great-authors", default="", help="Optional author list file for bucketing")
    ap.add_argument("--keep", default="all", help="Filter bucket: all|great|other")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    great_path = Path(str(args.great_authors)) if str(args.great_authors).strip() else None

    res = download_gutenberg_raw(
        out_dir=Path(str(args.out_dir)),
        seed=int(args.seed),
        start_index=int(args.start_index),
        max_pages=int(args.max_pages),
        max_books=int(args.max_books),
        max_bytes=int(args.max_bytes),
        sleep_s=float(args.sleep_s),
        normalize_text=bool(normalize_text),
        doc_type=str(args.doc_type),
        great_authors_path=great_path,
        keep=str(args.keep),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
