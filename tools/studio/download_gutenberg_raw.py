from __future__ import annotations

import argparse
import csv
import json
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


_DEFAULT_CATALOG_CSV_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"


def _norm_catalog_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(k or "").strip().lower())


def _catalog_lang_matches(lang_raw: str, *, want: str) -> bool:
    w = str(want or "").strip().lower()
    if not w:
        return True
    low = str(lang_raw or "").strip().lower()
    if not low:
        return False
    # Catalog language is usually an ISO-639-1 code ("en"), but sometimes appears as "English".
    if w == "en":
        return bool(re.search(r"\ben\b", low)) or ("english" in low)
    return bool(re.search(rf"\b{re.escape(w)}\b", low)) or (w in low)


def _catalog_authors(authors_raw: str) -> List[str]:
    raw = str(authors_raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"[;|]+", raw) if p.strip()]
    return parts if parts else [raw]


def _load_catalog_rows(
    *,
    catalog_url: str,
    language: str,
    type_filter: str,
    great: set[str],
    keep_norm: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    csv_text = http_get_text(str(catalog_url or _DEFAULT_CATALOG_CSV_URL), timeout_s=90.0)
    reader = csv.DictReader(StringIO(csv_text))

    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "catalog_url": str(catalog_url or _DEFAULT_CATALOG_CSV_URL),
        "catalog_language": str(language or ""),
        "catalog_type_filter": str(type_filter or ""),
        "catalog_rows_parsed": 0,
        "catalog_rows_skipped_parse": 0,
        "catalog_rows_skipped_type": 0,
        "catalog_rows_skipped_lang": 0,
        "catalog_rows_skipped_keep": 0,
        "catalog_rows_kept": 0,
    }

    want_type = str(type_filter or "").strip().lower()
    for raw_row in reader:
        stats["catalog_rows_parsed"] += 1
        # Normalize headers so we can be resilient to small changes.
        row = {_norm_catalog_key(k): (v.strip() if isinstance(v, str) else v) for k, v in (raw_row or {}).items()}
        try:
            book_id = int(str(row.get("text") or "").strip())
        except Exception:
            stats["catalog_rows_skipped_parse"] += 1
            continue
        if book_id <= 0:
            stats["catalog_rows_skipped_parse"] += 1
            continue

        typ = str(row.get("type") or "").strip().lower()
        if want_type and typ and typ != want_type:
            stats["catalog_rows_skipped_type"] += 1
            continue

        lang = str(row.get("language") or "").strip()
        if str(language or "").strip() and not _catalog_lang_matches(lang, want=str(language)):
            stats["catalog_rows_skipped_lang"] += 1
            continue

        authors_raw = str(row.get("authors") or row.get("author") or "").strip()
        authors = _catalog_authors(authors_raw)
        author_primary = authors[0] if authors else ""
        author_norm = normalize_author_name(author_primary)
        is_great = True if not great else any(normalize_author_name(a) in great for a in authors)

        if keep_norm == "great" and not is_great:
            stats["catalog_rows_skipped_keep"] += 1
            continue
        if keep_norm == "other" and is_great:
            stats["catalog_rows_skipped_keep"] += 1
            continue

        title = str(row.get("title") or "").strip()
        rows.append(
            {
                "book_id": int(book_id),
                "title": title,
                "author": author_primary,
                "author_norm": author_norm,
                "is_great": bool(is_great),
                "language": lang,
                "type": str(row.get("type") or ""),
            }
        )
        stats["catalog_rows_kept"] += 1

    return rows, stats


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
    id_source: str,
    catalog_url: str,
    catalog_language: str,
    shuffle_ids: bool,
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

    id_source_norm = str(id_source or "downloads").strip().lower()
    if id_source_norm not in ("downloads", "catalog"):
        raise ValueError("--id-source must be one of: downloads, catalog")

    ids: List[int] = []
    catalog_meta_by_id: Dict[int, Dict[str, Any]] = {}
    catalog_stats: Dict[str, Any] = {}
    if id_source_norm == "downloads":
        ids = gutenberg_most_downloaded_book_ids(
            max_ids=max(1, int(max_books) * 3),
            start_index=int(start_index),
            max_pages=int(max_pages),
        )
        ids = ids[: max(0, int(max_books))]
    else:
        rows, catalog_stats = _load_catalog_rows(
            catalog_url=str(catalog_url or _DEFAULT_CATALOG_CSV_URL),
            language=str(catalog_language or ""),
            type_filter="text",
            great=great,
            keep_norm=keep_norm,
        )
        rows = sorted(rows, key=lambda r: int(r.get("book_id") or 0))
        if bool(shuffle_ids):
            import random

            rng = random.Random(int(seed))
            rng.shuffle(rows)
        if int(start_index) > 1:
            rows = rows[int(start_index) - 1 :]
        if int(max_books) > 0:
            rows = rows[: int(max_books)]
        for r in rows:
            bid = int(r.get("book_id") or 0)
            if bid > 0:
                ids.append(bid)
                catalog_meta_by_id[bid] = dict(r)

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
        "id_source": id_source_norm,
        "catalog_url": str(catalog_url or ""),
        "catalog_language": str(catalog_language or ""),
        "shuffle_ids": bool(shuffle_ids),
        "catalog_stats": catalog_stats,
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
            fallback = catalog_meta_by_id.get(book_id) or {}
            title = _extract_gutenberg_title(raw) or str(fallback.get("title") or "") or f"Gutenberg {book_id}"
            author = _extract_gutenberg_author(raw) or str(fallback.get("author") or "")
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
    ap.add_argument(
        "--id-source",
        default="downloads",
        choices=["downloads", "catalog"],
        help="Where to collect book ids: downloads=most-downloaded search pages, catalog=official pg_catalog.csv",
    )
    ap.add_argument("--catalog-url", default=_DEFAULT_CATALOG_CSV_URL, help="Gutenberg catalog CSV URL (used when --id-source=catalog)")
    ap.add_argument("--catalog-language", default="en", help="ISO language code filter for catalog (e.g. en)")
    ap.add_argument("--shuffle-ids", action="store_true", help="Shuffle catalog ids before slicing (uses --seed)")
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
        id_source=str(args.id_source),
        catalog_url=str(args.catalog_url),
        catalog_language=str(args.catalog_language),
        shuffle_ids=bool(args.shuffle_ids),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
