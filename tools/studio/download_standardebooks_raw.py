from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from tools.studio.author_utils import load_author_list, normalize_author_name
from tools.studio.dataset_utils import append_jsonl, group_id as make_group_id, stable_sha1_hex
from tools.studio.standardebooks import download_epub_bytes, extract_text_from_epub, fetch_ebook, list_ebook_paths
from tools.studio.text_normalize import normalize_for_studio


def _safe_doc_id(*, gutenberg_id: Optional[int], url: str) -> str:
    if gutenberg_id:
        return f"pg{int(gutenberg_id)}"
    return "se_" + stable_sha1_hex([str(url)])[:16]


def _book_text_path(out_dir: Path, doc_id: str) -> Path:
    return out_dir / "books" / f"{doc_id}.txt"


def download_standardebooks_raw(
    *,
    out_dir: Path,
    seed: int,
    start_page: int,
    max_pages: int,
    max_books: int,
    sleep_s: float,
    normalize_text: bool,
    doc_type: str,
    great_authors_path: Optional[Path],
    only_great: bool,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    books_dir = out_dir / "books"
    meta_path = out_dir / "meta.jsonl"
    failures_path = out_dir / "failures.jsonl"
    stats_path = out_dir / "stats.json"

    great: set[str] = set()
    if great_authors_path is not None and great_authors_path.exists():
        great = load_author_list(great_authors_path.read_text(encoding="utf-8").splitlines())

    paths = list_ebook_paths(max_pages=int(max_pages), start_page=int(start_page))
    # Deterministic ordering; we rely on seed only for sampling windows later.
    paths = paths[: max(0, int(max_books))]

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "start_page": int(start_page),
        "max_pages": int(max_pages),
        "max_books": int(max_books),
        "sleep_s": float(sleep_s),
        "normalize_text": bool(normalize_text),
        "doc_type": str(doc_type),
        "great_authors_path": str(great_authors_path) if great_authors_path else "",
        "only_great": bool(only_great),
        "paths_listed": int(len(paths)),
        "books_attempted": 0,
        "books_saved": 0,
        "books_skipped_existing": 0,
        "books_skipped_not_great": 0,
        "books_skipped_too_short": 0,
        "failures": 0,
        "bytes_text_saved": 0,
    }

    now = int(time.time())
    for i, p in enumerate(paths, 1):
        stats["books_attempted"] += 1
        doc_stage = "init"
        eb = None
        try:
            doc_stage = "fetch_ebook"
            eb = fetch_ebook(p)
            doc_stage = "bucket"
            author_norm = normalize_author_name(eb.author or "")
            is_great = True if not great else (author_norm in great)
            if bool(only_great) and not bool(is_great):
                stats["books_skipped_not_great"] += 1
                continue

            doc_stage = "ids"
            gid = f"gutenberg:{int(eb.gutenberg_id)}" if eb.gutenberg_id else make_group_id("standardebooks", stable=eb.url)
            doc_id = _safe_doc_id(gutenberg_id=eb.gutenberg_id, url=eb.url)
            text_path = _book_text_path(out_dir, doc_id)
            if text_path.exists():
                stats["books_skipped_existing"] += 1
                continue

            doc_stage = "download_epub"
            epub = download_epub_bytes(eb)
            doc_stage = "extract_text"
            full = extract_text_from_epub(epub)
            if normalize_text:
                doc_stage = "normalize"
                full, norm_meta = normalize_for_studio(full, doc_type=str(doc_type), enabled=True)
            else:
                norm_meta = None

            if not full.strip() or len(full) < 6000:
                stats["books_skipped_too_short"] += 1
                continue

            doc_stage = "write_text"
            books_dir.mkdir(parents=True, exist_ok=True)
            text_path.write_text(full, encoding="utf-8")
            stats["books_saved"] += 1
            stats["bytes_text_saved"] += int(text_path.stat().st_size)

            doc_stage = "write_meta"
            append_jsonl(
                meta_path,
                [
                    {
                        "doc_id": doc_id,
                        "group_id": gid,
                        "source": "standardebooks_raw",
                        "bucket": "great_author" if is_great else "other_author",
                        "title": eb.title,
                        "author": eb.author,
                        "author_norm": author_norm,
                        "url": eb.url,
                        "standardebooks_path": eb.path,
                        "download_epub_url": eb.download_epub_url,
                        "gutenberg_id": eb.gutenberg_id,
                        "text_path": str(text_path.relative_to(out_dir)),
                        "chars": int(len(full)),
                        "fetched_at_unix": now,
                        "normalization": norm_meta,
                        "license_hint": "public_domain_standardebooks",
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
                            "standardebooks_path": str(p),
                            "url": (eb.url if eb else ""),
                            "gutenberg_id": (eb.gutenberg_id if eb else None),
                            "title": (eb.title if eb else ""),
                            "author": (eb.author if eb else ""),
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
        if i % 25 == 0:
            print(
                f"[standardebooks_raw] attempted={stats['books_attempted']} saved={stats['books_saved']} "
                f"skipped_existing={stats['books_skipped_existing']} skipped_not_great={stats['books_skipped_not_great']} "
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
    ap = argparse.ArgumentParser(description="Download Standard Ebooks EPUBs, extract + normalize full text, and store locally.")
    ap.add_argument("--out-dir", default="data/corpora/standardebooks_raw_v1")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--max-books", type=int, default=2000)
    ap.add_argument("--sleep-s", type=float, default=0.1)
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--doc-type", default="prose")

    ap.add_argument("--great-authors", default="", help="Optional text file (one author per line) to define the 'great' bucket")
    ap.add_argument("--only-great", action="store_true", help="Only keep authors listed in --great-authors (if provided)")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    great_path = Path(str(args.great_authors)) if str(args.great_authors).strip() else None

    res = download_standardebooks_raw(
        out_dir=Path(str(args.out_dir)),
        seed=int(args.seed),
        start_page=int(args.start_page),
        max_pages=int(args.max_pages),
        max_books=int(args.max_books),
        sleep_s=float(args.sleep_s),
        normalize_text=bool(normalize_text),
        doc_type=str(args.doc_type),
        great_authors_path=great_path,
        only_great=bool(args.only_great),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
