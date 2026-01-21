from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.studio.dataset_utils import FixedSample, iter_jsonl, make_sample_id, sample_window, write_jsonl
from tools.studio.split_eval_set import split_eval_set
from tools.studio.text_normalize import normalize_for_studio


def _middle_slice(text: str, *, lo_frac: float = 0.05, hi_frac: float = 0.95, min_chars: int = 25000) -> str:
    t = text or ""
    if len(t) < int(min_chars):
        return t
    lo = int(max(0, min(len(t), int(len(t) * float(lo_frac)))))
    hi = int(max(lo, min(len(t), int(len(t) * float(hi_frac)))))
    return t[lo:hi].strip() or t


def sample_windows_from_raw(
    *,
    raw_dir: Path,
    out_dir: Path,
    seed: int,
    max_chars: int,
    min_chars: int,
    windows_per_doc: int,
    max_docs: int,
    bucket: str,
    doc_type: str,
    renormalize: bool,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    meta_path = raw_dir / "meta.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"Missing raw meta: {meta_path}")

    rows = list(iter_jsonl(meta_path))
    if bucket.strip():
        want_bucket = str(bucket).strip().lower()
        rows = [r for r in rows if str(r.get("bucket") or "").strip().lower() == want_bucket]

    # Deterministic order (stable), sample subset with seed.
    rows = sorted(rows, key=lambda r: str(r.get("doc_id") or ""))
    if int(max_docs) > 0 and len(rows) > int(max_docs):
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        keep = set(idxs[: int(max_docs)])
        rows = [r for i, r in enumerate(rows) if i in keep]

    samples: List[FixedSample] = []
    seen: set[str] = set()
    now = int(time.time())
    per = max(1, int(windows_per_doc))
    dt = str(doc_type)

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "max_chars": int(max_chars),
        "min_chars": int(min_chars),
        "windows_per_doc": int(windows_per_doc),
        "max_docs": int(max_docs),
        "bucket_filter": str(bucket),
        "doc_type": dt,
        "renormalize": bool(renormalize),
        "docs_in_meta": int(len(list(iter_jsonl(meta_path)))),
        "docs_used": int(len(rows)),
        "windows_attempted": 0,
        "windows_kept": 0,
        "by_bucket": {},
        "by_origin": {},
    }

    for r in rows:
        doc_id = str(r.get("doc_id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        origin = str(r.get("source") or "").strip() or "raw"
        bkt = str(r.get("bucket") or "").strip().lower() or "unknown"
        title = str(r.get("title") or "")
        url = str(r.get("url") or "")
        text_path = raw_dir / str(r.get("text_path") or "")
        if not doc_id or not gid or not text_path.exists():
            continue
        try:
            full = text_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if not full.strip():
            continue
        if renormalize:
            full, norm_meta = normalize_for_studio(full, doc_type=dt, enabled=True)
        else:
            norm_meta = None

        source_slice = _middle_slice(full)

        for _ in range(per):
            stats["windows_attempted"] += 1
            excerpt = sample_window(source_slice, rng=rng, max_chars=int(max_chars), min_chars=int(min_chars))
            if not excerpt:
                continue
            sid = make_sample_id(bkt, title, url, excerpt)
            if sid in seen:
                continue
            seen.add(sid)
            samples.append(
                FixedSample(
                    sample_id=sid,
                    group_id=gid,
                    source=bkt,
                    title=title,
                    url=url,
                    text=excerpt,
                    fetched_at_unix=now,
                    meta={
                        "origin_source": origin,
                        "doc_id": doc_id,
                        "author": r.get("author"),
                        "author_norm": r.get("author_norm"),
                        "book_id": r.get("book_id") or r.get("gutenberg_id"),
                        "license_hint": r.get("license_hint"),
                        "normalization": norm_meta,
                    },
                )
            )
            stats["windows_kept"] += 1
            stats["by_bucket"][bkt] = int(stats["by_bucket"].get(bkt, 0)) + 1
            stats["by_origin"][origin] = int(stats["by_origin"].get(origin, 0)) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    splits_dir = out_dir / "splits"
    stats_path = out_dir / "stats.json"
    write_jsonl(samples_path, (asdict(s) for s in samples))
    split_eval_set(samples_path=samples_path, out_dir=splits_dir, seed=int(seed), train_frac=0.70, val_frac=0.15, stratify_by_source=False)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"samples_path": str(samples_path), "splits_dir": str(splits_dir), "stats_path": str(stats_path), "n_samples": int(len(samples))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sample fixed-size windows from a raw text store (meta.jsonl + books/).")
    ap.add_argument("--raw-dir", required=True, help="Directory containing meta.jsonl and books/")
    ap.add_argument("--out-dir", required=True, help="Corpus output directory (samples.jsonl + splits/)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-chars", type=int, default=3800)
    ap.add_argument("--min-chars", type=int, default=900)
    ap.add_argument("--windows-per-doc", type=int, default=8)
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs sampled from meta (0 = all)")
    ap.add_argument("--bucket", default="", help="Optional bucket filter (e.g. great_author)")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--renormalize", action="store_true", help="Apply normalize_for_studio again (debug)")
    args = ap.parse_args(argv)

    res = sample_windows_from_raw(
        raw_dir=Path(str(args.raw_dir)),
        out_dir=Path(str(args.out_dir)),
        seed=int(args.seed),
        max_chars=int(args.max_chars),
        min_chars=int(args.min_chars),
        windows_per_doc=int(args.windows_per_doc),
        max_docs=int(args.max_docs),
        bucket=str(args.bucket),
        doc_type=str(args.doc_type),
        renormalize=bool(args.renormalize),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

