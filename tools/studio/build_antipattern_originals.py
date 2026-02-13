from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tools.studio.author_utils import normalize_author_name
from tools.studio.dataset_utils import iter_jsonl, sample_window, stable_sha1_hex, write_jsonl
from tools.studio.text_normalize import normalize_for_studio


def _iter_existing_jsonl(paths: Sequence[Path], *, skip_missing: bool) -> Iterable[Tuple[Path, dict]]:
    for p in paths:
        path = Path(p)
        if not path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            if isinstance(row, dict):
                yield path, row


def _iter_text_files(
    dirs: Sequence[Path],
    *,
    skip_missing: bool,
    recurse: bool = True,
) -> Iterable[Tuple[Path, str]]:
    for d in dirs:
        base = Path(d)
        if not base.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(base)
        pattern = "**/*.txt" if recurse else "*.txt"
        for p in sorted(base.glob(pattern)):
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            yield p, text


def _guess_doc_type(*, source: str, path_hint: str, explicit: str) -> str:
    dt = str(explicit or "").strip().lower()
    if dt in ("poem", "shortstory", "prose", "novel"):
        return dt
    src = f"{source} {path_hint}".lower()
    if "poem" in src:
        return "poem"
    if "shortstory" in src or "short_story" in src or "short-story" in src:
        return "shortstory"
    if "novel" in src:
        return "novel"
    return "prose"


def _author_from_text_path(path: Path) -> str:
    # Examples:
    #   data/poem/william_butler_yeats/The_Second_Coming.txt
    #   data/shortstory/anton_chekhov/The_Bet.txt
    parts = path.parts
    if len(parts) < 2:
        return ""
    parent = parts[-2].replace("_", " ").strip()
    return " ".join(w.capitalize() for w in parent.split())


def _title_from_text_path(path: Path) -> str:
    stem = path.stem.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in stem.split()) or path.stem


def _truncate_or_sample(
    text: str,
    *,
    rng: random.Random,
    max_chars: int,
    min_chars: int,
) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    if len(t) <= int(max_chars):
        return t
    return sample_window(t, rng=rng, max_chars=int(max_chars), min_chars=int(min_chars))


def _norm_key(text: str) -> str:
    folded = " ".join((text or "").strip().lower().split())
    return stable_sha1_hex([folded])


def build_antipattern_originals(
    *,
    input_jsonl: Sequence[Path],
    input_text_dirs: Sequence[Path],
    out_path: Path,
    seed: int,
    max_samples: int,
    min_chars: int,
    max_chars: int,
    windows_per_text_file: int,
    normalize_text: bool,
    doc_type: str,
    skip_missing: bool,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))

    rows: List[dict] = []
    seen_sample_ids: set[str] = set()
    seen_text_keys: set[str] = set()
    seen_raw_text_keys: set[str] = set()

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "max_samples": int(max_samples),
        "min_chars": int(min_chars),
        "max_chars": int(max_chars),
        "windows_per_text_file": int(windows_per_text_file),
        "normalize_text": bool(normalize_text),
        "doc_type": str(doc_type),
        "rows_in_jsonl": 0,
        "rows_in_text_files": 0,
        "rows_out": 0,
        "skipped_short": 0,
        "skipped_empty": 0,
        "skipped_duplicate_text": 0,
        "skipped_duplicate_sample_id": 0,
        "by_source": {},
        "by_doc_type": {},
    }

    # 1) JSONL corpus rows.
    for src_path, r in _iter_existing_jsonl(input_jsonl, skip_missing=bool(skip_missing)):
        stats["rows_in_jsonl"] += 1
        text = str(r.get("text") or "")
        if normalize_text:
            text, _ = normalize_for_studio(text, doc_type=str(doc_type or "prose"), enabled=True)
        raw_text_key = _norm_key(text)
        if raw_text_key in seen_raw_text_keys:
            stats["skipped_duplicate_text"] += 1
            continue
        text = _truncate_or_sample(
            text,
            rng=rng,
            max_chars=int(max_chars),
            min_chars=int(min_chars),
        )
        if not text.strip():
            stats["skipped_empty"] += 1
            continue
        if len(text) < int(min_chars):
            stats["skipped_short"] += 1
            continue
        text_key = _norm_key(text)
        if text_key in seen_text_keys:
            stats["skipped_duplicate_text"] += 1
            continue

        sample_id = str(r.get("sample_id") or "").strip()
        if not sample_id:
            sample_id = stable_sha1_hex([str(src_path), str(r.get("group_id") or ""), text])[:12]
        if sample_id in seen_sample_ids:
            stats["skipped_duplicate_sample_id"] += 1
            continue

        meta = dict(r.get("meta") or {}) if isinstance(r.get("meta"), dict) else {}
        author = str(meta.get("author") or "")
        author_norm = normalize_author_name(author) if author else str(meta.get("author_norm") or "")
        source = str(r.get("source") or "unknown").strip() or "unknown"
        dt = _guess_doc_type(
            source=source,
            path_hint=str(src_path),
            explicit=str(doc_type),
        )
        row = {
            "sample_id": sample_id,
            "group_id": str(r.get("group_id") or ""),
            "source": source,
            "title": str(r.get("title") or ""),
            "url": str(r.get("url") or ""),
            "text": text,
            "author": author,
            "author_norm": author_norm,
            "doc_type": dt,
            "source_corpus": str(src_path),
            "fetched_at_unix": int(r.get("fetched_at_unix") or time.time()),
            "meta": {
                **meta,
                "antipattern_original": True,
                "source_path": str(src_path),
            },
        }
        rows.append(row)
        seen_sample_ids.add(sample_id)
        seen_text_keys.add(text_key)
        seen_raw_text_keys.add(raw_text_key)
        stats["by_source"][source] = int(stats["by_source"].get(source, 0)) + 1
        stats["by_doc_type"][dt] = int(stats["by_doc_type"].get(dt, 0)) + 1

    # 2) Plain-text files (poem/shortstory style folders).
    per_file = max(1, int(windows_per_text_file))
    for text_path, raw in _iter_text_files(input_text_dirs, skip_missing=bool(skip_missing)):
        stats["rows_in_text_files"] += 1
        base_author = _author_from_text_path(text_path)
        base_title = _title_from_text_path(text_path)
        base_doc_type = _guess_doc_type(
            source="text_file",
            path_hint=str(text_path),
            explicit=str(doc_type),
        )
        source = base_doc_type if base_doc_type in ("poem", "shortstory", "novel") else "text_file"
        text_full = str(raw or "")
        if normalize_text:
            text_full, _ = normalize_for_studio(text_full, doc_type=base_doc_type, enabled=True)
        if not text_full.strip():
            stats["skipped_empty"] += 1
            continue
        raw_text_key = _norm_key(text_full)
        if raw_text_key in seen_raw_text_keys:
            stats["skipped_duplicate_text"] += 1
            continue

        for k in range(per_file):
            text = _truncate_or_sample(
                text_full,
                rng=rng,
                max_chars=int(max_chars),
                min_chars=int(min_chars),
            )
            if not text.strip():
                stats["skipped_empty"] += 1
                continue
            if len(text) < int(min_chars):
                stats["skipped_short"] += 1
                continue
            text_key = _norm_key(text)
            if text_key in seen_text_keys:
                stats["skipped_duplicate_text"] += 1
                continue
            sample_id = stable_sha1_hex([str(text_path), str(k), text])[:12]
            if sample_id in seen_sample_ids:
                stats["skipped_duplicate_sample_id"] += 1
                continue
            group_id = stable_sha1_hex([str(text_path)])[:12]
            row = {
                "sample_id": sample_id,
                "group_id": f"textfile:{group_id}",
                "source": source,
                "title": base_title,
                "url": "",
                "text": text,
                "author": base_author,
                "author_norm": normalize_author_name(base_author),
                "doc_type": base_doc_type,
                "source_corpus": str(text_path.parent),
                "fetched_at_unix": int(time.time()),
                "meta": {
                    "antipattern_original": True,
                    "source_path": str(text_path),
                    "source_kind": "text_file",
                },
            }
            rows.append(row)
            seen_sample_ids.add(sample_id)
            seen_text_keys.add(text_key)
            seen_raw_text_keys.add(raw_text_key)
            stats["by_source"][source] = int(stats["by_source"].get(source, 0)) + 1
            stats["by_doc_type"][base_doc_type] = int(stats["by_doc_type"].get(base_doc_type, 0)) + 1

    rng.shuffle(rows)
    if int(max_samples) > 0:
        rows = rows[: int(max_samples)]
    stats["rows_out"] = int(len(rows))

    out_path = Path(out_path)
    write_jsonl(out_path, rows)
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_path": str(out_path), "stats_path": str(stats_path), "stats": stats}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Curate anti-pattern source passages from existing corpora/text folders. "
            "Outputs JSONL rows with author/doc-type metadata for downstream imitation generation."
        )
    )
    ap.add_argument("--input-jsonl", action="append", default=[], help="Input JSONL file path (repeatable)")
    ap.add_argument("--input-text-dir", action="append", default=[], help="Input text directory path (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-samples", type=int, default=0, help="Cap outputs (0 = all)")
    ap.add_argument("--min-chars", type=int, default=800)
    ap.add_argument("--max-chars", type=int, default=3800)
    ap.add_argument("--windows-per-text-file", type=int, default=1)
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--doc-type", default="auto", help="Optional forced doc type (auto|prose|poem|shortstory|novel)")
    ap.add_argument("--skip-missing", action="store_true", help="Skip missing input files/directories")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    res = build_antipattern_originals(
        input_jsonl=tuple(Path(str(x)) for x in (args.input_jsonl or []) if str(x).strip()),
        input_text_dirs=tuple(Path(str(x)) for x in (args.input_text_dir or []) if str(x).strip()),
        out_path=Path(str(args.out)),
        seed=int(args.seed),
        max_samples=int(args.max_samples),
        min_chars=int(args.min_chars),
        max_chars=int(args.max_chars),
        windows_per_text_file=int(args.windows_per_text_file),
        normalize_text=bool(normalize_text),
        doc_type=str(args.doc_type),
        skip_missing=bool(args.skip_missing),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
