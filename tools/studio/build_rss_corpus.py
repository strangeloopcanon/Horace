from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.studio.dataset_utils import (
    FixedSample,
    group_id as make_group_id,
    html_to_text,
    http_get_text,
    make_sample_id,
    sample_window,
    write_jsonl,
)
from tools.studio.rss import FeedEntry, parse_feed
from tools.studio.split_eval_set import split_eval_set
from tools.studio.text_normalize import normalize_for_studio


DEFAULT_FEEDS: List[Tuple[str, str]] = [
    ("theconversation_us", "https://theconversation.com/us/articles.atom"),
    ("propublica", "https://www.propublica.org/feeds/propublica/main"),
]


def _extract_text_from_entry(entry: FeedEntry) -> str:
    # Many feeds include HTML in description/content; extract best-effort text.
    raw = entry.content_html or ""
    if not raw.strip() and entry.url.strip():
        try:
            raw = http_get_text(entry.url)
        except Exception:
            raw = ""
    if not raw.strip():
        return ""
    return html_to_text(raw)


def build_rss_corpus(
    *,
    seed: int,
    max_chars: int,
    min_chars: int,
    max_items_per_feed: int,
    excerpts_per_item: int,
    sleep_s: float,
    normalize_text: bool,
    doc_type: str,
    feeds: Sequence[Tuple[str, str]],
) -> Tuple[List[FixedSample], Dict[str, Any]]:
    rng = random.Random(int(seed))
    now = int(time.time())

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "feeds": [{"name": n, "url": u} for n, u in feeds],
        "attempted_entries": 0,
        "kept_entries": 0,
        "attempted_excerpts": 0,
        "kept_excerpts": 0,
        "skipped_empty_text": 0,
        "skipped_too_short": 0,
        "by_feed": {},
    }

    samples: List[FixedSample] = []
    seen: set[str] = set()

    for feed_name, feed_url in feeds:
        feed_name_norm = str(feed_name or "").strip() or "rss_feed"
        feed_url_norm = str(feed_url or "").strip()
        if not feed_url_norm:
            continue

        feed_stats = {
            "feed_url": feed_url_norm,
            "entries_parsed": 0,
            "entries_used": 0,
            "excerpts_kept": 0,
        }

        try:
            xml = http_get_text(feed_url_norm)
        except Exception:
            stats["by_feed"][feed_name_norm] = {**feed_stats, "error": "fetch_failed"}
            continue

        entries = parse_feed(xml)
        if not entries:
            stats["by_feed"][feed_name_norm] = {**feed_stats, "error": "parse_empty"}
            continue

        # Deterministic ordering, then sample with seed.
        entries_sorted = sorted(entries, key=lambda e: (e.url or "", e.title or ""))
        feed_stats["entries_parsed"] = int(len(entries_sorted))

        max_items = max(1, int(max_items_per_feed))
        idxs = list(range(len(entries_sorted)))
        rng.shuffle(idxs)
        keep = set(idxs[: min(max_items, len(idxs))])
        chosen = [e for i, e in enumerate(entries_sorted) if i in keep]
        feed_stats["entries_used"] = int(len(chosen))

        for ent in chosen:
            stats["attempted_entries"] += 1
            text = _extract_text_from_entry(ent)
            if normalize_text and text:
                text, _ = normalize_for_studio(text, doc_type=str(doc_type), enabled=True)
            if not text.strip():
                stats["skipped_empty_text"] += 1
                continue
            if len(text) < max(200, int(min_chars)):
                stats["skipped_too_short"] += 1
                continue

            stats["kept_entries"] += 1

            gid = make_group_id(feed_name_norm, stable=ent.url or ent.title)
            per = max(1, int(excerpts_per_item))
            for _ in range(per):
                stats["attempted_excerpts"] += 1
                excerpt = sample_window(text, rng=rng, max_chars=int(max_chars), min_chars=int(min_chars))
                if not excerpt:
                    continue
                sid = make_sample_id(feed_name_norm, ent.title, ent.url, excerpt)
                if sid in seen:
                    continue
                seen.add(sid)
                samples.append(
                    FixedSample(
                        sample_id=sid,
                        group_id=gid,
                        source=feed_name_norm,
                        title=str(ent.title or ""),
                        url=str(ent.url or ""),
                        text=excerpt,
                        fetched_at_unix=now,
                        meta={
                            "feed_url": feed_url_norm,
                            "published_unix": ent.published_unix,
                            "license_hint": "web_rss",
                        },
                    )
                )
                stats["kept_excerpts"] += 1
                feed_stats["excerpts_kept"] += 1

        stats["by_feed"][feed_name_norm] = feed_stats

        pause = float(sleep_s)
        if pause > 0:
            time.sleep(pause)

    return samples, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a web RSS/Atom corpus snapshot (feed content windows) with leakage-safe splits.")
    ap.add_argument("--out-dir", default="data/corpora/rss_corpus_v1")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-chars", type=int, default=3800)
    ap.add_argument("--min-chars", type=int, default=900)
    ap.add_argument("--max-items-per-feed", type=int, default=80)
    ap.add_argument("--excerpts-per-item", type=int, default=1)
    ap.add_argument("--sleep-s", type=float, default=0.0)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--doc-type", default="prose")

    ap.add_argument("--feed", action="append", default=[], help="Feed in form name=url (repeatable)")
    ap.add_argument("--feeds-json", default="", help="Optional JSON file with [{name,url},...]")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out_dir))
    samples_path = out_dir / "samples.jsonl"
    splits_dir = out_dir / "splits"
    stats_path = out_dir / "stats.json"

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)

    feeds: List[Tuple[str, str]] = []
    if str(args.feeds_json).strip():
        p = Path(str(args.feeds_json))
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for it in obj:
                if not isinstance(it, dict):
                    continue
                n = str(it.get("name") or "").strip()
                u = str(it.get("url") or "").strip()
                if n and u:
                    feeds.append((n, u))
    for spec in args.feed or []:
        s = str(spec or "").strip()
        if not s or "=" not in s:
            continue
        name, url = s.split("=", 1)
        name = name.strip()
        url = url.strip()
        if name and url:
            feeds.append((name, url))
    if not feeds:
        feeds = list(DEFAULT_FEEDS)

    samples, stats = build_rss_corpus(
        seed=int(args.seed),
        max_chars=int(args.max_chars),
        min_chars=int(args.min_chars),
        max_items_per_feed=int(args.max_items_per_feed),
        excerpts_per_item=int(args.excerpts_per_item),
        sleep_s=float(args.sleep_s),
        normalize_text=bool(normalize_text),
        doc_type=str(args.doc_type),
        feeds=feeds,
    )

    if not samples:
        raise RuntimeError("No RSS samples produced. Try increasing --max-items-per-feed or adding --feed sources.")

    write_jsonl(samples_path, (asdict(s) for s in samples))
    split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        stratify_by_source=False,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(samples_path))
    print(f"n={len(samples)} splits={splits_dir} stats={stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

