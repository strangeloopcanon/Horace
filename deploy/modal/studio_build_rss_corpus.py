"""
Modal wrapper: build a small-but-diverse RSS/Atom corpus snapshot on a persistent volume.

So what: to avoid “Gutenberg vs not” shortcuts, we need modern prose (news/essays) as
cross-domain negatives/diagnostics. RSS feeds give us a lightweight way to ingest that
text without extra dependencies, and we keep the fetched corpus on Modal volumes.

Setup:
  make setup-modal
  make modal-token

Run (writes to /vol):
  modal run deploy/modal/studio_build_rss_corpus.py --out-dir /vol/corpora/rss_corpus_v1
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-build-rss-corpus"
REPO_REMOTE_PATH = "/root/horace"


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists():
            return p
    return Path.cwd()


_LOCAL_REPO_ROOT = _local_repo_root()

image = modal.Image.debian_slim(python_version="3.11")
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")
if (_LOCAL_REPO_ROOT / "configs").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "configs", remote_path=f"{REPO_REMOTE_PATH}/configs")

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HORACE_HTTP_CACHE_DIR", "/vol/http_cache")
    os.environ.setdefault("HORACE_HTTP_RETRIES", "2")
    os.environ.setdefault("HORACE_HTTP_RETRY_BASE_SLEEP_S", "0.8")
    os.environ.setdefault("HORACE_HTTP_RETRY_MAX_SLEEP_S", "20")


@app.function(image=image, timeout=60 * 60 * 2, volumes={"/vol": data_vol})
def build_rss_remote(
    *,
    out_dir: str,
    seed: int,
    max_chars: int,
    min_chars: int,
    max_items_per_feed: int,
    excerpts_per_item: int,
    sleep_s: float,
    feeds_json: str,
) -> str:
    _bootstrap_repo()
    from tools.studio.build_rss_corpus import DEFAULT_FEEDS, build_rss_corpus
    from tools.studio.dataset_utils import write_jsonl
    from tools.studio.split_eval_set import split_eval_set

    feeds = list(DEFAULT_FEEDS)
    fj = str(feeds_json or "").strip()
    if fj:
        try:
            p = Path(fj)
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    parsed = []
                    for it in obj:
                        if not isinstance(it, dict):
                            continue
                        name = str(it.get("name") or "").strip()
                        url = str(it.get("url") or "").strip()
                        if name and url:
                            parsed.append((name, url))
                    if parsed:
                        feeds = parsed
        except Exception:
            pass

    out = Path(str(out_dir))
    samples_path = out / "samples.jsonl"
    splits_dir = out / "splits"
    stats_path = out / "stats.json"

    if samples_path.exists():
        return json.dumps({"out_dir": str(out), "samples_path": str(samples_path), "skipped": True}, ensure_ascii=False)

    samples, stats = build_rss_corpus(
        seed=int(seed),
        max_chars=int(max_chars),
        min_chars=int(min_chars),
        max_items_per_feed=int(max_items_per_feed),
        excerpts_per_item=int(excerpts_per_item),
        sleep_s=float(sleep_s),
        normalize_text=True,
        doc_type="prose",
        feeds=tuple(feeds),
    )
    if not samples:
        raise RuntimeError("No RSS samples produced (feeds empty or extraction failed).")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples_path, (asdict(s) for s in samples))
    split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(seed),
        train_frac=0.70,
        val_frac=0.15,
        stratify_by_source=False,
    )
    stats["built_at_unix"] = int(time.time())
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    data_vol.commit()
    return json.dumps(
        {
            "out_dir": str(out),
            "samples_path": str(samples_path),
            "splits_dir": str(splits_dir),
            "stats_path": str(stats_path),
            "n_samples": int(len(samples)),
        },
        ensure_ascii=False,
        indent=2,
    )


@app.local_entrypoint()
def main(
    out_dir: str = "/vol/corpora/rss_corpus_v1",
    feeds_json: str = "configs/rss_feeds_v1.json",
    seed: int = 1337,
    max_chars: int = 3800,
    min_chars: int = 900,
    max_items_per_feed: int = 80,
    excerpts_per_item: int = 1,
    sleep_s: float = 0.0,
) -> None:  # pragma: no cover
    print(
        build_rss_remote.remote(
            out_dir=str(out_dir),
            feeds_json=str(feeds_json),
            seed=int(seed),
            max_chars=int(max_chars),
            min_chars=int(min_chars),
            max_items_per_feed=int(max_items_per_feed),
            excerpts_per_item=int(excerpts_per_item),
            sleep_s=float(sleep_s),
        )
    )
