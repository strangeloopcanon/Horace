"""
Modal wrapper: build a Standard Ebooks corpus snapshot on a persistent volume.

So what: for an accuracy-first scorer, we need more “great prose” positives than raw Gutenberg.
Standard Ebooks is a curated public-domain source with cleaner formatting, and we can sample
windows from EPUBs to create a leakage-safe training corpus.

Setup:
  make setup-modal
  make modal-token

Run (writes to /vol):
  modal run deploy/modal/studio_build_standardebooks_corpus.py --out-dir /vol/corpora/standardebooks_corpus_v1
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

APP_NAME = "horace-studio-build-standardebooks-corpus"
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

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HORACE_HTTP_CACHE_DIR", "/vol/http_cache")
    os.environ.setdefault("HORACE_HTTP_RETRIES", "3")
    os.environ.setdefault("HORACE_HTTP_RETRY_BASE_SLEEP_S", "1.2")
    os.environ.setdefault("HORACE_HTTP_RETRY_MAX_SLEEP_S", "30")


@app.function(image=image, timeout=60 * 60 * 6, volumes={"/vol": data_vol})
def build_standardebooks_remote(
    *,
    out_dir: str,
    seed: int,
    max_chars: int,
    min_chars: int,
    max_books: int,
    excerpts_per_book: int,
    start_page: int,
    max_pages: int,
    sleep_s: float,
) -> str:
    _bootstrap_repo()
    from tools.studio.build_standardebooks_corpus import build_standardebooks_corpus
    from tools.studio.dataset_utils import write_jsonl
    from tools.studio.split_eval_set import split_eval_set

    out = Path(str(out_dir))
    samples_path = out / "samples.jsonl"
    splits_dir = out / "splits"

    if samples_path.exists():
        return json.dumps({"out_dir": str(out), "samples_path": str(samples_path), "skipped": True}, ensure_ascii=False)

    samples, stats, failures = build_standardebooks_corpus(
        seed=int(seed),
        max_chars=int(max_chars),
        min_chars=int(min_chars),
        max_books=int(max_books),
        excerpts_per_book=int(excerpts_per_book),
        start_page=int(start_page),
        max_pages=int(max_pages),
        sleep_s=float(sleep_s),
        normalize_text=True,
        doc_type="prose",
    )
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples_path, (asdict(s) for s in samples))
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        write_jsonl(out / "failures.jsonl", failures)
    split_eval_set(
        samples_path=samples_path,
        out_dir=splits_dir,
        seed=int(seed),
        train_frac=0.70,
        val_frac=0.15,
        stratify_by_source=False,
    )

    data_vol.commit()
    return json.dumps(
        {
            "out_dir": str(out),
            "samples_path": str(samples_path),
            "splits_dir": str(splits_dir),
            "n_samples": int(len(samples)),
            "built_at_unix": int(time.time()),
        },
        ensure_ascii=False,
        indent=2,
    )


@app.local_entrypoint()
def main(
    out_dir: str = "/vol/corpora/standardebooks_corpus_v1",
    seed: int = 1337,
    max_chars: int = 3800,
    min_chars: int = 900,
    max_books: int = 600,
    excerpts_per_book: int = 3,
    start_page: int = 1,
    max_pages: int = 80,
    sleep_s: float = 0.1,
) -> None:  # pragma: no cover
    print(
        build_standardebooks_remote.remote(
            out_dir=str(out_dir),
            seed=int(seed),
            max_chars=int(max_chars),
            min_chars=int(min_chars),
            max_books=int(max_books),
            excerpts_per_book=int(excerpts_per_book),
            start_page=int(start_page),
            max_pages=int(max_pages),
            sleep_s=float(sleep_s),
        )
    )
