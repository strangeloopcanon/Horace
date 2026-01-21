"""
Modal wrapper for training a single text→score scorer model on the within-domain benchmark (v4).

So what: this produces a fast inference-time scorer (no token-level analysis needed) that can be used as
the primary "literary style" score in Studio. Labels are proxy: Gutenberg top-download excerpts are
treated as positives vs long-tail Gutenberg + controlled corruptions as negatives.

Setup (outside this repo):
  pip install modal
  modal token new

Run:
  modal run deploy/modal/studio_train_scorer_v4.py --out-dir /vol/models/scorer_v4 --base-model distilbert-base-uncased
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-train-scorer-v4"
REPO_REMOTE_PATH = "/root/horace"


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists():
            return p
    return Path.cwd()


_LOCAL_REPO_ROOT = _local_repo_root()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu121", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy>=1.24.0",
        "transformers>=4.40.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.2.0",
        "safetensors>=0.4.0",
    )
)
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("horace-hf-cache", create_if_missing=True)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HORACE_HTTP_CACHE_DIR", "/vol/http_cache")


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 4,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def train_remote(
    *,
    out_dir: str,
    base_model: str,
    seed: int,
    max_length: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    max_chars: int,
    top_books: int,
    top_excerpts_per_book: int,
    tail_books: int,
    tail_min_index: int,
    tail_max_index: int,
    tail_excerpts_per_book: int,
    tail_start_index: Optional[int] = None,
    rebuild_benchmark: bool = False,
) -> str:
    _bootstrap_repo()
    from tools.studio.build_benchmark_v4 import main as build_bench_main
    from tools.studio.train_scorer import train_scorer

    bench_root = Path("/vol/benchmarks/studio_benchmark_v4")
    if rebuild_benchmark and bench_root.exists():
        # Avoid deleting; just write a new timestamped snapshot.
        bench_root = Path(f"/vol/benchmarks/studio_benchmark_v4_{int(time.time())}")
    bench_root.mkdir(parents=True, exist_ok=True)

    samples_path = bench_root / "samples.jsonl"
    if rebuild_benchmark or not samples_path.exists():
        if tail_start_index is not None:
            tail_min_index = int(tail_start_index)
        args = [
            "--out-dir",
            str(bench_root),
            "--seed",
            str(int(seed)),
            "--max-chars",
            str(int(max_chars)),
            "--top-books",
            str(int(top_books)),
            "--top-excerpts-per-book",
            str(int(top_excerpts_per_book)),
            "--random-books",
            str(int(tail_books)),
            "--random-excerpts-per-book",
            str(int(tail_excerpts_per_book)),
            "--tail-min-index",
            str(int(tail_min_index)),
            "--tail-max-index",
            str(int(tail_max_index)),
        ]
        build_bench_main(args)

    splits = bench_root / "splits"
    out_dir_p = Path(str(out_dir))
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Labels: top excerpts are positives; long-tail and corruptions are negatives.
    summary = train_scorer(
        train_path=splits / "train.jsonl",
        val_path=splits / "val.jsonl",
        test_path=splits / "test.jsonl",
        out_dir=out_dir_p,
        base_model=str(base_model),
        doc_type="prose",
        normalize_text=True,
        positive_sources=("gutenberg_top_excerpt",),
        negative_sources=(
            "gutenberg_random_excerpt",
            "gutenberg_corrupt_shuffle_sentences_global",
            "gutenberg_corrupt_shuffle_paragraphs",
            "gutenberg_corrupt_repeat_sentences",
            "gutenberg_corrupt_flatten",
        ),
        label_key="label",
        max_length=int(max_length),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
        seed=int(seed),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    data_vol.commit()
    hf_cache_vol.commit()
    return json.dumps({"benchmark_dir": str(bench_root), "train_summary": summary.__dict__}, ensure_ascii=False, indent=2)


@app.local_entrypoint()
def main(
    out_dir: str,
    base_model: str = "distilbert-base-uncased",
    seed: int = 1337,
    max_length: int = 384,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    epochs: int = 2,
    max_chars: int = 3800,
    top_books: int = 220,
    top_excerpts_per_book: int = 2,
    tail_books: int = 450,
    tail_min_index: int = 2500,
    tail_max_index: int = 5000,
    tail_start_index: Optional[int] = None,
    tail_excerpts_per_book: int = 1,
    rebuild_benchmark: bool = False,
) -> None:  # pragma: no cover
    res = train_remote.remote(
        out_dir=str(out_dir),
        base_model=str(base_model),
        seed=int(seed),
        max_length=int(max_length),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
        max_chars=int(max_chars),
        top_books=int(top_books),
        top_excerpts_per_book=int(top_excerpts_per_book),
        tail_books=int(tail_books),
        tail_min_index=int(tail_min_index),
        tail_max_index=int(tail_max_index),
        tail_excerpts_per_book=int(tail_excerpts_per_book),
        tail_start_index=int(tail_start_index) if tail_start_index is not None else None,
        rebuild_benchmark=bool(rebuild_benchmark),
    )
    print(res)
