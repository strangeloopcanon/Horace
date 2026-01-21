"""
Modal wrapper: distill Horace's slow rubric score into a single fast scorer model.

So what: token-level analysis + rubric scoring is informative but too slow to run on every request.
This job labels a benchmark split with the rubric score (teacher) and trains an encoder (student)
to predict it: text → score (0–100) in one forward pass.

Setup (outside this repo):
  pip install modal
  modal token new

Run (GPU, writes to /vol):
  modal run deploy/modal/studio_distill_scorer_v4.py --out-dir /vol/models/scorer_v4_distilled
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

APP_NAME = "horace-studio-distill-scorer-v4"
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
if (_LOCAL_REPO_ROOT / "data" / "baselines").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "data" / "baselines", remote_path=f"{REPO_REMOTE_PATH}/data/baselines")

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
    os.environ.setdefault("HORACE_HF_FULL_LOGITS", "1")


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 8,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def distill_remote(
    *,
    out_dir: str,
    base_model: str,
    scorer_max_length: int,
    scorer_batch_size: int,
    scorer_lr: float,
    scorer_weight_decay: float,
    scorer_epochs: int,
    seed: int,
    # Teacher config (rubric)
    teacher_model: str,
    baseline_model: str,
    teacher_max_input_tokens: int,
    # Benchmark config
    max_chars: int,
    top_books: int,
    top_excerpts_per_book: int,
    tail_books: int,
    tail_min_index: int,
    tail_max_index: int,
    tail_excerpts_per_book: int,
    rebuild_benchmark: bool = False,
    max_label_samples: int = 0,
) -> str:
    _bootstrap_repo()
    from tools.studio.build_benchmark_v4 import main as build_bench_main
    from tools.studio.label_scorer_dataset import label_jsonl
    from tools.studio.train_scorer import train_scorer

    bench_root = Path("/vol/benchmarks/studio_benchmark_v4")
    if rebuild_benchmark and bench_root.exists():
        bench_root = Path(f"/vol/benchmarks/studio_benchmark_v4_{int(time.time())}")
    bench_root.mkdir(parents=True, exist_ok=True)

    samples_path = bench_root / "samples.jsonl"
    if rebuild_benchmark or not samples_path.exists():
        build_args = [
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
        build_bench_main(build_args)

    splits = bench_root / "splits"
    label_root = bench_root / "distill_labels"
    label_root.mkdir(parents=True, exist_ok=True)

    max_samples = int(max_label_samples) if int(max_label_samples) > 0 else None
    for split in ("train", "val", "test"):
        n = label_jsonl(
            in_path=splits / f"{split}.jsonl",
            out_path=label_root / f"{split}.jsonl",
            max_samples=max_samples if split == "train" else max_samples,
            seed=int(seed),
            teacher_model_id=str(teacher_model),
            baseline_model=str(baseline_model),
            doc_type="prose",
            backend="hf",
            max_input_tokens=int(teacher_max_input_tokens),
            normalize_text=True,
            compute_cohesion=False,
        )
        print(f"labeled {split}: n={n}")

    out_dir_p = Path(str(out_dir))
    out_dir_p.mkdir(parents=True, exist_ok=True)

    summary = train_scorer(
        train_path=label_root / "train.jsonl",
        val_path=label_root / "val.jsonl",
        test_path=label_root / "test.jsonl",
        out_dir=out_dir_p,
        base_model=str(base_model),
        doc_type="prose",
        normalize_text=True,
        positive_sources=("gutenberg_top_excerpt",),  # ignored because label_key is present
        negative_sources=None,
        label_key="label",
        max_length=int(scorer_max_length),
        batch_size=int(scorer_batch_size),
        lr=float(scorer_lr),
        weight_decay=float(scorer_weight_decay),
        epochs=int(scorer_epochs),
        seed=int(seed),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    data_vol.commit()
    hf_cache_vol.commit()
    return json.dumps({"benchmark_dir": str(bench_root), "label_dir": str(label_root), "train_summary": summary.__dict__}, ensure_ascii=False, indent=2)


@app.local_entrypoint()
def main(
    out_dir: str,
    base_model: str = "distilbert-base-uncased",
    scorer_max_length: int = 384,
    scorer_batch_size: int = 16,
    scorer_lr: float = 2e-5,
    scorer_weight_decay: float = 0.01,
    scorer_epochs: int = 2,
    seed: int = 1337,
    teacher_model: str = "gpt2",
    baseline_model: str = "gpt2_gutenberg_512",
    teacher_max_input_tokens: int = 512,
    max_chars: int = 3800,
    top_books: int = 220,
    top_excerpts_per_book: int = 2,
    tail_books: int = 450,
    tail_min_index: int = 2500,
    tail_max_index: int = 5000,
    tail_excerpts_per_book: int = 1,
    rebuild_benchmark: bool = False,
    max_label_samples: int = 0,
) -> None:  # pragma: no cover
    res = distill_remote.remote(
        out_dir=str(out_dir),
        base_model=str(base_model),
        scorer_max_length=int(scorer_max_length),
        scorer_batch_size=int(scorer_batch_size),
        scorer_lr=float(scorer_lr),
        scorer_weight_decay=float(scorer_weight_decay),
        scorer_epochs=int(scorer_epochs),
        seed=int(seed),
        teacher_model=str(teacher_model),
        baseline_model=str(baseline_model),
        teacher_max_input_tokens=int(teacher_max_input_tokens),
        max_chars=int(max_chars),
        top_books=int(top_books),
        top_excerpts_per_book=int(top_excerpts_per_book),
        tail_books=int(tail_books),
        tail_min_index=int(tail_min_index),
        tail_max_index=int(tail_max_index),
        tail_excerpts_per_book=int(tail_excerpts_per_book),
        rebuild_benchmark=bool(rebuild_benchmark),
        max_label_samples=int(max_label_samples),
    )
    print(res)
