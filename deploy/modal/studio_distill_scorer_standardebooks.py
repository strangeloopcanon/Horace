"""
Modal wrapper: distill Horace's rubric score into a single fast scorer model using Standard Ebooks.

So what: for accuracy, we want the scorer to learn “great prose” patterns. Standard Ebooks is a
curated public-domain corpus with cleaner formatting than raw Gutenberg. This job:
  1) Builds/loads a Standard Ebooks corpus snapshot on /vol
  2) Builds a baseline distribution from the *train split only* (no tainted eval)
  3) Labels train/val/test with the rubric overall score (teacher)
  4) Trains an encoder student: text → score (0–100) in one pass

Setup:
  make setup-modal
  make modal-token

Run (GPU; writes to /vol):
  modal run deploy/modal/studio_distill_scorer_standardebooks.py --out-dir /vol/models/scorer_standardebooks_distilled
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-distill-scorer-standardebooks"
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
    os.environ.setdefault("HORACE_HF_FULL_LOGITS", "1")


def _pick_subset(rows: List[dict], *, seed: int, max_samples: int) -> List[dict]:
    n = int(max_samples)
    if n <= 0 or len(rows) <= n:
        return rows
    rng = random.Random(int(seed))
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    keep = set(idxs[:n])
    return [r for i, r in enumerate(rows) if i in keep]


def _build_baseline_from_split(
    *,
    samples_path: Path,
    out_path: Path,
    teacher_model: str,
    teacher_max_input_tokens: int,
    baseline_max_samples: int,
    seed: int,
) -> Dict[str, Any]:
    from tools.studio.analyze import analyze_text
    from tools.studio.baselines import build_baseline_from_rows, safe_model_id
    from tools.studio.dataset_utils import iter_jsonl

    rows = list(iter_jsonl(samples_path))
    rows = _pick_subset(rows, seed=int(seed), max_samples=int(baseline_max_samples))

    metric_rows: List[Dict[str, Any]] = []
    for r in rows:
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        res = analyze_text(
            text,
            model_id=str(teacher_model),
            doc_type="prose",
            backend="hf",
            max_input_tokens=int(teacher_max_input_tokens),
            normalize_text=True,
            compute_cohesion=False,
        )
        dm = res.get("doc_metrics") or {}
        if not isinstance(dm, dict) or not dm:
            continue
        metric_rows.append(
            {
                **dm,
                "source": str(r.get("source") or "unknown"),
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "sample_id": str(r.get("sample_id") or ""),
            }
        )

    if not metric_rows:
        raise RuntimeError("No baseline rows collected (teacher/model failure?)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_baseline_from_rows(str(teacher_model), metric_rows, out_path=out_path)

    # Attach build metadata (best-effort).
    try:
        obj = json.loads(out_path.read_text(encoding="utf-8"))
        obj["build_meta"] = {
            "source": "standardebooks_train_split",
            "teacher_model": str(teacher_model),
            "teacher_max_input_tokens": int(teacher_max_input_tokens),
            "baseline_max_samples": int(baseline_max_samples),
            "seed": int(seed),
            "created_at_unix": int(time.time()),
            "samples_path": str(samples_path),
        }
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"baseline_path": str(out_path), "n_rows": int(len(metric_rows)), "safe_model_id": safe_model_id(teacher_model)}


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 12,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def distill_standardebooks_remote(
    *,
    out_dir: str,
    corpus_dir: str,
    rebuild_corpus: bool,
    seed: int,
    max_chars: int,
    min_chars: int,
    max_books: int,
    excerpts_per_book: int,
    start_page: int,
    max_pages: int,
    sleep_s: float,
    teacher_model: str,
    teacher_max_input_tokens: int,
    baseline_max_samples: int,
    max_label_samples: int,
    base_model: str,
    scorer_max_length: int,
    scorer_batch_size: int,
    scorer_lr: float,
    scorer_weight_decay: float,
    scorer_epochs: int,
) -> str:
    _bootstrap_repo()
    from tools.studio.build_standardebooks_corpus import main as build_corpus_main
    from tools.studio.label_scorer_dataset import label_jsonl
    from tools.studio.train_scorer import train_scorer
    from tools.studio.baselines import safe_model_id

    corpus_root = Path(str(corpus_dir))
    if rebuild_corpus and corpus_root.exists():
        corpus_root = Path(f"{corpus_root}_{int(time.time())}")
    corpus_root.mkdir(parents=True, exist_ok=True)

    samples_path = corpus_root / "samples.jsonl"
    if rebuild_corpus or not samples_path.exists():
        build_args = [
            "--out-dir",
            str(corpus_root),
            "--seed",
            str(int(seed)),
            "--max-chars",
            str(int(max_chars)),
            "--min-chars",
            str(int(min_chars)),
            "--max-books",
            str(int(max_books)),
            "--excerpts-per-book",
            str(int(excerpts_per_book)),
            "--start-page",
            str(int(start_page)),
            "--max-pages",
            str(int(max_pages)),
            "--sleep-s",
            str(float(sleep_s)),
            "--normalize-text",
        ]
        build_corpus_main(build_args)

    splits = corpus_root / "splits"
    train_split = splits / "train.jsonl"
    if not train_split.exists():
        raise RuntimeError(f"Missing corpus split: {train_split}")

    baseline_path = Path(f"/vol/baselines/{safe_model_id(teacher_model)}_standardebooks_{int(teacher_max_input_tokens)}_docs.json")
    baseline_info = _build_baseline_from_split(
        samples_path=train_split,
        out_path=baseline_path,
        teacher_model=str(teacher_model),
        teacher_max_input_tokens=int(teacher_max_input_tokens),
        baseline_max_samples=int(baseline_max_samples),
        seed=int(seed),
    )

    label_root = corpus_root / f"distill_labels_{safe_model_id(teacher_model)}_{int(time.time())}"
    label_root.mkdir(parents=True, exist_ok=True)

    max_samples = int(max_label_samples) if int(max_label_samples) > 0 else None
    for split in ("train", "val", "test"):
        in_path = splits / f"{split}.jsonl"
        if not in_path.exists():
            continue
        n = label_jsonl(
            in_path=in_path,
            out_path=label_root / f"{split}.jsonl",
            max_samples=max_samples,
            seed=int(seed),
            teacher_model_id=str(teacher_model),
            baseline_model=str(baseline_path),
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
        positive_sources=("standardebooks_excerpt",),
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
    return json.dumps(
        {
            "corpus_dir": str(corpus_root),
            "baseline": baseline_info,
            "label_dir": str(label_root),
            "model_dir": str(out_dir_p),
            "train_summary": summary.__dict__,
        },
        ensure_ascii=False,
        indent=2,
    )


@app.local_entrypoint()
def main(
    out_dir: str,
    corpus_dir: str = "/vol/corpora/standardebooks_corpus_v1",
    rebuild_corpus: bool = False,
    seed: int = 1337,
    max_chars: int = 3800,
    min_chars: int = 900,
    max_books: int = 600,
    excerpts_per_book: int = 3,
    start_page: int = 1,
    max_pages: int = 80,
    sleep_s: float = 0.1,
    teacher_model: str = "gpt2",
    teacher_max_input_tokens: int = 512,
    baseline_max_samples: int = 1500,
    max_label_samples: int = 0,
    base_model: str = "distilbert-base-uncased",
    scorer_max_length: int = 384,
    scorer_batch_size: int = 16,
    scorer_lr: float = 2e-5,
    scorer_weight_decay: float = 0.01,
    scorer_epochs: int = 2,
) -> None:  # pragma: no cover
    print(
        distill_standardebooks_remote.remote(
            out_dir=str(out_dir),
            corpus_dir=str(corpus_dir),
            rebuild_corpus=bool(rebuild_corpus),
            seed=int(seed),
            max_chars=int(max_chars),
            min_chars=int(min_chars),
            max_books=int(max_books),
            excerpts_per_book=int(excerpts_per_book),
            start_page=int(start_page),
            max_pages=int(max_pages),
            sleep_s=float(sleep_s),
            teacher_model=str(teacher_model),
            teacher_max_input_tokens=int(teacher_max_input_tokens),
            baseline_max_samples=int(baseline_max_samples),
            max_label_samples=int(max_label_samples),
            base_model=str(base_model),
            scorer_max_length=int(scorer_max_length),
            scorer_batch_size=int(scorer_batch_size),
            scorer_lr=float(scorer_lr),
            scorer_weight_decay=float(scorer_weight_decay),
            scorer_epochs=int(scorer_epochs),
        )
    )
