"""
Modal wrapper: distill Horace's rubric score into a single fast scorer using *mixed* corpora.

So what:
- Distilling on literary-only positives (e.g. Standard Ebooks) produces a scorer with a narrow label
  range, which can make cross-domain separation weak.
- Mixing in modern prose (RSS/news/essays) + synthetic degradations creates a wider supervision
  range and improves robustness: text → score (0–100) in one model.

Run:
  make setup-modal
  make modal-token
  modal run deploy/modal/studio_distill_scorer_mixed.py --out-dir /vol/models/scorer_mixed_distilled
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-distill-scorer-mixed"
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
        "peft>=0.10.0",
    )
)
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")
if (_LOCAL_REPO_ROOT / "configs").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "configs", remote_path=f"{REPO_REMOTE_PATH}/configs")

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
    os.environ.setdefault("HORACE_HTTP_RETRIES", "3")
    os.environ.setdefault("HORACE_HTTP_RETRY_BASE_SLEEP_S", "1.2")
    os.environ.setdefault("HORACE_HTTP_RETRY_MAX_SLEEP_S", "30")


def _iter_jsonl(path: Path) -> List[dict]:
    from tools.studio.dataset_utils import iter_jsonl

    return list(iter_jsonl(path))


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    from tools.studio.dataset_utils import write_jsonl

    write_jsonl(path, rows)


def _gibberish_rows(n: int, *, seed: int, max_chars: int) -> List[dict]:
    rng = random.Random(int(seed) ^ 0xA5A5)
    out: List[dict] = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    punctuation = " ,.;:?!\n"
    for i in range(max(0, int(n))):
        chunks: List[str] = []
        while sum(len(c) for c in chunks) < max_chars:
            wlen = rng.randint(2, 12)
            word = "".join(rng.choice(alphabet) for _ in range(wlen))
            if rng.random() < 0.08:
                word = word.capitalize()
            chunks.append(word)
            chunks.append(rng.choice(punctuation))
        text = "".join(chunks)[:max_chars].strip()
        out.append(
            {
                "sample_id": f"gib_{i}_{abs(hash(text))%1000000}",
                "group_id": f"gibberish:{i}",
                "source": "gibberish_control",
                "title": f"gibberish_{i+1}",
                "url": "",
                "text": text,
                "fetched_at_unix": int(time.time()),
                "meta": {"license_hint": "synthetic"},
            }
        )
    return out


def _augment_corruptions(rows: List[dict], *, seed: int, kinds: List[str], max_per_sample: int) -> List[dict]:
    from tools.studio.text_corrupt import corrupt_text

    rng = random.Random(int(seed) ^ 0xC0FFEE)
    out: List[dict] = []
    per = max(0, int(max_per_sample))
    kinds = [k for k in (kinds or []) if str(k).strip()]
    if not kinds or per <= 0:
        return []

    for r in rows:
        if (r.get("source") or "") != "standardebooks_excerpt":
            continue
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        chosen = list(kinds)
        rng.shuffle(chosen)
        chosen = chosen[:per]
        for k in chosen:
            try:
                corr = corrupt_text(text, rng=rng, kind=str(k))
            except Exception:
                continue
            if not corr.strip() or corr == text:
                continue
            out.append(
                {
                    **r,
                    "sample_id": f"{r.get('sample_id')}_corrupt_{k}",
                    "source": f"standardebooks_corrupt_{k}",
                    "text": corr,
                    "meta": {**(r.get("meta") or {}), "corruption_kind": str(k)},
                }
            )
    return out


def _ensure_corpus(
    *,
    kind: str,
    out_dir: Path,
    args: List[str],
    builder_main,
) -> None:
    samples_path = out_dir / "samples.jsonl"
    if samples_path.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    builder_main(args)
    if not samples_path.exists():
        raise RuntimeError(f"{kind} corpus build failed (missing {samples_path})")


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

    rows = _iter_jsonl(samples_path)
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    if int(baseline_max_samples) > 0:
        rows = rows[: int(baseline_max_samples)]

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
    return {"baseline_path": str(out_path), "n_rows": int(len(metric_rows)), "safe_model_id": safe_model_id(teacher_model)}


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 12,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def distill_mixed_remote(
    *,
    out_dir: str,
    mixed_corpus_dir: str,
    standardebooks_dir: str,
    rss_dir: str,
    seed: int,
    teacher_model: str,
    teacher_max_input_tokens: int,
    baseline_max_samples: int,
    corruption_kinds: str,
    corruptions_per_sample: int,
    gibberish_train: int,
    base_model: str,
    scorer_max_length: int,
    scorer_batch_size: int,
    scorer_lr: float,
    scorer_weight_decay: float,
    scorer_epochs: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    grad_accum_steps: int,
    gradient_checkpointing: bool,
    bf16: bool,
    merge_lora: bool,
) -> str:
    _bootstrap_repo()
    from tools.studio.baselines import safe_model_id
    from tools.studio.label_scorer_dataset import label_jsonl
    from tools.studio.train_scorer import train_scorer
    from tools.studio.build_standardebooks_corpus import main as build_standardebooks_main
    from tools.studio.build_rss_corpus import main as build_rss_main

    se_root = Path(str(standardebooks_dir))
    rss_root = Path(str(rss_dir))
    mixed_root = Path(str(mixed_corpus_dir))
    mixed_root.mkdir(parents=True, exist_ok=True)

    _ensure_corpus(
        kind="standardebooks",
        out_dir=se_root,
        args=[
            "--out-dir",
            str(se_root),
            "--max-books",
            "240",
            "--max-pages",
            "30",
            "--excerpts-per-book",
            "2",
            "--sleep-s",
            "0.6",
            "--normalize-text",
        ],
        builder_main=build_standardebooks_main,
    )
    _ensure_corpus(
        kind="rss",
        out_dir=rss_root,
        args=[
            "--out-dir",
            str(rss_root),
            "--feeds-json",
            "configs/rss_feeds_v1.json",
            "--max-items-per-feed",
            "120",
            "--excerpts-per-item",
            "1",
            "--normalize-text",
        ],
        builder_main=build_rss_main,
    )

    se_splits = se_root / "splits"
    rss_splits = rss_root / "splits"
    mixed_splits = mixed_root / "splits"
    mixed_splits.mkdir(parents=True, exist_ok=True)

    kinds = [k.strip() for k in str(corruption_kinds or "").split(",") if k.strip()]
    split_counts: Dict[str, Dict[str, int]] = {}

    for split in ("train", "val", "test"):
        se_rows = _iter_jsonl(se_splits / f"{split}.jsonl")
        rss_rows = _iter_jsonl(rss_splits / f"{split}.jsonl")
        merged = list(se_rows) + list(rss_rows)

        # Only augment train: corruptions + gibberish widen the label range without tainting eval splits.
        if split == "train":
            merged.extend(_augment_corruptions(se_rows, seed=int(seed), kinds=kinds, max_per_sample=int(corruptions_per_sample)))
            merged.extend(_gibberish_rows(int(gibberish_train), seed=int(seed), max_chars=1200))

        out_path = mixed_splits / f"{split}.jsonl"
        _write_jsonl(out_path, merged)
        split_counts[split] = {"standardebooks": int(len(se_rows)), "rss": int(len(rss_rows)), "total": int(len(merged))}

    # Baseline from Standard Ebooks train split only.
    baseline_path = Path(f"/vol/baselines/{safe_model_id(teacher_model)}_mixed_{int(teacher_max_input_tokens)}_docs.json")
    baseline_info = _build_baseline_from_split(
        samples_path=se_splits / "train.jsonl",
        out_path=baseline_path,
        teacher_model=str(teacher_model),
        teacher_max_input_tokens=int(teacher_max_input_tokens),
        baseline_max_samples=int(baseline_max_samples),
        seed=int(seed),
    )

    label_root = mixed_root / f"distill_labels_{safe_model_id(teacher_model)}_{int(time.time())}"
    label_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        in_path = mixed_splits / f"{split}.jsonl"
        n = label_jsonl(
            in_path=in_path,
            out_path=label_root / f"{split}.jsonl",
            max_samples=None,
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
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        gradient_checkpointing=bool(gradient_checkpointing),
        bf16=bool(bf16),
        grad_accum_steps=int(grad_accum_steps),
        merge_lora=bool(merge_lora),
    )

    data_vol.commit()
    hf_cache_vol.commit()
    return json.dumps(
        {
            "standardebooks_dir": str(se_root),
            "rss_dir": str(rss_root),
            "mixed_corpus_dir": str(mixed_root),
            "split_counts": split_counts,
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
    out_dir: str = "/vol/models/scorer_mixed_distilled",
    mixed_corpus_dir: str = "/vol/corpora/mixed_corpus_v1",
    standardebooks_dir: str = "/vol/corpora/standardebooks_corpus_v1",
    rss_dir: str = "/vol/corpora/rss_corpus_v1",
    seed: int = 1337,
    teacher_model: str = "gpt2",
    teacher_max_input_tokens: int = 512,
    baseline_max_samples: int = 1200,
    corruption_kinds: str = "shuffle_sentences_global,shuffle_paragraphs,repeat_sentences,flatten",
    corruptions_per_sample: int = 2,
    gibberish_train: int = 120,
    base_model: str = "distilbert-base-uncased",
    scorer_max_length: int = 384,
    scorer_batch_size: int = 16,
    scorer_lr: float = 2e-5,
    scorer_weight_decay: float = 0.01,
    scorer_epochs: int = 2,
    lora_r: int = 0,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    grad_accum_steps: int = 1,
    gradient_checkpointing: bool = False,
    bf16: bool = False,
    merge_lora: bool = False,
) -> None:  # pragma: no cover
    print(
        distill_mixed_remote.remote(
            out_dir=str(out_dir),
            mixed_corpus_dir=str(mixed_corpus_dir),
            standardebooks_dir=str(standardebooks_dir),
            rss_dir=str(rss_dir),
            seed=int(seed),
            teacher_model=str(teacher_model),
            teacher_max_input_tokens=int(teacher_max_input_tokens),
            baseline_max_samples=int(baseline_max_samples),
            corruption_kinds=str(corruption_kinds),
            corruptions_per_sample=int(corruptions_per_sample),
            gibberish_train=int(gibberish_train),
            base_model=str(base_model),
            scorer_max_length=int(scorer_max_length),
            scorer_batch_size=int(scorer_batch_size),
            scorer_lr=float(scorer_lr),
            scorer_weight_decay=float(scorer_weight_decay),
            scorer_epochs=int(scorer_epochs),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            grad_accum_steps=int(grad_accum_steps),
            gradient_checkpointing=bool(gradient_checkpointing),
            bf16=bool(bf16),
            merge_lora=bool(merge_lora),
        )
    )
