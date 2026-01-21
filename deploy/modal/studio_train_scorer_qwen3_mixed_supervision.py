"""
Modal job: train a single Qwen3 text→score scorer with *mixed supervision*.

So what:
- Our best aligned fast scorer so far is trained on the held-out "great_author vs other_author" window corpus.
- Pure rubric distillation does NOT preserve that separation (domain/label mismatch).
- This job keeps the great/other split as an anchor while adding rubric-v2 teacher labels on mixed
  Standard Ebooks (high) + RSS/news (broad) + light corruptions (low), so the scorer generalizes
  beyond Gutenberg while staying aligned with "great writers vs others".

Outputs (written to /vol):
- Labeled teacher set: /vol/corpora/<teacher_corpus_dir>/labels_<teacher>_<tokens>/{train,val,test}.jsonl
- Trained scorer: /vol/models/<out_dir>

Run:
  make setup-modal
  make modal-token
  modal run deploy/modal/studio_train_scorer_qwen3_mixed_supervision.py --out-dir /vol/models/scorer_qwen3_mixed_supervision_v1
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

APP_NAME = "horace-studio-train-scorer-qwen3-mixed-supervision"
REPO_REMOTE_PATH = "/root/horace"


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists() and (p / "data").exists():
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
if (_LOCAL_REPO_ROOT / "data" / "corpora" / "mixed_windows_v1").exists():
    image = image.add_local_dir(
        _LOCAL_REPO_ROOT / "data" / "corpora" / "mixed_windows_v1",
        remote_path=f"{REPO_REMOTE_PATH}/data/corpora/mixed_windows_v1",
    )

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
    os.environ.setdefault("HORACE_TQDM_DISABLE", "1")


def _iter_jsonl(path: Path) -> List[dict]:
    from tools.studio.dataset_utils import iter_jsonl

    return list(iter_jsonl(path))


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    from tools.studio.dataset_utils import write_jsonl

    write_jsonl(path, rows)


def _ensure_corpus(*, out_dir: Path, args: List[str], builder_main) -> None:
    samples_path = out_dir / "samples.jsonl"
    if samples_path.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    builder_main(args)
    if not samples_path.exists():
        raise RuntimeError(f"Corpus build failed (missing {samples_path})")


def _repeat_rows(rows: List[dict], *, factor: int, seed: int) -> List[dict]:
    f = int(factor)
    if f <= 1 or not rows:
        return rows
    rng = random.Random(int(seed) ^ 0xBADA55)
    out: List[dict] = []
    for _ in range(f):
        chunk = list(rows)
        rng.shuffle(chunk)
        out.extend(chunk)
    return out


def _eval_teacher_regression(
    *,
    model_path_or_id: str,
    labeled_rows: List[dict],
    max_length: int,
    batch_size: int,
) -> Dict[str, Any]:
    import math

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from tools.studio.text_normalize import normalize_for_studio

    xs: List[str] = []
    ys: List[float] = []
    for r in labeled_rows:
        y = r.get("label")
        if not isinstance(y, (int, float)) or not math.isfinite(float(y)):
            continue
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        t, _ = normalize_for_studio(text, doc_type="prose", enabled=True)
        xs.append(t)
        ys.append(float(y))

    if not xs:
        return {"n": 0, "mse": None, "mae": None, "pearson": None}

    m = str(model_path_or_id)
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(m, trust_remote_code=True)
    if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
        model.config.pad_token_id = tok.pad_token_id
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model.to(device)
    model.eval()

    preds: List[float] = []
    with torch.no_grad():
        for i in range(0, len(xs), int(batch_size)):
            batch = xs[i : i + int(batch_size)]
            enc = tok(batch, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            p = torch.sigmoid(out.logits.squeeze(-1)).detach().cpu().numpy().reshape(-1)
            preds.extend([float(x) for x in p.tolist()])

    y = np.asarray(ys[: len(preds)], dtype=np.float64)
    p = np.asarray(preds[: len(ys)], dtype=np.float64)
    mse = float(np.mean((p - y) ** 2)) if y.size else None
    mae = float(np.mean(np.abs(p - y))) if y.size else None
    pearson = None
    if y.size >= 2:
        y0 = y - float(np.mean(y))
        p0 = p - float(np.mean(p))
        denom = float(np.sqrt(np.sum(y0 * y0) * np.sum(p0 * p0)))
        pearson = float(np.sum(y0 * p0) / denom) if denom > 0 else None
    return {"n": int(y.size), "mse": mse, "mae": mae, "pearson": pearson}


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 18,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def train_remote(cfg_json: str) -> str:
    _bootstrap_repo()
    cfg = json.loads(cfg_json)

    from tools.studio.baselines import safe_model_id
    from tools.studio.build_rss_corpus import main as build_rss_main
    from tools.studio.build_standardebooks_corpus import main as build_se_main
    from tools.studio.eval_scorer import eval_scorer
    from tools.studio.label_scorer_dataset import label_jsonl
    from tools.studio.text_corrupt import corrupt_text
    from tools.studio.train_scorer import train_scorer

    seed = int(cfg.get("seed") or 1337)
    teacher_model = str(cfg.get("teacher_model") or "gpt2")
    teacher_max_input_tokens = int(cfg.get("teacher_max_input_tokens") or 512)
    baseline_max_samples = int(cfg.get("baseline_max_samples") or 1200)
    label_version = str(cfg.get("label_version") or "rubricv2")

    se_root = Path(str(cfg.get("standardebooks_dir") or "/vol/corpora/standardebooks_corpus_v1"))
    rss_root = Path(str(cfg.get("rss_dir") or "/vol/corpora/rss_corpus_v1"))
    teacher_corpus_root = Path(str(cfg.get("teacher_corpus_dir") or "/vol/corpora/mixed_teacher_v1"))
    teacher_splits = teacher_corpus_root / "splits"
    teacher_splits.mkdir(parents=True, exist_ok=True)

    _ensure_corpus(
        out_dir=se_root,
        args=[
            "--out-dir",
            str(se_root),
            "--max-books",
            str(int(cfg.get("standardebooks_max_books") or 240)),
            "--max-pages",
            str(int(cfg.get("standardebooks_max_pages") or 30)),
            "--excerpts-per-book",
            str(int(cfg.get("standardebooks_excerpts_per_book") or 2)),
            "--sleep-s",
            str(float(cfg.get("standardebooks_sleep_s") or 0.6)),
            "--normalize-text",
        ],
        builder_main=build_se_main,
    )
    _ensure_corpus(
        out_dir=rss_root,
        args=[
            "--out-dir",
            str(rss_root),
            "--feeds-json",
            str(cfg.get("feeds_json") or "configs/rss_feeds_v1.json"),
            "--max-items-per-feed",
            str(int(cfg.get("rss_max_items_per_feed") or 120)),
            "--excerpts-per-item",
            str(int(cfg.get("rss_excerpts_per_item") or 1)),
            "--normalize-text",
        ],
        builder_main=build_rss_main,
    )
    data_vol.commit()

    # Build teacher corpus splits (SE + RSS) and add *train-only* corruptions to widen the score range.
    rng = random.Random(int(seed) ^ 0x1234)
    corruption_kinds = [k.strip() for k in str(cfg.get("corruption_kinds") or "flatten,drop_punct,repeat_sentences").split(",") if k.strip()]
    corruptions_per_sample = int(cfg.get("corruptions_per_sample") or 1)
    gibberish_train = int(cfg.get("gibberish_train") or 80)

    se_splits = se_root / "splits"
    rss_splits = rss_root / "splits"

    def _gibberish_rows(n: int, *, max_chars: int) -> List[dict]:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        punctuation = " ,.;:?!\n"
        out: List[dict] = []
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

    def _augment_corruptions(rows: List[dict]) -> List[dict]:
        if not corruption_kinds or corruptions_per_sample <= 0:
            return []
        out: List[dict] = []
        for r in rows:
            if str(r.get("source") or "") != "standardebooks_excerpt":
                continue
            text = str(r.get("text") or "")
            if not text.strip():
                continue
            chosen = list(corruption_kinds)
            rng.shuffle(chosen)
            chosen = chosen[: max(0, int(corruptions_per_sample))]
            for kind in chosen:
                try:
                    corr = corrupt_text(text, rng=rng, kind=str(kind))
                except Exception:
                    continue
                if not corr.strip() or corr == text:
                    continue
                out.append(
                    {
                        **r,
                        "sample_id": f"{r.get('sample_id')}_corrupt_{kind}",
                        "source": f"standardebooks_corrupt_{kind}",
                        "text": corr,
                        "meta": {**(r.get("meta") or {}), "corruption_kind": str(kind)},
                    }
                )
        return out

    teacher_counts: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        se_rows = _iter_jsonl(se_splits / f"{split}.jsonl")
        rss_rows = _iter_jsonl(rss_splits / f"{split}.jsonl")
        merged = list(se_rows) + list(rss_rows)
        if split == "train":
            merged.extend(_augment_corruptions(se_rows))
            merged.extend(_gibberish_rows(gibberish_train, max_chars=1200))
        out_path = teacher_splits / f"{split}.jsonl"
        _write_jsonl(out_path, merged)
        teacher_counts[split] = {"standardebooks": int(len(se_rows)), "rss": int(len(rss_rows)), "total": int(len(merged))}

    # Baseline from Standard Ebooks train split only (avoids tainting eval).
    baseline_path = Path(f"/vol/baselines/{safe_model_id(teacher_model)}_se_{teacher_max_input_tokens}_docs.json")
    if not baseline_path.exists():
        from tools.studio.analyze import analyze_text
        from tools.studio.baselines import build_baseline_from_rows

        rows = _iter_jsonl(se_splits / "train.jsonl")
        rng.shuffle(rows)
        if baseline_max_samples > 0:
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
            if isinstance(dm, dict) and dm:
                metric_rows.append(dm)
        if not metric_rows:
            raise RuntimeError("Failed to build baseline (no metric rows)")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        build_baseline_from_rows(str(teacher_model), metric_rows, out_path=baseline_path)
        data_vol.commit()
        hf_cache_vol.commit()

    labels_root = teacher_corpus_root / f"labels_{label_version}_{safe_model_id(teacher_model)}_{teacher_max_input_tokens}"
    labels_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        out_path = labels_root / f"{split}.jsonl"
        if out_path.exists():
            continue
        label_jsonl(
            in_path=teacher_splits / f"{split}.jsonl",
            out_path=out_path,
            max_samples=int(cfg.get("label_max_samples") or 0) or None,
            seed=int(seed),
            teacher_model_id=str(teacher_model),
            baseline_model=str(baseline_path),
            doc_type="prose",
            backend="hf",
            max_input_tokens=int(teacher_max_input_tokens),
            normalize_text=True,
            compute_cohesion=False,
        )
        data_vol.commit()
        hf_cache_vol.commit()

    # Mix supervision:
    # - Anchor: great_author vs other_author corpus (binary).
    # - Teacher: labeled SE+RSS (+ corruptions) corpus (soft label in [0,1]).
    go_root = Path(f"{REPO_REMOTE_PATH}/data/corpora/mixed_windows_v1/splits")
    go_train = _iter_jsonl(go_root / "train.jsonl")
    go_val = _iter_jsonl(go_root / "val.jsonl")
    go_test = _iter_jsonl(go_root / "test.jsonl")

    teacher_train = _iter_jsonl(labels_root / "train.jsonl")
    teacher_val = _iter_jsonl(labels_root / "val.jsonl")
    teacher_test = _iter_jsonl(labels_root / "test.jsonl")

    teacher_upsample = int(cfg.get("teacher_upsample") or 2)
    mixed_train = list(go_train) + _repeat_rows(teacher_train, factor=teacher_upsample, seed=seed)
    mixed_val = list(go_val) + list(teacher_val)
    mixed_root = Path(str(cfg.get("mixed_supervision_dir") or "/vol/corpora/mixed_supervision_v1"))
    mixed_splits = mixed_root / "splits"
    mixed_splits.mkdir(parents=True, exist_ok=True)
    _write_jsonl(mixed_splits / "train.jsonl", mixed_train)
    _write_jsonl(mixed_splits / "val.jsonl", mixed_val)

    # Keep held-out test *untainted* for the anchor task.
    _write_jsonl(mixed_splits / "test_great_other.jsonl", list(go_test))
    _write_jsonl(mixed_splits / "test_teacher.jsonl", list(teacher_test))
    data_vol.commit()

    out_dir = Path(str(cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    init_model = str(cfg.get("init_model") or "/vol/models/scorer_qwen3_great_other_v1")
    max_length = int(cfg.get("max_length") or 512)
    batch_size = int(cfg.get("batch_size") or 1)
    lr = float(cfg.get("lr") or 8e-5)
    epochs = int(cfg.get("epochs") or 1)

    lora_r = int(cfg.get("lora_r") or 16)
    lora_alpha = int(cfg.get("lora_alpha") or 32)
    lora_dropout = float(cfg.get("lora_dropout") or 0.05)
    grad_accum_steps = int(cfg.get("grad_accum_steps") or 16)

    train_summary = train_scorer(
        train_path=mixed_splits / "train.jsonl",
        val_path=mixed_splits / "val.jsonl",
        test_path=mixed_splits / "test_great_other.jsonl",
        out_dir=out_dir,
        base_model=str(init_model),
        doc_type="prose",
        normalize_text=True,
        positive_sources=("great_author",),
        negative_sources=("other_author",),
        label_key="label",
        max_length=max_length,
        batch_size=batch_size,
        lr=lr,
        weight_decay=float(cfg.get("weight_decay") or 0.01),
        epochs=epochs,
        seed=seed,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
        trust_remote_code=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        bf16=bool(cfg.get("bf16", True)),
        grad_accum_steps=grad_accum_steps,
        merge_lora=True,
    )

    # Held-out anchor eval (great vs other).
    auc_res, auc_meta = eval_scorer(
        model_path_or_id=str(out_dir),
        samples_path=go_root / "test.jsonl",
        positive_sources=("great_author",),
        negative_sources=("other_author",),
        doc_type="prose",
        normalize_text=True,
        max_length=max_length,
        batch_size=int(cfg.get("eval_batch_size") or 2),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    # Teacher-set diagnostic (does the model track the rubric label at all?).
    teacher_eval = _eval_teacher_regression(
        model_path_or_id=str(out_dir),
        labeled_rows=teacher_test,
        max_length=max_length,
        batch_size=int(cfg.get("teacher_eval_batch_size") or 2),
    )

    data_vol.commit()
    hf_cache_vol.commit()
    return json.dumps(
        {
            "out_dir": str(out_dir),
            "init_model": str(init_model),
            "great_other_corpus": str(go_root.parent),
            "teacher_corpus_dir": str(teacher_corpus_root),
            "teacher_split_counts": teacher_counts,
            "baseline_path": str(baseline_path),
            "labels_dir": str(labels_root),
            "mixed_supervision_dir": str(mixed_root),
            "mixed_counts": {
                "great_other_train": int(len(go_train)),
                "teacher_train": int(len(teacher_train)),
                "teacher_upsample": int(teacher_upsample),
                "mixed_train": int(len(mixed_train)),
                "great_other_val": int(len(go_val)),
                "teacher_val": int(len(teacher_val)),
                "mixed_val": int(len(mixed_val)),
                "great_other_test": int(len(go_test)),
                "teacher_test": int(len(teacher_test)),
            },
            "train_summary": train_summary.__dict__,
            "eval_great_other_test": {"result": auc_res.__dict__, "meta": auc_meta},
            "eval_teacher_test": teacher_eval,
        },
        ensure_ascii=False,
        indent=2,
    )


@app.local_entrypoint()
def main(  # pragma: no cover
    out_dir: str = "/vol/models/scorer_qwen3_mixed_supervision_v1",
    init_model: str = "/vol/models/scorer_qwen3_great_other_v1",
    seed: int = 1337,
    teacher_model: str = "gpt2",
    teacher_max_input_tokens: int = 512,
    baseline_max_samples: int = 1200,
    feeds_json: str = "configs/rss_feeds_v1.json",
    rss_max_items_per_feed: int = 120,
    rss_excerpts_per_item: int = 1,
    standardebooks_max_books: int = 240,
    standardebooks_max_pages: int = 30,
    standardebooks_excerpts_per_book: int = 2,
    standardebooks_sleep_s: float = 0.6,
    corruption_kinds: str = "flatten,drop_punct,repeat_sentences",
    corruptions_per_sample: int = 1,
    gibberish_train: int = 80,
    teacher_upsample: int = 2,
    label_max_samples: int = 0,
    label_version: str = "rubricv2",
    mixed_supervision_dir: str = "/vol/corpora/mixed_supervision_v1",
    teacher_corpus_dir: str = "/vol/corpora/mixed_teacher_v1",
    standardebooks_dir: str = "/vol/corpora/standardebooks_corpus_v1",
    rss_dir: str = "/vol/corpora/rss_corpus_v1",
    max_length: int = 512,
    batch_size: int = 1,
    lr: float = 8e-5,
    weight_decay: float = 0.01,
    epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    grad_accum_steps: int = 16,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    eval_batch_size: int = 2,
    teacher_eval_batch_size: int = 2,
) -> None:
    cfg = {
        "out_dir": str(out_dir),
        "init_model": str(init_model),
        "seed": int(seed),
        "teacher_model": str(teacher_model),
        "teacher_max_input_tokens": int(teacher_max_input_tokens),
        "baseline_max_samples": int(baseline_max_samples),
        "feeds_json": str(feeds_json),
        "rss_max_items_per_feed": int(rss_max_items_per_feed),
        "rss_excerpts_per_item": int(rss_excerpts_per_item),
        "standardebooks_max_books": int(standardebooks_max_books),
        "standardebooks_max_pages": int(standardebooks_max_pages),
        "standardebooks_excerpts_per_book": int(standardebooks_excerpts_per_book),
        "standardebooks_sleep_s": float(standardebooks_sleep_s),
        "corruption_kinds": str(corruption_kinds),
        "corruptions_per_sample": int(corruptions_per_sample),
        "gibberish_train": int(gibberish_train),
        "teacher_upsample": int(teacher_upsample),
        "label_max_samples": int(label_max_samples),
        "label_version": str(label_version),
        "mixed_supervision_dir": str(mixed_supervision_dir),
        "teacher_corpus_dir": str(teacher_corpus_dir),
        "standardebooks_dir": str(standardebooks_dir),
        "rss_dir": str(rss_dir),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "grad_accum_steps": int(grad_accum_steps),
        "bf16": bool(bf16),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "eval_batch_size": int(eval_batch_size),
        "teacher_eval_batch_size": int(teacher_eval_batch_size),
    }
    print(train_remote.remote(json.dumps(cfg)))
