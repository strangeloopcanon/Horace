"""
Modal pipeline: train a single text→score model with *hybrid supervision*.

So what:
- The token/logit rubric is diagnostic and useful, but may not fully capture "literary quality".
- We keep it as one supervision source (teacher distillation), and add within-content
  preference pairs (original > dulled rewrite/cadence corruption) so the scorer learns
  cadence/voice directly without relying on topic/domain shortcuts.

Outputs (written to /vol):
- Distilled scorer (rubric teacher → encoder): /vol/models/<distilled_dir>
- Preference fine-tuned scorer: /vol/models/<out_dir>
- Preference pairs: /vol/corpora/<pairs_dir>/{train,val,test}.jsonl

Run:
  make setup-modal
  make modal-token
  modal run deploy/modal/studio_train_scorer_hybrid.py --out-dir /vol/models/scorer_hybrid_v1

Note: this job pins `torch==2.5.1+cu121` on Modal. Transformers blocks loading `.bin` weights with
torch<2.6 (CVE-2025-32434), so pick a `base_model` that ships `model.safetensors` (e.g. roberta/distilbert).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-train-scorer-hybrid"
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


def _ensure_corpus(*, out_dir: Path, args: List[str], builder_main) -> None:
    samples_path = out_dir / "samples.jsonl"
    if samples_path.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    builder_main(args)
    if not samples_path.exists():
        raise RuntimeError(f"Corpus build failed (missing {samples_path})")


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 18,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def train_hybrid_remote(cfg_json: str) -> str:
    _bootstrap_repo()
    cfg = json.loads(cfg_json)

    from tools.studio.build_rss_corpus import main as build_rss_main
    from tools.studio.build_standardebooks_corpus import main as build_se_main
    from tools.studio.label_scorer_dataset import label_jsonl
    from tools.studio.train_preference_scorer import train_preference_scorer
    from tools.studio.train_scorer import train_scorer

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        if v is None:
            return int(default)
        return int(v)

    se_root = Path(str(cfg["standardebooks_dir"]))
    rss_root = Path(str(cfg["rss_dir"]))
    mixed_root = Path(str(cfg["mixed_corpus_dir"]))
    pairs_root = Path(str(cfg["pairs_dir"]))

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
    hf_cache_vol.commit()

    # Build mixed split JSONLs (SE + RSS); keep train-only augmentations out of val/test.
    se_splits = se_root / "splits"
    rss_splits = rss_root / "splits"
    mixed_splits = mixed_root / "splits"
    mixed_splits.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        merged = _iter_jsonl(se_splits / f"{split}.jsonl") + _iter_jsonl(rss_splits / f"{split}.jsonl")
        _write_jsonl(mixed_splits / f"{split}.jsonl", merged)
    data_vol.commit()

    teacher_model = str(cfg.get("teacher_model") or "Qwen/Qwen2.5-1.5B")
    teacher_max_input_tokens = int(cfg.get("teacher_max_input_tokens") or 512)
    base_model = str(cfg.get("base_model") or "roberta-base")

    # Baseline from Standard Ebooks train split only (avoids tainted eval).
    from tools.studio.baselines import safe_model_id

    baseline_path = Path(f"/vol/baselines/{safe_model_id(teacher_model)}_se_{teacher_max_input_tokens}_docs.json")
    if not baseline_path.exists():
        from tools.studio.analyze import analyze_text
        from tools.studio.baselines import build_baseline_from_rows

        se_train = _iter_jsonl(se_splits / "train.jsonl")
        # Cap baseline rows for time/cost.
        max_baseline = _int_cfg("baseline_max_samples", 600)
        se_train = se_train[: max(0, max_baseline)] if max_baseline > 0 else se_train

        metric_rows: List[Dict[str, Any]] = []
        for r in se_train:
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
            metric_rows.append(dm)
        if not metric_rows:
            raise RuntimeError("Failed to build baseline (no metric rows)")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        build_baseline_from_rows(str(teacher_model), metric_rows, out_path=baseline_path)
        data_vol.commit()
        hf_cache_vol.commit()

    label_root = mixed_root / f"hybrid_labels_{safe_model_id(teacher_model)}_{int(time.time())}"
    label_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        in_path = mixed_splits / f"{split}.jsonl"
        out_path = label_root / f"{split}.jsonl"
        if out_path.exists():
            continue
        label_jsonl(
            in_path=in_path,
            out_path=out_path,
            max_samples=int(cfg.get("label_max_samples") or 0) or None,
            seed=int(cfg.get("seed") or 1337),
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

    distilled_dir = Path(str(cfg.get("distilled_dir") or "/vol/models/scorer_distilled_tmp"))
    distilled_dir.mkdir(parents=True, exist_ok=True)
    train_scorer(
        train_path=label_root / "train.jsonl",
        val_path=label_root / "val.jsonl",
        test_path=label_root / "test.jsonl",
        out_dir=distilled_dir,
        base_model=str(base_model),
        doc_type="prose",
        normalize_text=True,
        positive_sources=("standardebooks_excerpt",),
        negative_sources=None,
        label_key="label",
        max_length=int(cfg.get("scorer_max_length") or 384),
        batch_size=int(cfg.get("scorer_batch_size") or 16),
        lr=float(cfg.get("scorer_lr") or 2e-5),
        weight_decay=float(cfg.get("scorer_weight_decay") or 0.01),
        epochs=int(cfg.get("scorer_epochs") or 2),
        seed=int(cfg.get("seed") or 1337),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )
    data_vol.commit()

    # Build preference pairs from Standard Ebooks splits (within-content).
    from tools.studio.build_dull_rewrite_pairs import build_dull_rewrite_pairs

    pairs_root.mkdir(parents=True, exist_ok=True)
    pairs: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        out_pairs = pairs_root / f"{split}.jsonl"
        if out_pairs.exists():
            pairs[split] = str(out_pairs)
            continue
        res = build_dull_rewrite_pairs(
            in_path=se_splits / f"{split}.jsonl",
            out_path=out_pairs,
            seed=int(cfg.get("seed") or 1337),
            rewrite_model_id=str(cfg.get("rewrite_model") or "Qwen/Qwen2.5-0.5B-Instruct"),
            doc_type="prose",
            strength=str(cfg.get("dull_strength") or "mild"),
            rewrites_per_sample=_int_cfg("rewrites_per_sample", 1),
            max_samples=_int_cfg("pairs_max_samples_per_split", 80),
            corruption_kinds=tuple(str(cfg.get("corruption_kinds") or "flatten,drop_punct").split(",")),
            corruptions_per_sample=_int_cfg("corruptions_per_sample", 1),
        )
        pairs[split] = str(res["pairs_path"])
        data_vol.commit()

    # Fine-tune with pairwise ranking, starting from the distilled scorer.
    out_dir = Path(str(cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    pref_summary = train_preference_scorer(
        train_pairs_path=Path(pairs["train"]),
        val_pairs_path=Path(pairs["val"]),
        test_pairs_path=Path(pairs["test"]),
        out_dir=out_dir,
        base_model=str(base_model),
        init_model=str(distilled_dir),
        doc_type="prose",
        normalize_text=True,
        max_length=int(cfg.get("pref_max_length") or 384),
        batch_size=int(cfg.get("pref_batch_size") or 12),
        lr=float(cfg.get("pref_lr") or 1e-5),
        weight_decay=float(cfg.get("pref_weight_decay") or 0.01),
        epochs=int(cfg.get("pref_epochs") or 1),
        seed=int(cfg.get("seed") or 1337),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    # Evaluate on mixed test split (diagnostic): SE vs non-SE.
    from tools.studio.eval_scorer import eval_scorer

    mixed_test = mixed_splits / "test.jsonl"
    eval_res, eval_meta = eval_scorer(
        model_path_or_id=str(out_dir),
        samples_path=mixed_test,
        positive_sources=("standardebooks_excerpt",),
        negative_sources=None,
        doc_type="prose",
        normalize_text=True,
        max_length=int(cfg.get("pref_max_length") or 384),
        batch_size=int(cfg.get("pref_batch_size") or 12),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    payload = {
        "out_dir": str(out_dir),
        "distilled_dir": str(distilled_dir),
        "baseline_path": str(baseline_path),
        "label_dir": str(label_root),
        "pairs_dir": str(pairs_root),
        "pairs": pairs,
        "preference_train_summary": asdict(pref_summary),
        "eval_mixed_test": {"result": eval_res.__dict__, "meta": eval_meta},
    }

    data_vol.commit()
    hf_cache_vol.commit()
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.local_entrypoint()
def main(
    out_dir: str = "/vol/models/scorer_hybrid_v1",
    distilled_dir: str = "/vol/models/scorer_hybrid_distilled_tmp",
    mixed_corpus_dir: str = "/vol/corpora/mixed_corpus_hybrid_v1",
    standardebooks_dir: str = "/vol/corpora/standardebooks_corpus_v1",
    rss_dir: str = "/vol/corpora/rss_corpus_v1",
    pairs_dir: str = "/vol/corpora/dull_pairs_v1",
    feeds_json: str = "configs/rss_feeds_v1.json",
    seed: int = 1337,
    teacher_model: str = "Qwen/Qwen2.5-1.5B",
    teacher_max_input_tokens: int = 512,
    base_model: str = "roberta-base",
    baseline_max_samples: int = 600,
    standardebooks_max_books: int = 240,
    standardebooks_max_pages: int = 30,
    standardebooks_excerpts_per_book: int = 2,
    standardebooks_sleep_s: float = 0.6,
    rss_max_items_per_feed: int = 120,
    rss_excerpts_per_item: int = 1,
    rewrite_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dull_strength: str = "mild",
    rewrites_per_sample: int = 1,
    pairs_max_samples_per_split: int = 80,
    corruption_kinds: str = "flatten,drop_punct",
    corruptions_per_sample: int = 1,
    scorer_epochs: int = 2,
    pref_epochs: int = 1,
) -> None:  # pragma: no cover
    cfg = {
        "out_dir": str(out_dir),
        "distilled_dir": str(distilled_dir),
        "mixed_corpus_dir": str(mixed_corpus_dir),
        "standardebooks_dir": str(standardebooks_dir),
        "rss_dir": str(rss_dir),
        "pairs_dir": str(pairs_dir),
        "feeds_json": str(feeds_json),
        "seed": int(seed),
        "teacher_model": str(teacher_model),
        "teacher_max_input_tokens": int(teacher_max_input_tokens),
        "base_model": str(base_model),
        "baseline_max_samples": int(baseline_max_samples),
        "standardebooks_max_books": int(standardebooks_max_books),
        "standardebooks_max_pages": int(standardebooks_max_pages),
        "standardebooks_excerpts_per_book": int(standardebooks_excerpts_per_book),
        "standardebooks_sleep_s": float(standardebooks_sleep_s),
        "rss_max_items_per_feed": int(rss_max_items_per_feed),
        "rss_excerpts_per_item": int(rss_excerpts_per_item),
        "rewrite_model": str(rewrite_model),
        "dull_strength": str(dull_strength),
        "rewrites_per_sample": int(rewrites_per_sample),
        "pairs_max_samples_per_split": int(pairs_max_samples_per_split),
        "corruption_kinds": str(corruption_kinds),
        "corruptions_per_sample": int(corruptions_per_sample),
        "scorer_epochs": int(scorer_epochs),
        "pref_epochs": int(pref_epochs),
    }
    print(train_hybrid_remote.remote(json.dumps(cfg)))
