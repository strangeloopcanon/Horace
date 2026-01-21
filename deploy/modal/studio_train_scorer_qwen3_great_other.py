"""
Modal job: train a single text→score scorer using Qwen3 + LoRA on the "great_author vs other_author" corpus.

So what:
- This is the "single model scorer" path: one forward pass → score.
- We fine-tune Qwen3 as a sequence-classification regressor (sigmoid → [0,1]) with LoRA to keep it feasible.

Run:
  make setup-modal
  modal run deploy/modal/studio_train_scorer_qwen3_great_other.py --out-dir /vol/models/scorer_qwen3_great_other_v1
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-train-scorer-qwen3-great-other"
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
if (_LOCAL_REPO_ROOT / "data" / "corpora" / "mixed_windows_v1").exists():
    image = image.add_local_dir(
        _LOCAL_REPO_ROOT / "data" / "corpora" / "mixed_windows_v1",
        remote_path=f"{REPO_REMOTE_PATH}/data/corpora/mixed_windows_v1",
    )
if (_LOCAL_REPO_ROOT / "data" / "benchmarks" / "studio_benchmark_v3").exists():
    image = image.add_local_dir(
        _LOCAL_REPO_ROOT / "data" / "benchmarks" / "studio_benchmark_v3",
        remote_path=f"{REPO_REMOTE_PATH}/data/benchmarks/studio_benchmark_v3",
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


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 12,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def train_remote(cfg_json: str) -> str:
    _bootstrap_repo()
    cfg = json.loads(cfg_json)

    from tools.studio.eval_scorer import eval_scorer
    from tools.studio.train_scorer import train_scorer

    corpus_dir = Path(str(cfg.get("corpus_dir") or f"{REPO_REMOTE_PATH}/data/corpora/mixed_windows_v1"))
    splits = corpus_dir / "splits"
    train_path = splits / "train.jsonl"
    val_path = splits / "val.jsonl"
    test_path = splits / "test.jsonl"
    if not train_path.exists():
        raise RuntimeError(f"Missing corpus train split: {train_path}")

    out_dir = Path(str(cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = str(cfg.get("base_model") or "Qwen/Qwen3-1.7B")
    max_length = int(cfg.get("max_length") or 512)
    batch_size = int(cfg.get("batch_size") or 1)
    lr = float(cfg.get("lr") or 1e-4)
    epochs = int(cfg.get("epochs") or 1)
    seed = int(cfg.get("seed") or 1337)

    lora_r = int(cfg.get("lora_r") or 16)
    lora_alpha = int(cfg.get("lora_alpha") or 32)
    lora_dropout = float(cfg.get("lora_dropout") or 0.05)
    grad_accum_steps = int(cfg.get("grad_accum_steps") or 16)

    summary = train_scorer(
        train_path=train_path,
        val_path=val_path if val_path.exists() else None,
        test_path=test_path if test_path.exists() else None,
        out_dir=out_dir,
        base_model=base_model,
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
    data_vol.commit()
    hf_cache_vol.commit()

    eval_res, eval_meta = eval_scorer(
        model_path_or_id=str(out_dir),
        samples_path=test_path,
        positive_sources=("great_author",),
        negative_sources=("other_author",),
        doc_type="prose",
        normalize_text=True,
        max_length=max_length,
        batch_size=int(cfg.get("eval_batch_size") or 2),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    bench_test = Path(f"{REPO_REMOTE_PATH}/data/benchmarks/studio_benchmark_v3/splits/test.jsonl")
    bench_eval = None
    if bench_test.exists():
        r, m = eval_scorer(
            model_path_or_id=str(out_dir),
            samples_path=bench_test,
            positive_sources=("gutenberg_excerpt",),
            negative_sources=None,
            doc_type="prose",
            normalize_text=True,
            max_length=max_length,
            batch_size=int(cfg.get("eval_batch_size") or 2),
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
        )
        bench_eval = {"result": r.__dict__, "meta": m}

    payload = {
        "out_dir": str(out_dir),
        "corpus_dir": str(corpus_dir),
        "train_summary": summary.__dict__,
        "eval_test": {"result": eval_res.__dict__, "meta": eval_meta},
        "eval_benchmark_v3_test": bench_eval,
    }
    data_vol.commit()
    hf_cache_vol.commit()
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.local_entrypoint()
def main(
    out_dir: str,
    base_model: str = "Qwen/Qwen3-1.7B",
    corpus_dir: str = "",
    seed: int = 1337,
    max_length: int = 512,
    batch_size: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    grad_accum_steps: int = 16,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    eval_batch_size: int = 2,
) -> None:  # pragma: no cover
    cfg = {
        "out_dir": str(out_dir),
        "base_model": str(base_model),
        "corpus_dir": str(corpus_dir) if str(corpus_dir).strip() else None,
        "seed": int(seed),
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
    }
    print(train_remote.remote(json.dumps(cfg)))

