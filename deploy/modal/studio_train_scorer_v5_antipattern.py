"""
Modal wrapper for training a single text→score scorer on benchmark v5 (anti-pattern aware).

So what:
- This is the same anti-pattern objective as the local `make train-scorer-v5-antipattern`
  command, but moved to Modal for practical Qwen3-sized fine-tuning.
- Use LoRA to keep memory/latency sane on GPU and make model swapping cheap.

Run:
  make setup-modal
  make modal-train-scorer-v5-antipattern-qwen3
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-train-scorer-v5-antipattern"
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


def _positive_sources() -> tuple[str, ...]:
    return ("gutenberg_top_excerpt",)


def _negative_sources() -> tuple[str, ...]:
    return (
        "gutenberg_random_excerpt",
        "gutenberg_corrupt_shuffle_sentences_global",
        "gutenberg_corrupt_shuffle_paragraphs",
        "gutenberg_corrupt_repeat_sentences",
        "gutenberg_corrupt_flatten",
        "llm_antipattern_write_like",
        "llm_antipattern_continue_from",
        "llm_antipattern_rewrite_from_memory",
    )


def _resolve_split(bench_dir: Path, name: str) -> Path:
    path = bench_dir / "splits" / f"{name}.jsonl"
    if not path.exists():
        raise RuntimeError(f"Missing benchmark split: {path}")
    return path


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 18,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def train_v5_remote(
    *,
    out_dir: str,
    bench_dir: str,
    base_model: str,
    seed: int,
    max_length: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    grad_accum_steps: int,
    eval_batch_size: int,
    trust_remote_code: bool,
    gradient_checkpointing: bool,
    bf16: bool,
    merge_lora: bool,
) -> str:
    _bootstrap_repo()
    from tools.studio.train_scorer import train_scorer
    from tools.studio.eval_scorer import eval_scorer

    bench_root = Path(str(bench_dir)).expanduser()
    if not bench_root.exists():
        raise RuntimeError(f"Missing benchmark root: {bench_root}")

    out_dir_p = Path(str(out_dir)).expanduser()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    train_path = _resolve_split(bench_root, "train")
    val_path = _resolve_split(bench_root, "val")
    test_path = _resolve_split(bench_root, "test")

    summary = train_scorer(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        out_dir=out_dir_p,
        base_model=str(base_model),
        doc_type="prose",
        normalize_text=True,
        positive_sources=_positive_sources(),
        negative_sources=_negative_sources(),
        label_key="label",
        max_length=int(max_length),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
        seed=int(seed),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
        trust_remote_code=bool(trust_remote_code),
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        grad_accum_steps=int(grad_accum_steps),
        gradient_checkpointing=bool(gradient_checkpointing),
        bf16=bool(bf16),
        merge_lora=bool(merge_lora),
    )

    eval_res, eval_meta = eval_scorer(
        model_path_or_id=str(out_dir_p),
        samples_path=test_path,
        positive_sources=_positive_sources(),
        negative_sources=None,
        doc_type="prose",
        normalize_text=True,
        max_length=int(max_length),
        batch_size=int(eval_batch_size),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    payload: Dict[str, Any] = {
        "out_dir": str(out_dir_p),
        "bench_dir": str(bench_root),
        "train_summary": summary.__dict__,
        "train_summary_lora": {
            "r": int(lora_r),
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
            "grad_accum_steps": int(grad_accum_steps),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "bf16": bool(bf16),
            "merge_lora": bool(merge_lora),
            "trust_remote_code": bool(trust_remote_code),
        },
        "eval_test": {"result": eval_res.__dict__, "meta": eval_meta},
    }
    data_vol.commit()
    hf_cache_vol.commit()
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.local_entrypoint()
def main(
    out_dir: str,
    bench_dir: str = "/vol/benchmarks/studio_benchmark_v5",
    base_model: str = "Qwen/Qwen3-1.7B",
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
    eval_batch_size: int = 2,
    trust_remote_code: bool = True,
    gradient_checkpointing: bool = True,
    bf16: bool = True,
    merge_lora: bool = True,
) -> None:  # pragma: no cover
    print(
        train_v5_remote.remote(
            out_dir=str(out_dir),
            bench_dir=str(bench_dir),
            base_model=str(base_model),
            seed=int(seed),
            max_length=int(max_length),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            epochs=int(epochs),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            grad_accum_steps=int(grad_accum_steps),
            eval_batch_size=int(eval_batch_size),
            trust_remote_code=bool(trust_remote_code),
            gradient_checkpointing=bool(gradient_checkpointing),
            bf16=bool(bf16),
            merge_lora=bool(merge_lora),
        )
    )
