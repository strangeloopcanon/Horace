"""
Modal job: evaluate a trained scorer model on an eval/benchmark JSONL split.

So what: lets us sanity-check that the distilled scorer separates "literary" vs
non-literary sources on a held-out split without running the slow rubric.

Example:
  modal run deploy/modal/studio_eval_trained_scorer.py \\
    --model /vol/models/scorer_standardebooks_distilled \\
    --samples data/eval_sets/studio_fixed_v1_splits/test.jsonl \\
    --pos gutenberg_excerpt
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-eval-trained-scorer"
REPO_REMOTE_PATH = "/root/horace"


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists() and (p / "data").exists():
            return p
    return Path.cwd()


_LOCAL_REPO_ROOT = _local_repo_root()

data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("horace-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu121", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy>=1.24.0", "transformers>=4.40.0", "safetensors>=0.4.0")
)
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")
if (_LOCAL_REPO_ROOT / "data" / "eval_sets").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "data" / "eval_sets", remote_path=f"{REPO_REMOTE_PATH}/data/eval_sets")
if (_LOCAL_REPO_ROOT / "data" / "benchmarks").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "data" / "benchmarks", remote_path=f"{REPO_REMOTE_PATH}/data/benchmarks")
if (_LOCAL_REPO_ROOT / "data" / "corpora" / "mixed_windows_v1").exists():
    image = image.add_local_dir(
        _LOCAL_REPO_ROOT / "data" / "corpora" / "mixed_windows_v1",
        remote_path=f"{REPO_REMOTE_PATH}/data/corpora/mixed_windows_v1",
    )

app = modal.App(APP_NAME)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HF_HOME", "/cache/hf")


@app.function(image=image, gpu="any", timeout=60 * 30, volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol})
def eval_remote(cfg_json: str) -> Dict[str, Any]:
    _bootstrap_repo()
    cfg = json.loads(cfg_json)

    from tools.studio.eval_scorer import eval_scorer

    res, meta = eval_scorer(
        model_path_or_id=str(cfg["model"]),
        samples_path=Path(str(cfg["samples"])),
        positive_sources=tuple(cfg.get("pos_sources") or ["gutenberg_excerpt"]),
        negative_sources=tuple(cfg.get("neg_sources") or []) or None,
        doc_type=str(cfg.get("doc_type") or "prose"),
        normalize_text=bool(cfg.get("normalize_text", True)),
        max_length=int(cfg.get("max_length") or 384),
        batch_size=int(cfg.get("batch_size") or 24),
        device=None,
    )

    hf_cache_vol.commit()
    return {"result": res.__dict__, "meta": meta}


@app.local_entrypoint()
def main(  # pragma: no cover
    out_dir: str = "reports/studio_eval_trained_scorer_modal",
    model: str = "/vol/models/scorer_standardebooks_distilled",
    samples: str = "data/eval_sets/studio_fixed_v1_splits/test.jsonl",
    doc_type: str = "prose",
    max_length: int = 384,
    batch_size: int = 24,
    normalize_text: bool = True,
    pos: str = "gutenberg_excerpt",
    neg: str = "",
) -> None:
    cfg = {
        "model": str(model),
        "samples": str(samples),
        "doc_type": str(doc_type),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "normalize_text": bool(normalize_text),
        "pos_sources": [s.strip() for s in str(pos).split(",") if s.strip()],
        "neg_sources": [s.strip() for s in str(neg).split(",") if s.strip()],
    }
    report = eval_remote.remote(json.dumps(cfg))

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = out_base / "eval_trained_scorer_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote report: {out_path}")
