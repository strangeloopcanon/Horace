"""
Modal job: train a tiny calibrator over scorer head probabilities.

So what: calibrates a multi-head scorer into a single "vs great authors" score.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
from dataclasses import asdict


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-train-head-calibrator"
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


@app.function(image=image, gpu="any", timeout=60 * 60 * 6, volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol})
def train_remote(cfg_json: str) -> Dict[str, Any]:
    _bootstrap_repo()
    cfg = json.loads(cfg_json)

    from tools.studio.train_head_calibrator import train_head_calibrator

    report = train_head_calibrator(
        model_path_or_id=str(cfg["model"]),
        train_path=Path(str(cfg["train"])),
        test_path=Path(str(cfg["test"])),
        out_path=Path(str(cfg["out"])),
        positive_sources=tuple(cfg.get("pos_sources") or ["great_author"]),
        negative_sources=tuple(cfg.get("neg_sources") or ["other_author"]),
        max_length=int(cfg.get("max_length") or 512),
        batch_size=int(cfg.get("batch_size") or 24),
        doc_type=str(cfg.get("doc_type") or "prose"),
        normalize_text=bool(cfg.get("normalize_text", True)),
        l2=float(cfg.get("l2") or 1e-2),
        lr=float(cfg.get("lr") or 0.5),
        steps=int(cfg.get("steps") or 600),
        seed=int(cfg.get("seed") or 1337),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )

    data_vol.commit()
    hf_cache_vol.commit()
    return {"report": asdict(report)}


@app.local_entrypoint()
def main(  # pragma: no cover
    model: str = "/vol/models/scorer_qwen3_multihead_v7b_rubricv3_bigteacher",
    out_dir: str = "/vol/calibrators",
    train: str = "data/corpora/mixed_windows_v1/splits/train.jsonl",
    test: str = "data/corpora/mixed_windows_v1/splits/test.jsonl",
    doc_type: str = "prose",
    max_length: int = 512,
    batch_size: int = 24,
    normalize_text: bool = True,
    pos: str = "great_author",
    neg: str = "other_author",
    l2: float = 1e-2,
    lr: float = 0.5,
    steps: int = 600,
    seed: int = 1337,
) -> None:
    out_path = Path(str(out_dir)) / "scorer_head_calibrator.json"
    cfg = {
        "model": str(model),
        "train": str(train),
        "test": str(test),
        "out": str(out_path),
        "doc_type": str(doc_type),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "normalize_text": bool(normalize_text),
        "pos_sources": [s.strip() for s in str(pos).split(",") if s.strip()],
        "neg_sources": [s.strip() for s in str(neg).split(",") if s.strip()],
        "l2": float(l2),
        "lr": float(lr),
        "steps": int(steps),
        "seed": int(seed),
    }
    report = train_remote.remote(json.dumps(cfg))
    print(json.dumps(report, ensure_ascii=False, indent=2))
