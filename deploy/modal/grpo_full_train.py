"""
Modal wrapper for running Horace GRPO full training on GPU.

This reuses `tools/grpo_full_train.py` and persists outputs to a Modal Volume.

Setup (outside this repo):
  pip install modal
  modal token new

Run (example):
  modal run deploy/modal/grpo_full_train.py --config configs/grpo_full_hf_smoke.json --out-dir /vol/grpo_runs/demo
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-grpo-full-train"
REPO_REMOTE_PATH = "/root/horace"

def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists() and (p / "configs").exists():
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


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 60 * 6,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def train_remote(config_json: str, *, out_dir: str) -> str:
    _bootstrap_repo()
    cfg = json.loads(config_json)
    cfg["backend"] = "hf"
    cfg["out_dir"] = out_dir
    # Guardrail: adapter configs often have huge LRs (e.g., 0.05) which will explode full-weight training.
    try:
        lr = float(cfg.get("train", {}).get("lr", 0.0))
        if lr > 1e-3:
            cfg.setdefault("train", {})["lr"] = 5e-6
    except Exception:
        pass
    cfg_path = Path("/tmp/grpo_full_train_config.json")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    from tools.grpo_full_train import train_full_grpo

    out = train_full_grpo(cfg_path)
    data_vol.commit()
    hf_cache_vol.commit()
    return str(out)


@app.local_entrypoint()
def main(config: str, out_dir: str, limit_steps: Optional[int] = None) -> None:  # pragma: no cover
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    if limit_steps is not None:
        cfg.setdefault("train", {})["steps"] = int(limit_steps)
    res = train_remote.remote(json.dumps(cfg), out_dir=str(out_dir))
    print(res)
