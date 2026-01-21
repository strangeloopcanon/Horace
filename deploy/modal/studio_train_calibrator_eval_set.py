"""
Modal job: score a fixed eval set, then train a tiny learned calibrator.

Run (first time):
  make setup-modal
  modal token new

Run:
  modal run deploy/modal/studio_train_calibrator_eval_set.py --out-dir reports/calibrators_eval_set --samples data/eval_sets/studio_fixed_v1.jsonl

Train from a split of the larger benchmark snapshot:
  modal run deploy/modal/studio_train_calibrator_eval_set.py --samples data/benchmarks/studio_benchmark_v3/splits/train.jsonl --out-dir reports/calibrators_benchmark_modal
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

APP_NAME = "horace-studio-train-calibrator-eval-set"
REPO_REMOTE_PATH = "/root/horace"


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists() and (p / "data").exists():
            return p
    return Path.cwd()


_LOCAL_REPO_ROOT = _local_repo_root()

hf_cache_vol = modal.Volume.from_name("horace-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu121", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy>=1.24.0",
        "transformers>=4.40.0",
        "sentencepiece>=0.2.0",
        "safetensors>=0.4.0",
    )
)
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")
if (_LOCAL_REPO_ROOT / "data" / "baselines").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "data" / "baselines", remote_path=f"{REPO_REMOTE_PATH}/data/baselines")
if (_LOCAL_REPO_ROOT / "data" / "eval_sets").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "data" / "eval_sets", remote_path=f"{REPO_REMOTE_PATH}/data/eval_sets")
if (_LOCAL_REPO_ROOT / "data" / "benchmarks").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "data" / "benchmarks", remote_path=f"{REPO_REMOTE_PATH}/data/benchmarks")

app = modal.App(APP_NAME)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HORACE_HF_FULL_LOGITS", "1")


@app.function(image=image, gpu="any", timeout=60 * 90, volumes={"/cache/hf": hf_cache_vol})
def train_remote(config_json: str) -> Dict[str, Any]:
    _bootstrap_repo()
    cfg = json.loads(config_json)

    from tools.studio.eval_set import run_eval_set
    from tools.studio.train_calibrator import train_from_eval_report

    samples_path = Path(str(cfg["samples"]))
    report_path = Path("/tmp/horace_studio_eval_set_report.json")
    run_eval_set(
        samples_path=samples_path,
        model_id=str(cfg["model_id"]),
        baseline_model=str(cfg["baseline_model"]),
        doc_type=str(cfg["doc_type"]),
        backend="hf",
        max_input_tokens=int(cfg["max_input_tokens"]),
        normalize_text=bool(cfg.get("normalize_text", True)),
        compute_cohesion=bool(cfg.get("compute_cohesion", False)),
        positive_sources=tuple(cfg.get("pos_sources") or ["gutenberg_excerpt"]),
        negative_sources=tuple(cfg.get("neg_sources") or []),
        report_out=report_path,
    )

    cal_path = Path("/tmp/horace_studio_calibrator.json")
    train_res = train_from_eval_report(
        report_path,
        out_path=cal_path,
        positive_sources=tuple(cfg.get("pos_sources") or ["gutenberg_excerpt"]),
        negative_sources=tuple(cfg.get("neg_sources") or []),
        missing_value=float(cfg.get("missing_value", 0.5)),
        l2=float(cfg.get("l2", 1e-2)),
        lr=float(cfg.get("lr", 0.5)),
        steps=int(cfg.get("steps", 600)),
        seed=int(cfg.get("train_seed", 1337)),
    )

    hf_cache_vol.commit()
    report_obj = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "report_summary": {"run_meta": report_obj.get("run_meta"), "sources": report_obj.get("sources")},
        "train_result": {
            "out_path": str(train_res.out_path),
            "n_rows": int(train_res.n_rows),
            "n_pos": int(train_res.n_pos),
            "n_neg": int(train_res.n_neg),
            "train_acc": float(train_res.train_acc),
            "train_auc": float(train_res.train_auc) if train_res.train_auc is not None else None,
            "feature_dim": int(train_res.feature_dim),
        },
        "calibrator_json": json.loads(cal_path.read_text(encoding="utf-8")),
    }


@app.local_entrypoint()
def main(  # pragma: no cover
    out_dir: str = "reports/calibrators_eval_set",
    samples: str = "data/eval_sets/studio_fixed_v1.jsonl",
    model_id: str = "gpt2",
    baseline_model: str = "gpt2_gutenberg_512",
    doc_type: str = "prose",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    compute_cohesion: bool = False,
    train_seed: Optional[int] = None,
    steps: int = 600,
    lr: float = 0.5,
    l2: float = 1e-2,
    missing_value: float = 0.5,
    pos_sources: str = "gutenberg_excerpt",
    neg_sources: str = "wikipedia_summary,wikinews_published,rfc_excerpt,nasa_breaking_news,gibberish_control",
) -> None:
    cfg = {
        "samples": str(samples),
        "model_id": str(model_id),
        "baseline_model": str(baseline_model),
        "doc_type": str(doc_type),
        "max_input_tokens": int(max_input_tokens),
        "normalize_text": bool(normalize_text),
        "compute_cohesion": bool(compute_cohesion),
        "pos_sources": [s.strip() for s in str(pos_sources).split(",") if s.strip()],
        "neg_sources": [s.strip() for s in str(neg_sources).split(",") if s.strip()],
        "train_seed": int(train_seed) if train_seed is not None else 1337,
        "steps": int(steps),
        "lr": float(lr),
        "l2": float(l2),
        "missing_value": float(missing_value),
    }

    res = train_remote.remote(json.dumps(cfg))
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    cal_out = out_base / "calibrator.json"
    rep_out = out_base / "report_summary.json"
    cal_out.write_text(json.dumps(res["calibrator_json"], ensure_ascii=False, indent=2), encoding="utf-8")
    rep_out.write_text(json.dumps(res["report_summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("== Horace Studio calibrator train (fixed eval set, Modal) ==")
    print(json.dumps(res.get("train_result") or {}, indent=2))
    print(f"Wrote calibrator: {cal_out}")
    print(f"Wrote report:     {rep_out}")
