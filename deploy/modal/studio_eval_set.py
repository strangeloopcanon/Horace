"""
Modal job: run Horace Studio scoring on a fixed eval set (JSONL).

Run (first time):
  make setup-modal
  modal token new

Run:
  modal run deploy/modal/studio_eval_set.py --out-dir reports/studio_eval_set_modal --samples data/eval_sets/studio_fixed_v1.jsonl

Evaluate a split from the larger benchmark snapshot:
  modal run deploy/modal/studio_eval_set.py --samples data/benchmarks/studio_benchmark_v3/splits/test.jsonl --out-dir reports/benchmark_modal
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

APP_NAME = "horace-studio-eval-set"
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
def eval_remote(config_json: str) -> Dict[str, Any]:
    _bootstrap_repo()
    cfg = json.loads(config_json)

    from tools.studio.eval_set import run_eval_set

    samples_path = Path(str(cfg["samples"]))
    cal_path = None
    cal_json = str(cfg.get("calibrator_json") or "").strip()
    if cal_json:
        cal_path = Path("/tmp/horace_studio_calibrator.json")
        cal_path.write_text(cal_json, encoding="utf-8")
    report_path = Path("/tmp/horace_studio_eval_set_report.json")
    report = run_eval_set(
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
        calibrator_path=cal_path,
        report_out=report_path,
    )

    hf_cache_vol.commit()
    return report


@app.local_entrypoint()
def main(  # pragma: no cover
    out_dir: str = "reports/studio_eval_set_modal",
    samples: str = "data/eval_sets/studio_fixed_v1.jsonl",
    calibrator: str = "",
    model_id: str = "gpt2",
    baseline_model: str = "gpt2_gutenberg_512",
    doc_type: str = "prose",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    compute_cohesion: bool = False,
    pos_sources: str = "gutenberg_excerpt",
    neg_sources: str = "",
) -> None:
    calibrator_json = ""
    if (calibrator or "").strip():
        calibrator_json = Path(str(calibrator)).read_text(encoding="utf-8")
    cfg = {
        "samples": str(samples),
        "calibrator_json": calibrator_json,
        "model_id": str(model_id),
        "baseline_model": str(baseline_model),
        "doc_type": str(doc_type),
        "max_input_tokens": int(max_input_tokens),
        "normalize_text": bool(normalize_text),
        "compute_cohesion": bool(compute_cohesion),
        "pos_sources": [s.strip() for s in str(pos_sources).split(",") if s.strip()],
        "neg_sources": [s.strip() for s in str(neg_sources).split(",") if s.strip()],
    }
    report = eval_remote.remote(json.dumps(cfg))

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = out_base / "studio_eval_set_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    auc = ((report.get("literary_test") or {}).get("auc")) if isinstance(report, dict) else None
    auc_cal = ((report.get("literary_test_calibrated") or {}).get("auc")) if isinstance(report, dict) else None
    print("== Horace Studio fixed-set eval (Modal) ==")
    print(json.dumps((report.get("run_meta") or {}) if isinstance(report, dict) else {}, indent=2))
    if auc is not None:
        print(f"AUC (literary test): {auc}")
    if auc_cal is not None:
        print(f"AUC (calibrated): {auc_cal}")
    print(f"Wrote report: {out_path}")
