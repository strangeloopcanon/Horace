"""
Modal job: fetch web text samples, run Studio scoring, then train a tiny learned calibrator.

This is intentionally lightweight (logistic regression on rubric outputs) so we can iterate quickly
before committing to training a larger model.

Run (first time):
  make setup-modal
  modal token new

Run (writes local files):
  modal run deploy/modal/studio_train_calibrator_web.py --out-dir reports/calibrators --max-input-tokens 512
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

APP_NAME = "horace-studio-train-calibrator-web"
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

    from tools.studio.eval_web import run_eval
    from tools.studio.train_calibrator import train_from_eval_report

    samples_path = Path("/tmp/horace_studio_cal_samples.jsonl")
    report_path = Path("/tmp/horace_studio_cal_report.json")
    report = run_eval(
        seed=int(cfg["seed"]),
        wikipedia_n=int(cfg["wikipedia_n"]),
        gutenberg_n=int(cfg["gutenberg_n"]),
        rfc_n=int(cfg["rfc_n"]),
        gibberish_n=int(cfg["gibberish_n"]),
        model_id=str(cfg["model_id"]),
        baseline_model=str(cfg["baseline_model"]),
        doc_type=str(cfg["doc_type"]),
        backend="hf",
        max_input_tokens=int(cfg["max_input_tokens"]),
        normalize_text=bool(cfg.get("normalize_text", True)),
        compute_cohesion=bool(cfg.get("compute_cohesion", False)),
        excerpt_chars=int(cfg["excerpt_chars"]),
        samples_out=samples_path,
        report_out=report_path,
    )

    cal_path = Path("/tmp/horace_studio_calibrator.json")
    train_res = train_from_eval_report(
        report_path,
        out_path=cal_path,
        positive_sources=tuple(cfg.get("pos_sources") or ["gutenberg_excerpt"]),
        negative_sources=tuple(cfg.get("neg_sources") or ["wikipedia_random_summary", "rfc_excerpt", "gibberish_control"]),
        missing_value=float(cfg.get("missing_value", 0.5)),
        l2=float(cfg.get("l2", 1e-2)),
        lr=float(cfg.get("lr", 0.5)),
        steps=int(cfg.get("steps", 600)),
        seed=int(cfg.get("train_seed", cfg["seed"])),
    )

    hf_cache_vol.commit()
    return {
        "report_summary": {"run_meta": report.get("run_meta"), "sources": report.get("sources")},
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
    out_dir: str = "reports/calibrators",
    seed: int = 1337,
    wikipedia: int = 40,
    gutenberg: int = 40,
    rfc: int = 20,
    gibberish: int = 20,
    model_id: str = "gpt2",
    baseline_model: str = "gpt2_gutenberg_512",
    doc_type: str = "prose",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    compute_cohesion: bool = False,
    excerpt_chars: int = 3800,
    train_seed: Optional[int] = None,
    steps: int = 600,
    lr: float = 0.5,
    l2: float = 1e-2,
    missing_value: float = 0.5,
    pos_sources: str = "gutenberg_excerpt",
    neg_sources: str = "wikipedia_random_summary,rfc_excerpt,gibberish_control",
) -> None:
    cfg = {
        "seed": int(seed),
        "wikipedia_n": int(wikipedia),
        "gutenberg_n": int(gutenberg),
        "rfc_n": int(rfc),
        "gibberish_n": int(gibberish),
        "model_id": str(model_id),
        "baseline_model": str(baseline_model),
        "doc_type": str(doc_type),
        "max_input_tokens": int(max_input_tokens),
        "normalize_text": bool(normalize_text),
        "compute_cohesion": bool(compute_cohesion),
        "excerpt_chars": int(excerpt_chars),
        "train_seed": int(train_seed) if train_seed is not None else int(seed),
        "steps": int(steps),
        "lr": float(lr),
        "l2": float(l2),
        "missing_value": float(missing_value),
        "pos_sources": [s.strip() for s in str(pos_sources).split(",") if s.strip()],
        "neg_sources": [s.strip() for s in str(neg_sources).split(",") if s.strip()],
    }

    res = train_remote.remote(json.dumps(cfg))
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    cal_out = out_base / "calibrator.json"
    rep_out = out_base / "report_summary.json"

    cal_out.write_text(json.dumps(res["calibrator_json"], ensure_ascii=False, indent=2), encoding="utf-8")
    rep_out.write_text(json.dumps(res["report_summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("== Horace Studio calibrator train (Modal) ==")
    print(json.dumps(res.get("train_result") or {}, indent=2))
    print(f"Wrote calibrator: {cal_out}")
    print(f"Wrote report:     {rep_out}")
