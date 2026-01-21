"""
Modal job: fetch web text samples and evaluate Horace Studio scoring.

This is meant to answer: does the current metric separate "literary" prose from
other web writing (Wikipedia/RFC) and from gibberish controls?

Run (first time):
  make setup-modal
  modal token new

Run (writes local files under ./reports):
  modal run deploy/modal/studio_eval_web.py --wikipedia 12 --gutenberg 10 --rfc 6 --gibberish 10
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

APP_NAME = "horace-studio-eval-web"
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
    image = image.add_local_dir(
        _LOCAL_REPO_ROOT / "data" / "baselines", remote_path=f"{REPO_REMOTE_PATH}/data/baselines"
    )

app = modal.App(APP_NAME)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HORACE_HF_FULL_LOGITS", "1")


@app.function(image=image, gpu="any", timeout=60 * 45, volumes={"/cache/hf": hf_cache_vol})
def eval_remote(config_json: str) -> Dict[str, Any]:
    _bootstrap_repo()
    cfg = json.loads(config_json)

    from tools.studio.eval_web import run_eval

    samples_path = Path("/tmp/horace_studio_eval_samples.jsonl")
    report_path = Path("/tmp/horace_studio_eval_report.json")
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
        compute_cohesion=bool(cfg["compute_cohesion"]),
        excerpt_chars=int(cfg["excerpt_chars"]),
        samples_out=samples_path,
        report_out=report_path,
    )
    samples: list[dict] = []
    try:
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    except Exception:
        pass
    hf_cache_vol.commit()
    return {"report": report, "samples": samples}


@app.local_entrypoint()
def main(  # pragma: no cover
    out_dir: str = "reports",
    seed: int = 1337,
    wikipedia: int = 12,
    gutenberg: int = 10,
    rfc: int = 6,
    gibberish: int = 10,
    model_id: str = "gpt2",
    baseline_model: str = "gpt2_gutenberg_512",
    doc_type: str = "prose",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    compute_cohesion: bool = False,
    excerpt_chars: int = 3800,
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
    }
    res = eval_remote.remote(json.dumps(cfg))
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    samples_out = out_base / "studio_eval_web_samples.jsonl"
    report_out = out_base / "studio_eval_web_report.json"

    samples = res.get("samples") or []
    report = res.get("report") or {}

    with samples_out.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== Horace Studio web eval (Modal) ==")
    print(json.dumps(report.get("run_meta") or {}, indent=2))
    for src, info in (report.get("sources") or {}).items():
        summ = info.get("summary") or {}
        print(
            f"- {src}: n={summ.get('n')}, mean={summ.get('mean'):.1f}, "
            f"p10={summ.get('p10'):.1f}, p50={summ.get('p50'):.1f}, p90={summ.get('p90'):.1f}"
        )
    print(f"\nWrote samples: {samples_out}")
    print(f"Wrote report:  {report_out}")
