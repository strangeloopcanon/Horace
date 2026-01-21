"""
Modal job: build a prose baseline by sampling Project Gutenberg excerpts.

Why: the default baseline in `data/analysis/<model>/docs_clean.jsonl` is based on
the repo's local corpus, which currently mixes full texts and short summaries.
For the Studio promise ("match top literature patterns"), we want a baseline
constructed from real literary prose windows.

Run (first time):
  make setup-modal
  modal token new

Run (writes local baseline JSON under ./reports):
  modal run deploy/modal/studio_build_baseline_web.py --out-dir reports/baselines --n 200 --max-input-tokens 512
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

APP_NAME = "horace-studio-build-baseline-web"
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

app = modal.App(APP_NAME)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HORACE_HF_FULL_LOGITS", "1")


@app.function(image=image, gpu="any", timeout=60 * 90, volumes={"/cache/hf": hf_cache_vol})
def build_remote(config_json: str) -> Dict[str, Any]:
    _bootstrap_repo()
    cfg = json.loads(config_json)

    from tools.studio.build_baseline_web import build_gutenberg_prose_baseline

    out_path = Path("/tmp/horace_studio_baseline.json")
    rows_path = Path("/tmp/horace_studio_baseline_rows.jsonl") if bool(cfg.get("rows_out")) else None
    res = build_gutenberg_prose_baseline(
        model_id=str(cfg["model_id"]),
        backend="hf",
        doc_type=str(cfg["doc_type"]),
        max_input_tokens=int(cfg["max_input_tokens"]),
        excerpt_chars=int(cfg["excerpt_chars"]),
        n=int(cfg["n"]),
        seed=int(cfg["seed"]),
        normalize_text=bool(cfg.get("normalize_text", True)),
        compute_cohesion=bool(cfg.get("compute_cohesion", False)),
        out_path=out_path,
        rows_out=rows_path,
    )
    hf_cache_vol.commit()
    return {
        "result": {
            "baseline_path": str(res.baseline_path),
            "rows_path": str(res.rows_path) if res.rows_path else None,
            "n_rows": int(res.n_rows),
        },
        "baseline_json": json.loads(out_path.read_text(encoding="utf-8")),
        "rows_jsonl": rows_path.read_text(encoding="utf-8") if rows_path else None,
    }


@app.local_entrypoint()
def main(  # pragma: no cover
    out_dir: str = "reports/baselines",
    model_id: str = "gpt2",
    doc_type: str = "prose",
    max_input_tokens: int = 512,
    excerpt_chars: int = 3800,
    n: int = 200,
    seed: int = 1337,
    normalize_text: bool = True,
    compute_cohesion: bool = False,
    rows_out: bool = False,
) -> None:
    from tools.studio.baselines import safe_model_id

    cfg = {
        "model_id": str(model_id),
        "doc_type": str(doc_type),
        "max_input_tokens": int(max_input_tokens),
        "excerpt_chars": int(excerpt_chars),
        "n": int(n),
        "seed": int(seed),
        "normalize_text": bool(normalize_text),
        "compute_cohesion": bool(compute_cohesion),
        "rows_out": bool(rows_out),
    }
    res = build_remote.remote(json.dumps(cfg))

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    base_name = f"{safe_model_id(str(model_id))}_gutenberg_{max_input_tokens}_docs.json"
    baseline_out = out_base / base_name
    baseline_out.write_text(json.dumps(res["baseline_json"], ensure_ascii=False, indent=2), encoding="utf-8")

    if rows_out and res.get("rows_jsonl"):
        (out_base / f"{safe_model_id(str(model_id))}_gutenberg_{max_input_tokens}_rows.jsonl").write_text(
            str(res["rows_jsonl"]), encoding="utf-8"
        )

    print("== Horace Studio baseline build (Modal) ==")
    print(json.dumps(res.get("result") or {}, indent=2))
    print(f"Wrote baseline: {baseline_out}")
