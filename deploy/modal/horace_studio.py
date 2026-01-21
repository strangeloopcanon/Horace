"""
Modal deployment skeleton for Horace Studio.

This keeps the UX simple:
- `/analyze` -> metrics + score + suggestions (+ optional `trained_score`; `fast_only=true` skips token analysis)
- `/rewrite` -> N rewrites, reranked by score

Notes:
- Modal runs Linux + CUDA; use HF backend (no MLX).
- Baselines should be prebuilt into `data/baselines/` (or generated on first call).

Setup (outside this repo):
  pip install modal
  modal token new

Run locally:
  modal run deploy/modal/horace_studio.py

Deploy:
  modal deploy deploy/modal/horace_studio.py
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

APP_NAME = "horace-studio"

hf_cache_vol = modal.Volume.from_name("horace-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)

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
        "safetensors>=0.4.0",
        "fastapi>=0.110.0",
        "pydantic>=2.6.0",
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


def _ensure_baseline(model_id: str):
    from tools.studio.baselines import load_baseline_cached

    try:
        return load_baseline_cached(model_id)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing baseline file. Build locally (e.g. `python -c \"from tools.studio.baselines import build_baseline; build_baseline('gpt2')\"`) "
            "and ensure `data/baselines/` is mounted into Modal."
        ) from e


@app.function(image=image, gpu="any", timeout=600, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
def analyze_remote(
    text: str,
    *,
    doc_type: str = "prose",
    scorer_model_path: str = "",
    scorer_max_length: int = 384,
    fast_only: bool = False,
    scoring_model_id: str = "gpt2",
    baseline_model_id: str = "gpt2_gutenberg_512",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    calibrator_path: str = "",
) -> Dict[str, Any]:
    _bootstrap_repo()
    trained_score = None
    trained_err = None
    if (scorer_model_path or "").strip():
        try:
            from tools.studio.scorer_model import score_with_scorer

            ts = score_with_scorer(
                text,
                model_path_or_id=str(scorer_model_path),
                doc_type=str(doc_type),
                normalize_text=bool(normalize_text),
                max_length=int(scorer_max_length),
                device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
            )
            trained_score = ts.__dict__
        except Exception as e:
            trained_err = f"{type(e).__name__}: {e}"

    if bool(fast_only):
        if trained_score is None:
            return {"error": "fast_only=true requires scorer_model_path", "trained_score_error": trained_err}
        out: Dict[str, Any] = {
            "primary_score": {"overall_0_100": float(trained_score["score_0_100"]), "source": "trained_scorer"},
            "trained_score": trained_score,
        }
        if trained_err is not None:
            out["trained_score_error"] = trained_err
        return out

    from tools.studio.analyze import analyze_text
    from tools.studio.calibrator import featurize_from_report_row, load_logistic_calibrator
    from tools.studio.score import score_text
    from tools.studio.critique import suggest_edits

    analysis = analyze_text(
        text,
        model_id=scoring_model_id,
        doc_type=doc_type,
        backend="hf",
        max_input_tokens=max_input_tokens,
        normalize_text=bool(normalize_text),
        compute_cohesion=True,
    )
    baseline = _ensure_baseline(baseline_model_id)
    score = score_text(analysis["doc_metrics"], baseline, doc_type=doc_type)
    critique = suggest_edits(
        doc_metrics=analysis["doc_metrics"],
        score=score,
        spikes=analysis.get("spikes") or [],
        segments=analysis.get("segments") or {},
    )
    calibrated = None
    cal_err = None
    if (calibrator_path or "").strip():
        try:
            cal = load_logistic_calibrator(Path(str(calibrator_path)))
            missing_value = float((cal.meta or {}).get("missing_value", 0.5))
            rubric_metrics = {k: {"score_0_1": v.score_0_1} for k, v in score.metrics.items()}
            feats = featurize_from_report_row(
                feature_names=cal.feature_names,
                categories=score.categories,
                rubric_metrics=rubric_metrics,
                doc_metrics=analysis.get("doc_metrics") or {},
                max_input_tokens=int(max_input_tokens),
                missing_value=missing_value,
            )
            calibrated = {
                "overall_0_100": float(cal.score_0_100(feats)),
                "calibrator_path": str(calibrator_path),
            }
        except Exception as e:
            cal_err = f"{type(e).__name__}: {e}"
    hf_cache_vol.commit()
    out: Dict[str, Any] = {
        "analysis": analysis,
        "score": {
            "overall_0_100": score.overall_0_100,
            "categories": score.categories,
            "metrics": {k: {"value": v.value, "percentile": v.percentile, "score_0_1": v.score_0_1, "mode": v.mode} for k, v in score.metrics.items()},
        },
        "critique": critique,
    }
    if trained_score is not None:
        out["trained_score"] = trained_score
    if trained_err is not None:
        out["trained_score_error"] = trained_err
    if calibrated is not None:
        out["calibrated_score"] = calibrated
    if cal_err is not None:
        out["calibrated_score_error"] = cal_err
    if trained_score is not None:
        out["primary_score"] = {"overall_0_100": float(trained_score["score_0_100"]), "source": "trained_scorer"}
    elif calibrated is not None:
        out["primary_score"] = {
            "overall_0_100": float(calibrated["overall_0_100"]),
            "source": "rubric_calibrated",
            "calibrator_path": str(calibrated["calibrator_path"]),
        }
    else:
        out["primary_score"] = {"overall_0_100": float(score.overall_0_100), "source": "rubric"}
    return out


@app.function(image=image, gpu="any", timeout=900, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
def rewrite_remote(
    text: str,
    *,
    doc_type: str = "prose",
    rewrite_model_id: str = "gpt2",
    scoring_model_id: str = "gpt2",
    baseline_model_id: str = "gpt2_gutenberg_512",
    calibrator_path: str = "",
    n_candidates: int = 4,
    keep_top: int = 3,
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_p: float = 0.92,
    seed: Optional[int] = 7,
) -> Dict[str, Any]:
    _bootstrap_repo()
    from tools.studio.rewrite import rewrite_and_rerank

    out = rewrite_and_rerank(
        text,
        doc_type=doc_type,
        rewrite_model_id=rewrite_model_id,
        scoring_model_id=scoring_model_id,
        baseline_model_id=baseline_model_id,
        calibrator_path=calibrator_path,
        backend="hf",
        n_candidates=n_candidates,
        keep_top=keep_top,
        max_input_tokens=max_input_tokens,
        normalize_text=bool(normalize_text),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    hf_cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
@modal.asgi_app()
def fastapi_app():  # pragma: no cover
    _bootstrap_repo()
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    from fastapi.responses import HTMLResponse

    from tools.studio.site import STUDIO_HTML

    web = FastAPI(title="Horace Studio API")

    class AnalyzeReq(BaseModel):
        text: str = Field(min_length=1, max_length=50_000)
        doc_type: str = "prose"
        scorer_model_path: str = ""
        scorer_max_length: int = 384
        fast_only: bool = False
        scoring_model_id: str = "gpt2"
        baseline_model: str = "gpt2_gutenberg_512"
        baseline_model_id: Optional[str] = None
        max_input_tokens: int = 512
        normalize_text: bool = True
        calibrator_path: str = ""

    class RewriteReq(BaseModel):
        text: str = Field(min_length=1, max_length=50_000)
        doc_type: str = "prose"
        rewrite_model_id: str = "gpt2"
        scoring_model_id: str = "gpt2"
        baseline_model: str = "gpt2_gutenberg_512"
        baseline_model_id: Optional[str] = None
        n_candidates: int = 4
        keep_top: int = 3
        max_input_tokens: int = 512
        max_new_tokens: int = 300
        temperature: float = 0.8
        top_p: float = 0.92
        seed: Optional[int] = 7
        normalize_text: bool = True
        calibrator_path: str = ""

    @web.post("/analyze")
    async def analyze(req: AnalyzeReq):
        baseline = (req.baseline_model_id or req.baseline_model or "gpt2").strip()
        return analyze_remote.remote(
            req.text,
            doc_type=req.doc_type,
            scorer_model_path=str(req.scorer_model_path or ""),
            scorer_max_length=int(req.scorer_max_length),
            fast_only=bool(req.fast_only),
            scoring_model_id=req.scoring_model_id,
            baseline_model_id=baseline,
            max_input_tokens=req.max_input_tokens,
            normalize_text=bool(req.normalize_text),
            calibrator_path=str(req.calibrator_path or ""),
        )

    @web.post("/rewrite")
    async def rewrite(req: RewriteReq):
        baseline = (req.baseline_model_id or req.baseline_model or "gpt2").strip()
        return rewrite_remote.remote(
            req.text,
            doc_type=req.doc_type,
            rewrite_model_id=req.rewrite_model_id,
            scoring_model_id=req.scoring_model_id,
            baseline_model_id=baseline,
            calibrator_path=str(req.calibrator_path or ""),
            n_candidates=req.n_candidates,
            keep_top=req.keep_top,
            max_input_tokens=req.max_input_tokens,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            normalize_text=bool(req.normalize_text),
        )

    @web.get("/")
    async def root():
        return HTMLResponse(content=STUDIO_HTML)

    return web


@app.local_entrypoint()
def main(  # pragma: no cover
    text: str = "At dawn, the city leans into light.\nA gull lifts, then drops, then lifts again.\n",
    doc_type: str = "prose",
    scorer_model_path: str = "",
    fast_only: bool = False,
    scorer_max_length: int = 384,
    scoring_model_id: str = "gpt2",
    baseline_model_id: str = "gpt2_gutenberg_512",
    max_input_tokens: int = 512,
    do_rewrite: bool = False,
) -> None:
    """Quick smoke runner for Modal.

    Use `modal deploy deploy/modal/horace_studio.py` to deploy the web app.
    """
    if do_rewrite:
        out = rewrite_remote.remote(
            text,
            doc_type=doc_type,
            rewrite_model_id="gpt2",
            scoring_model_id=scoring_model_id,
            baseline_model_id=baseline_model_id,
            n_candidates=2,
            keep_top=1,
            max_input_tokens=max_input_tokens,
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.92,
            seed=7,
        )
        print(json.dumps(out.get("meta") or {}, ensure_ascii=False, indent=2))
        if out.get("rewrites"):
            print("\n--- top rewrite ---\n")
            print(out["rewrites"][0].get("text", "").strip())
        return

    out = analyze_remote.remote(
        text,
        doc_type=doc_type,
        scorer_model_path=str(scorer_model_path or ""),
        scorer_max_length=int(scorer_max_length),
        fast_only=bool(fast_only),
        scoring_model_id=scoring_model_id,
        baseline_model_id=baseline_model_id,
        max_input_tokens=max_input_tokens,
    )
    print(json.dumps(out.get("score") or {}, ensure_ascii=False, indent=2))
    try:
        print("\n" + str((out.get("critique") or {}).get("summary") or ""))
    except Exception:
        pass
