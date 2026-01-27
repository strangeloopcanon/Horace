#!/usr/bin/env python3
"""
Horace Studio API (local FastAPI app).

This mirrors the Modal endpoints so a real frontend can be built on top.

Run:
  python -m tools.studio_api --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional


def _lazy_import_fastapi():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field

        return FastAPI, BaseModel, Field
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is not installed. Install via `make setup` (brings gradio deps), "
            "or `uv pip install fastapi uvicorn`."
        ) from e


FastAPI, BaseModel, Field = _lazy_import_fastapi()

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline, load_baseline_cached
from tools.studio.critique import suggest_edits
from tools.studio.llm_critic import llm_critique
from tools.studio.meaning_lock import MeaningLockConfig
from tools.studio.rewrite import rewrite_and_rerank
from tools.studio.score import score_text
from tools.studio.span_patcher import patch_span as patch_one_span
from tools.studio.span_patcher import suggest_dead_zones
from tools.studio.site import STUDIO_HTML
from tools.studio.write_like import write_like as write_like_gen


def _ensure_baseline(baseline_model_or_path: str):
    ident = (baseline_model_or_path or "").strip() or "gpt2"
    p = Path(ident)
    if p.exists():
        return load_baseline_cached(ident, path=p)
    try:
        return load_baseline_cached(ident)
    except Exception:
        build_baseline(ident)
        return load_baseline_cached(ident)


app = FastAPI(title="Horace Studio API")


class AnalyzeReq(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    doc_type: str = "prose"
    # Optional: a single trained text→score model directory (HF save_pretrained).
    # If provided, the API will return `trained_score`; if fast_only=true it will skip token-level analysis.
    scorer_model_path: str = ""
    scorer_max_length: int = 384
    fast_only: bool = False
    scoring_model_id: str = "gpt2"
    baseline_model: str = "gpt2_gutenberg_512"  # model id or baseline json path
    calibrator_path: str = ""  # optional JSON calibrator trained from eval reports
    backend: str = "auto"
    max_input_tokens: int = 512
    normalize_text: bool = True
    compute_cohesion: bool = False
    use_llm_critic: bool = False
    critic_model_id: str = ""
    critic_max_new_tokens: int = 450
    critic_temperature: float = 0.7
    critic_top_p: float = 0.95
    critic_seed: Optional[int] = None


class RewriteReq(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    doc_type: str = "prose"
    rewrite_model_id: str = "gpt2"
    scoring_model_id: str = "gpt2"
    baseline_model: str = "gpt2_gutenberg_512"  # model id or baseline json path
    calibrator_path: str = ""  # optional JSON calibrator trained from eval reports
    n_candidates: int = 4
    keep_top: int = 3
    backend: str = "auto"
    max_input_tokens: int = 512
    normalize_text: bool = True
    compute_cohesion: bool = False
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_p: float = 0.92
    seed: Optional[int] = 7


class PatchSuggestReq(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    doc_type: str = "prose"
    scoring_model_id: str = "gpt2"
    backend: str = "auto"
    max_input_tokens: int = 512
    normalize_text: bool = True
    window_sentences: int = 4
    max_zones: int = 6


class PatchSpanReq(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    doc_type: str = "prose"
    start_char: int = 0
    end_char: int = 0
    rewrite_mode: str = "strict"  # strict | creative
    intensity: float = 0.5  # 0=clearer, 1=punchier
    rewrite_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    scoring_model_id: str = "gpt2"
    baseline_model: str = "gpt2_gutenberg_512"  # model id or baseline json path
    calibrator_path: str = ""  # optional JSON calibrator trained from eval reports
    scorer_model_path: str = ""  # optional trained scorer (fast primary score)
    scorer_max_length: int = 384
    score_top_n: int = 3
    backend: str = "auto"
    max_input_tokens: int = 384
    normalize_text: bool = True
    n_candidates: int = 6
    max_new_tokens: int = 260
    temperature: float = 0.8
    top_p: float = 0.92
    seed: Optional[int] = 7
    meaning_lock_embedder_model_id: str = "distilbert-base-uncased"
    meaning_lock_embedder_max_length: int = 256
    meaning_lock_min_cosine_sim: float = 0.86
    meaning_lock_max_length_ratio: float = 1.45
    meaning_lock_max_edit_ratio: float = 0.55
    meaning_lock_allow_new_numbers: bool = False
    meaning_lock_allow_new_proper_nouns: bool = False
    meaning_lock_allow_negation_change: bool = False


class WriteLikeReq(BaseModel):
    prompt: str = Field(default="", max_length=10_000)
    reference_text: str = Field(min_length=1, max_length=50_000)
    doc_type: str = "prose"
    model_id: str = "gpt2"
    backend: str = "auto"
    max_new_tokens: int = 200
    seed: Optional[int] = 7


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/")
def index():  # pragma: no cover
    from fastapi.responses import HTMLResponse

    return HTMLResponse(content=STUDIO_HTML)


@app.post("/analyze")
def analyze(req: AnalyzeReq) -> Dict[str, Any]:
    trained_score = None
    trained_err = None
    if (req.scorer_model_path or "").strip():
        try:
            from tools.studio.scorer_model import score_with_scorer

            ts = score_with_scorer(
                req.text,
                model_path_or_id=str(req.scorer_model_path),
                doc_type=req.doc_type,
                normalize_text=bool(req.normalize_text),
                max_length=int(req.scorer_max_length),
                device=None,
            )
            trained_score = {
                "overall_0_100": ts.score_0_100,
                "prob_0_1": ts.prob_0_1,
                "model_path_or_id": ts.model_path_or_id,
                "device": ts.device,
                "max_length": ts.max_length,
            }
        except Exception as e:
            trained_err = f"{type(e).__name__}: {e}"

    if bool(req.fast_only):
        if trained_score is None:
            return {"error": "fast_only=true requires scorer_model_path", "trained_score_error": trained_err}
        out = {
            "primary_score": {"overall_0_100": float(trained_score["overall_0_100"]), "source": "trained_scorer"},
            "trained_score": trained_score,
        }
        if trained_err is not None:
            out["trained_score_error"] = trained_err
        return out

    analysis = analyze_text(
        req.text,
        model_id=req.scoring_model_id,
        doc_type=req.doc_type,
        backend=req.backend,
        max_input_tokens=int(req.max_input_tokens),
        normalize_text=bool(req.normalize_text),
        compute_cohesion=bool(req.compute_cohesion),
    )
    baseline = _ensure_baseline(req.baseline_model)
    score = score_text(analysis["doc_metrics"], baseline, doc_type=req.doc_type)
    critique = suggest_edits(
        doc_metrics=analysis["doc_metrics"],
        score=score,
        spikes=analysis.get("spikes") or [],
        segments=analysis.get("segments") or {},
    )
    out: Dict[str, Any] = {
        "analysis": analysis,
        "score": {
            "overall_0_100": score.overall_0_100,
            "categories": score.categories,
            "metrics": {
                k: {"value": v.value, "percentile": v.percentile, "score_0_1": v.score_0_1, "mode": v.mode}
                for k, v in score.metrics.items()
            },
        },
        "critique": critique,
    }
    if trained_score is not None:
        out["trained_score"] = trained_score
    if trained_err is not None:
        out["trained_score_error"] = trained_err
    if (req.calibrator_path or "").strip():
        try:
            from tools.studio.calibrator import featurize_from_report_row, load_logistic_calibrator

            cal = load_logistic_calibrator(Path(str(req.calibrator_path)))
            missing_value = float((cal.meta or {}).get("missing_value", 0.5))
            rubric_metrics = {k: {"score_0_1": v.score_0_1} for k, v in score.metrics.items()}
            feats = featurize_from_report_row(
                feature_names=cal.feature_names,
                categories=score.categories,
                rubric_metrics=rubric_metrics,
                doc_metrics=analysis.get("doc_metrics") or {},
                max_input_tokens=int(req.max_input_tokens),
                missing_value=missing_value,
            )
            out["calibrated_score"] = {
                "overall_0_100": float(cal.score_0_100(feats)),
                "calibrator_path": str(req.calibrator_path),
            }
        except Exception as e:
            out["calibrated_score_error"] = str(e)

    if trained_score is not None:
        out["primary_score"] = {"overall_0_100": float(trained_score["overall_0_100"]), "source": "trained_scorer"}
    elif out.get("calibrated_score") is not None:
        out["primary_score"] = {
            "overall_0_100": float(out["calibrated_score"]["overall_0_100"]),
            "source": "rubric_calibrated",
            "calibrator_path": str(out["calibrated_score"]["calibrator_path"]),
        }
    else:
        out["primary_score"] = {"overall_0_100": float(score.overall_0_100), "source": "rubric"}

    if bool(req.use_llm_critic):
        mid = (req.critic_model_id or "").strip()
        if not mid:
            out["llm_critique_error"] = "use_llm_critic=true but critic_model_id is empty"
        else:
            out["llm_critique"] = llm_critique(
                text=req.text,
                doc_type=req.doc_type,
                score=score,
                doc_metrics=analysis["doc_metrics"],
                spikes=analysis.get("spikes") or [],
                segments=analysis.get("segments") or {},
                model_id=mid,
                max_new_tokens=int(req.critic_max_new_tokens),
                temperature=float(req.critic_temperature),
                top_p=float(req.critic_top_p),
                seed=int(req.critic_seed) if req.critic_seed is not None else None,
            )
    return out


@app.post("/rewrite")
def rewrite(req: RewriteReq) -> Dict[str, Any]:
    # rewrite_and_rerank handles baseline model ids or paths.
    return rewrite_and_rerank(
        req.text,
        doc_type=req.doc_type,
        rewrite_model_id=req.rewrite_model_id,
        scoring_model_id=req.scoring_model_id,
        baseline_model_id=req.baseline_model,
        calibrator_path=req.calibrator_path,
        backend=req.backend,
        n_candidates=int(req.n_candidates),
        keep_top=int(req.keep_top),
        max_input_tokens=int(req.max_input_tokens),
        normalize_text=bool(req.normalize_text),
        max_new_tokens=int(req.max_new_tokens),
        temperature=float(req.temperature),
        top_p=float(req.top_p),
        seed=int(req.seed) if req.seed is not None else None,
        compute_cohesion=bool(req.compute_cohesion),
    )


@app.post("/patch/suggest")
def patch_suggest(req: PatchSuggestReq) -> Dict[str, Any]:
    return suggest_dead_zones(
        req.text,
        doc_type=req.doc_type,
        scoring_model_id=req.scoring_model_id,
        backend=req.backend,
        max_input_tokens=int(req.max_input_tokens),
        normalize_text=bool(req.normalize_text),
        window_sentences=int(req.window_sentences),
        max_zones=int(req.max_zones),
    )


@app.post("/patch/span")
def patch_span(req: PatchSpanReq) -> Dict[str, Any]:
    cfg = MeaningLockConfig(
        embedder_model_id=str(req.meaning_lock_embedder_model_id),
        embedder_max_length=int(req.meaning_lock_embedder_max_length),
        min_cosine_sim=float(req.meaning_lock_min_cosine_sim),
        max_length_ratio=float(req.meaning_lock_max_length_ratio),
        max_edit_ratio=float(req.meaning_lock_max_edit_ratio),
        allow_new_numbers=bool(req.meaning_lock_allow_new_numbers),
        allow_new_proper_nouns=bool(req.meaning_lock_allow_new_proper_nouns),
        allow_negation_change=bool(req.meaning_lock_allow_negation_change),
    )
    return patch_one_span(
        req.text,
        start_char=int(req.start_char),
        end_char=int(req.end_char),
        doc_type=req.doc_type,
        rewrite_mode=str(req.rewrite_mode),
        intensity=float(req.intensity),
        rewrite_model_id=req.rewrite_model_id,
        scoring_model_id=req.scoring_model_id,
        baseline_model_id=req.baseline_model,
        calibrator_path=req.calibrator_path,
        scorer_model_path=req.scorer_model_path,
        scorer_max_length=int(req.scorer_max_length),
        score_top_n=int(req.score_top_n),
        backend=req.backend,
        max_input_tokens=int(req.max_input_tokens),
        normalize_text=bool(req.normalize_text),
        n_candidates=int(req.n_candidates),
        max_new_tokens=int(req.max_new_tokens),
        temperature=float(req.temperature),
        top_p=float(req.top_p),
        seed=int(req.seed) if req.seed is not None else None,
        meaning_lock=cfg,
    )


@app.post("/write-like")
def write_like(req: WriteLikeReq) -> Dict[str, Any]:
    """Generate text matching the cadence of reference text."""
    try:
        result = write_like_gen(
            prompt=req.prompt or " ",
            reference_text=req.reference_text,
            model_id=req.model_id or "gpt2",
            backend=req.backend or "auto",
            doc_type=req.doc_type or "prose",
            max_new_tokens=int(req.max_new_tokens) if req.max_new_tokens else 200,
            seed=int(req.seed) if req.seed else 7,
        )
        return result.to_dict()
    except Exception as e:
        return {"error": str(e)}


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Horace Studio API (FastAPI)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    try:
        import uvicorn  # type: ignore
    except Exception as e:
        raise RuntimeError("uvicorn is not installed. Install via `make setup` or `uv pip install uvicorn`.") from e

    uvicorn.run("tools.studio_api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
