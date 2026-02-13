#!/usr/bin/env python3
"""
Horace Studio API (local FastAPI app).

This mirrors the Modal endpoints so a real frontend can be built on top.

Run:
  python -m tools.studio_api --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
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

from tools.studio.analyze import analyze_text, DEFAULT_SCORING_MODEL, DEFAULT_BASELINE_MODEL
from tools.studio.baselines import build_baseline, load_baseline_cached
from tools.studio.critique import suggest_edits
from tools.studio.llm_critic import llm_critique
from tools.studio.meaning_lock import MeaningLockConfig
from tools.studio.rewrite import rewrite_and_rerank
from tools.studio.score import score_text
from tools.studio.span_patcher import patch_span as patch_one_span
from tools.studio.span_patcher import suggest_dead_zones
from tools.studio.site import API_HTML, STUDIO_HTML
from tools.studio.write_like import write_like as write_like_gen


def _ensure_baseline(baseline_model_or_path: str):
    ident = (baseline_model_or_path or "").strip() or DEFAULT_BASELINE_MODEL
    p = Path(ident)
    if p.exists():
        return load_baseline_cached(ident, path=p)
    try:
        return load_baseline_cached(ident)
    except Exception:
        build_baseline(ident)
        return load_baseline_cached(ident)


def _antipattern_model_warning(path: str, anti_prob: Optional[float]) -> Optional[str]:
    s = str(path or "").strip().lower()
    if not s:
        return None
    if anti_prob is None:
        return None
    if "authenticity" in s:
        return None
    if "v5_antipattern" in s and anti_prob < 0.20:
        return (
            "antipattern model looks like quality-style training (not authenticity-focused); "
            "expected model like scorer_v5_authenticity_v1"
        )
    return None


def _scorer_model_warning(path: str) -> Optional[str]:
    s = str(path or "").strip().lower()
    if not s:
        return None
    if ("antipattern" in s or "scorer_v5_antipattern" in s) and "authenticity" not in s:
        return (
            "scorer_model_path looks like an anti-pattern checkpoint; "
            "use it in antipattern_model_path for authenticity penalty, not as primary scorer"
        )
    return None


def _antipattern_prob_is_inverted(path: str) -> Optional[bool]:
    model_path = str(path or "").strip()
    if not model_path:
        return None

    base = Path(model_path)
    if base.exists() and base.is_dir():
        report_path = base / "train_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                pos_sources = [str(x).strip().lower() for x in (report.get("run_meta", {}).get("positive_sources") or [])]
                if "human_original" in pos_sources:
                    return True
                if any(s.startswith("llm_antipattern") for s in pos_sources):
                    return False
            except Exception:
                pass

    lowered = model_path.lower()
    if "v5_antipattern_mix_plusfull" in lowered or "v5_antipattern_mix_" in lowered or "v5_antipattern_pilot" in lowered:
        return True
    if "authenticity" in lowered:
        return False
    return None


def _resolve_antipattern_prob(path: str, anti_prob: Optional[float]) -> tuple[float, bool, Optional[str]]:
    if anti_prob is None:
        return 0.0, False, None
    p = float(anti_prob)
    p = min(1.0, max(0.0, p))
    mode = _antipattern_prob_is_inverted(path)
    if mode is None:
        return p, False, None
    if mode:
        return (1.0 - p), True, "antipattern model appears to be trained with human-positive labels; score was inverted for AI-likelihood."
    return p, False, None


def _compute_antipattern_adjustment(
    *,
    base_score_0_100: float,
    anti_prob_0_1: float,
    threshold_0_1: float,
    weight: float,
    mode: str = "adaptive",
) -> dict:
    base = max(0.0, min(100.0, float(base_score_0_100)))
    p = max(0.0, min(1.0, float(anti_prob_0_1)))
    hard_t = max(0.05, min(0.98, float(threshold_0_1)))
    w = max(0.0, min(2.0, float(weight)))
    selected_mode = str(mode or "adaptive").strip().lower()
    if selected_mode not in ("adaptive", "legacy"):
        selected_mode = "adaptive"

    # Legacy behavior kept for backward compatibility/testing.
    if selected_mode == "legacy":
        penalty_0_100 = 0.0
        if p > hard_t and hard_t < 1.0:
            rel = (p - hard_t) / max(1e-6, 1.0 - hard_t)
            penalty_0_100 = max(0.0, min(100.0, rel * w * 100.0))
        adjusted = max(0.0, base - penalty_0_100)
        return {
            "mode": selected_mode,
            "adjusted_score_0_100": adjusted,
            "penalty_0_100": max(0.0, min(base, penalty_0_100)),
            "soft_threshold_0_1": None,
            "hard_threshold_0_1": hard_t,
            "authenticity_cap_0_100": None,
            "risk_soft_0_1": None,
            "risk_hard_0_1": None,
        }

    # Adaptive behavior: penalties begin below hard threshold and tighten with risk.
    soft_t = max(0.45, min(0.65, hard_t - 0.30))
    soft_rel = 0.0
    if p > soft_t:
        soft_rel = (p - soft_t) / max(1e-6, hard_t - soft_t)
    soft_rel = max(0.0, min(1.0, soft_rel))

    hard_rel = 0.0
    if p > hard_t:
        hard_rel = (p - hard_t) / max(1e-6, 1.0 - hard_t)
    hard_rel = max(0.0, min(1.0, hard_rel))

    soft_penalty = (20.0 * w) * (soft_rel**1.6)
    hard_penalty = (80.0 * w) * (hard_rel**1.1)
    pre_cap = max(0.0, base - soft_penalty - hard_penalty)

    cap_floor = max(20.0, 100.0 - (70.0 * w))
    cap_rel = 0.0
    if p > soft_t:
        cap_rel = (p - soft_t) / max(1e-6, 1.0 - soft_t)
    cap_rel = max(0.0, min(1.0, cap_rel))
    authenticity_cap = 100.0 - (100.0 - cap_floor) * (cap_rel**1.35)

    adjusted = min(pre_cap, authenticity_cap)
    adjusted = max(0.0, min(100.0, adjusted))
    penalty = max(0.0, min(base, base - adjusted))
    return {
        "mode": selected_mode,
        "adjusted_score_0_100": adjusted,
        "penalty_0_100": penalty,
        "soft_threshold_0_1": float(soft_t),
        "hard_threshold_0_1": float(hard_t),
        "authenticity_cap_0_100": float(authenticity_cap),
        "risk_soft_0_1": float(soft_rel),
        "risk_hard_0_1": float(hard_rel),
    }


app = FastAPI(title="Horace Studio API")

_HORACE_API_KEY = (os.environ.get("HORACE_API_KEY") or "").strip()


@app.middleware("http")
async def api_key_middleware(request, call_next):  # pragma: no cover
    if not _HORACE_API_KEY:
        return await call_next(request)

    # Keep docs / landing open.
    if request.url.path in ("/", "/api", "/docs", "/openapi.json", "/healthz"):
        return await call_next(request)

    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    else:
        supplied = (request.headers.get("x-api-key") or "").strip()

    if not supplied or supplied != _HORACE_API_KEY:
        from fastapi.responses import JSONResponse

        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


class AnalyzeReq(BaseModel):
    text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    # Optional: a single trained text→score model directory (HF save_pretrained).
    # If provided, the API will return `trained_score`; if fast_only=true it will skip token-level analysis.
    scorer_model_path: str = ""
    scorer_max_length: int = 384
    antipattern_model_path: str = ""  # optional model trained on human vs LLM-imitation
    antipattern_max_length: int = 384
    # Anti-pattern is a likelihood-of-LLM-imitation score for this text.
    # Lowering threshold makes AI-text penalties kick in earlier; raising it
    # makes penalties rarer and more selective.
    antipattern_penalty_weight: float = 0.85
    antipattern_penalty_threshold: float = 0.85
    antipattern_combiner_mode: str = "adaptive"  # adaptive | legacy
    # Dual-score mode default: report quality + authenticity separately.
    # Set true to fold authenticity penalty into primary overall score.
    apply_antipattern_penalty: bool = False
    # Primary score selection:
    # - auto: trained scorer (if provided) else rubric (or calibrated rubric)
    # - rubric: rubric (or calibrated rubric)
    # - trained: trained scorer (fallback to rubric on error)
    # - blend: blend rubric + trained (fallback to rubric on error)
    primary_score_mode: str = "auto"
    primary_blend_weight: float = 0.35
    fast_only: bool = False
    scoring_model_id: str = DEFAULT_SCORING_MODEL
    baseline_model: str = DEFAULT_BASELINE_MODEL  # model id or baseline json path
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
    text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    rewrite_model_id: str = DEFAULT_SCORING_MODEL
    scoring_model_id: str = DEFAULT_SCORING_MODEL
    baseline_model: str = DEFAULT_BASELINE_MODEL  # model id or baseline json path
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
    text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    scoring_model_id: str = DEFAULT_SCORING_MODEL
    backend: str = "auto"
    max_input_tokens: int = 512
    normalize_text: bool = True
    window_sentences: int = 4
    max_zones: int = 6


class PatchSpanReq(BaseModel):
    text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    start_char: int = 0
    end_char: int = 0
    rewrite_mode: str = "strict"  # strict | creative
    intensity: float = 0.5  # 0=clearer, 1=punchier
    rewrite_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    scoring_model_id: str = DEFAULT_SCORING_MODEL
    baseline_model: str = DEFAULT_BASELINE_MODEL  # model id or baseline json path
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
    meaning_lock_min_length_ratio: float = 0.75
    meaning_lock_max_length_ratio: float = 1.45
    meaning_lock_max_edit_ratio: float = 0.55
    meaning_lock_allow_new_numbers: bool = False
    meaning_lock_allow_new_proper_nouns: bool = False
    meaning_lock_allow_negation_change: bool = False


class WriteLikeReq(BaseModel):
    prompt: str = Field(default="", max_length=10_000)
    reference_text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    model_id: str = DEFAULT_SCORING_MODEL
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


@app.get("/api")
def api_docs():  # pragma: no cover
    from fastapi.responses import HTMLResponse

    return HTMLResponse(content=API_HTML)


@app.post("/analyze")
def analyze(req: AnalyzeReq) -> Dict[str, Any]:
    trained_score = None
    trained_err = None
    antipattern_score = None
    antipattern_err = None
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

    if (req.antipattern_model_path or "").strip():
        try:
            from tools.studio.scorer_model import score_with_scorer

            aps = score_with_scorer(
                req.text,
                model_path_or_id=str(req.antipattern_model_path),
                doc_type=req.doc_type,
                normalize_text=bool(req.normalize_text),
                max_length=int(req.antipattern_max_length),
                device=None,
            )
            antipattern_score = {
                "score_0_100": aps.score_0_100,
                "prob_0_1": aps.prob_0_1,
                "model_path_or_id": aps.model_path_or_id,
                "device": aps.device,
                "max_length": aps.max_length,
            }
        except Exception as e:
            antipattern_err = f"{type(e).__name__}: {e}"

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
            "top_improvements": [
                {
                    "category": h.category,
                    "metric": h.metric,
                    "current_score": h.current_score,
                    "potential_gain": h.potential_gain,
                    "direction": h.direction,
                }
                for h in (score.top_improvements or [])
            ],
        },
        "critique": critique,
    }
    if trained_score is not None:
        out["trained_score"] = trained_score
    if trained_err is not None:
        out["trained_score_error"] = trained_err
    if antipattern_score is not None:
        out["antipattern_score"] = antipattern_score
    if antipattern_err is not None:
        out["antipattern_score_error"] = antipattern_err
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

    rubric_score_0_100 = float(score.overall_0_100)
    rubric_source = "rubric"
    if out.get("calibrated_score") is not None:
        rubric_score_0_100 = float(out["calibrated_score"]["overall_0_100"])
        rubric_source = "rubric_calibrated"

    trained_score_0_100 = float(trained_score["overall_0_100"]) if trained_score is not None else None
    mode = str(getattr(req, "primary_score_mode", "auto") or "auto").strip().lower()
    blend_w = float(getattr(req, "primary_blend_weight", 0.35) or 0.35)
    blend_w = max(0.0, min(1.0, float(blend_w)))

    base_score_0_100 = float(rubric_score_0_100)
    base_source = str(rubric_source)
    if mode == "auto":
        if trained_score_0_100 is not None:
            base_score_0_100 = float(trained_score_0_100)
            base_source = "trained_scorer"
    elif mode == "rubric":
        base_score_0_100 = float(rubric_score_0_100)
        base_source = str(rubric_source)
    elif mode == "trained":
        if trained_score_0_100 is not None:
            base_score_0_100 = float(trained_score_0_100)
            base_source = "trained_scorer"
        else:
            out["primary_score_warning"] = "primary_score_mode=trained but scorer_model_path missing/failed; falling back to rubric"
    elif mode == "blend":
        if trained_score_0_100 is not None:
            base_score_0_100 = float((1.0 - blend_w) * float(rubric_score_0_100) + blend_w * float(trained_score_0_100))
            base_source = "blend"
        else:
            out["primary_score_warning"] = "primary_score_mode=blend but scorer_model_path missing/failed; falling back to rubric"
    else:
        out["primary_score_warning"] = f"unknown primary_score_mode={mode!r}; using auto"
        if trained_score_0_100 is not None:
            base_score_0_100 = float(trained_score_0_100)
            base_source = "trained_scorer"

    apply_antipattern_penalty = bool(getattr(req, "apply_antipattern_penalty", False))
    adjusted_score_0_100 = float(base_score_0_100)
    penalty_0_100 = 0.0
    suggested_penalty_0_100 = 0.0
    combined_preview_0_100 = float(base_score_0_100)
    anti_soft_threshold = None
    anti_hard_threshold = None
    anti_cap_0_100 = None
    anti_risk_soft = None
    anti_risk_hard = None
    anti_combiner_mode = str(getattr(req, "antipattern_combiner_mode", "adaptive") or "adaptive").strip().lower()
    anti_prob = None
    anti_prob_raw = None
    anti_prob_inverted = False
    if antipattern_score is not None:
        anti_prob_raw = float(antipattern_score.get("prob_0_1") or 0.0)
        anti_prob, anti_prob_inverted, anti_prob_inversion_msg = _resolve_antipattern_prob(
            req.antipattern_model_path,
            anti_prob_raw,
        )
        if anti_prob_inversion_msg is not None:
            out.setdefault("primary_score_warning", "antipattern score polarity corrected for AI-likelihood.")
            prev = str(out["primary_score_warning"] or "").strip()
            if prev and anti_prob_inversion_msg not in prev:
                out["primary_score_warning"] = f"{prev}; {anti_prob_inversion_msg}"
            else:
                out["primary_score_warning"] = anti_prob_inversion_msg

        anti_adj = _compute_antipattern_adjustment(
            base_score_0_100=base_score_0_100,
            anti_prob_0_1=float(anti_prob),
            threshold_0_1=float(req.antipattern_penalty_threshold),
            weight=float(req.antipattern_penalty_weight),
            mode=anti_combiner_mode,
        )
        combined_preview_0_100 = float(anti_adj["adjusted_score_0_100"])
        suggested_penalty_0_100 = float(anti_adj["penalty_0_100"])
        if apply_antipattern_penalty:
            adjusted_score_0_100 = float(combined_preview_0_100)
            penalty_0_100 = float(suggested_penalty_0_100)
        else:
            adjusted_score_0_100 = float(base_score_0_100)
            penalty_0_100 = 0.0
        anti_soft_threshold = anti_adj["soft_threshold_0_1"]
        anti_hard_threshold = anti_adj["hard_threshold_0_1"]
        anti_cap_0_100 = anti_adj["authenticity_cap_0_100"]
        anti_risk_soft = anti_adj["risk_soft_0_1"]
        anti_risk_hard = anti_adj["risk_hard_0_1"]
        anti_combiner_mode = str(anti_adj["mode"])
    if antipattern_err is not None:
        warning = (
            "anti-pattern scorer failed; authenticity penalty disabled for this request"
        )
        prev_warning = str(out.get("primary_score_warning") or "").strip()
        out["primary_score_warning"] = (
            warning if not prev_warning else f"{prev_warning}; {warning}"
        )
    anti_warning = _antipattern_model_warning(req.antipattern_model_path, anti_prob)
    if anti_warning is not None:
        prev_warning = str(out.get("primary_score_warning") or "").strip()
        out["primary_score_warning"] = (
            anti_warning if not prev_warning else f"{prev_warning}; {anti_warning}"
        )
    scorer_warning = _scorer_model_warning(req.scorer_model_path)
    if scorer_warning is not None:
        prev_warning = str(out.get("primary_score_warning") or "").strip()
        out["primary_score_warning"] = (
            scorer_warning if not prev_warning else f"{prev_warning}; {scorer_warning}"
        )
    if antipattern_score is not None and not apply_antipattern_penalty:
        prev_warning = str(out.get("primary_score_warning") or "").strip()
        dual_warning = "dual-score mode active: authenticity risk reported separately (no penalty applied to primary score)"
        out["primary_score_warning"] = (
            dual_warning if not prev_warning else f"{prev_warning}; {dual_warning}"
        )
    if antipattern_score is not None and anti_combiner_mode == "legacy":
        prev_warning = str(out.get("primary_score_warning") or "").strip()
        legacy_warning = "using legacy anti-pattern combiner (threshold-only); prefer adaptive mode for stricter AI-risk handling"
        out["primary_score_warning"] = (
            legacy_warning if not prev_warning else f"{prev_warning}; {legacy_warning}"
        )

    primary: Dict[str, Any] = {
        "overall_0_100": float(adjusted_score_0_100),
        "source": base_source,
        "base_overall_0_100": float(base_score_0_100),
        "mode": str(mode),
        "rubric_overall_0_100": float(rubric_score_0_100),
    }
    if out.get("calibrated_score") is not None:
        primary["calibrator_path"] = str(out["calibrated_score"]["calibrator_path"])
    if trained_score_0_100 is not None:
        primary["trained_overall_0_100"] = float(trained_score_0_100)
        primary["trained_model_path_or_id"] = str(trained_score.get("model_path_or_id") or "")
    if mode == "blend" and trained_score_0_100 is not None:
        primary["blend_weight"] = float(blend_w)
    primary["apply_antipattern_penalty"] = bool(apply_antipattern_penalty)
    if antipattern_score is not None:
        primary["antipattern_prob_0_1"] = float(anti_prob or 0.0)
        primary["antipattern_penalty_0_100"] = float(penalty_0_100)
        primary["antipattern_suggested_penalty_0_100"] = float(suggested_penalty_0_100)
        primary["combined_preview_overall_0_100"] = float(combined_preview_0_100)
        primary["antipattern_prob_raw_0_1"] = float(anti_prob_raw or 0.0)
        primary["antipattern_prob_inverted"] = bool(anti_prob_inverted)
        primary["antipattern_penalty_mode"] = str(anti_combiner_mode)
        if anti_soft_threshold is not None:
            primary["antipattern_soft_threshold_0_1"] = float(anti_soft_threshold)
        if anti_hard_threshold is not None:
            primary["antipattern_hard_threshold_0_1"] = float(anti_hard_threshold)
        if anti_cap_0_100 is not None:
            primary["antipattern_authenticity_cap_0_100"] = float(anti_cap_0_100)
        if anti_risk_soft is not None:
            primary["antipattern_risk_soft_0_1"] = float(anti_risk_soft)
        if anti_risk_hard is not None:
            primary["antipattern_risk_hard_0_1"] = float(anti_risk_hard)
        if apply_antipattern_penalty:
            primary["source"] = f"{base_source}_antipattern_adjusted"
        else:
            primary["source"] = f"{base_source}_dual_scores"
    out["primary_score"] = primary

    out["quality_score"] = {
        "overall_0_100": float(base_score_0_100),
        "source": str(base_source),
        "mode": str(mode),
        "rubric_overall_0_100": float(rubric_score_0_100),
        "trained_overall_0_100": (float(trained_score_0_100) if trained_score_0_100 is not None else None),
    }
    out["authenticity_score"] = {
        "enabled": bool((req.antipattern_model_path or "").strip()),
        "available": bool(antipattern_score is not None),
        "llm_likelihood_0_1": (float(anti_prob) if anti_prob is not None else None),
        "llm_likelihood_raw_0_1": (float(anti_prob_raw) if anti_prob_raw is not None else None),
        "llm_likelihood_inverted": bool(anti_prob_inverted),
        "combiner_mode": str(anti_combiner_mode),
        "soft_threshold_0_1": (float(anti_soft_threshold) if anti_soft_threshold is not None else None),
        "hard_threshold_0_1": (float(anti_hard_threshold) if anti_hard_threshold is not None else None),
        "authenticity_cap_0_100": (float(anti_cap_0_100) if anti_cap_0_100 is not None else None),
        "risk_soft_0_1": (float(anti_risk_soft) if anti_risk_soft is not None else None),
        "risk_hard_0_1": (float(anti_risk_hard) if anti_risk_hard is not None else None),
        "suggested_penalty_0_100": float(suggested_penalty_0_100),
        "combined_preview_overall_0_100": float(combined_preview_0_100),
        "penalty_applied": bool(apply_antipattern_penalty and antipattern_score is not None),
        "error": (str(antipattern_err) if antipattern_err is not None else None),
    }

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
        min_length_ratio=float(req.meaning_lock_min_length_ratio),
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
@app.post("/cadence-match")
def cadence_match(req: WriteLikeReq) -> Dict[str, Any]:
    """Generate text matching the cadence of reference text."""
    try:
        result = write_like_gen(
            prompt=req.prompt or " ",
            reference_text=req.reference_text,
            model_id=req.model_id or DEFAULT_SCORING_MODEL,
            backend=req.backend or "auto",
            doc_type=req.doc_type or "prose",
            max_new_tokens=int(req.max_new_tokens) if req.max_new_tokens else 200,
            seed=int(req.seed) if req.seed is not None else 7,
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
