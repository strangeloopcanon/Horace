"""
Modal deployment skeleton for Horace Studio.

This keeps the UX simple:
- `/analyze` -> metrics + score + suggestions (+ optional `trained_score`; `fast_only=true` skips token analysis)
- `/rewrite` -> N rewrites, reranked by score
- `/cadence-match` (alias: `/write-like`) -> cadence-matched generation from a reference sample

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

from pydantic import BaseModel, Field

# Force rebuild version: 2026-02-14-v6-preference-scorer

# Request models defined at module level for FastAPI compatibility
class AnalyzeReq(BaseModel):
    text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    scorer_model_path: str = ""
    scorer_max_length: int = 384
    preference_model_path: str = ""  # v6 preference model JSON path
    antipattern_model_path: str = ""
    antipattern_max_length: int = 384
    # Anti-pattern is a likelihood-of-LLM-imitation score for this text.
    # Lowering threshold makes AI-text penalties kick in earlier; raising it makes
    # penalties rarer and more selective.
    antipattern_penalty_weight: float = 0.85
    antipattern_penalty_threshold: float = 0.85
    antipattern_combiner_mode: str = "adaptive"  # adaptive | legacy
    apply_antipattern_penalty: bool = False
    primary_score_mode: str = "auto"  # auto | rubric | trained | preference | blend
    primary_blend_weight: float = 0.35
    fast_only: bool = False
    scoring_model_id: str = "gpt2"
    baseline_model: str = "gpt2_gutenberg_512"
    baseline_model_id: Optional[str] = None
    max_input_tokens: int = 512
    normalize_text: bool = True
    calibrator_path: str = ""


class RewriteReq(BaseModel):
    text: str = Field(min_length=1, max_length=250_000)
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


class WriteLikeReq(BaseModel):
    prompt: str = Field(default="", max_length=10_000)
    reference_text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    model_id: str = "gpt2"
    max_new_tokens: int = 200
    seed: Optional[int] = 7


class PatchSuggestReq(BaseModel):
    text: str = Field(min_length=1, max_length=250_000)
    doc_type: str = "prose"
    scoring_model_id: str = "gpt2"
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
    scoring_model_id: str = "gpt2"
    baseline_model: str = "gpt2_gutenberg_512"
    baseline_model_id: Optional[str] = None
    calibrator_path: str = ""
    scorer_model_path: str = ""
    scorer_max_length: int = 384
    score_top_n: int = 3
    max_input_tokens: int = 384
    normalize_text: bool = True
    n_candidates: int = 6
    max_new_tokens: int = 260
    temperature: float = 0.8
    top_p: float = 0.92
    seed: Optional[int] = 7


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
        "scikit-learn>=1.3.0",
        "fastapi==0.115.6",
        "pydantic>=2.6.0",
    )
)
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")
if (_LOCAL_REPO_ROOT / "data" / "baselines").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "data" / "baselines", remote_path=f"{REPO_REMOTE_PATH}/data/baselines")
if (_LOCAL_REPO_ROOT / "models").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "models", remote_path=f"{REPO_REMOTE_PATH}/models")

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
    if "antipattern" in s or "authenticity" in s:
        return (
            "scorer_model_path looks like an authenticity/anti-pattern checkpoint; "
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


@app.function(image=image, gpu="any", timeout=600, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
def analyze_remote(
    text: str,
    *,
    doc_type: str = "prose",
    scorer_model_path: str = "",
    scorer_max_length: int = 384,
    preference_model_path: str = "",
    antipattern_model_path: str = "",
    antipattern_max_length: int = 384,
    antipattern_penalty_weight: float = 0.85,
    antipattern_penalty_threshold: float = 0.85,
    antipattern_combiner_mode: str = "adaptive",
    apply_antipattern_penalty: bool = False,
    primary_score_mode: str = "auto",
    primary_blend_weight: float = 0.35,
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
    antipattern_score = None
    antipattern_err = None
    scorer_warning = _scorer_model_warning(scorer_model_path)
    if scorer_warning is not None:
        trained_err = scorer_warning
    elif (scorer_model_path or "").strip():
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

    if (antipattern_model_path or "").strip():
        try:
            from tools.studio.scorer_model import score_with_scorer

            aps = score_with_scorer(
                text,
                model_path_or_id=str(antipattern_model_path),
                doc_type=str(doc_type),
                normalize_text=bool(normalize_text),
                max_length=int(antipattern_max_length),
                device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
            )
            antipattern_score = aps.__dict__
        except Exception as e:
            antipattern_err = f"{type(e).__name__}: {e}"

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
    if calibrated is not None:
        out["calibrated_score"] = calibrated
    if cal_err is not None:
        out["calibrated_score_error"] = cal_err

    # v6 preference scorer (feature-based)
    if (preference_model_path or "").strip():
        try:
            from tools.studio.preference_features import (
                FeaturePreferenceModel,
                extract_features,
                generate_feedback,
            )

            pref_model = FeaturePreferenceModel.load(Path(str(preference_model_path)))
            pref_features = extract_features(analysis.get("doc_metrics") or {}, text)
            pref_score_val = pref_model.score_0_100(pref_features)
            pref_raw = pref_model.score(pref_features)
            pref_gaps = pref_model.feature_gaps(pref_features)
            pref_feedback = generate_feedback(pref_model, pref_features, max_suggestions=5)
            out["preference_score"] = {
                "overall_0_100": int(pref_score_val),
                "raw_score": round(float(pref_raw), 3),
                "source": "preference_v6",
                "model_path": str(preference_model_path),
            }
            out["preference_feature_gaps"] = pref_gaps[:10]
            out["preference_feedback"] = pref_feedback
        except Exception as e:
            out["preference_score_error"] = f"{type(e).__name__}: {e}"

    rubric_score_0_100 = float(score.overall_0_100)
    rubric_source = "rubric"
    if calibrated is not None:
        rubric_score_0_100 = float(calibrated["overall_0_100"])
        rubric_source = "rubric_calibrated"

    trained_score_0_100 = float(trained_score["score_0_100"]) if trained_score is not None else None
    preference_score_0_100 = float(out["preference_score"]["overall_0_100"]) if out.get("preference_score") else None
    mode = str(primary_score_mode or "auto").strip().lower()
    blend_w = max(0.0, min(1.0, float(primary_blend_weight or 0.35)))

    base_score_0_100 = float(rubric_score_0_100)
    base_source = str(rubric_source)
    if mode == "auto":
        if preference_score_0_100 is not None:
            base_score_0_100 = float(preference_score_0_100)
            base_source = "preference_v6"
        elif trained_score_0_100 is not None:
            base_score_0_100 = float(trained_score_0_100)
            base_source = "trained_scorer"
    elif mode == "preference":
        if preference_score_0_100 is not None:
            base_score_0_100 = float(preference_score_0_100)
            base_source = "preference_v6"
        else:
            out["primary_score_warning"] = "primary_score_mode=preference but preference_model_path missing/failed; falling back to rubric"
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
        if preference_score_0_100 is not None:
            base_score_0_100 = float(preference_score_0_100)
            base_source = "preference_v6"
        elif trained_score_0_100 is not None:
            base_score_0_100 = float(trained_score_0_100)
            base_source = "trained_scorer"

    apply_antipattern_penalty = bool(apply_antipattern_penalty)
    adjusted_score_0_100 = float(base_score_0_100)
    penalty_0_100 = 0.0
    suggested_penalty_0_100 = 0.0
    combined_preview_0_100 = float(base_score_0_100)
    anti_soft_threshold = None
    anti_hard_threshold = None
    anti_cap_0_100 = None
    anti_risk_soft = None
    anti_risk_hard = None
    anti_combiner_mode = str(antipattern_combiner_mode or "adaptive").strip().lower()
    anti_prob = None
    anti_prob_raw = None
    anti_prob_inverted = False
    if antipattern_score is not None:
        anti_prob = float(antipattern_score.get("prob_0_1") or 0.0)
        anti_prob_raw = anti_prob
        anti_prob, anti_prob_inverted, anti_prob_inversion_msg = _resolve_antipattern_prob(
            antipattern_model_path,
            anti_prob_raw,
        )
        if anti_prob_inversion_msg is not None:
            out.setdefault("primary_score_warning", "antipattern score polarity corrected for AI-likelihood.")
            prev = str(out.get("primary_score_warning") or "").strip()
            if prev and anti_prob_inversion_msg not in prev:
                out["primary_score_warning"] = f"{prev}; {anti_prob_inversion_msg}"
            else:
                out["primary_score_warning"] = anti_prob_inversion_msg
        anti_adj = _compute_antipattern_adjustment(
            base_score_0_100=base_score_0_100,
            anti_prob_0_1=float(anti_prob),
            threshold_0_1=float(antipattern_penalty_threshold),
            weight=float(antipattern_penalty_weight),
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
    anti_warning = _antipattern_model_warning(antipattern_model_path, anti_prob)
    if anti_warning is not None:
        prev_warning = str(out.get("primary_score_warning") or "").strip()
        out["primary_score_warning"] = (
            anti_warning if not prev_warning else f"{prev_warning}; {anti_warning}"
        )
    if scorer_warning is not None:
        prev_warning = str(out.get("primary_score_warning") or "").strip()
        if scorer_warning not in prev_warning:
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
    if calibrated is not None:
        primary["calibrator_path"] = str(calibrated["calibrator_path"])
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
        "enabled": bool((antipattern_model_path or "").strip()),
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


@app.function(image=image, gpu="any", timeout=900, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
def write_like_remote(
    prompt: str,
    *,
    reference_text: str,
    doc_type: str = "prose",
    model_id: str = "gpt2",
    max_new_tokens: int = 200,
    seed: Optional[int] = 7,
) -> Dict[str, Any]:
    _bootstrap_repo()
    from tools.studio.write_like import write_like

    out = write_like(
        prompt=prompt or " ",
        reference_text=reference_text,
        model_id=model_id,
        backend="hf",
        doc_type=doc_type,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    hf_cache_vol.commit()
    return out.to_dict()


@app.function(image=image, gpu="any", timeout=600, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
def patch_suggest_remote(
    text: str,
    *,
    doc_type: str = "prose",
    scoring_model_id: str = "gpt2",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    window_sentences: int = 4,
    max_zones: int = 6,
) -> Dict[str, Any]:
    _bootstrap_repo()
    from tools.studio.span_patcher import suggest_dead_zones

    out = suggest_dead_zones(
        text,
        doc_type=str(doc_type),
        scoring_model_id=str(scoring_model_id),
        backend="hf",
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        window_sentences=int(window_sentences),
        max_zones=int(max_zones),
    )
    hf_cache_vol.commit()
    return out


@app.function(image=image, gpu="any", timeout=900, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
def patch_span_remote(
    text: str,
    *,
    doc_type: str = "prose",
    start_char: int = 0,
    end_char: int = 0,
    rewrite_mode: str = "strict",
    intensity: float = 0.5,
    rewrite_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    scoring_model_id: str = "gpt2",
    baseline_model_id: str = "gpt2_gutenberg_512",
    calibrator_path: str = "",
    scorer_model_path: str = "",
    scorer_max_length: int = 384,
    score_top_n: int = 3,
    max_input_tokens: int = 384,
    normalize_text: bool = True,
    n_candidates: int = 6,
    max_new_tokens: int = 260,
    temperature: float = 0.8,
    top_p: float = 0.92,
    seed: Optional[int] = 7,
) -> Dict[str, Any]:
    _bootstrap_repo()
    # Surface rewrite model failures (otherwise generation exceptions are swallowed
    # and the UI only sees "no_rewrites_generated").
    os.environ.setdefault("HORACE_RAISE_REWRITE_ERRORS", "1")
    from tools.studio.meaning_lock import MeaningLockConfig
    from tools.studio.span_patcher import patch_span

    cfg = MeaningLockConfig()
    out = patch_span(
        text,
        start_char=int(start_char),
        end_char=int(end_char),
        doc_type=str(doc_type),
        rewrite_mode=str(rewrite_mode),
        intensity=float(intensity),
        rewrite_model_id=str(rewrite_model_id),
        scoring_model_id=str(scoring_model_id),
        baseline_model_id=str(baseline_model_id),
        calibrator_path=str(calibrator_path or ""),
        scorer_model_path=str(scorer_model_path or ""),
        scorer_max_length=int(scorer_max_length),
        score_top_n=int(score_top_n),
        backend="hf",
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        n_candidates=int(n_candidates),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed) if seed is not None else None,
        meaning_lock=cfg,
    )
    hf_cache_vol.commit()
    return out


@app.function(image=image, volumes={"/cache/hf": hf_cache_vol, "/vol": data_vol})
@modal.asgi_app()
def fastapi_app():  # pragma: no cover
    _bootstrap_repo()
    import time
    from collections import defaultdict
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import HTMLResponse, JSONResponse
    from typing import Dict, Any

    import tools.studio.site
    import importlib
    importlib.reload(tools.studio.site)
    from tools.studio.site import API_HTML, STUDIO_HTML

    web = FastAPI(title="Horace")

    @web.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in (err.get("loc") or []) if x != "body")
            msg = str(err.get("msg") or err.get("type") or "invalid input")
            details.append(f"{loc}: {msg}" if loc else msg)
        return JSONResponse(
            {"error": "invalid_request", "details": "; ".join(details) or "Request validation failed"},
            status_code=422,
        )

    # Simple in-memory rate limiting (per container)
    _rate_limits: Dict[str, list] = defaultdict(list)
    RATE_LIMIT = 30  # requests per minute per IP
    RATE_WINDOW = 60  # seconds
    API_KEY = (os.environ.get("HORACE_API_KEY") or "").strip()
    OPEN_PATHS = ("/", "/api", "/docs", "/openapi.json", "/healthz")

    @web.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path in OPEN_PATHS:
            return await call_next(request)

        if API_KEY:
            auth = (request.headers.get("authorization") or "").strip()
            if auth.lower().startswith("bearer "):
                supplied = auth[7:].strip()
            else:
                supplied = (request.headers.get("x-api-key") or "").strip()
            if not supplied or supplied != API_KEY:
                return JSONResponse({"error": "unauthorized"}, status_code=401)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries and add new request
        _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if now - t < RATE_WINDOW]

        if len(_rate_limits[client_ip]) >= RATE_LIMIT:
            return JSONResponse(
                {"error": "Rate limit exceeded. Please wait a minute before trying again."},
                status_code=429
            )

        _rate_limits[client_ip].append(now)
        return await call_next(request)

    @web.get("/healthz")
    def healthz():
        return {"ok": True}

    @web.get("/api")
    async def api_docs():
        return HTMLResponse(content=API_HTML)

    @web.post("/analyze")
    async def analyze(req: AnalyzeReq):
        baseline = (req.baseline_model_id or req.baseline_model or "gpt2").strip()
        try:
            return analyze_remote.remote(
                req.text,
                doc_type=req.doc_type,
                scorer_model_path=str(req.scorer_model_path or ""),
                scorer_max_length=int(req.scorer_max_length),
                preference_model_path=str(req.preference_model_path or ""),
                antipattern_model_path=str(req.antipattern_model_path or ""),
                antipattern_max_length=int(req.antipattern_max_length),
                antipattern_penalty_weight=float(req.antipattern_penalty_weight),
                antipattern_penalty_threshold=float(req.antipattern_penalty_threshold),
                antipattern_combiner_mode=str(req.antipattern_combiner_mode or "adaptive"),
                apply_antipattern_penalty=bool(req.apply_antipattern_penalty),
                primary_score_mode=str(req.primary_score_mode or "auto"),
                primary_blend_weight=float(req.primary_blend_weight),
                fast_only=bool(req.fast_only),
                scoring_model_id=req.scoring_model_id,
                baseline_model_id=baseline,
                max_input_tokens=req.max_input_tokens,
                normalize_text=bool(req.normalize_text),
                calibrator_path=str(req.calibrator_path or ""),
            )
        except Exception as e:
            return JSONResponse(
                {"error": "analyze_failed", "details": f"{type(e).__name__}: {e}"},
                status_code=500,
            )

    @web.post("/rewrite")
    async def rewrite(req: RewriteReq):
        baseline = (req.baseline_model_id or req.baseline_model or "gpt2").strip()
        try:
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
        except Exception as e:
            return JSONResponse(
                {"error": "rewrite_failed", "details": f"{type(e).__name__}: {e}"},
                status_code=500,
            )

    @web.post("/write-like")
    @web.post("/cadence-match")
    async def cadence_match(req: WriteLikeReq):
        try:
            return write_like_remote.remote(
                req.prompt,
                reference_text=req.reference_text,
                doc_type=req.doc_type,
                model_id=req.model_id,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed,
            )
        except Exception as e:
            return JSONResponse(
                {"error": "cadence_match_failed", "details": f"{type(e).__name__}: {e}"},
                status_code=500,
            )

    @web.post("/patch/suggest")
    async def patch_suggest(req: PatchSuggestReq):
        try:
            return patch_suggest_remote.remote(
                req.text,
                doc_type=req.doc_type,
                scoring_model_id=req.scoring_model_id,
                max_input_tokens=req.max_input_tokens,
                normalize_text=bool(req.normalize_text),
                window_sentences=req.window_sentences,
                max_zones=req.max_zones,
            )
        except Exception as e:
            return JSONResponse(
                {"error": "patch_suggest_failed", "details": f"{type(e).__name__}: {e}"},
                status_code=500,
            )

    @web.post("/patch/span")
    async def patch_span(req: PatchSpanReq):
        baseline = (req.baseline_model_id or req.baseline_model or "gpt2").strip()
        try:
            return patch_span_remote.remote(
                req.text,
                doc_type=req.doc_type,
                start_char=int(req.start_char),
                end_char=int(req.end_char),
                rewrite_mode=str(req.rewrite_mode or "strict"),
                intensity=float(req.intensity),
                rewrite_model_id=str(req.rewrite_model_id or "Qwen/Qwen2.5-0.5B-Instruct"),
                scoring_model_id=str(req.scoring_model_id or "gpt2"),
                baseline_model_id=baseline,
                calibrator_path=str(req.calibrator_path or ""),
                scorer_model_path=str(req.scorer_model_path or ""),
                scorer_max_length=int(req.scorer_max_length),
                score_top_n=int(req.score_top_n),
                max_input_tokens=int(req.max_input_tokens),
                normalize_text=bool(req.normalize_text),
                n_candidates=int(req.n_candidates),
                max_new_tokens=int(req.max_new_tokens),
                temperature=float(req.temperature),
                top_p=float(req.top_p),
                seed=req.seed,
            )
        except Exception as e:
            return JSONResponse(
                {"error": "patch_span_failed", "details": f"{type(e).__name__}: {e}"},
                status_code=500,
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
