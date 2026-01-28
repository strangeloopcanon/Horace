#!/usr/bin/env python3
"""
Horace Studio (local Gradio app)

Paste a piece of writing and get:
- Score (0-100) + sub-scores
- Profile vs reference baselines (percentiles)
- Suggestions grounded in metrics + spike examples
- Optional rewrite + rerank (slow; uses an instruct model)

Run:
  python tools/studio_ui.py --host 127.0.0.1 --port 7861
"""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

# Allow running both as a module (-m tools.studio_ui) and as a script (python tools/studio_ui.py)
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline, load_baseline_cached
from tools.studio.score import score_text
from tools.studio.critique import suggest_edits
from tools.studio.llm_critic import llm_critique
from tools.studio.rewrite import rewrite_and_rerank
from tools.studio.span_patcher import patch_span as patch_one_span
from tools.studio.span_patcher import suggest_dead_zones
from tools.studio.write_like import write_like, extract_cadence_for_display
from tools.studio.windowed_cadence import windowed_cadence_for_text


def _ensure_baseline(model_id: str):
    ident = (model_id or "").strip()
    if not ident:
        ident = "gpt2"
    p = Path(ident)
    if p.exists():
        return load_baseline_cached(ident, path=p)
    try:
        return load_baseline_cached(ident)
    except Exception:
        build_baseline(ident)
        return load_baseline_cached(ident)


def _plot_surprisal(series: Dict[str, Any]) -> Optional[Image.Image]:
    if not series:
        return None
    s = series.get("surprisal") or []
    if not s:
        return None
    thr = float(series.get("threshold_surprisal") or 0.0)
    fig = plt.figure(figsize=(7.0, 2.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.array(s, dtype=np.float32), lw=0.8)
    if thr > 0:
        ax.axhline(thr, color="red", linestyle="--", alpha=0.6, label="high-surprise threshold")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Token (first window)")
    ax.set_ylabel("Surprisal")
    ax.set_title("Cadence (surprisal series)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _plot_cadence_timeline(timeline) -> Optional[Image.Image]:
    """Plot cadence timeline as a horizontal bar chart showing window quality."""
    if not timeline or not timeline.windows:
        return None
    
    n = len(timeline.windows)
    scores = [w.cadence_score for w in timeline.windows]
    colors = []
    for i, w in enumerate(timeline.windows):
        if w.is_worst:
            colors.append("#ff6b6b")  # Red for worst
        elif w.is_best:
            colors.append("#51cf66")  # Green for best
        else:
            # Gradient from yellow to green based on score
            normalized = min(1.0, max(0.0, w.cadence_score / 100))
            colors.append(plt.cm.RdYlGn(normalized))
    
    fig, ax = plt.subplots(figsize=(8.0, 1.8))
    
    # Horizontal bar for each window
    bars = ax.barh(range(n), scores, color=colors, edgecolor="white", height=0.7)
    
    # Labels
    ax.set_yticks(range(n))
    labels = []
    for i, w in enumerate(timeline.windows):
        lbl = f"W{i+1}"
        if w.is_worst:
            lbl += " ⚠"
        elif w.is_best:
            lbl += " ★"
        labels.append(lbl)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cadence Score")
    ax.set_xlim(0, 100)
    ax.set_title(f"Cadence Timeline (pacing variety: {timeline.pacing_variety:.2f})")
    ax.invert_yaxis()  # Top to bottom
    
    # Add score annotations
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f"{score:.0f}", va="center", fontsize=8)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _md_score(score, baseline_model: str, doc_type: str, *, tokens_count: int, truncated: bool) -> str:
    lines = []
    lines.append(f"### Score: **{score.overall_0_100:.1f}/100**")
    if score.categories:
        lines.append("")
        lines.append("**Sub-scores**")
        for k, v in score.categories.items():
            lines.append(f"- `{k}`: {v*100:.0f}/100")
    lines.append("")
    lines.append(f"Tokens analyzed: `{int(tokens_count)}`" + (" (truncated)" if truncated else ""))
    if int(tokens_count) > 0 and int(tokens_count) < 80:
        lines.append("_Note: short input → low stability; scores/percentiles can swing a lot._")
    lines.append(f"Baseline: `{baseline_model}` (slice `{doc_type}`)")
    return "\n".join(lines)


def _md_profile(score) -> str:
    # Show rubric metrics with percentiles
    lines = []
    lines.append("**Profile (rubric metrics)**")
    for k, ms in score.metrics.items():
        p = "N/A" if ms.percentile is None else f"{ms.percentile:.0f}th"
        s = "N/A" if ms.score_0_1 is None else f"{ms.score_0_1*100:.0f}/100"
        lines.append(f"- `{k}`: value={ms.value:.3f}, pctl={p}, metric_score={s} ({ms.mode})")
    return "\n".join(lines)


def _md_spikes(spikes) -> str:
    if not spikes:
        return "_No high-surprise spikes found (short text or very flat surprisal)._"
    lines = []
    lines.append("**Top spikes (surprisal ≥ mean+std)**")
    for i, s in enumerate(spikes[:8], 1):
        ctx = (s.get("context") or "").replace("\n", "\\n")
        surface = (s.get("surface") or "").replace("\n", "\\n")
        lines.append(
            f"{i}. s={s.get('surprisal'):.2f}, entropy={s.get('entropy'):.2f}, "
            f"content={bool(s.get('is_content'))}, punct={bool(s.get('is_punct'))}, "
            f"line_pos={s.get('line_pos')} — `...{surface}...`"
        )
        lines.append(f"   - {ctx}")
    return "\n".join(lines)


def run_analyze(
    text: str,
    doc_type: str,
    scorer_model_path: str,
    scorer_max_length: int,
    fast_only: bool,
    scoring_model: str,
    baseline_model: str,
    calibrator_path: str,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
    use_llm_critic: bool,
    critic_model: str,
    critic_max_new_tokens: int,
    critic_temperature: float,
    critic_top_p: float,
    critic_seed: Optional[int],
):
    if not text or not text.strip():
        return "", "", "", "", None, None, {}

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
                device=None,
            )
            trained_score = ts
        except Exception as e:
            trained_err = f"{type(e).__name__}: {e}"

    if bool(fast_only):
        if trained_score is None:
            msg = "_Fast mode requires a valid trained scorer model path._"
            if trained_err:
                msg += f"\n\nError: `{trained_err}`"
            return msg, "", "", "", None, None, {"trained_score_error": trained_err}
        score_md = []
        score_md.append(f"### Trained scorer: **{trained_score.score_0_100:.1f}/100**")
        score_md.append(f"- prob: `{trained_score.prob_0_1:.3f}`")
        score_md.append(f"- model: `{trained_score.model_path_or_id}`")
        score_md.append(f"- device: `{trained_score.device}`")
        out_json = {"trained_score": asdict(trained_score)}
        return "\n".join(score_md), "", "", "", None, None, out_json

    analysis = analyze_text(
        text,
        model_id=scoring_model.strip() or "gpt2",
        doc_type=doc_type,
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        compute_cohesion=bool(compute_cohesion),
    )
    baseline = _ensure_baseline(baseline_model.strip() or "gpt2")
    score = score_text(analysis["doc_metrics"], baseline, doc_type=doc_type)
    critique = suggest_edits(
        doc_metrics=analysis["doc_metrics"],
        score=score,
        spikes=analysis.get("spikes") or [],
        segments=analysis.get("segments") or {},
    )

    score_md = _md_score(
        score,
        baseline_model=baseline.model_id,
        doc_type=doc_type,
        tokens_count=int(analysis["doc_metrics"].get("tokens_count") or 0),
        truncated=bool(analysis.get("truncated")),
    )
    if (calibrator_path or "").strip():
        try:
            from tools.studio.calibrator import featurize_from_report_row, load_logistic_calibrator

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
            cal_score = float(cal.score_0_100(feats))
            score_md += f"\n\nCalibrated score: **{cal_score:.1f}/100** (from `{calibrator_path}`)"
        except Exception as e:
            score_md += f"\n\n_Calibrator failed: {type(e).__name__}: {e}_"
    if trained_score is not None:
        score_md = (
            f"### Primary score: **{trained_score.score_0_100:.1f}/100** (trained scorer)\n"
            f"- model: `{trained_score.model_path_or_id}`\n\n"
            + score_md
        )
    if trained_err is not None:
        score_md += f"\n\n_Trained scorer failed: {trained_err}_"
    profile_md = _md_profile(score)
    spikes_md = _md_spikes(analysis.get("spikes") or [])
    plot_img = _plot_surprisal(analysis.get("series") or {})
    
    out_json = {
        "analysis": analysis,
        "score": {
            "overall_0_100": score.overall_0_100,
            "categories": score.categories,
            "metrics": {k: {"value": v.value, "percentile": v.percentile, "score_0_1": v.score_0_1, "mode": v.mode} for k, v in score.metrics.items()},
        },
        "critique": critique,
        "primary_score": {"overall_0_100": float(score.overall_0_100), "source": "rubric"},
    }
    if trained_score is not None:
        out_json["trained_score"] = asdict(trained_score)
        out_json["primary_score"] = {"overall_0_100": float(trained_score.score_0_100), "source": "trained_scorer"}
    if trained_err is not None:
        out_json["trained_score_error"] = trained_err
    
    # Compute cadence timeline for longer texts
    timeline_img = None
    tokens_count = int(analysis["doc_metrics"].get("tokens_count") or 0)
    if tokens_count >= 100:  # Only compute timeline for longer texts
        try:
            timeline = windowed_cadence_for_text(
                text,
                model_id=scoring_model.strip() or "gpt2",
                doc_type=doc_type,
                max_input_tokens=int(max_input_tokens),
                normalize_text=bool(normalize_text),
            )
            timeline_img = _plot_cadence_timeline(timeline)
            out_json["timeline"] = timeline.to_dict()
        except Exception:
            pass  # Timeline is optional, don't fail on errors

    llm_md = ""
    if bool(use_llm_critic):
        mid = (critic_model or "").strip()
        if not mid:
            llm_md = "_LLM critic enabled, but no critic model id was provided._"
        else:
            try:
                llm = llm_critique(
                    text=text,
                    doc_type=doc_type,
                    score=score,
                    doc_metrics=analysis["doc_metrics"],
                    spikes=analysis.get("spikes") or [],
                    segments=analysis.get("segments") or {},
                    model_id=mid,
                    max_new_tokens=int(critic_max_new_tokens),
                    temperature=float(critic_temperature),
                    top_p=float(critic_top_p),
                    seed=int(critic_seed) if critic_seed is not None else None,
                )
                out_json["llm_critique"] = llm
                llm_md = "### LLM Critique\n"
                llm_md += (llm.get("summary") or "").strip() + "\n\n"
                for s in llm.get("suggestions") or []:
                    llm_md += f"- **{s.get('title','')}** — {s.get('why','')}\n  - Try: {s.get('what_to_try','')}\n"
                    if s.get("evidence"):
                        llm_md += f"  - Evidence: {s.get('evidence')}\n"
            except Exception as e:
                llm_md = f"_LLM critic failed: {type(e).__name__}: {e}_"

    # Put suggestions into markdown for readability
    suggestions_md = "### Suggestions\n"
    suggestions_md += critique.get("summary", "") + "\n\n"
    for s in critique.get("suggestions") or []:
        suggestions_md += f"- **{s.get('title','')}** — {s.get('why','')}\n  - Try: {s.get('what_to_try','')}\n"
        if s.get("evidence"):
            suggestions_md += f"  - Evidence: {s.get('evidence')}\n"
    return score_md, profile_md, suggestions_md + "\n\n" + spikes_md, llm_md, plot_img, timeline_img, out_json


def run_rewrite(
    text: str,
    doc_type: str,
    rewrite_model: str,
    scoring_model: str,
    baseline_model: str,
    calibrator_path: str,
    n_candidates: int,
    keep_top: int,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
):
    if not text or not text.strip():
        return "", {}
    res = rewrite_and_rerank(
        text,
        doc_type=doc_type,
        rewrite_model_id=rewrite_model.strip() or "gpt2",
        scoring_model_id=scoring_model.strip() or "gpt2",
        baseline_model_id=baseline_model.strip() or "gpt2",
        calibrator_path=str(calibrator_path or ""),
        n_candidates=int(n_candidates),
        keep_top=int(keep_top),
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed) if seed is not None else None,
        compute_cohesion=bool(compute_cohesion),
    )

    md = []
    rank_by = str((res.get("meta") or {}).get("rank_by") or "score")
    orig_best = res["original"].get(rank_by) if isinstance(res.get("original"), dict) else None
    if not isinstance(orig_best, (int, float)):
        orig_best = res["original"].get("score")
    md.append(f"### Original score ({rank_by}): **{float(orig_best):.1f}/100**")
    md.append("")
    for i, r in enumerate(res.get("rewrites") or [], 1):
        best = r.get(rank_by)
        if not isinstance(best, (int, float)):
            best = r.get("score", 0.0)
        delta = ((r.get("delta") or {}).get("overall_delta_0_100")) if isinstance(r.get("delta"), dict) else None
        delta_s = ""
        if isinstance(delta, (int, float)):
            delta_s = f" (Δ {float(delta):+.1f})"
        md.append(f"### Rewrite {i} score ({rank_by}): **{float(best):.1f}/100**{delta_s}")
        cat_d = ((r.get("delta") or {}).get("categories_delta_0_100")) if isinstance(r.get("delta"), dict) else None
        if isinstance(cat_d, dict) and cat_d:
            items = sorted(cat_d.items(), key=lambda kv: abs(float(kv[1] or 0.0)), reverse=True)[:4]
            md.append("- Category deltas: " + ", ".join([f"`{k}` {float(v):+.0f}" for k, v in items]))
        gains = ((r.get("delta") or {}).get("top_metric_gains")) if isinstance(r.get("delta"), dict) else None
        if isinstance(gains, list) and gains:
            md.append(
                "- Top metric gains: "
                + ", ".join([f"`{g.get('metric')}` {float(g.get('delta_score_0_1') or 0.0)*100:+.0f}" for g in gains[:3]])
            )
        md.append("```")
        md.append(r.get("text", "").strip())
        md.append("```")
    return "\n".join(md), res


def _md_dead_zones(zones: list[dict]) -> str:
    if not zones:
        return "_No obvious dead zones found (or text too short)._"
    lines = []
    lines.append("**Dead zones (low texture candidates)**")
    for z in zones:
        zid = z.get("zone_id")
        sev = z.get("severity")
        s = z.get("start_char")
        e = z.get("end_char")
        reasons = ", ".join(z.get("reasons") or [])
        excerpt = (z.get("excerpt") or "").replace("\n", " ").strip()
        lines.append(f"- `{zid}` sev={sev:.2f} chars={s}-{e} ({reasons}) — {excerpt}")
    return "\n".join(lines)


def patch_mode_preset(mode: str):
    m = (mode or "strict").strip().lower()
    if m == "creative":
        return (
            gr.Slider.update(value=0.95),
            gr.Slider.update(value=0.95),
            gr.Slider.update(value=0.82),
            gr.Slider.update(value=1.80),
            gr.Slider.update(value=0.75),
        )
    return (
        gr.Slider.update(value=0.8),
        gr.Slider.update(value=0.92),
        gr.Slider.update(value=0.86),
        gr.Slider.update(value=1.45),
        gr.Slider.update(value=0.55),
    )


def _md_history(history: list[str]) -> str:
    if not history:
        return "_No patches applied yet._"
    lines = []
    lines.append(f"**Patch history** ({len(history)} undo levels)")
    for i, txt in enumerate(history[-3:], 1):
        preview = (txt or "").replace("\n", " ").strip()
        if len(preview) > 140:
            preview = preview[:140].rstrip() + "…"
        lines.append(f"- {max(1, len(history) - 3 + i)}: {len(txt)} chars — {preview}")
    return "\n".join(lines)


def apply_best_patch(current_text: str, best_text: str, history: list[str]):
    if not best_text or not best_text.strip():
        return current_text, history or [], _md_history(history or [])
    hist = list(history or [])
    hist.append(current_text or "")
    md = _md_history(hist)
    a = (current_text or "").splitlines(keepends=True)
    b = (best_text or "").splitlines(keepends=True)
    dd = difflib.unified_diff(a, b, fromfile="doc_before", tofile="doc_after", lineterm="")
    out_dd = list(dd)
    if len(out_dd) > 140:
        out_dd = out_dd[:140] + ["… (diff truncated)"]
    diff_txt = "\n".join(out_dd).strip()
    if diff_txt:
        md += (
            "\n\n<details><summary>Last apply diff</summary>\n\n```diff\n"
            + diff_txt
            + "\n```\n</details>"
        )
    return best_text, hist, md


def undo_last_patch(current_text: str, history: list[str]):
    hist = list(history or [])
    if not hist:
        return current_text, hist, _md_history(hist)
    prev = hist.pop()
    md = _md_history(hist)
    a = (current_text or "").splitlines(keepends=True)
    b = (prev or "").splitlines(keepends=True)
    dd = difflib.unified_diff(a, b, fromfile="doc_before", tofile="doc_after", lineterm="")
    out_dd = list(dd)
    if len(out_dd) > 140:
        out_dd = out_dd[:140] + ["… (diff truncated)"]
    diff_txt = "\n".join(out_dd).strip()
    if diff_txt:
        md += (
            "\n\n<details><summary>Last undo diff</summary>\n\n```diff\n"
            + diff_txt
            + "\n```\n</details>"
        )
    return prev, hist, md


def run_patch_suggest(
    text: str,
    doc_type: str,
    scoring_model: str,
    max_input_tokens: int,
    normalize_text: bool,
    window_sentences: int,
    max_zones: int,
):
    if not text or not text.strip():
        return text, "_No text provided._", {}, gr.Dropdown.update(choices=[], value=None), []
    out = suggest_dead_zones(
        text,
        doc_type=str(doc_type),
        scoring_model_id=str(scoring_model).strip() or "gpt2",
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        window_sentences=int(window_sentences),
        max_zones=int(max_zones),
    )
    zones = out.get("dead_zones") or []
    md = _md_dead_zones(zones)
    choices = [
        f"{z.get('zone_id')}: sev={float(z.get('severity') or 0.0):.2f} ({', '.join(z.get('reasons') or [])})"
        for z in zones
    ]
    dd = gr.Dropdown.update(choices=choices, value=(choices[0] if choices else None))
    return out.get("text") or text, md, out, dd, zones


def run_patch_span(
    text: str,
    doc_type: str,
    selected_zone: str,
    zones_state: list[dict],
    rewrite_mode: str,
    intensity: float,
    rewrite_model_id: str,
    scoring_model: str,
    baseline_model: str,
    calibrator_path: str,
    scorer_model_path: str,
    scorer_max_length: int,
    max_input_tokens: int,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    min_cosine_sim: float,
    max_length_ratio: float,
    max_edit_ratio: float,
    allow_new_numbers: bool,
    allow_new_proper_nouns: bool,
    allow_negation_change: bool,
):
    if not text or not text.strip():
        return "_No text provided._", "", {}
    if not selected_zone:
        return "_Select a dead zone first (or re-run suggest)._", "", {}
    try:
        zid = int(str(selected_zone).split(":", 1)[0].strip())
    except Exception:
        return "_Could not parse selected zone id._", "", {}
    zone = None
    for z in zones_state or []:
        if int(z.get("zone_id") or -1) == zid:
            zone = z
            break
    if zone is None:
        return "_Selected zone not found; re-run suggest._", "", {}

    from tools.studio.meaning_lock import MeaningLockConfig

    cfg = MeaningLockConfig(
        min_cosine_sim=float(min_cosine_sim),
        max_length_ratio=float(max_length_ratio),
        max_edit_ratio=float(max_edit_ratio),
        allow_new_numbers=bool(allow_new_numbers),
        allow_new_proper_nouns=bool(allow_new_proper_nouns),
        allow_negation_change=bool(allow_negation_change),
    )
    out = patch_one_span(
        text,
        start_char=int(zone.get("start_char") or 0),
        end_char=int(zone.get("end_char") or 0),
        doc_type=str(doc_type),
        rewrite_mode=str(rewrite_mode or "strict"),
        intensity=float(intensity),
        rewrite_model_id=str(rewrite_model_id).strip() or "Qwen/Qwen2.5-0.5B-Instruct",
        scoring_model_id=str(scoring_model).strip() or "gpt2",
        baseline_model_id=str(baseline_model).strip() or "gpt2_gutenberg_512",
        calibrator_path=str(calibrator_path or ""),
        scorer_model_path=str(scorer_model_path or ""),
        scorer_max_length=int(scorer_max_length),
        score_top_n=3,
        max_input_tokens=int(max_input_tokens),
        normalize_text=False,  # already normalized in this tab
        n_candidates=int(n_candidates),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed) if seed is not None else None,
        meaning_lock=cfg,
    )

    cands = out.get("candidates") or []
    if not cands:
        msg = "_No candidates passed MeaningLock; try relaxing thresholds or increasing candidates._"
        if out.get("error"):
            msg += f"\n\nError: `{out.get('error')}`"
        return msg, "", out

    lines = []
    pb = out.get("primary_before")
    pb_err = out.get("primary_before_error")
    if isinstance(pb, dict) and isinstance(pb.get("overall_0_100"), (int, float)):
        src = pb.get("source") or "unknown"
        lines.append(f"**Primary score before**: `{float(pb['overall_0_100']):.1f}/100` ({src})")
    elif pb_err:
        lines.append(f"_Primary score failed: {pb_err}_")
    lines.append("")
    ctrl = out.get("control") or {}
    inten = ctrl.get("intensity_0_1")
    inten_s = "N/A" if not isinstance(inten, (int, float)) else f"{float(inten):.2f}"
    lines.append(f"**Patch candidates** (sorted by `rank_score`; intensity={inten_s}; 0–100 score is secondary)")
    best_text = ""
    best_doc_diff = ""
    for i, c in enumerate(cands[:5], 1):
        gain = float(c.get("texture_gain") or 0.0)
        rank = c.get("rank_score")
        dr = c.get("droning_delta")
        pdelta = c.get("primary_delta_0_100")
        pafter = (c.get("primary_after") or {}).get("overall_0_100") if isinstance(c.get("primary_after"), dict) else None
        sim = (c.get("meaning_lock") or {}).get("cosine_sim")
        er = (c.get("meaning_lock") or {}).get("edit_ratio")
        ml = c.get("meaning_lock") or {}
        sim_s = "N/A" if not isinstance(sim, (int, float)) else f"{float(sim):.3f}"
        er_s = "N/A" if not isinstance(er, (int, float)) else f"{float(er):.2f}"
        score_s = ""
        if isinstance(pafter, (int, float)):
            score_s = f"  score={float(pafter):.1f}"
        if isinstance(pdelta, (int, float)):
            score_s += f" (Δ {float(pdelta):+.1f})"
        rank_s = "N/A" if not isinstance(rank, (int, float)) else f"{float(rank):+.3f}"
        dr_s = "N/A" if not isinstance(dr, (int, float)) else f"{float(dr):+.3f}"
        lines.append(f"{i}. rank={rank_s}  Δtexture={gain:+.3f}  Δdroning={dr_s}{score_s}  sim={sim_s}  edit={er_s}")
        facts = []
        for key, label in [
            ("numbers_added", "+nums"),
            ("numbers_removed", "-nums"),
            ("proper_nouns_added", "+PN"),
            ("proper_nouns_removed", "-PN"),
            ("negations_added", "+neg"),
            ("negations_removed", "-neg"),
        ]:
            vals = ml.get(key) if isinstance(ml, dict) else None
            if isinstance(vals, list) and vals:
                facts.append(f"{label}={vals}")
        if facts:
            lines.append(f"   - fact diff: `{'; '.join(facts)}`")
        lines.append("```diff")
        lines.append(c.get("span_diff") or "")
        lines.append("```")
        if i == 1:
            best_text = c.get("patched_text") or ""
            if best_text.strip():
                a = (text or "").splitlines(keepends=True)
                b = best_text.splitlines(keepends=True)
                dd = difflib.unified_diff(a, b, fromfile="doc_before", tofile="doc_after", lineterm="")
                out_dd = list(dd)
                if len(out_dd) > 140:
                    out_dd = out_dd[:140] + ["… (diff truncated)"]
                best_doc_diff = "\n".join(out_dd).strip()

    if best_doc_diff:
        lines.append("")
        lines.append("<details><summary>Document diff (best candidate)</summary>")
        lines.append("")
        lines.append("```diff")
        lines.append(best_doc_diff)
        lines.append("```")
        lines.append("</details>")

    return "\n".join(lines), best_text, out


def _plot_cadence_comparison(
    ref_surprisal: Optional[list],
    gen_surprisal: Optional[list],
) -> Optional[Image.Image]:
    """Plot reference vs generated cadence side by side."""
    if not ref_surprisal and not gen_surprisal:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))

    if ref_surprisal:
        axes[0].plot(ref_surprisal, lw=0.8, color="#4CAF50")
        axes[0].set_title("Reference Cadence")
        axes[0].set_xlabel("Token")
        axes[0].set_ylabel("Surprisal")
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")

    if gen_surprisal:
        axes[1].plot(gen_surprisal, lw=0.8, color="#2196F3")
        axes[1].set_title("Generated Cadence")
        axes[1].set_xlabel("Token")
        axes[1].set_ylabel("Surprisal")
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def run_write_like(
    prompt: str,
    reference_text: str,
    doc_type: str,
    model_id: str,
    max_new_tokens: int,
    seed: Optional[int],
):
    """Handler for Cadence Match tab."""
    if not reference_text or not reference_text.strip():
        return "Please provide reference text.", "", None, {}

    if not prompt or not prompt.strip():
        prompt = " "

    try:
        result = write_like(
            prompt=prompt,
            reference_text=reference_text,
            model_id=model_id or "gpt2",
            doc_type=doc_type or "prose",
            max_new_tokens=int(max_new_tokens) if max_new_tokens else 200,
            seed=int(seed) if seed is not None else 7,
        )

        # Build markdown report
        lines = []
        lines.append("### Generated Text")
        lines.append("")
        lines.append(result.generated_text)
        lines.append("")
        lines.append("---")
        lines.append("### Cadence Match")
        match = result.cadence_match
        similarity = match.get("similarity_0_1", 0.0)
        lines.append(f"**Similarity:** {similarity:.1%}")
        lines.append("")
        lines.append("| Metric | Generated | Reference |")
        lines.append("|--------|-----------|-----------|")
        details = match.get("details") or {}
        for k, v in details.items():
            if isinstance(v, dict):
                gen_v = v.get("generated", "—")
                ref_v = v.get("reference", "—")
                if isinstance(gen_v, float):
                    gen_v = f"{gen_v:.2f}"
                if isinstance(ref_v, float):
                    ref_v = f"{ref_v:.2f}"
                lines.append(f"| {k} | {gen_v} | {ref_v} |")

        md = "\n".join(lines)

        # Get surprisal for plots
        ref_display = extract_cadence_for_display(
            reference_text,
            model_id=model_id or "gpt2",
            doc_type=doc_type or "prose",
        )
        gen_display = extract_cadence_for_display(
            result.generated_text,
            model_id=model_id or "gpt2",
            doc_type=doc_type or "prose",
        )

        ref_surp = ref_display.get("token_surprisal") or []
        gen_surp = gen_display.get("token_surprisal") or []

        plot = _plot_cadence_comparison(ref_surp, gen_surp)

        return md, result.generated_text, plot, result.to_dict()

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", "", None, {"error": f"{type(e).__name__}: {e}"}


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Horace Studio") as demo:
        gr.Markdown(
            """
            # Horace Studio (prototype)
            Paste writing → get a score, a profile vs reference baselines, suggestions, and rewrites reranked by Horace metrics.
            """
        )

        with gr.Row():
            doc_type = gr.Dropdown(
                choices=["poem", "prose", "essay", "shortstory", "novel"],
                value="prose",
                label="Type",
            )
            scoring_model = gr.Textbox(value="gpt2", label="Scoring model (analysis)")
            baseline_model = gr.Textbox(value="gpt2_gutenberg_512", label="Baseline model id (or baseline JSON path)")
            calibrator_path = gr.Textbox(value="", label="Calibrator JSON path (optional)")
            max_input_tokens = gr.Slider(128, 2048, value=512, step=64, label="Max input tokens (cap)")
            normalize_text = gr.Checkbox(value=True, label="Normalize formatting (fix hard wraps)")
            compute_cohesion = gr.Checkbox(value=False, label="Compute cohesion (slower)")

        with gr.Row():
            scorer_model_path = gr.Textbox(value="", label="Trained scorer model path (optional)")
            scorer_max_length = gr.Slider(64, 1024, value=384, step=32, label="Trained scorer max length")
            fast_only = gr.Checkbox(value=False, label="Fast mode (trained scorer only)")

        text = gr.Textbox(lines=14, label="Your text", value="At dawn, the city leans into light:\n")

        with gr.Tab("Analyze"):
            run_btn = gr.Button("Analyze")
            score_md = gr.Markdown()
            profile_md = gr.Markdown()
            suggestions_md = gr.Markdown()
            with gr.Accordion("LLM Critique (optional, slow)", open=False):
                use_llm_critic = gr.Checkbox(value=False, label="Enable LLM critique")
                critic_model = gr.Textbox(
                    value="",
                    label="Critic model id (HF)",
                    placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct",
                )
                with gr.Row():
                    critic_temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                    critic_top_p = gr.Slider(0.05, 0.99, value=0.95, step=0.01, label="Top-p")
                with gr.Row():
                    critic_max_new_tokens = gr.Slider(64, 900, value=450, step=16, label="Max new tokens")
                    critic_seed = gr.Number(value=None, precision=0, label="Seed (optional)")
                llm_md = gr.Markdown()
            plot_img = gr.Image(label="Cadence plot", type="pil")
            timeline_img = gr.Image(label="Cadence timeline (page pacing)", type="pil")
            out_json = gr.JSON(label="Raw JSON")

            run_btn.click(
                fn=run_analyze,
                inputs=[
                    text,
                    doc_type,
                    scorer_model_path,
                    scorer_max_length,
                    fast_only,
                    scoring_model,
                    baseline_model,
                    calibrator_path,
                    max_input_tokens,
                    normalize_text,
                    compute_cohesion,
                    use_llm_critic,
                    critic_model,
                    critic_max_new_tokens,
                    critic_temperature,
                    critic_top_p,
                    critic_seed,
                ],
                outputs=[score_md, profile_md, suggestions_md, llm_md, plot_img, timeline_img, out_json],
            )

        with gr.Tab("Rewrite + Rerank"):
            with gr.Row():
                rewrite_model = gr.Textbox(
                    value="gpt2",
                    label="Rewrite model (instruct works best; e.g. Qwen/Qwen2.5-0.5B-Instruct)",
                )
                n_candidates = gr.Slider(1, 8, value=4, step=1, label="Candidates")
                keep_top = gr.Slider(1, 5, value=3, step=1, label="Keep top")
            with gr.Row():
                max_new_tokens = gr.Slider(64, 800, value=300, step=16, label="Max new tokens (rewrite)")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.05, 0.99, value=0.92, step=0.01, label="Top-p")
                seed = gr.Number(value=7, precision=0, label="Seed")
            rewrite_btn = gr.Button("Rewrite + rerank (slow)")
            rewrite_md = gr.Markdown()
            rewrite_json = gr.JSON(label="Raw JSON")

            rewrite_btn.click(
                fn=run_rewrite,
                inputs=[
                    text,
                    doc_type,
                    rewrite_model,
                    scoring_model,
                    baseline_model,
                    calibrator_path,
                    n_candidates,
                    keep_top,
                    max_input_tokens,
                    normalize_text,
                    compute_cohesion,
                    max_new_tokens,
                    temperature,
                    top_p,
                    seed,
                ],
                outputs=[rewrite_md, rewrite_json],
            )

        with gr.Tab("Patch (dead zones)"):
            gr.Markdown(
                "Find low-texture spans and patch them with MeaningLock (semantic similarity + no-new-facts heuristics). "
                "Optimize locally with an intensity knob (clearer↔punchier); keep the 0–100 score as a secondary readout."
            )
            zones_state = gr.State([])
            history_state = gr.State([])
            with gr.Row():
                patch_mode = gr.Dropdown(choices=["strict", "creative"], value="strict", label="Patch mode")
                patch_intensity = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="Intensity (clearer ↔ punchier)")
                patch_rewrite_model = gr.Textbox(
                    value="Qwen/Qwen2.5-0.5B-Instruct",
                    label="Rewrite model (span patching; instruct works best)",
                )
                patch_candidates = gr.Slider(2, 12, value=6, step=1, label="Candidates")
                patch_max_new_tokens = gr.Slider(64, 520, value=260, step=16, label="Max new tokens (span)")
            with gr.Row():
                patch_temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                patch_top_p = gr.Slider(0.05, 0.99, value=0.92, step=0.01, label="Top-p")
                patch_seed = gr.Number(value=7, precision=0, label="Seed")
            with gr.Row():
                window_sentences = gr.Slider(3, 8, value=4, step=1, label="Dead-zone window (sentences)")
                max_zones = gr.Slider(1, 10, value=6, step=1, label="Max zones")
                suggest_btn = gr.Button("Suggest dead zones")

            zones_md = gr.Markdown()
            zones_json = gr.JSON(label="Raw JSON (zones + analysis)")
            zone_dd = gr.Dropdown(choices=[], value=None, label="Pick a zone")

            with gr.Accordion("MeaningLock (constraints)", open=False):
                with gr.Row():
                    min_cosine_sim = gr.Slider(0.70, 0.99, value=0.86, step=0.01, label="Min semantic similarity (cosine)")
                    max_length_ratio = gr.Slider(1.05, 2.50, value=1.45, step=0.05, label="Max length ratio")
                    max_edit_ratio = gr.Slider(0.10, 0.95, value=0.55, step=0.05, label="Max edit ratio (1-sim)")
                with gr.Row():
                    allow_new_numbers = gr.Checkbox(value=False, label="Allow number changes")
                    allow_new_proper = gr.Checkbox(value=False, label="Allow proper-noun changes")
                    allow_negation_change = gr.Checkbox(value=False, label="Allow negation changes")

            patch_btn = gr.Button("Patch selected zone (slow)")
            with gr.Row():
                apply_btn = gr.Button("Apply best patch to editor")
                undo_btn = gr.Button("Undo last apply")
            patch_md = gr.Markdown()
            patched_text = gr.Textbox(lines=14, label="Best patched text (candidate #1)")
            patch_json = gr.JSON(label="Raw JSON (candidates)")
            history_md = gr.Markdown()

            suggest_btn.click(
                fn=run_patch_suggest,
                inputs=[text, doc_type, scoring_model, max_input_tokens, normalize_text, window_sentences, max_zones],
                outputs=[text, zones_md, zones_json, zone_dd, zones_state],
            )

            patch_mode.change(
                fn=patch_mode_preset,
                inputs=[patch_mode],
                outputs=[patch_temperature, patch_top_p, min_cosine_sim, max_length_ratio, max_edit_ratio],
            )

            patch_btn.click(
                fn=run_patch_span,
                inputs=[
                    text,
                    doc_type,
                    zone_dd,
                    zones_state,
                    patch_mode,
                    patch_intensity,
                    patch_rewrite_model,
                    scoring_model,
                    baseline_model,
                    calibrator_path,
                    scorer_model_path,
                    scorer_max_length,
                    max_input_tokens,
                    patch_candidates,
                    patch_max_new_tokens,
                    patch_temperature,
                    patch_top_p,
                    patch_seed,
                    min_cosine_sim,
                    max_length_ratio,
                    max_edit_ratio,
                    allow_new_numbers,
                    allow_new_proper,
                    allow_negation_change,
                ],
                outputs=[patch_md, patched_text, patch_json],
            )

            apply_btn.click(
                fn=apply_best_patch,
                inputs=[text, patched_text, history_state],
                outputs=[text, history_state, history_md],
            )

            undo_btn.click(
                fn=undo_last_patch,
                inputs=[text, history_state],
                outputs=[text, history_state, history_md],
            )

        with gr.Tab("Cadence Match"):
            gr.Markdown(
                """
                ### Cadence Match
                Generate text that matches the *cadence* (spikes, lulls, cooldowns) of a reference passage.

                This is intentionally **cadence-only**: it tries to match rhythm, not copy vocabulary, facts, or voice.
                """
            )
            with gr.Row():
                wl_reference = gr.Textbox(
                    lines=8,
                    label="Reference Text (cadence to match)",
                    placeholder="Paste a paragraph whose rhythm you like...",
                )
                wl_prompt = gr.Textbox(
                    lines=4,
                    label="Your Prompt (starting text)",
                    placeholder="Begin your text here...",
                    value="The morning light crept through the window",
                )
            with gr.Row():
                wl_model = gr.Textbox(value="gpt2", label="Generation model")
                wl_max_tokens = gr.Slider(50, 500, value=200, step=10, label="Max tokens")
                wl_seed = gr.Number(value=7, precision=0, label="Seed")
            wl_btn = gr.Button("Generate", variant="primary")
            wl_md = gr.Markdown()
            wl_output = gr.Textbox(lines=6, label="Generated Text", interactive=False)
            wl_plot = gr.Image(label="Cadence Comparison", type="pil")
            wl_json = gr.JSON(label="Raw Output")

            wl_btn.click(
                fn=run_write_like,
                inputs=[
                    wl_prompt,
                    wl_reference,
                    doc_type,
                    wl_model,
                    wl_max_tokens,
                    wl_seed,
                ],
                outputs=[wl_md, wl_output, wl_plot, wl_json],
            )

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Horace Studio UI (Gradio)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
