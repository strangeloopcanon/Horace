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
        return "", "", "", "", None, {}

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
            return msg, "", "", "", None, {"trained_score_error": trained_err}
        score_md = []
        score_md.append(f"### Trained scorer: **{trained_score.score_0_100:.1f}/100**")
        score_md.append(f"- prob: `{trained_score.prob_0_1:.3f}`")
        score_md.append(f"- model: `{trained_score.model_path_or_id}`")
        score_md.append(f"- device: `{trained_score.device}`")
        out_json = {"trained_score": asdict(trained_score)}
        return "\n".join(score_md), "", "", "", None, out_json

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
    return score_md, profile_md, suggestions_md + "\n\n" + spikes_md, llm_md, plot_img, out_json


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
                choices=["poem", "prose", "shortstory", "novel"],
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
                outputs=[score_md, profile_md, suggestions_md, llm_md, plot_img, out_json],
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
