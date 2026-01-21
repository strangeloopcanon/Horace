from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from tools.studio.score import ScoreReport


@dataclass(frozen=True)
class Suggestion:
    title: str
    why: str
    what_to_try: str
    evidence: Optional[str] = None


def _pctl_hint(pctl: Optional[float]) -> str:
    if pctl is None:
        return "N/A"
    p = float(pctl)
    if p < 20:
        return f"{p:.0f}th (low)"
    if p > 80:
        return f"{p:.0f}th (high)"
    return f"{p:.0f}th"


def suggest_edits(
    *,
    doc_metrics: Dict[str, Any],
    score: ScoreReport,
    spikes: List[Dict[str, Any]],
    segments: Dict[str, Any],
) -> Dict[str, Any]:
    """Heuristic critique (deterministic fallback).

    Studio can swap this out for an LLM-based critic later; keep this one
    grounded in measured metrics and examples.
    """
    suggestions: List[Suggestion] = []

    # Quick helpers
    def m(metric: str):
        return score.metrics.get(metric)

    # 1) Focus / openness
    focus = score.categories.get("focus")
    ent = m("entropy_mean")
    nuc = m("nucleus_w_mean")
    if focus is not None and focus < 0.55:
        why = "The distribution looks less like the reference band for your selected genre."
        if ent and ent.percentile is not None and ent.percentile > 70:
            why = f"Your entropy is {_pctl_hint(ent.percentile)} vs baseline (more open / less committed choices)."
        elif ent and ent.percentile is not None and ent.percentile < 30:
            why = f"Your entropy is {_pctl_hint(ent.percentile)} vs baseline (very tight / possibly flat)."
        what = (
            "Revise 2–3 sentences to make the *verb and the concrete noun* do more work. "
            "Remove 1–2 generic intensifiers, and swap one abstract noun for a sensory detail."
        )
        suggestions.append(Suggestion(title="Sharpen focus of choices", why=why, what_to_try=what))

    # 2) Cadence and turns
    cadence = score.categories.get("cadence")
    spike_rate = m("high_surprise_rate_per_100")
    ipi = m("ipi_mean")
    cd = m("cooldown_entropy_drop_3")
    if cadence is not None and cadence < 0.55:
        why_parts = []
        if spike_rate:
            why_parts.append(f"spike rate {_pctl_hint(spike_rate.percentile)}")
        if ipi:
            why_parts.append(f"IPI {_pctl_hint(ipi.percentile)}")
        if cd:
            why_parts.append(f"cooldown {_pctl_hint(cd.percentile)}")
        why = "Cadence differs from the reference rhythm" + (": " + ", ".join(why_parts) if why_parts else ".")
        what = (
            "Insert one deliberate *turn* per paragraph (or per 4–8 lines in poetry): a precise image, a reversal, or a new constraint. "
            "After the turn, add 1–2 grounding clauses that re-anchor the scene before the next escalation."
        )
        suggestions.append(Suggestion(title="Add purposeful turns + cooldowns", why=why, what_to_try=what))

    # 3) Spike alignment (don’t spend turns on punctuation)
    align = score.categories.get("alignment")
    sp_prev_punct = m("spike_prev_punct_rate")
    sp_next_content = m("spike_next_content_rate")
    if align is not None and align < 0.60:
        why_parts = []
        if sp_prev_punct:
            why_parts.append(f"spikes near punctuation {_pctl_hint(sp_prev_punct.percentile)}")
        if sp_next_content:
            why_parts.append(f"post-spike content {_pctl_hint(sp_next_content.percentile)}")
        why = "Surprise is landing in less meaningful places" + (": " + ", ".join(why_parts) if why_parts else ".")
        what = (
            "Move the strongest lexical surprise onto a content word (noun/verb/adjective), not the line break or comma. "
            "Then follow it with a concrete continuation (a specific noun or action), not a generic connective."
        )
        evidence = None
        if spikes:
            # Show one problematic spike context for grounding
            s0 = spikes[0]
            evidence = f"Example spike context: {s0.get('context','').strip()}"
        suggestions.append(Suggestion(title="Land spikes on content pivots", why=why, what_to_try=what, evidence=evidence))

    # 4) Cohesion (order sensitivity)
    coh = score.categories.get("cohesion")
    coh_delta = m("cohesion_delta")
    if coh is not None and coh < 0.55:
        why = "Word-order cohesion is weaker than the baseline; shuffled order would read similarly."
        if coh_delta and coh_delta.percentile is not None:
            why = f"Cohesion delta is {_pctl_hint(coh_delta.percentile)} vs baseline."
        what = (
            "Thread 1–2 motifs across the piece (a repeated object, a verb family, or a sensory register). "
            "Add one intentional callback near the end that reuses earlier diction in a new meaning."
        )
        suggestions.append(Suggestion(title="Increase connective tissue", why=why, what_to_try=what))

    # 5) Broader-level burstiness (sentence rhythm)
    burst_cv = ((segments.get("sentences") or {}).get("burst_cv")) if segments else None
    if isinstance(burst_cv, (int, float)) and burst_cv > 0.7:
        suggestions.append(
            Suggestion(
                title="Smooth sentence-level rhythm",
                why=f"Sentence-level burstiness is high (CV ≈ {float(burst_cv):.2f}).",
                what_to_try=(
                    "Find the 1–2 most erratic sentences (very long or very clipped) and match them to neighbors: "
                    "split one long sentence, and expand one short sentence with a concrete clause."
                ),
            )
        )

    # If nothing triggered, give a small, constructive default
    if not suggestions:
        suggestions.append(
            Suggestion(
                title="Tighten one paragraph, then re-check",
                why="Your metrics sit reasonably close to the reference bands; improvements now are in micro-editing.",
                what_to_try=(
                    "Pick one paragraph/stanza and: (1) delete one generic phrase, (2) replace one abstract noun with a tangible object, "
                    "(3) add one deliberate turn followed by a short cooldown line."
                ),
            )
        )

    summary = (
        f"Overall score: {score.overall_0_100:.1f}/100. "
        + " ".join([f"{k}: {v*100:.0f}/100." for k, v in score.categories.items()])
    )

    return {
        "summary": summary,
        "suggestions": [asdict(s) for s in suggestions],
    }
