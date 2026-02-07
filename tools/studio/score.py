from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tools.studio.baselines import Baseline, get_slice, percentile


@dataclass(frozen=True)
class MetricScore:
    value: float
    percentile: Optional[float]
    score_0_1: Optional[float]
    mode: str


@dataclass(frozen=True)
class ImprovementHint:
    """A single actionable improvement suggestion tied to a metric."""
    category: str
    metric: str
    current_score: float
    potential_gain: float
    direction: str  # 'too_low', 'too_high', or 'flat'


@dataclass(frozen=True)
class ScoreReport:
    overall_0_100: float
    categories: Dict[str, float]
    metrics: Dict[str, MetricScore]
    top_improvements: List[ImprovementHint] = ()  # type: ignore[assignment]


def _metric_score(pctl: Optional[float], mode: str) -> Optional[float]:
    """Convert percentile to 0-1 score based on scoring mode.

    Modes:
    - higher_is_better: 100th percentile = 1.0, 0th = 0.0
    - lower_is_better: 0th percentile = 1.0, 100th = 0.0
    - match_baseline: Plateau scoring with excellence bonus — full score for
      25th-75th percentile, linear falloff outside, plus a gentle bonus for
      75th-90th to reward distinctive-but-controlled excellence.
    """
    if pctl is None:
        return None
    p = max(0.0, min(100.0, float(pctl)))
    if mode == "higher_is_better":
        return p / 100.0
    if mode == "lower_is_better":
        return 1.0 - (p / 100.0)
    # match_baseline: plateau scoring with excellence bonus
    PLATEAU_LOW = 25.0
    PLATEAU_HIGH = 75.0
    EXCELLENCE_CEIL = 90.0
    if PLATEAU_LOW <= p <= PLATEAU_HIGH:
        return 1.0
    elif p < PLATEAU_LOW:
        return p / PLATEAU_LOW
    elif p <= EXCELLENCE_CEIL:
        # Excellence bonus zone: 75th-90th percentile gets a gentle bonus
        # up to 1.08 (8% above plateau), rewarding controlled distinction.
        bonus = 0.08 * ((p - PLATEAU_HIGH) / (EXCELLENCE_CEIL - PLATEAU_HIGH))
        return 1.0 + bonus
    else:
        # Beyond 90th: linear falloff from 1.08 back to 0.0 at 100th
        return 1.08 * (100.0 - p) / (100.0 - EXCELLENCE_CEIL)


_RUBRIC: Dict[str, Dict[str, Any]] = {
    "focus": {
        "weight": 0.25,
        "metrics": {
            "entropy_mean": {"weight": 0.375, "mode": "match_baseline"},
            "nucleus_w_mean": {"weight": 0.375, "mode": "match_baseline"},
            "word_ttr": {"weight": 0.25, "mode": "match_baseline"},
        },
    },
    "cadence": {
        "weight": 0.25,
        "metrics": {
            "high_surprise_rate_per_100": {"weight": 0.25, "mode": "match_baseline"},
            "ipi_mean": {"weight": 0.25, "mode": "match_baseline"},
            # cooldown_entropy_drop_3: Changed from higher_is_better to match_baseline
            # because low cooldown (maintaining tension) is a valid stylistic choice
            # (e.g., Hemingway's minimalist style), not inherently worse.
            "cooldown_entropy_drop_3": {"weight": 0.20, "mode": "match_baseline"},
            "sent_burst_cv": {"weight": 0.10, "mode": "match_baseline"},
            "para_burst_cv": {"weight": 0.10, "mode": "match_baseline"},
            "sent_words_cv": {"weight": 0.07, "mode": "match_baseline"},
            "syllables_per_word_cv": {"weight": 0.03, "mode": "match_baseline"},
        },
    },
    "cohesion": {
        "weight": 0.20,
        "metrics": {
            "cohesion_delta": {"weight": 0.70, "mode": "lower_is_better"},
            "trigram_repeat_rate": {"weight": 0.30, "mode": "lower_is_better"},
        },
    },
    "alignment": {
        "weight": 0.15,
        "metrics": {
            "spike_next_content_rate": {"weight": 0.6, "mode": "higher_is_better"},
            "spike_prev_punct_rate": {"weight": 0.4, "mode": "lower_is_better"},
        },
    },
    "distinctiveness": {
        "weight": 0.15,
        "metrics": {
            "content_fraction": {"weight": 0.40, "mode": "match_baseline"},
            "word_hapax_ratio": {"weight": 0.35, "mode": "match_baseline"},
            "punct_variety_per_1000_chars": {"weight": 0.25, "mode": "match_baseline"},
        },
    },
}


def _compute_top_improvements(
    metric_scores: Dict[str, MetricScore],
    category_scores: Dict[str, float],
    n: int = 3,
) -> List[ImprovementHint]:
    """Find the N metrics with the most room for improvement, weighted by category impact."""
    candidates: List[Tuple[float, ImprovementHint]] = []

    for cat_name, cat_spec in _RUBRIC.items():
        cat_weight = float(cat_spec["weight"])
        for mkey, mspec in cat_spec["metrics"].items():
            ms = metric_scores.get(mkey)
            if ms is None or ms.score_0_1 is None or ms.percentile is None:
                continue
            metric_weight = float(mspec.get("weight", 1.0))
            # Room for improvement: how far from a perfect 1.0 score
            gap = max(0.0, 1.0 - float(ms.score_0_1))
            # Impact: category weight * metric weight * gap
            impact = cat_weight * metric_weight * gap
            if impact < 0.005:
                continue

            p = float(ms.percentile)
            mode = str(mspec.get("mode", "match_baseline"))
            if mode == "higher_is_better":
                direction = "too_low"
            elif mode == "lower_is_better":
                direction = "too_high"
            elif p < 25:
                direction = "too_low"
            elif p > 75:
                direction = "too_high"
            else:
                direction = "flat"

            candidates.append((
                impact,
                ImprovementHint(
                    category=cat_name,
                    metric=mkey,
                    current_score=float(ms.score_0_1),
                    potential_gain=impact,
                    direction=direction,
                ),
            ))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [hint for _, hint in candidates[:n]]


def score_text(doc_metrics: Dict[str, Any], baseline: Baseline, *, doc_type: str) -> ScoreReport:
    """Score a single analyzed text against baseline distributions.

    Notes:
    - Rewards matching the baseline, with a gentle excellence bonus (75th-90th pctl).
    - Missing metrics are skipped and weights renormalized within each category.
    - Returns top-3 improvement hints ranked by potential impact.
    """
    base_slice = get_slice(baseline, doc_type)
    metric_scores: Dict[str, MetricScore] = {}

    # Precompute metric percentiles for rubric metrics
    for cat in _RUBRIC.values():
        for mkey, mspec in cat["metrics"].items():
            v = doc_metrics.get(mkey)
            if not isinstance(v, (int, float)):
                continue
            summ = base_slice.metrics.get(mkey)
            if not summ:
                continue
            pctl = percentile(summ.values, float(v))
            mode = str(mspec.get("mode") or "match_baseline")
            metric_scores[mkey] = MetricScore(value=float(v), percentile=pctl, score_0_1=_metric_score(pctl, mode), mode=mode)

    category_scores: Dict[str, float] = {}
    weighted_total = 0.0
    weight_sum = 0.0

    for cat_name, cat_spec in _RUBRIC.items():
        cat_weight = float(cat_spec["weight"])
        mspecs: Dict[str, Dict[str, Any]] = cat_spec["metrics"]

        # Weighted average inside the category
        inner_total = 0.0
        inner_wsum = 0.0
        for mkey, mspec in mspecs.items():
            ms = metric_scores.get(mkey)
            if ms is None or ms.score_0_1 is None:
                continue
            w = float(mspec.get("weight", 1.0))
            inner_total += w * float(ms.score_0_1)
            inner_wsum += w
        if inner_wsum <= 0:
            continue
        cat_score = inner_total / inner_wsum
        category_scores[cat_name] = float(cat_score)
        weighted_total += cat_weight * float(cat_score)
        weight_sum += cat_weight

    overall = 100.0 * (weighted_total / weight_sum) if weight_sum > 0 else 0.0
    top_improvements = _compute_top_improvements(metric_scores, category_scores)
    return ScoreReport(
        overall_0_100=float(overall),
        categories=category_scores,
        metrics=metric_scores,
        top_improvements=top_improvements,
    )
