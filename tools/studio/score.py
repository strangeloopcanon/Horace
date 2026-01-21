from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.studio.baselines import Baseline, get_slice, percentile


@dataclass(frozen=True)
class MetricScore:
    value: float
    percentile: Optional[float]
    score_0_1: Optional[float]
    mode: str


@dataclass(frozen=True)
class ScoreReport:
    overall_0_100: float
    categories: Dict[str, float]
    metrics: Dict[str, MetricScore]


def _metric_score(pctl: Optional[float], mode: str) -> Optional[float]:
    if pctl is None:
        return None
    p = max(0.0, min(100.0, float(pctl)))
    if mode == "higher_is_better":
        return p / 100.0
    if mode == "lower_is_better":
        return 1.0 - (p / 100.0)
    # match_baseline: best near median
    return max(0.0, 1.0 - abs(p - 50.0) / 50.0)


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
            "cooldown_entropy_drop_3": {"weight": 0.20, "mode": "higher_is_better"},
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


def score_text(doc_metrics: Dict[str, Any], baseline: Baseline, *, doc_type: str) -> ScoreReport:
    """Score a single analyzed text against baseline distributions.

    Notes:
    - This is intentionally conservative: it mostly rewards matching the baseline,
      except for a few metrics with clearer directionality.
    - Missing metrics are skipped and weights renormalized within each category.
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
    return ScoreReport(overall_0_100=float(overall), categories=category_scores, metrics=metric_scores)
