from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tools.studio.analyze import analyze_text
from tools.studio.baselines import Baseline
from tools.studio.score import MetricScore, ScoreReport, score_text
from tools.studio.text_normalize import normalize_for_studio


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def _paragraph_starts(text: str) -> List[int]:
    t = text or ""
    starts = [0]
    for m in re.finditer(r"\n\s*\n+", t):
        starts.append(int(m.end()))
    starts = sorted({s for s in starts if 0 <= s <= len(t)})
    return starts or [0]


def _select_windows(
    text: str,
    *,
    window_chars: int,
    max_windows: int,
) -> List[Tuple[int, int, str]]:
    t = (text or "").strip()
    if not t:
        return [(0, 0, "")]
    win = max(400, int(window_chars))
    cap = max(1, int(max_windows))
    if len(t) <= win or cap <= 1:
        return [(0, len(t), t)]

    max_start = max(0, len(t) - win)
    targets = [int(round(i * max_start / max(1, cap - 1))) for i in range(cap)]
    boundaries = _paragraph_starts(t)

    starts: List[int] = []
    for pos in targets:
        i = bisect.bisect_right(boundaries, pos) - 1
        start = boundaries[i] if i >= 0 else 0
        start = min(int(start), int(max_start))
        if starts and abs(start - starts[-1]) < max(120, win // 4):
            continue
        starts.append(start)
    if 0 not in starts:
        starts.insert(0, 0)
    deduped: List[int] = []
    seen: set[int] = set()
    for s in starts:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)

    out: List[Tuple[int, int, str]] = []
    for s in deduped[:cap]:
        e = min(len(t), int(s) + win)
        out.append((int(s), int(e), t[int(s) : int(e)].strip()))
    return out


def _prose_weight(doc_metrics: Dict[str, Any]) -> float:
    alpha = float(doc_metrics.get("alpha_char_fraction") or 0.0)
    symbol = float(doc_metrics.get("symbol_char_fraction") or 0.0)
    code = float(doc_metrics.get("line_code_fraction") or 0.0)
    bullet = float(doc_metrics.get("line_list_fraction") or 0.0)
    heading = float(doc_metrics.get("line_heading_fraction") or 0.0)
    quote = float(doc_metrics.get("line_quote_fraction") or 0.0)

    base = (alpha - 0.35) / 0.45  # 0 at 0.35, 1 at 0.80
    base = _clamp(base, 0.05, 1.0)
    nonprose = 0.90 * code + 0.40 * bullet + 0.25 * heading + 0.15 * quote + 1.40 * symbol
    return _clamp(base * (1.0 - nonprose), 0.05, 1.0)


@dataclass(frozen=True)
class WindowedRubric:
    aggregate: ScoreReport
    worst: ScoreReport
    best: ScoreReport
    worst_window_index: int
    best_window_index: int
    windows: List[Dict[str, Any]]
    worst_analysis: Dict[str, Any]
    text_normalization: Dict[str, Any]


def windowed_rubric_for_text(
    text: str,
    *,
    baseline: Baseline,
    scoring_model_id: str,
    doc_type: str,
    backend: str,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
    window_chars: int = 9000,
    max_windows: int = 8,
) -> WindowedRubric:
    norm_text, norm_meta = normalize_for_studio(text or "", doc_type=str(doc_type), enabled=bool(normalize_text))
    windows = _select_windows(norm_text, window_chars=int(window_chars), max_windows=int(max_windows))

    per: List[Dict[str, Any]] = []
    weights: List[float] = []
    scores: List[ScoreReport] = []
    analyses: List[Dict[str, Any]] = []

    for i, (s, e, chunk) in enumerate(windows):
        analysis = analyze_text(
            chunk,
            model_id=str(scoring_model_id),
            doc_type=str(doc_type),
            backend=str(backend),
            max_input_tokens=int(max_input_tokens),
            normalize_text=False,  # already normalized once above
            compute_cohesion=bool(compute_cohesion),
        )
        score = score_text((analysis.get("doc_metrics") or {}), baseline, doc_type=str(doc_type))
        dm = analysis.get("doc_metrics") or {}
        w = _prose_weight(dm) if isinstance(dm, dict) else 0.2
        weights.append(float(w))
        scores.append(score)
        analyses.append(analysis)
        per.append(
            {
                "index": int(i),
                "start_char": int(s),
                "end_char": int(e),
                "chars": int(len(chunk)),
                "prose_weight": float(w),
                "tokens_count": int((dm or {}).get("tokens_count") or 0),
                "truncated": bool(analysis.get("truncated")),
                "overall_0_100": float(score.overall_0_100),
                "categories": dict(score.categories),
            }
        )

    if not scores:
        empty = ScoreReport(overall_0_100=0.0, categories={}, metrics={})
        return WindowedRubric(
            aggregate=empty,
            worst=empty,
            best=empty,
            worst_window_index=0,
            best_window_index=0,
            windows=[],
            worst_analysis={},
            text_normalization=dict(norm_meta),
        )

    def _weighted_mean(xs: List[float], ws: List[float]) -> float:
        num = 0.0
        den = 0.0
        for x, w in zip(xs, ws):
            if not isinstance(x, (int, float)):
                continue
            ww = float(w)
            if ww <= 0:
                continue
            num += ww * float(x)
            den += ww
        if den <= 0:
            return float(sum(float(x) for x in xs) / len(xs))
        return float(num / den)

    agg_overall = _weighted_mean([s.overall_0_100 for s in scores], weights)

    all_cats: set[str] = set()
    for s in scores:
        all_cats.update((s.categories or {}).keys())
    agg_cats: Dict[str, float] = {}
    for c in sorted(all_cats):
        vals: List[float] = []
        ws: List[float] = []
        for sc, w in zip(scores, weights):
            v = (sc.categories or {}).get(c)
            if v is None:
                continue
            vals.append(float(v))
            ws.append(float(w))
        if vals:
            agg_cats[c] = float(_weighted_mean(vals, ws))

    all_metrics: set[str] = set()
    for s in scores:
        all_metrics.update((s.metrics or {}).keys())
    agg_metrics: Dict[str, MetricScore] = {}
    for m in sorted(all_metrics):
        vals: List[float] = []
        vals_w: List[float] = []
        pcts: List[float] = []
        pcts_w: List[float] = []
        score01: List[float] = []
        score01_w: List[float] = []
        mode: Optional[str] = None
        for sc, w in zip(scores, weights):
            ms = (sc.metrics or {}).get(m)
            if ms is None:
                continue
            ww = float(w)
            if ww <= 0:
                continue
            if mode is None:
                mode = str(ms.mode)
            if isinstance(ms.value, (int, float)):
                vals.append(float(ms.value))
                vals_w.append(ww)
            if isinstance(ms.percentile, (int, float)):
                pcts.append(float(ms.percentile))
                pcts_w.append(ww)
            if isinstance(ms.score_0_1, (int, float)):
                score01.append(float(ms.score_0_1))
                score01_w.append(ww)
        if not vals:
            continue
        agg_metrics[m] = MetricScore(
            value=float(_weighted_mean(vals, vals_w)),
            percentile=float(_weighted_mean(pcts, pcts_w)) if pcts else None,
            score_0_1=float(_weighted_mean(score01, score01_w)) if score01 else None,
            mode=mode or "match_baseline",
        )

    eligible = [i for i, w in enumerate(weights) if float(w) >= 0.25]
    if not eligible:
        eligible = list(range(len(scores)))

    worst_i = min(eligible, key=lambda i: float(scores[i].overall_0_100))
    best_i = max(eligible, key=lambda i: float(scores[i].overall_0_100))

    aggregate = ScoreReport(overall_0_100=float(agg_overall), categories=agg_cats, metrics=agg_metrics)
    return WindowedRubric(
        aggregate=aggregate,
        worst=scores[worst_i],
        best=scores[best_i],
        worst_window_index=int(worst_i),
        best_window_index=int(best_i),
        windows=per,
        worst_analysis=analyses[worst_i],
        text_normalization=dict(norm_meta),
    )
