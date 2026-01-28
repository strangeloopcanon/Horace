"""Windowed cadence timeline for page-level pacing analysis.

Divides a document into paragraph-aligned windows and extracts cadence
profiles per window, identifying best/worst sections.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.studio.analyze import analyze_text
from tools.studio.cadence_profile import CadenceProfile, extract_cadence_profile
from tools.studio.paragraph_cadence import (
    DocumentParagraphCadence,
    ParagraphCadence,
    extract_paragraph_cadence,
)
from tools.studio.text_normalize import normalize_for_studio


def _paragraph_starts(text: str) -> List[int]:
    """Find character positions where paragraphs start."""
    t = text or ""
    starts = [0]
    for m in re.finditer(r"\n\s*\n+", t):
        starts.append(int(m.end()))
    starts = sorted({s for s in starts if 0 <= s <= len(t)})
    return starts or [0]


def _select_windows(
    text: str,
    *,
    window_chars: int = 6000,
    max_windows: int = 8,
) -> List[Tuple[int, int, str]]:
    """
    Select paragraph-aligned windows across the document.

    Returns list of (start_char, end_char, window_text) tuples.
    """
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


@dataclass
class WindowCadence:
    """Cadence profile for a single window of text."""

    window_index: int
    start_char: int
    end_char: int

    # Cadence metrics
    spike_rate: float = 0.0
    surprisal_mean: float = 0.0
    surprisal_cv: float = 0.0
    ipi_mean: float = 0.0  # Inter-peak interval

    # Paragraph-level metrics for this window
    para_count: int = 0
    mean_sentence_length_cv: float = 0.0
    pacing_variety: float = 0.0

    # Scores (0-100)
    cadence_score: float = 0.0
    texture_score: float = 0.0

    # Flags
    is_worst: bool = False
    is_best: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "spike_rate": self.spike_rate,
            "surprisal_mean": self.surprisal_mean,
            "surprisal_cv": self.surprisal_cv,
            "ipi_mean": self.ipi_mean,
            "para_count": self.para_count,
            "mean_sentence_length_cv": self.mean_sentence_length_cv,
            "pacing_variety": self.pacing_variety,
            "cadence_score": self.cadence_score,
            "texture_score": self.texture_score,
            "is_worst": self.is_worst,
            "is_best": self.is_best,
            "reasons": self.reasons,
        }


@dataclass
class CadenceTimeline:
    """Document-level cadence timeline with per-window analysis."""

    windows: List[WindowCadence] = field(default_factory=list)
    worst_window_index: int = 0
    best_window_index: int = 0

    # Aggregate metrics
    overall_cadence_score: float = 0.0
    pacing_variety: float = 0.0  # CV of window scores (variation = good pacing)
    tension_arc: float = 0.0  # Correlation of position with surprisal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "windows": [w.to_dict() for w in self.windows],
            "worst_window_index": self.worst_window_index,
            "best_window_index": self.best_window_index,
            "overall_cadence_score": self.overall_cadence_score,
            "pacing_variety": self.pacing_variety,
            "tension_arc": self.tension_arc,
        }


def _compute_cadence_score(
    dm: Dict[str, Any],
    *,
    doc_median_spike_rate: Optional[float] = None,
    doc_median_cv: Optional[float] = None,
) -> float:
    """
    Compute a cadence score (0-100) from doc_metrics.

    Higher = better cadence (good spike rate, variety, not flat).
    
    When doc_median values are provided, scoring is relative to the document's
    own baseline rather than absolute targets. This is better for within-document
    comparison across windows.
    """
    spike_rate = float(dm.get("spike_rate") or dm.get("high_surprise_rate_per_100") or 0)
    surprisal_cv = float(dm.get("surprisal_cv") or 0)
    sent_burst_cv = float(dm.get("sent_burst_cv") or 0)

    # Spike rate scoring
    if doc_median_spike_rate is not None and doc_median_spike_rate > 0:
        # Relative: score by deviation from doc median
        # Windows close to median get ~80, deviation down to ~40
        deviation = abs(spike_rate - doc_median_spike_rate) / (doc_median_spike_rate + 1e-6)
        spike_score = 80 - min(40, deviation * 40)
    else:
        # Absolute: target spike rate around 6-10%
        spike_score = 100 * (1 - abs(spike_rate - 8) / 10)
        spike_score = max(0, min(100, spike_score))

    # Surprisal CV: some variation is good
    if doc_median_cv is not None and doc_median_cv > 0:
        # Relative: windows matching doc median get ~70
        deviation = abs(surprisal_cv - doc_median_cv) / (doc_median_cv + 1e-6)
        cv_score = 70 - min(30, deviation * 30)
    else:
        cv_score = min(100, surprisal_cv * 100)

    # Sentence burstiness: some variation is good
    burst_score = min(100, sent_burst_cv * 200)

    return float(0.5 * spike_score + 0.3 * cv_score + 0.2 * burst_score)


def _compute_texture_score(dm: Dict[str, Any]) -> float:
    """
    Compute a texture score (0-100) from doc_metrics.

    Higher = more varied vocabulary, less repetition.
    """
    content_frac = float(dm.get("content_fraction") or 0.4)
    nucleus_mean = float(dm.get("nucleus_w_mean") or 100)

    content_score = min(100, content_frac * 200)
    nucleus_score = min(100, nucleus_mean / 2)

    return float(0.6 * content_score + 0.4 * nucleus_score)


def _identify_reasons(wc: WindowCadence, dm: Dict[str, Any]) -> List[str]:
    """Identify issues with this window's cadence."""
    reasons: List[str] = []

    spike_rate = wc.spike_rate
    if spike_rate < 4:
        reasons.append("Low spike rate (flat, predictable)")
    elif spike_rate > 15:
        reasons.append("Very high spike rate (may feel chaotic)")

    if wc.surprisal_cv < 0.15:
        reasons.append("Low surprisal variation (monotonous)")

    if wc.mean_sentence_length_cv < 0.15:
        reasons.append("Uniform sentence lengths")

    ipi = wc.ipi_mean
    if ipi > 25:
        reasons.append("Long gaps between spikes")
    elif ipi < 5 and spike_rate > 10:
        reasons.append("Spikes too close together")

    return reasons


def windowed_cadence_for_text(
    text: str,
    *,
    model_id: str = "gpt2",
    doc_type: str = "prose",
    backend: str = "auto",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    window_chars: int = 6000,
    max_windows: int = 8,
    relative_scoring: bool = True,
) -> CadenceTimeline:
    """
    Extract a cadence timeline for a document.

    Divides text into paragraph-aligned windows, analyzes each,
    and returns per-window cadence profiles plus aggregate metrics.

    Args:
        text: Document text
        model_id: Scoring model
        doc_type: Document type (prose, poem, etc.)
        backend: Model backend
        max_input_tokens: Max tokens per window analysis
        normalize_text: Whether to normalize text first
        window_chars: Target characters per window
        max_windows: Maximum number of windows
        relative_scoring: If True, score windows relative to doc median (better for
            within-document comparison). If False, use absolute targets.

    Returns:
        CadenceTimeline with per-window analysis and worst/best identification
    """
    norm_text, norm_meta = normalize_for_studio(
        text or "", doc_type=str(doc_type), enabled=bool(normalize_text)
    )
    windows = _select_windows(
        norm_text, window_chars=int(window_chars), max_windows=int(max_windows)
    )

    if not windows or (len(windows) == 1 and not windows[0][2]):
        return CadenceTimeline()

    # First pass: collect metrics from all windows
    window_data: List[Dict[str, Any]] = []
    for i, (start, end, chunk) in enumerate(windows):
        if not chunk.strip():
            continue

        analysis = analyze_text(
            chunk,
            model_id=str(model_id),
            doc_type=str(doc_type),
            backend=str(backend),
            max_input_tokens=int(max_input_tokens),
            normalize_text=False,
            compute_cohesion=False,
            include_token_metrics=True,
        )

        dm = analysis.get("doc_metrics") or {}
        para_cad = extract_paragraph_cadence(analysis)

        window_data.append({
            "index": i,
            "start": start,
            "end": end,
            "dm": dm,
            "para_cad": para_cad,
        })

    if not window_data:
        return CadenceTimeline()

    # Compute doc-level medians for relative scoring
    doc_median_spike_rate: Optional[float] = None
    doc_median_cv: Optional[float] = None
    if relative_scoring and len(window_data) > 1:
        spike_rates = [float(wd["dm"].get("spike_rate") or wd["dm"].get("high_surprise_rate_per_100") or 0) 
                       for wd in window_data]
        cvs = [float(wd["dm"].get("surprisal_cv") or 0) for wd in window_data]
        doc_median_spike_rate = float(np.median(spike_rates)) if spike_rates else None
        doc_median_cv = float(np.median(cvs)) if cvs else None

    # Second pass: score windows (relative or absolute)
    window_cadences: List[WindowCadence] = []
    cadence_scores: List[float] = []
    surprisal_means: List[float] = []

    for wd in window_data:
        dm = wd["dm"]
        para_cad = wd["para_cad"]

        cadence_score = _compute_cadence_score(
            dm,
            doc_median_spike_rate=doc_median_spike_rate,
            doc_median_cv=doc_median_cv,
        )
        texture_score = _compute_texture_score(dm)

        wc = WindowCadence(
            window_index=wd["index"],
            start_char=wd["start"],
            end_char=wd["end"],
            spike_rate=float(dm.get("spike_rate") or dm.get("high_surprise_rate_per_100") or 0),
            surprisal_mean=float(dm.get("surprisal_mean") or 0),
            surprisal_cv=float(dm.get("surprisal_cv") or 0),
            ipi_mean=float(dm.get("ipi_mean") or 0),
            para_count=para_cad.para_count,
            mean_sentence_length_cv=para_cad.mean_sentence_length_cv,
            pacing_variety=para_cad.pacing_variety,
            cadence_score=cadence_score,
            texture_score=texture_score,
        )

        wc.reasons = _identify_reasons(wc, dm)
        window_cadences.append(wc)
        cadence_scores.append(cadence_score)
        surprisal_means.append(float(dm.get("surprisal_mean") or 0))

    if not window_cadences:
        return CadenceTimeline()

    # Identify worst/best
    worst_i = int(np.argmin(cadence_scores))
    best_i = int(np.argmax(cadence_scores))

    window_cadences[worst_i].is_worst = True
    window_cadences[best_i].is_best = True

    # Compute aggregate metrics
    scores_arr = np.array(cadence_scores, dtype=np.float32)
    means_arr = np.array(surprisal_means, dtype=np.float32)

    overall_score = float(np.mean(scores_arr))

    # Pacing variety: CV of window scores
    pacing_variety = 0.0
    if len(scores_arr) > 1 and np.mean(scores_arr) > 1e-6:
        pacing_variety = float(np.std(scores_arr) / np.mean(scores_arr))

    # Tension arc: correlation of window position with surprisal
    tension_arc = 0.0
    if len(means_arr) > 2:
        x = np.arange(len(means_arr), dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (means_arr - means_arr.mean()) / (means_arr.std() + 1e-9)
        tension_arc = float(np.clip(np.mean(x * y), -1.0, 1.0))

    return CadenceTimeline(
        windows=window_cadences,
        worst_window_index=worst_i,
        best_window_index=best_i,
        overall_cadence_score=overall_score,
        pacing_variety=pacing_variety,
        tension_arc=tension_arc,
    )

