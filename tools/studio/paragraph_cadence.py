"""Paragraph-level cadence profile extraction.

Extracts rhythm metrics at the paragraph scale: sentence variety, opening/closing
punch, spike distribution within paragraphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ParagraphCadence:
    """Cadence profile for a single paragraph."""

    # Shape metrics
    sentence_count: int = 0
    sentence_length_cv: float = 0.0  # Coefficient of variation in sentence lengths
    mean_surprisal: float = 0.0  # Mean of sentence mean surprisals (paragraph tone)
    surprisal_range: float = 0.0  # Max - min sentence surprisal
    surprisal_cv: float = 0.0  # CV of sentence surprisals

    # Position metrics
    opening_surprisal: float = 0.0  # First sentence mean surprisal
    closing_surprisal: float = 0.0  # Last sentence mean surprisal
    opening_spike_rate: float = 0.0  # First sentence spike density
    closing_spike_rate: float = 0.0  # Last sentence spike density

    # Flow metrics
    spike_front_loading: float = 0.0  # -1 = back-loaded, +1 = front-loaded
    momentum: float = 0.0  # Trend in surprisal (-1 = declining, +1 = rising)

    # Position in document
    para_index: int = 0
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_count": self.sentence_count,
            "sentence_length_cv": self.sentence_length_cv,
            "mean_surprisal": self.mean_surprisal,
            "surprisal_range": self.surprisal_range,
            "surprisal_cv": self.surprisal_cv,
            "opening_surprisal": self.opening_surprisal,
            "closing_surprisal": self.closing_surprisal,
            "opening_spike_rate": self.opening_spike_rate,
            "closing_spike_rate": self.closing_spike_rate,
            "spike_front_loading": self.spike_front_loading,
            "momentum": self.momentum,
            "para_index": self.para_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class DocumentParagraphCadence:
    """Aggregate paragraph cadence for a document."""

    paragraphs: List[ParagraphCadence] = field(default_factory=list)

    # Aggregate metrics
    para_count: int = 0
    mean_sentence_length_cv: float = 0.0  # Average within-para variety
    mean_surprisal_range: float = 0.0
    pacing_variety: float = 0.0  # CV of paragraph surprisal means (good = variation)
    front_loading_trend: float = 0.0  # Do paragraphs tend to front-load spikes?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "para_count": self.para_count,
            "mean_sentence_length_cv": self.mean_sentence_length_cv,
            "mean_surprisal_range": self.mean_surprisal_range,
            "pacing_variety": self.pacing_variety,
            "front_loading_trend": self.front_loading_trend,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }


def _coefficient_of_variation(arr: np.ndarray) -> float:
    """Compute CV (std/mean), safe for empty or constant arrays."""
    if len(arr) < 2:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-9:
        return 0.0
    std = float(np.std(arr))
    return float(std / abs(mean))


def _momentum(arr: np.ndarray) -> float:
    """Compute trend: correlation of values with position. +1 = rising, -1 = falling."""
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float32)
    # Normalize both
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (arr - arr.mean()) / (arr.std() + 1e-9)
    corr = float(np.mean(x * y))
    return float(np.clip(corr, -1.0, 1.0))


def _spike_front_loading(spike_positions: List[int], total_tokens: int) -> float:
    """
    Compute front-loading score: +1 if spikes are early, -1 if late.
    Based on comparing median spike position to midpoint.
    """
    if not spike_positions or total_tokens < 2:
        return 0.0
    median_pos = float(np.median(spike_positions))
    midpoint = total_tokens / 2.0
    # Normalize to -1 to +1
    loading = (midpoint - median_pos) / (total_tokens / 2.0)
    return float(np.clip(loading, -1.0, 1.0))


def extract_paragraph_cadence(
    analysis: Dict[str, Any],
    *,
    spike_threshold: Optional[float] = None,
) -> DocumentParagraphCadence:
    """
    Extract paragraph-level cadence profiles from an analysis result.

    Args:
        analysis: Output from analyze_text()
        spike_threshold: Surprisal threshold for counting spikes. If None, uses the
            analysis' own high-surprise threshold (mean + std), falling back to
            mean + std computed from token surprisals.

    Returns:
        DocumentParagraphCadence with per-paragraph and aggregate metrics
    """
    segments = analysis.get("segments") or {}
    para_segment = segments.get("paragraphs") or {}
    sent_segment = segments.get("sentences") or {}

    para_items = para_segment.get("items") or []
    sent_items = sent_segment.get("items") or []
    sent_means = sent_segment.get("mean_surprisal") or []
    sent_counts = sent_segment.get("token_counts") or []

    token_metrics = analysis.get("token_metrics") or {}
    surprisals = token_metrics.get("surprisal") or []

    if not para_items:
        return DocumentParagraphCadence()

    thr: float
    if spike_threshold is not None:
        thr = float(spike_threshold)
    else:
        series = analysis.get("series") or {}
        thr_val = series.get("threshold_surprisal") if isinstance(series, dict) else None
        if isinstance(thr_val, (int, float)) and math.isfinite(float(thr_val)):
            thr = float(thr_val)
        elif surprisals:
            s_arr = np.array([float(x) for x in surprisals], dtype=np.float32)
            thr = float(np.mean(s_arr) + np.std(s_arr))
        else:
            thr = 0.0

    # Build sentence-to-paragraph mapping
    para_cadences: List[ParagraphCadence] = []

    for pi, para in enumerate(para_items):
        para_start_token = int(para.get("start_token") or 0)
        para_end_token = int(para.get("end_token") or para_start_token)
        para_start_char = int(para.get("start_char") or 0)
        para_end_char = int(para.get("end_char") or para_start_char)

        # Find sentences in this paragraph
        para_sents: List[int] = []
        for si, sent in enumerate(sent_items):
            sent_start = int(sent.get("start_token") or 0)
            # Sentence is in paragraph if it starts within paragraph bounds
            if para_start_token <= sent_start < para_end_token:
                para_sents.append(si)

        if not para_sents:
            # Paragraph with no detected sentences
            para_cadences.append(
                ParagraphCadence(
                    para_index=pi,
                    start_char=para_start_char,
                    end_char=para_end_char,
                )
            )
            continue

        # Extract sentence lengths and surprisals for this paragraph
        lengths = np.array(
            [float(sent_counts[si]) for si in para_sents if si < len(sent_counts)],
            dtype=np.float32,
        )
        means = np.array(
            [float(sent_means[si]) for si in para_sents if si < len(sent_means)],
            dtype=np.float32,
        )

        # Shape metrics
        mean_surprisal = float(np.mean(means)) if len(means) > 0 else 0.0
        sentence_length_cv = _coefficient_of_variation(lengths) if len(lengths) > 1 else 0.0
        surprisal_range = float(np.max(means) - np.min(means)) if len(means) > 1 else 0.0
        surprisal_cv = _coefficient_of_variation(means) if len(means) > 1 else 0.0

        # Position metrics
        opening_surprisal = float(means[0]) if len(means) > 0 else 0.0
        closing_surprisal = float(means[-1]) if len(means) > 0 else 0.0

        # Spike rates for opening/closing sentences
        def _spike_rate_for_sent(si: int) -> float:
            if si >= len(sent_items):
                return 0.0
            sent = sent_items[si]
            st = int(sent.get("start_token") or 0)
            et = int(sent.get("end_token") or st)
            if et <= st or et > len(surprisals):
                return 0.0
            sent_surp = [float(surprisals[i]) for i in range(st, min(et, len(surprisals)))]
            if not sent_surp:
                return 0.0
            spikes = sum(1 for s in sent_surp if float(s) >= thr)
            return float(spikes / len(sent_surp)) * 100  # per 100 tokens

        opening_spike_rate = _spike_rate_for_sent(para_sents[0])
        closing_spike_rate = _spike_rate_for_sent(para_sents[-1])

        # Flow metrics
        momentum = _momentum(means) if len(means) > 2 else 0.0

        # Spike front-loading within paragraph
        para_token_count = para_end_token - para_start_token
        spike_positions: List[int] = []
        for ti in range(para_start_token, min(para_end_token, len(surprisals))):
            if ti < len(surprisals) and float(surprisals[ti]) >= thr:
                spike_positions.append(ti - para_start_token)
        front_loading = _spike_front_loading(spike_positions, para_token_count)

        para_cadences.append(
            ParagraphCadence(
                sentence_count=len(para_sents),
                sentence_length_cv=sentence_length_cv,
                mean_surprisal=mean_surprisal,
                surprisal_range=surprisal_range,
                surprisal_cv=surprisal_cv,
                opening_surprisal=opening_surprisal,
                closing_surprisal=closing_surprisal,
                opening_spike_rate=opening_spike_rate,
                closing_spike_rate=closing_spike_rate,
                spike_front_loading=front_loading,
                momentum=momentum,
                para_index=pi,
                start_char=para_start_char,
                end_char=para_end_char,
            )
        )

    # Aggregate metrics
    if not para_cadences:
        return DocumentParagraphCadence()

    length_cvs = [p.sentence_length_cv for p in para_cadences if p.sentence_count > 1]
    ranges = [p.surprisal_range for p in para_cadences if p.sentence_count > 1]
    front_loadings = [p.spike_front_loading for p in para_cadences]

    # Pacing variety: CV of paragraph mean surprisals
    para_means = np.array([p.mean_surprisal for p in para_cadences if p.sentence_count > 0], dtype=np.float32)
    pacing_variety = _coefficient_of_variation(para_means) if len(para_means) > 1 else 0.0

    return DocumentParagraphCadence(
        paragraphs=para_cadences,
        para_count=len(para_cadences),
        mean_sentence_length_cv=float(np.mean(length_cvs)) if length_cvs else 0.0,
        mean_surprisal_range=float(np.mean(ranges)) if ranges else 0.0,
        pacing_variety=pacing_variety,
        front_loading_trend=float(np.mean(front_loadings)) if front_loadings else 0.0,
    )
