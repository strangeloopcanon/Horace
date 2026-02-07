"""Cadence profile extraction and conversion to CadenceSampler configuration.

So what: this module bridges the analysis layer (which measures cadence) with the
generation layer (CadenceSampler), enabling cadence-aware text generation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.studio.analyze import analyze_text


@dataclass
class CadenceProfile:
    """Cadence signature extracted from text, usable for generation control.

    All ranges are (min, max) tuples for sampling variation.
    """

    # Core cadence parameters
    interval_range: Tuple[int, int] = (10, 16)  # tokens between spikes
    cooldown_range: Tuple[int, int] = (3, 6)  # tokens to cool down after spike

    # Phase-specific parameters
    spike_content_boost: float = 0.25  # logit boost for content tokens during spike
    spike_stop_punct_penalty: float = 1.4  # penalty for stopping on punct during spike
    base_top_p: float = 0.90  # nucleus sampling for base phase
    cool_top_p: float = 0.84  # nucleus sampling for cooldown phase
    base_temperature: float = 0.85
    spike_temperature: float = 1.05
    cool_temperature: float = 0.75

    # Source metrics (for debugging/display)
    source_ipi_mean: Optional[float] = None
    source_cooldown_drop: Optional[float] = None
    source_content_fraction: Optional[float] = None
    source_spike_prev_punct_rate: Optional[float] = None
    source_nucleus_w_mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default profiles for common document types
DEFAULT_PROSE_PROFILE = CadenceProfile(
    interval_range=(12, 18),
    cooldown_range=(3, 5),
    spike_content_boost=0.22,
    spike_stop_punct_penalty=1.3,
    base_top_p=0.90,
    cool_top_p=0.85,
    base_temperature=0.85,
    spike_temperature=1.0,
    cool_temperature=0.78,
)

DEFAULT_POETRY_PROFILE = CadenceProfile(
    interval_range=(8, 14),
    cooldown_range=(2, 4),
    spike_content_boost=0.30,
    spike_stop_punct_penalty=1.6,
    base_top_p=0.88,
    cool_top_p=0.80,
    base_temperature=0.88,
    spike_temperature=1.10,
    cool_temperature=0.72,
)

# More punchy profile (shorter intervals, sharper spikes)
PUNCHY_PROFILE = CadenceProfile(
    interval_range=(7, 12),
    cooldown_range=(2, 4),
    spike_content_boost=0.32,
    spike_stop_punct_penalty=1.8,
    base_top_p=0.86,
    cool_top_p=0.78,
    base_temperature=0.82,
    spike_temperature=1.12,
    cool_temperature=0.70,
)

# More flowing profile (longer runs, gentler variation)
FLOWING_PROFILE = CadenceProfile(
    interval_range=(14, 22),
    cooldown_range=(4, 7),
    spike_content_boost=0.20,
    spike_stop_punct_penalty=1.2,
    base_top_p=0.92,
    cool_top_p=0.88,
    base_temperature=0.88,
    spike_temperature=0.98,
    cool_temperature=0.82,
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float, handling None and NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        if not math.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def extract_cadence_profile(
    text: str,
    *,
    model_id: str = "Qwen/Qwen3-0.6B",
    backend: str = "auto",
    max_input_tokens: int = 512,
    doc_type: str = "prose",
) -> CadenceProfile:
    """Extract a cadence profile from text via token-level analysis.

    This uses analyze_text() to compute cadence metrics, then maps them to
    CadenceSampler parameters.
    """
    analysis = analyze_text(
        text,
        model_id=model_id,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=max_input_tokens,
        normalize_text=True,
        compute_cohesion=False,
    )
    doc_metrics = analysis.get("doc_metrics") or {}

    # Start from a sensible default based on doc type
    if doc_type == "poem":
        base = copy.deepcopy(DEFAULT_POETRY_PROFILE)
    else:
        base = copy.deepcopy(DEFAULT_PROSE_PROFILE)

    # Extract source metrics
    ipi_mean = _safe_float(doc_metrics.get("ipi_mean") or doc_metrics.get("high_surprise_ipi_mean"))
    cooldown_drop = _safe_float(doc_metrics.get("cooldown_entropy_drop_3"))
    content_fraction = _safe_float(doc_metrics.get("content_fraction"))
    spike_prev_punct = _safe_float(doc_metrics.get("spike_prev_punct_rate"))
    nucleus_w = _safe_float(doc_metrics.get("nucleus_w_mean"))

    # Store source metrics for debugging
    base.source_ipi_mean = ipi_mean if ipi_mean > 0 else None
    base.source_cooldown_drop = cooldown_drop if cooldown_drop != 0 else None
    base.source_content_fraction = content_fraction if content_fraction > 0 else None
    base.source_spike_prev_punct_rate = spike_prev_punct if spike_prev_punct > 0 else None
    base.source_nucleus_w_mean = nucleus_w if nucleus_w > 0 else None

    # Map IPI (inter-peak interval) to interval_range
    if ipi_mean > 0:
        lo = max(4, int(round(0.75 * ipi_mean)))
        hi = max(lo + 2, int(round(1.25 * ipi_mean)))
        base.interval_range = (lo, hi)

    # Map cooldown entropy drop to cooldown_range
    if cooldown_drop > 0:
        if cooldown_drop >= 1.4:
            base.cooldown_range = (5, 8)
        elif cooldown_drop >= 1.1:
            base.cooldown_range = (3, 6)
        elif cooldown_drop >= 0.8:
            base.cooldown_range = (2, 5)
        else:
            base.cooldown_range = (2, 4)

    # Map content fraction to spike content boost
    if content_fraction > 0:
        if content_fraction >= 0.45:
            base.spike_content_boost = 0.32
        elif content_fraction >= 0.38:
            base.spike_content_boost = 0.26
        elif content_fraction >= 0.30:
            base.spike_content_boost = 0.22
        else:
            base.spike_content_boost = 0.18

    # Map spike-prev-punct rate to stop_punct_penalty
    if spike_prev_punct > 0:
        if spike_prev_punct >= 0.28:
            base.spike_stop_punct_penalty = 2.0
        elif spike_prev_punct >= 0.20:
            base.spike_stop_punct_penalty = 1.6
        elif spike_prev_punct >= 0.12:
            base.spike_stop_punct_penalty = 1.3
        else:
            base.spike_stop_punct_penalty = 1.1

    # Map nucleus width to top_p settings
    if nucleus_w > 0:
        if nucleus_w <= 80:
            base.base_top_p, base.cool_top_p = 0.86, 0.80
        elif nucleus_w <= 120:
            base.base_top_p, base.cool_top_p = 0.88, 0.82
        elif nucleus_w <= 180:
            base.base_top_p, base.cool_top_p = 0.90, 0.84
        else:
            base.base_top_p, base.cool_top_p = 0.92, 0.86

    return base


def profile_to_poetry_config(profile: CadenceProfile):
    """Convert a CadenceProfile to a PoetryConfig for CadenceSampler.

    Returns a PoetryConfig instance that can be passed to CadenceSampler.
    """
    # Import here to avoid circular dependency
    from tools.sampler import PhaseParams, PoetryConfig

    base = PhaseParams(
        top_p=profile.base_top_p,
        temperature=profile.base_temperature,
        content_boost=0.0,
        stop_punct_penalty=0.0,
    )
    spike = PhaseParams(
        top_p=0.95,  # More open during spikes
        temperature=profile.spike_temperature,
        content_boost=profile.spike_content_boost,
        stop_punct_penalty=profile.spike_stop_punct_penalty,
    )
    cool = PhaseParams(
        top_p=profile.cool_top_p,
        temperature=profile.cool_temperature,
        content_boost=0.0,
        stop_punct_penalty=0.0,
    )

    return PoetryConfig(
        base=base,
        spike=spike,
        cool=cool,
        interval_range=profile.interval_range,
        cooldown_range=profile.cooldown_range,
        defer_spike_on_punct=True,
    )


def blend_profiles(
    primary: CadenceProfile,
    secondary: CadenceProfile,
    alpha: float = 0.5,
) -> CadenceProfile:
    """Blend two cadence profiles together.

    alpha=0.0 → pure primary, alpha=1.0 → pure secondary
    """

    def blend_tuple(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        lo = int(round(a[0] * (1 - alpha) + b[0] * alpha))
        hi = int(round(a[1] * (1 - alpha) + b[1] * alpha))
        return (lo, max(lo + 1, hi))

    def blend_float(a: float, b: float) -> float:
        return float(a * (1 - alpha) + b * alpha)

    return CadenceProfile(
        interval_range=blend_tuple(primary.interval_range, secondary.interval_range),
        cooldown_range=blend_tuple(primary.cooldown_range, secondary.cooldown_range),
        spike_content_boost=blend_float(primary.spike_content_boost, secondary.spike_content_boost),
        spike_stop_punct_penalty=blend_float(
            primary.spike_stop_punct_penalty, secondary.spike_stop_punct_penalty
        ),
        base_top_p=blend_float(primary.base_top_p, secondary.base_top_p),
        cool_top_p=blend_float(primary.cool_top_p, secondary.cool_top_p),
        base_temperature=blend_float(primary.base_temperature, secondary.base_temperature),
        spike_temperature=blend_float(primary.spike_temperature, secondary.spike_temperature),
        cool_temperature=blend_float(primary.cool_temperature, secondary.cool_temperature),
    )


def dtw_distance(
    series_a: List[float],
    series_b: List[float],
    *,
    window: Optional[int] = None,
) -> float:
    """Compute Dynamic Time Warping distance between two surprisal time series.

    Uses the Sakoe-Chiba band constraint for efficiency. Returns the normalized
    DTW distance (divided by path length).
    """
    a = np.array(series_a, dtype=np.float64)
    b = np.array(series_b, dtype=np.float64)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    w = window or max(n, m)  # default: no constraint
    w = max(w, abs(n - m))  # must be at least |n-m|

    # DTW cost matrix with Sakoe-Chiba band
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            d = abs(float(a[i - 1]) - float(b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Normalize by path length (approximated by n+m)
    return float(cost[n, m]) / float(n + m)


def dtw_similarity(
    series_a: List[float],
    series_b: List[float],
    *,
    window: Optional[int] = None,
    scale: float = 3.0,
) -> float:
    """Compute DTW-based cadence similarity in [0, 1].

    Higher means the two surprisal time series have more similar shapes.
    Uses an exponential mapping: similarity = exp(-distance / scale).
    """
    dist = dtw_distance(series_a, series_b, window=window)
    if not math.isfinite(dist):
        return 0.0
    return float(math.exp(-dist / max(0.01, scale)))


def compare_cadence_profiles(
    profile_a: CadenceProfile,
    profile_b: CadenceProfile,
    *,
    series_a: Optional[List[float]] = None,
    series_b: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compare two cadence profiles and return distance metrics.

    If surprisal time series are provided, also computes DTW-based similarity
    which is more sensitive to the actual shape of the cadence curve.
    """

    def tuple_dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2)

    interval_dist = tuple_dist(profile_a.interval_range, profile_b.interval_range)
    cooldown_dist = tuple_dist(profile_a.cooldown_range, profile_b.cooldown_range)
    content_boost_dist = abs(profile_a.spike_content_boost - profile_b.spike_content_boost)
    temperature_dist = abs(profile_a.spike_temperature - profile_b.spike_temperature)
    top_p_dist = abs(profile_a.base_top_p - profile_b.base_top_p)

    # Profile-level distance (original method)
    profile_dist = (
        0.30 * (interval_dist / 10.0)
        + 0.20 * (cooldown_dist / 4.0)
        + 0.20 * (content_boost_dist / 0.3)
        + 0.15 * (temperature_dist / 0.3)
        + 0.15 * (top_p_dist / 0.1)
    )

    # DTW-based similarity if series are available
    dtw_sim = None
    dtw_dist = None
    if series_a is not None and series_b is not None and len(series_a) >= 5 and len(series_b) >= 5:
        dtw_dist = dtw_distance(series_a, series_b, window=max(len(series_a), len(series_b)) // 4)
        dtw_sim = dtw_similarity(series_a, series_b, window=max(len(series_a), len(series_b)) // 4)

    # Blend profile and DTW similarity if DTW is available
    if dtw_sim is not None:
        profile_sim = max(0.0, 1.0 - profile_dist)
        overall_sim = 0.4 * profile_sim + 0.6 * dtw_sim  # DTW is the primary signal
        overall_dist = 1.0 - overall_sim
    else:
        overall_dist = profile_dist
        overall_sim = max(0.0, 1.0 - profile_dist)

    result: Dict[str, Any] = {
        "interval_distance": interval_dist,
        "cooldown_distance": cooldown_dist,
        "content_boost_distance": content_boost_dist,
        "temperature_distance": temperature_dist,
        "top_p_distance": top_p_dist,
        "profile_distance": float(profile_dist),
        "overall_distance": float(overall_dist),
        "similarity_0_1": float(overall_sim),
        "details": {
            "interval_range": {"generated": profile_a.interval_range, "reference": profile_b.interval_range},
            "cooldown_range": {"generated": profile_a.cooldown_range, "reference": profile_b.cooldown_range},
            "spike_content_boost": {"generated": profile_a.spike_content_boost, "reference": profile_b.spike_content_boost},
            "spike_stop_punct_penalty": {"generated": profile_a.spike_stop_punct_penalty, "reference": profile_b.spike_stop_punct_penalty},
            "base_top_p": {"generated": profile_a.base_top_p, "reference": profile_b.base_top_p},
            "cool_top_p": {"generated": profile_a.cool_top_p, "reference": profile_b.cool_top_p},
            "base_temperature": {"generated": profile_a.base_temperature, "reference": profile_b.base_temperature},
            "spike_temperature": {"generated": profile_a.spike_temperature, "reference": profile_b.spike_temperature},
            "cool_temperature": {"generated": profile_a.cool_temperature, "reference": profile_b.cool_temperature},
        },
    }

    if dtw_sim is not None:
        result["dtw_distance"] = float(dtw_dist)
        result["dtw_similarity"] = float(dtw_sim)

    return result
