from __future__ import annotations

import difflib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline, load_baseline_cached
from tools.studio.cadence_profile import CadenceProfile
from tools.studio.calibrator import featurize_from_report_row, load_logistic_calibrator
from tools.studio.meaning_lock import MeaningLockConfig, MeaningLockReport, check_meaning_lock
from tools.studio.rewrite import generate_cadence_span_rewrites, generate_span_rewrites
from tools.studio.score import score_text
from tools.studio.text_normalize import normalize_for_studio

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NOMINALIZATION_RE = re.compile(
    r"\b[a-z]{4,}(?:tion|sion|ment|ness|ity|ism|ization|isation|ality|ability|ance|ence)\b",
    flags=re.I,
)
_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\b\s+(?:\w+\s+){0,2}?\w+(?:ed|en)\b",
    flags=re.I,
)
_HEDGE_RE = re.compile(
    r"\b(?:may|might|could|perhaps|arguably|likely|generally|typically|often|sometimes|somewhat|relatively)\b",
    flags=re.I,
)
_SENT_END_RE = re.compile(r"[.!?]+")


def _bureaucracy_stats(text: str) -> Dict[str, float]:
    t = (text or "").strip()
    words = [m.group(0) for m in _WORD_RE.finditer(t)]
    n = len(words)
    if n <= 0:
        return {"mean_word_len": 0.0, "long_word_rate": 0.0, "nominal_rate": 0.0, "passive_hits": 0.0, "hedge_rate": 0.0}
    wl = [len(w) for w in words]
    mean_wl = float(np.mean(np.array(wl, dtype=np.float32))) if wl else 0.0
    long_rate = float(sum(1 for w in wl if w >= 8)) / float(n)
    nom_rate = float(len(_NOMINALIZATION_RE.findall(t))) / float(n)
    hedge_rate = float(len(_HEDGE_RE.findall(t))) / float(n)
    passive_hits = float(len(_PASSIVE_RE.findall(t)))
    return {
        "mean_word_len": float(mean_wl),
        "long_word_rate": float(long_rate),
        "nominal_rate": float(nom_rate),
        "passive_hits": float(passive_hits),
        "hedge_rate": float(hedge_rate),
    }


def _sentence_count_est(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 1
    n = len(_SENT_END_RE.findall(t))
    return max(1, int(n))


def _droning_score(text: str) -> float:
    """Heuristic 'droning density' score; higher is worse.

    This is Horace-native: the intent is to penalize bureaucratic density, hedging,
    and nominalization-heavy phrasing that often reads like "competent vanilla".
    """
    stats = _bureaucracy_stats(text)
    sents = float(_sentence_count_est(text))
    passive_per_sent = float(stats.get("passive_hits") or 0.0) / max(1.0, sents)
    mean_wl = float(stats.get("mean_word_len") or 0.0)
    score = 0.0
    score += 5.0 * float(stats.get("nominal_rate") or 0.0)
    score += 4.0 * float(stats.get("hedge_rate") or 0.0)
    score += 1.5 * float(stats.get("long_word_rate") or 0.0)
    score += 0.30 * passive_per_sent
    score += 0.10 * max(0.0, mean_wl - 4.7)
    return float(score)


def _dead_zone_penalty_and_reasons(zone_text: str, *, sent_len_cv: Optional[float]) -> Tuple[bool, float, List[str]]:
    """Return (keep_candidate, extra_penalty, reasons) for a flat-cadence span.

    "First, do no harm": flat cadence alone is not a problem. We only surface spans
    that are flat *and* show either repetition or droning density.
    """
    wstats = _word_stats(zone_text)
    bstats = _bureaucracy_stats(zone_text)

    trigram_rep = wstats.get("trigram_repeat")
    adjacent_rep = wstats.get("adjacent_repeat")

    reasons = ["flat_cadence"]
    pen = 0.0

    has_rep = False
    if isinstance(trigram_rep, (int, float)) and float(trigram_rep) >= 0.18:
        reasons.append("high_trigram_repetition")
        pen += min(0.35, float(trigram_rep))
        has_rep = True
    if isinstance(adjacent_rep, (int, float)) and float(adjacent_rep) >= 0.06:
        reasons.append("adjacent_word_repetition")
        pen += min(0.20, float(adjacent_rep))
        has_rep = True

    is_droning = (
        float(bstats.get("nominal_rate") or 0.0) >= 0.060
        or float(bstats.get("hedge_rate") or 0.0) >= 0.028
        or float(bstats.get("passive_hits") or 0.0) >= 2.0
        or float(bstats.get("long_word_rate") or 0.0) >= 0.22
    )
    if is_droning:
        reasons.append("droning_density")
        pen += 0.15

    has_uniform_len = isinstance(sent_len_cv, (int, float)) and float(sent_len_cv) <= 0.08
    if has_uniform_len:
        reasons.append("uniform_sentence_length")
        pen += 0.12

    keep = bool(has_rep or is_droning)
    return keep, float(pen), reasons

_CALIBRATOR_CACHE: Dict[str, Any] = {}


@dataclass(frozen=True)
class DeadZone:
    zone_id: int
    start_char: int
    end_char: int
    sent_start: int
    sent_end: int
    severity: float
    reasons: List[str]
    excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchCandidate:
    candidate_id: int
    replacement: str
    patched_text: str
    texture_before: float
    texture_after: float
    texture_gain: float
    droning_before: float
    droning_after: float
    droning_delta: float
    brevity_gain: float
    clarity_gain: float
    rank_score: float
    meaning_lock: Dict[str, Any]
    span_diff: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _word_stats(text: str) -> Dict[str, Optional[float]]:
    t = text or ""
    words = [m.group(0).lower() for m in _WORD_RE.finditer(t)]
    if not words:
        return {"ttr": None, "adjacent_repeat": None, "trigram_repeat": None}
    n = len(words)
    uniq = len(set(words))
    ttr = float(uniq) / float(n) if n > 0 else None
    adjacent_repeat = None
    if n >= 2:
        adjacent = sum(1 for i in range(1, n) if words[i] == words[i - 1])
        adjacent_repeat = float(adjacent) / float(n - 1)
    trigram_repeat = None
    if n >= 4:
        trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(n - 2)]
        trigram_repeat = 1.0 - (float(len(set(trigrams))) / float(len(trigrams))) if trigrams else None
    return {"ttr": ttr, "adjacent_repeat": adjacent_repeat, "trigram_repeat": trigram_repeat}


def _texture_score_from_metrics(doc_metrics: Dict[str, Any]) -> float:
    # A simple, local "texture" heuristic. Higher is better.
    # This is intentionally not a global 0-100 quality score.
    s_cv = doc_metrics.get("surprisal_cv")
    s_masd = doc_metrics.get("surprisal_masd")
    burst = doc_metrics.get("sent_burst_cv")
    len_cv = doc_metrics.get("sent_len_cv")
    trigram_rep = doc_metrics.get("trigram_repeat_rate")
    adjacent_rep = doc_metrics.get("adjacent_word_repeat_rate")
    word_ttr = doc_metrics.get("word_ttr")

    def f(x: Any, default: float = 0.0) -> float:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return float(x)
        return float(default)

    # Encourage some variability; penalize repetition.
    score = 0.0
    score += 0.55 * f(s_cv)
    score += 0.25 * f(burst)
    score += 0.10 * f(len_cv)
    score += 0.15 * f(s_masd)
    score += 0.08 * f(word_ttr)
    score -= 0.70 * f(trigram_rep)
    score -= 0.35 * f(adjacent_rep)
    return float(score)


def _span_excerpt(text: str, start: int, end: int, *, max_chars: int = 220) -> str:
    t = text or ""
    s = max(0, min(int(start), len(t)))
    e = max(s, min(int(end), len(t)))
    seg = t[s:e].strip()
    if len(seg) <= int(max_chars):
        return seg
    return seg[: int(max_chars)].rstrip() + "…"


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    rs = sorted((int(a), int(b)) for a, b in ranges if int(a) <= int(b))
    out: List[Tuple[int, int]] = []
    cur_s, cur_e = rs[0]
    for s, e in rs[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
            continue
        out.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out


def suggest_dead_zones(
    text: str,
    *,
    doc_type: str = "prose",
    scoring_model_id: str = "gpt2",
    backend: str = "auto",
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    window_sentences: int = 4,
    max_zones: int = 6,
    window_start_char: Optional[int] = None,
    window_end_char: Optional[int] = None,
) -> Dict[str, Any]:
    """Suggest dead zones (flat/monotonous areas) in text.

    Args:
        text: Input text to analyze
        doc_type: Document type for analysis
        scoring_model_id: Model for surprisal scoring
        backend: Model backend
        max_input_tokens: Max tokens for analysis
        normalize_text: Whether to normalize text first
        window_sentences: Number of sentences per sliding window
        max_zones: Maximum number of dead zones to return
        window_start_char: Optional start character to limit analysis to a window
        window_end_char: Optional end character to limit analysis to a window

    Returns:
        Dict with text, analysis, and list of dead zones
    """
    raw = text or ""
    norm_text, norm_meta = normalize_for_studio(raw, doc_type=str(doc_type), enabled=bool(normalize_text))
    analysis = analyze_text(
        norm_text,
        model_id=str(scoring_model_id),
        doc_type=str(doc_type),
        backend=str(backend),
        max_input_tokens=int(max_input_tokens),
        normalize_text=False,  # already normalized above
        compute_cohesion=False,
    )

    seg = (analysis.get("segments") or {}).get("sentences") or {}
    means = seg.get("mean_surprisal") or []
    tok_counts = seg.get("token_counts") or []
    items = seg.get("items") or []
    if not means or not items or len(means) < 6:
        return {"text": norm_text, "text_normalization": norm_meta, "analysis": analysis, "dead_zones": []}

    # Filter sentences to window bounds if specified
    window_info: Optional[Dict[str, Any]] = None
    if window_start_char is not None or window_end_char is not None:
        ws = int(window_start_char) if window_start_char is not None else 0
        we = int(window_end_char) if window_end_char is not None else len(norm_text)
        # Find sentences overlapping the window
        filtered_indices: List[int] = []
        for idx, item in enumerate(items):
            sc = int(item.get("start_char") or 0)
            ec = int(item.get("end_char") or sc)
            # Sentence overlaps window if it starts before window ends AND ends after window starts
            if sc < we and ec > ws:
                filtered_indices.append(idx)
        if len(filtered_indices) < 4:
            # Too few sentences in window for meaningful analysis
            return {"text": norm_text, "text_normalization": norm_meta, "analysis": analysis, "dead_zones": [],
                    "window": {"start_char": ws, "end_char": we, "sentence_count": len(filtered_indices)}}
        # Build filtered arrays
        items = [items[i] for i in filtered_indices]
        means = [means[i] for i in filtered_indices if i < len(means)]
        tok_counts = [tok_counts[i] for i in filtered_indices if i < len(tok_counts)]
        window_info = {"start_char": ws, "end_char": we, "sentence_count": len(filtered_indices),
                       "original_indices": filtered_indices}

    arr = np.array([float(x) for x in means[: len(items)]], dtype=np.float32)
    counts_arr = None
    try:
        if tok_counts and len(tok_counts) >= len(items):
            counts_arr = np.array([float(x) for x in tok_counts[: len(items)]], dtype=np.float32)
    except Exception:
        counts_arr = None
    global_sd = float(np.std(arr))
    if not math.isfinite(global_sd) or global_sd <= 1e-6:
        result: Dict[str, Any] = {"text": norm_text, "text_normalization": norm_meta, "analysis": analysis, "dead_zones": []}
        if window_info:
            result["window"] = window_info
        return result

    W = max(3, int(window_sentences))
    threshold = max(0.10, 0.25 * global_sd)

    candidates: List[Tuple[Tuple[int, int], float, List[str]]] = []
    for i in range(0, len(items) - W + 1):
        window = arr[i : i + W]
        w_sd = float(np.std(window))
        if not math.isfinite(w_sd):
            continue
        if w_sd > threshold:
            continue

        w_len_cv = None
        if counts_arr is not None and (i + W) <= len(counts_arr):
            wc = counts_arr[i : i + W]
            mu = float(np.mean(wc))
            sd = float(np.std(wc))
            if math.isfinite(mu) and abs(mu) > 1e-12 and math.isfinite(sd):
                w_len_cv = float(sd / (abs(mu) + 1e-12))

        s0 = int(items[i].get("start_char") or 0)
        s1 = int(items[i + W - 1].get("end_char") or s0)
        zone_text = norm_text[s0:s1]
        keep, extra_pen, reasons = _dead_zone_penalty_and_reasons(zone_text, sent_len_cv=w_len_cv)
        if not keep:
            continue

        sev = float((threshold - w_sd) / max(threshold, 1e-6))
        sev = float(np.clip(sev + float(extra_pen), 0.0, 1.25))
        candidates.append(((i, i + W - 1), sev, reasons))

    if not candidates:
        result = {"text": norm_text, "text_normalization": norm_meta, "analysis": analysis, "dead_zones": []}
        if window_info:
            result["window"] = window_info
        return result

    candidates.sort(key=lambda x: float(x[1]), reverse=True)

    sent_ranges = [c[0] for c in candidates[: max(10, int(max_zones) * 2)]]
    merged = _merge_ranges(sent_ranges)

    zones: List[DeadZone] = []
    zid = 0
    for ss, se in merged:
        if zid >= int(max_zones):
            break
        ss = max(0, ss)
        se = min(len(items) - 1, se)
        start_char = int(items[ss].get("start_char") or 0)
        end_char = int(items[se].get("end_char") or start_char)
        # Recompute severity and reasons over the merged span.
        span_means = arr[ss : se + 1]
        span_sd = float(np.std(span_means))
        span_len_cv = None
        if counts_arr is not None and 0 <= ss <= se < len(counts_arr):
            wc = counts_arr[ss : se + 1]
            mu = float(np.mean(wc))
            sd = float(np.std(wc))
            if math.isfinite(mu) and abs(mu) > 1e-12 and math.isfinite(sd):
                span_len_cv = float(sd / (abs(mu) + 1e-12))
        keep, extra_pen, reasons = _dead_zone_penalty_and_reasons(
            norm_text[start_char:end_char],
            sent_len_cv=span_len_cv,
        )
        if not keep:
            continue

        sev = float((threshold - span_sd) / max(threshold, 1e-6))
        sev = float(np.clip(sev + float(extra_pen), 0.0, 1.25))
        excerpt = _span_excerpt(norm_text, start_char, end_char)
        zones.append(
            DeadZone(
                zone_id=int(zid),
                start_char=int(start_char),
                end_char=int(end_char),
                sent_start=int(ss),
                sent_end=int(se),
                severity=float(sev),
                reasons=reasons,
                excerpt=excerpt,
            )
        )
        zid += 1

    result = {
        "text": norm_text,
        "text_normalization": norm_meta,
        "analysis": analysis,
        "dead_zones": [z.to_dict() for z in zones],
    }
    if window_info:
        result["window"] = window_info
    return result


def _unified_span_diff(original: str, replacement: str) -> str:
    a = (original or "").splitlines(keepends=True)
    b = (replacement or "").splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile="span_before", tofile="span_after", lineterm="")
    # keep it short for UI
    out = list(diff)
    if len(out) > 80:
        out = out[:80] + ["… (diff truncated)"]
    return "\n".join(out)


def _ensure_baseline(baseline_model_or_path: str):
    ident = (baseline_model_or_path or "").strip() or "gpt2"
    p = Path(ident)
    if p.exists():
        return load_baseline_cached(ident, path=p)
    try:
        return load_baseline_cached(ident)
    except Exception:
        build_baseline(ident)
        return load_baseline_cached(ident)


def _ensure_calibrator(calibrator_path: str):
    ident = (calibrator_path or "").strip()
    if not ident:
        return None
    cached = _CALIBRATOR_CACHE.get(ident)
    if cached is not None:
        return cached
    p = Path(ident)
    if not p.exists():
        raise FileNotFoundError(ident)
    cal = load_logistic_calibrator(p)
    _CALIBRATOR_CACHE[ident] = cal
    return cal


def _primary_score_for_text(
    text: str,
    *,
    doc_type: str,
    scoring_model_id: str,
    baseline_model_id: str,
    calibrator_path: str,
    scorer_model_path: str,
    scorer_max_length: int,
    backend: str,
    max_input_tokens: int,
) -> Dict[str, Any]:
    """Compute a stable-ish 0–100 score used for UX (not optimization)."""
    if (scorer_model_path or "").strip():
        try:
            from tools.studio.scorer_model import score_with_scorer

            ts = score_with_scorer(
                text,
                model_path_or_id=str(scorer_model_path),
                doc_type=str(doc_type),
                normalize_text=False,
                max_length=int(scorer_max_length),
                device=None,
            )
            return {"overall_0_100": float(ts.score_0_100), "source": "trained_scorer"}
        except Exception as e:
            return {"error": f"trained_scorer_failed: {type(e).__name__}: {e}"}

    analysis = analyze_text(
        text,
        model_id=str(scoring_model_id),
        doc_type=str(doc_type),
        backend=str(backend),
        max_input_tokens=int(max_input_tokens),
        normalize_text=False,
        compute_cohesion=False,
    )
    baseline = _ensure_baseline(baseline_model_id)
    score = score_text(analysis.get("doc_metrics") or {}, baseline, doc_type=doc_type)

    out: Dict[str, Any] = {"rubric_0_100": float(score.overall_0_100)}

    cal = _ensure_calibrator(calibrator_path)
    if cal is None:
        out["overall_0_100"] = float(score.overall_0_100)
        out["source"] = "rubric"
        return out

    missing_value = float(getattr(cal, "meta", {}).get("missing_value", 0.5))
    rubric_metrics = {k: {"score_0_1": v.score_0_1} for k, v in (score.metrics or {}).items()}
    feats = featurize_from_report_row(
        feature_names=cal.feature_names,
        categories=score.categories or {},
        rubric_metrics=rubric_metrics,
        doc_metrics=analysis.get("doc_metrics") or {},
        max_input_tokens=int(max_input_tokens),
        missing_value=missing_value,
    )
    out["calibrated_0_100"] = float(cal.score_0_100(feats))
    out["overall_0_100"] = float(out["calibrated_0_100"])
    out["source"] = "rubric_calibrated"
    return out


def patch_span(
    text: str,
    *,
    start_char: int,
    end_char: int,
    doc_type: str = "prose",
    rewrite_mode: str = "strict",
    intensity: float = 0.5,  # 0=clearer, 1=punchier
    rewrite_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    scoring_model_id: str = "gpt2",
    baseline_model_id: str = "gpt2_gutenberg_512",
    calibrator_path: str = "",
    scorer_model_path: str = "",
    scorer_max_length: int = 384,
    score_top_n: int = 3,
    backend: str = "auto",
    max_input_tokens: int = 384,
    normalize_text: bool = True,
    n_candidates: int = 6,
    max_new_tokens: int = 260,
    temperature: float = 0.8,
    top_p: float = 0.92,
    seed: Optional[int] = 7,
    meaning_lock: Optional[MeaningLockConfig] = None,
    use_cadence_sampler: bool = False,
    cadence_profile: Optional[CadenceProfile] = None,
) -> Dict[str, Any]:
    raw = text or ""
    norm_text, norm_meta = normalize_for_studio(raw, doc_type=str(doc_type), enabled=bool(normalize_text))
    s = max(0, min(int(start_char), len(norm_text)))
    e = max(s, min(int(end_char), len(norm_text)))
    span_slice = norm_text[s:e]
    if not span_slice:
        return {"error": "empty_span", "text": norm_text, "text_normalization": norm_meta, "candidates": []}
    lead = len(span_slice) - len(span_slice.lstrip())
    trail = len(span_slice) - len(span_slice.rstrip())
    core_s = min(len(norm_text), s + int(lead))
    core_e = max(core_s, e - int(trail))
    span = norm_text[core_s:core_e]
    if not span:
        return {"error": "empty_span", "text": norm_text, "text_normalization": norm_meta, "candidates": []}

    # Texture score before (span only).
    before_analysis = analyze_text(
        span,
        model_id=str(scoring_model_id),
        doc_type=str(doc_type),
        backend=str(backend),
        max_input_tokens=int(max_input_tokens),
        normalize_text=False,
        compute_cohesion=False,
    )
    texture_before = _texture_score_from_metrics(before_analysis.get("doc_metrics") or {})
    droning_before = _droning_score(span)
    intensity_norm = float(np.clip(float(intensity), 0.0, 1.0))

    # Choose generation method: cadence sampler or LLM prompting
    if use_cadence_sampler:
        rewrites = generate_cadence_span_rewrites(
            norm_text,
            start_char=core_s,
            end_char=core_e,
            doc_type=str(doc_type),
            target_profile=cadence_profile,
            model_id=str(scoring_model_id),  # Use scoring model for cadence gen
            backend=str(backend),
            n_candidates=int(n_candidates),
            max_new_tokens=int(max_new_tokens),
            seed=int(seed) if seed is not None else None,
        )
    else:
        rewrites = generate_span_rewrites(
            norm_text,
            start_char=core_s,
            end_char=core_e,
            doc_type=str(doc_type),
            rewrite_mode=str(rewrite_mode),
            intensity=float(intensity_norm),
            rewrite_model_id=str(rewrite_model_id),
            n=int(n_candidates),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=int(seed) if seed is not None else None,
        )
    if not rewrites:
        return {
            "text": norm_text,
            "text_normalization": norm_meta,
            "span": {"start_char": s, "end_char": e, "core_start_char": core_s, "core_end_char": core_e, "text": span},
            "candidates": [],
            "error": "no_rewrites_generated",
        }

    cfg = meaning_lock or MeaningLockConfig()
    scored: List[PatchCandidate] = []
    cid = 0
    for repl in rewrites:
        ml: MeaningLockReport = check_meaning_lock(span, repl, cfg=cfg)
        if not ml.ok:
            continue
        after_analysis = analyze_text(
            repl,
            model_id=str(scoring_model_id),
            doc_type=str(doc_type),
            backend=str(backend),
            max_input_tokens=int(max_input_tokens),
            normalize_text=False,
            compute_cohesion=False,
        )
        texture_after = _texture_score_from_metrics(after_analysis.get("doc_metrics") or {})
        droning_after = _droning_score(repl)
        droning_delta = float(droning_after - droning_before)
        len_ratio = float(len(repl) / max(1, len(span)))
        # Reward modest tightening, but do not reward aggressive deletion.
        brevity_gain = float(min(max(0.0, 1.0 - len_ratio), 0.12))
        deletion_penalty = float(max(0.0, 0.90 - len_ratio))
        clarity_gain = float((-droning_delta) + 0.25 * brevity_gain - deletion_penalty)
        rank_score = float(
            (intensity_norm * float(texture_after - texture_before))
            + ((1.0 - intensity_norm) * float(clarity_gain))
            - 0.10 * max(0.0, droning_delta)
        )
        patched = norm_text[:core_s] + repl + norm_text[core_e:]
        scored.append(
            PatchCandidate(
                candidate_id=int(cid),
                replacement=str(repl),
                patched_text=str(patched),
                texture_before=float(texture_before),
                texture_after=float(texture_after),
                texture_gain=float(texture_after - texture_before),
                droning_before=float(droning_before),
                droning_after=float(droning_after),
                droning_delta=float(droning_delta),
                brevity_gain=float(brevity_gain),
                clarity_gain=float(clarity_gain),
                rank_score=float(rank_score),
                meaning_lock=ml.to_dict(),
                span_diff=_unified_span_diff(span, repl),
            )
        )
        cid += 1

    scored.sort(
        key=lambda c: (float(c.rank_score), float(c.texture_gain), float(c.meaning_lock.get("cosine_sim") or 0.0)),
        reverse=True,
    )

    candidates_out: List[Dict[str, Any]] = []

    primary_before = None
    primary_before_err = None
    try:
        primary_before = _primary_score_for_text(
            norm_text,
            doc_type=str(doc_type),
            scoring_model_id=str(scoring_model_id),
            baseline_model_id=str(baseline_model_id),
            calibrator_path=str(calibrator_path or ""),
            scorer_model_path=str(scorer_model_path or ""),
            scorer_max_length=int(scorer_max_length),
            backend=str(backend),
            max_input_tokens=int(max_input_tokens),
        )
        if isinstance(primary_before, dict) and primary_before.get("error"):
            primary_before_err = str(primary_before.get("error"))
    except Exception as e:
        primary_before_err = f"{type(e).__name__}: {e}"

    # Compute doc-level 0–100 for a small top set (or all, if using a trained scorer).
    use_trained = bool((scorer_model_path or "").strip())
    n_score = len(scored) if use_trained else min(max(0, int(score_top_n)), len(scored))

    primary_after: List[Optional[Dict[str, Any]]] = [None] * len(scored)
    for i in range(n_score):
        try:
            primary_after[i] = _primary_score_for_text(
                scored[i].patched_text,
                doc_type=str(doc_type),
                scoring_model_id=str(scoring_model_id),
                baseline_model_id=str(baseline_model_id),
                calibrator_path=str(calibrator_path or ""),
                scorer_model_path=str(scorer_model_path or ""),
                scorer_max_length=int(scorer_max_length),
                backend=str(backend),
                max_input_tokens=int(max_input_tokens),
            )
        except Exception as e:
            primary_after[i] = {"error": f"{type(e).__name__}: {e}"}

    before_0_100 = primary_before.get("overall_0_100") if isinstance(primary_before, dict) else None
    primary_before_status = "not_scored"
    if isinstance(primary_before, dict) and isinstance(primary_before.get("overall_0_100"), (int, float)):
        primary_before_status = "scored"
    elif primary_before_err:
        primary_before_status = "error"

    for i, c in enumerate(scored[: max(1, int(n_candidates))]):
        d = c.to_dict()
        after = primary_after[i] if i < len(primary_after) else None
        if i >= n_score:
            d["primary_after_status"] = "not_scored"
            d["primary_after_reason"] = "score_top_n_limit"
        elif isinstance(after, dict):
            d["primary_after"] = after
            if after.get("error"):
                d["primary_after_status"] = "error"
                d["primary_after_error"] = str(after.get("error"))
            else:
                d["primary_after_status"] = "scored"
                after_0_100 = after.get("overall_0_100")
                if isinstance(before_0_100, (int, float)) and isinstance(after_0_100, (int, float)):
                    d["primary_delta_0_100"] = float(after_0_100 - before_0_100)
        else:
            d["primary_after_status"] = "not_scored"
            d["primary_after_reason"] = "missing_result"
        candidates_out.append(d)

    return {
        "text": norm_text,
        "text_normalization": norm_meta,
        "span": {"start_char": s, "end_char": e, "core_start_char": core_s, "core_end_char": core_e, "text": span},
        "control": {"intensity_0_1": float(intensity_norm)},
        "primary_before": primary_before,
        "primary_before_status": primary_before_status,
        "primary_before_error": primary_before_err,
        "candidates": candidates_out,
    }
