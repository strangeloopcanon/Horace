"""
Horace GRPO Reward (v0 skeleton)

Simple, toggleable reward composed of five core signals:
- cadence, surprise, distinctive, corridor, wander_return
Plus regularizers: KL, safety, format.

This module provides a minimal API and placeholders so training code
can call compute_reward() and log component breakdowns. Implementations
return zeros by default; fill in using Horace analysis utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


@dataclass
class RewardWeights:
    cad: float = 0.3
    sup: float = 0.25
    dis: float = 0.15
    cor: float = 0.2
    arc: float = 0.1
    kl: float = 0.05
    safety: float = 1.0
    format: float = 0.5
    # Optional plugin weights
    rhyme: float = 0.0
    meter: float = 0.0


@dataclass
class Corridor:
    r_lo: float = 0.2
    r_hi: float = 0.6
    soft_margin: float = 0.1


@dataclass
class PresetConfig:
    name: str
    weights: RewardWeights
    corridor: Corridor
    format_lines: Optional[int] = None
    enable_rhyme: bool = False
    enable_meter: bool = False
    notes: Optional[str] = None


def load_presets(path: str) -> Dict[str, PresetConfig]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    presets: Dict[str, PresetConfig] = {}
    for name, cfg in raw.items():
        weights = RewardWeights(**cfg.get("weights", {}))
        cor = Corridor(**cfg.get("corridor", {}))
        presets[name] = PresetConfig(
            name=name,
            weights=weights,
            corridor=cor,
            format_lines=cfg.get("format_lines"),
            enable_rhyme=cfg.get("enable_rhyme", False),
            enable_meter=cfg.get("enable_meter", False),
            notes=cfg.get("notes"),
        )
    return presets


def normalize_components(components: Dict[str, float], stats: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Normalize component scores to [0,1].
    If stats not provided, clamp into [0,1]. Replace with min-max or z→sigmoid using baseline stats.
    """
    out = {}
    for k, v in components.items():
        if stats and k in stats:
            s = stats[k]
            if "min" in s and "max" in s and s["max"] > s["min"]:
                v = (v - s["min"]) / (s["max"] - s["min"])
        out[k] = max(0.0, min(1.0, float(v)))
    return out


def cadence_score(sample: Dict[str, Any], preset: PresetConfig) -> float:
    """Compare line-level entropy traces to a target envelope.
    Heuristic envelope: one spike per line near 60% of tokens, then cooldown.
    Returns [0,1]. Requires sample['logits_per_step'] and sample['line_steps'].
    """
    logits_steps = sample.get("logits_per_step")
    line_steps: List[Tuple[int, int]] = sample.get("line_steps") or []
    if not logits_steps or not line_steps:
        return 0.0

    # Entropy per step
    def entropy_from_logits(logits: np.ndarray) -> float:
        x = logits.astype(np.float32)
        x = x - np.max(x)
        p = np.exp(x)
        p = p / (p.sum() + 1e-12)
        return float(-(p * (np.log(p + 1e-12))).sum())

    H = [entropy_from_logits(l) for l in logits_steps]

    def pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3 or b.size != a.size:
            return 0.0
        va = a.var()
        vb = b.var()
        if va < 1e-8 or vb < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    per_line_scores: List[float] = []
    early_end_penalties: List[float] = []
    for start, end in line_steps:
        if end <= start:
            continue
        h = np.array(H[start:end], dtype=np.float32)
        L = h.size
        if L < 4:
            continue
        # Target envelope: gaussian peak at mu=0.6*L, sigma=0.2*L
        t = np.arange(L, dtype=np.float32)
        mu = 0.6 * (L - 1)
        sigma = max(1.0, 0.2 * (L - 1))
        env = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        # Normalize both
        if h.std() > 1e-6:
            h = (h - h.mean()) / (h.std() + 1e-6)
        if env.std() > 1e-6:
            env = (env - env.mean()) / (env.std() + 1e-6)
        corr = pearson(h, env)
        # Map [-1,1] -> [0,1]
        s = 0.5 * (corr + 1.0)
        per_line_scores.append(s)
        # Early-end penalty: too-short lines relative to a soft target of 8 tokens
        target_min = 6
        early_end_penalties.append(0.0 if L >= target_min else (target_min - L) / target_min)

    if not per_line_scores:
        return 0.0
    base = float(np.mean(per_line_scores))
    pen = float(np.mean(early_end_penalties))
    return max(0.0, min(1.0, base * (1.0 - 0.4 * pen)))


def surprise_score(sample: Dict[str, Any], preset: PresetConfig) -> float:
    """Reward 1–2 high-surprisal tokens per line; penalize flat or continuous spikes.
    Uses chosen-token surprisal from logits_per_step and tokens.
    """
    logits_steps = sample.get("logits_per_step")
    toks: List[int] = sample.get("gen_token_ids") or sample.get("tokens") or []
    line_steps: List[Tuple[int, int]] = sample.get("line_steps") or []
    if not logits_steps or not toks or not line_steps:
        return 0.0

    def surprisal(logits: np.ndarray, tid: int) -> float:
        x = logits.astype(np.float32)
        x = x - np.max(x)
        p = np.exp(x)
        p = p / (p.sum() + 1e-12)
        tid = int(tid) if 0 <= int(tid) < p.shape[-1] else int(np.argmax(p))
        pt = float(p[tid])
        return float(-math.log(max(pt, 1e-12)))

    svals = [surprisal(l, t) for l, t in zip(logits_steps, toks[: len(logits_steps)])]
    per_line = []
    for start, end in line_steps:
        end = min(end, len(svals))
        if end <= start:
            continue
        sv = np.array(svals[start:end], dtype=np.float32)
        if sv.size < 3:
            continue
        mu, sd = float(np.mean(sv)), float(np.std(sv) + 1e-6)
        z = (sv - mu) / sd
        spikes = int((z > 1.0).sum())  # 1+ sigma surprises
        L = sv.size
        rate = spikes / max(1, L)
        target_rate = 1.5 / max(4, L)
        # Penalize continuous spike runs (z>1 across >3 tokens)
        run_pen = 0.0
        run, best = 0, 0
        for zi in z:
            if zi > 1.0:
                run += 1
                best = max(best, run)
            else:
                run = 0
        if best >= 4:
            run_pen = min(1.0, 0.15 * (best - 3))
        score = math.exp(-abs(rate - target_rate) / (target_rate + 1e-6)) * (1.0 - run_pen)
        per_line.append(score)
    if not per_line:
        return 0.0
    return float(np.clip(np.mean(per_line), 0.0, 1.0))


def distinctive_score(sample: Dict[str, Any], preset: PresetConfig) -> float:
    """Heuristic distinctiveness: type-token ratio for content tokens and bigram novelty.
    Avoids external IDF/PMI deps. Returns [0,1].
    """
    text = sample.get("gen_text") or sample.get("text") or ""
    if not text:
        return 0.0
    import re
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    content = [w for w in words if len(w) >= 5 and w not in _COMMON]
    if len(content) < 6:
        return 0.0
    types = len(set(content))
    ttr = types / len(content)
    # Bigram novelty
    bigrams = list(zip(content, content[1:]))
    uniq_bi = len(set(bigrams)) / max(1, len(bigrams))
    # Clip extremes to avoid reward hacking
    s = 0.6 * min(1.0, ttr / 0.6) + 0.4 * min(1.0, uniq_bi / 0.7)
    return float(np.clip(s, 0.0, 1.0))


def corridor_score(sample: Dict[str, Any], preset: PresetConfig) -> float:
    """Fraction of lines with theme-distance inside a corridor band.
    Uses hashed bag-of-words cosine distance as a lightweight embedding proxy.
    """
    lines: List[str] = sample.get("lines") or []
    prompt: str = sample.get("prompt") or ""
    if not lines:
        return 0.0

    def hash_vec(s: str, dim: int = 4096) -> np.ndarray:
        import re
        v = np.zeros(dim, dtype=np.float32)
        toks = re.findall(r"[A-Za-z']+", s.lower())
        for t in toks:
            h = (hash(t) % dim)
            v[h] += 1.0
        n = float(np.linalg.norm(v) + 1e-6)
        return v / n

    theme_src = prompt
    # Include first line as part of theme (bootstrap)
    if lines:
        theme_src = (prompt + "\n" + lines[0]).strip()
    theme = hash_vec(theme_src)
    rlo, rhi, m = preset.corridor.r_lo, preset.corridor.r_hi, preset.corridor.soft_margin

    ds: List[float] = []
    for ln in lines:
        v = hash_vec(ln)
        sim = float(np.dot(theme, v))
        d = 1.0 - sim  # cosine distance surrogate
        ds.append(d)
    if not ds:
        return 0.0
    ok = 0
    for d in ds:
        if rlo - m <= d <= rhi + m:
            ok += 1
    return float(np.clip(ok / len(ds), 0.0, 1.0))


def wander_return_score(sample: Dict[str, Any], preset: PresetConfig) -> float:
    """Detect 1–2 excursion arcs that return/reframe near the theme band.
    Uses the same hashed distance sequence over lines.
    """
    lines: List[str] = sample.get("lines") or []
    prompt: str = sample.get("prompt") or ""
    if len(lines) < 3:
        return 0.0
    # Reuse corridor distances
    def hash_vec(s: str, dim: int = 4096) -> np.ndarray:
        import re
        v = np.zeros(dim, dtype=np.float32)
        toks = re.findall(r"[A-Za-z']+", s.lower())
        for t in toks:
            v[hash(t) % dim] += 1.0
        n = float(np.linalg.norm(v) + 1e-6)
        return v / n
    theme = hash_vec((prompt + "\n" + lines[0]).strip())
    ds = [1.0 - float(np.dot(theme, hash_vec(ln))) for ln in lines]

    # Find a peak then a return near the end
    peak_idx = int(np.argmax(ds))
    d0, dmax, dend = ds[0], ds[peak_idx], ds[-1]
    # Require excursion and some return
    if dmax - d0 < 0.1:
        return 0.0
    # Return quality: closeness to start band
    rlo = preset.corridor.r_lo
    ret_ok = 1.0 - float(min(1.0, abs(dend - rlo) / max(1e-3, rlo))) if rlo > 0 else (1.0 - min(1.0, dend))
    # Arc timing: discourage peak at very first/last lines
    t_ok = 1.0 - float(min(1.0, min(peak_idx, len(ds) - 1 - peak_idx) / max(1, len(ds) // 2)))
    s = max(0.0, min(1.0, 0.6 * ret_ok + 0.4 * t_ok))
    return s


def safety_penalty(sample: Dict[str, Any], preset: PresetConfig) -> float:
    """Placeholder: toxicity/NSFW gates; return penalty in [0,1]."""
    return 0.0


def format_penalty(sample: Dict[str, Any], preset: PresetConfig) -> float:
    """Penalty if required format (e.g., 14 lines) not met; else 0."""
    if preset.format_lines is None:
        return 0.0
    lines: List[str] = sample.get("lines") or []
    return 1.0 if len(lines) != preset.format_lines else 0.0


def kl_divergence(sample: Dict[str, Any], refs: Dict[str, Any]) -> float:
    """Approximate per-token KL to reference policy using logits traces if provided.
    Expects refs['ref_logits_per_step'] with same length as sample logits.
    Returns a small non-negative number scaled into [0,1] via tanh.
    """
    logits_steps = sample.get("logits_per_step")
    ref_steps = (refs or {}).get("ref_logits_per_step") if refs else None
    if not logits_steps or ref_steps is None or len(ref_steps) != len(logits_steps):
        return 0.0
    def kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
        def _softmax(x):
            x = x - np.max(x)
            e = np.exp(x)
            return e / (e.sum() + 1e-12)
        p = _softmax(p_logits.astype(np.float32))
        q = _softmax(q_logits.astype(np.float32))
        return float(np.sum(p * (np.log((p + 1e-12) / (q + 1e-12)))))
    vals = [kl(p, q) for p, q in zip(logits_steps, ref_steps)]
    mean_kl = float(np.mean(vals))
    # squash to [0,1) roughly
    return float(np.tanh(0.1 * mean_kl))


def compute_reward(
    sample: Dict[str, Any],
    preset: PresetConfig,
    refs: Optional[Dict[str, Any]] = None,
    norm_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite reward and component breakdown.

    sample expects keys:
      - tokens: List[int]
      - text: str
      - lines: List[str]
      - logits_per_step: Optional[List[List[float]]]
      - line_embeddings: Optional[List[List[float]]]
      - misc features as needed by component scorers

    refs may contain:
      - ref_logits_per_step, target_cadence_envelope, embedding_model, idf_table, etc.
    """
    w = preset.weights

    raw = {
        "cad": cadence_score(sample, preset),
        "sup": surprise_score(sample, preset),
        "dis": distinctive_score(sample, preset),
        "cor": corridor_score(sample, preset),
        "arc": wander_return_score(sample, preset),
        "kl": kl_divergence(sample, refs or {}),
        "saf": safety_penalty(sample, preset),
        "fmt": format_penalty(sample, preset),
    }

    comp = normalize_components(raw, norm_stats)

    reward = (
        w.cad * comp["cad"]
        + w.sup * comp["sup"]
        + w.dis * comp["dis"]
        + w.cor * comp["cor"]
        + w.arc * comp["arc"]
        - w.kl * comp["kl"]
        - w.safety * comp["saf"]
        - w.format * comp["fmt"]
    )

    # Optional plugins (weights default to 0.0)
    if preset.enable_rhyme and w.rhyme:
        comp.setdefault("rhy", 0.0)
        reward += w.rhyme * comp["rhy"]
    if preset.enable_meter and w.meter:
        comp.setdefault("met", 0.0)
        reward += w.meter * comp["met"]

    return float(reward), comp


def example_usage() -> None:
    """Quick smoke test to ensure API composes without external deps."""
    presets = load_presets("configs/reward_presets.json")
    preset = presets["freeverse"]
    sample = {"tokens": [], "text": "", "lines": ["a line"], "logits_per_step": None}
    r, parts = compute_reward(sample, preset)
    print("reward:", r)
    print("components:", parts)


if __name__ == "__main__":
    example_usage()

# Common small stopword-ish list for distinctiveness
_COMMON = set(
    "a an the and or but if then else when while for nor so yet about above below under over into out upon with without within across around between among through during before after where what who which how why whose this that these those here there ever never always often sometimes usually would could should might must can may".split()
)
