"""Spike pattern extraction and analysis.

So what: learns *what* makes a good spike and *when* to place one by analyzing
the corpus of literary text, enabling more intentional spike injection during generation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.studio.analyze import analyze_text


@dataclass
class SpikePattern:
    """Pattern describing a detected spike in text."""

    # Position information
    token_index: int
    char_start: int
    char_end: int
    position_in_sentence: str  # 'start', 'middle', 'end'
    sentence_index: int

    # Token information
    token_text: str
    token_type: str  # 'content', 'function', 'punctuation'

    # Context tokens (before/after)
    prev_token: Optional[str]
    prev_token_type: Optional[str]
    next_token: Optional[str]
    next_token_type: Optional[str]

    # Surprise metrics
    surprisal: float
    surprisal_delta: float  # vs local average
    entropy_context: float  # entropy of surrounding tokens
    rank: int  # rank in vocabulary

    # Semantic hints
    is_content_word: bool
    ends_clause: bool  # followed by comma/period

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpikeStats:
    """Aggregated statistics about spikes in a document or corpus."""

    total_spikes: int
    spike_rate: float  # spikes per 100 tokens

    # Position distribution
    position_counts: Dict[str, int]  # start/middle/end counts
    position_rates: Dict[str, float]

    # Type distribution
    type_counts: Dict[str, int]  # content/function/punctuation
    type_rates: Dict[str, float]

    # Surprise statistics
    mean_surprisal: float
    std_surprisal: float
    mean_surprisal_delta: float
    mean_rank: float

    # Inter-spike intervals
    ipi_mean: float
    ipi_std: float
    ipi_median: float

    # Cooldown patterns
    avg_cooldown_length: float  # tokens until entropy recovery

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_STOPWORDS = set(
    """a an the and or but if then else when while for nor so yet at by from in
    of on to up with as is are was were be been being do does did have has had
    i me my we us our you your he him his she her it its they them their this
    that which what where when why how not no""".split()
)


def _classify_token(token: str) -> str:
    """Classify token as content, function, or punctuation."""
    t = token.strip()
    if not t:
        return "punctuation"
    # Check if purely punctuation
    if all(not c.isalnum() for c in t):
        return "punctuation"
    # Check if stopword/function word
    word = "".join(c for c in t.lower() if c.isalpha())
    if word in _STOPWORDS:
        return "function"
    if len(word) <= 2:
        return "function"
    return "content"


def _position_in_sentence(
    token_idx: int, sentence_start_idx: int, sentence_end_idx: int
) -> str:
    """Classify position within sentence."""
    length = sentence_end_idx - sentence_start_idx
    if length <= 2:
        return "middle"
    relative = token_idx - sentence_start_idx
    if relative <= 1:
        return "start"
    if relative >= length - 2:
        return "end"
    return "middle"


def extract_spike_patterns(
    text: str,
    *,
    model_id: str = "gpt2",
    backend: str = "auto",
    max_input_tokens: int = 512,
    doc_type: str = "prose",
    surprisal_threshold: float = 2.0,  # tokens with surprisal > threshold are spikes
) -> List[SpikePattern]:
    """Extract spike patterns from text.

    Args:
        text: Input text to analyze
        model_id: Model for analysis
        backend: 'auto', 'mlx', or 'hf'
        max_input_tokens: Max tokens to process
        doc_type: 'prose' or 'poem'
        surprisal_threshold: Surprisal cutoff for spike detection

    Returns:
        List of SpikePattern objects describing each detected spike
    """
    analysis = analyze_text(
        text,
        model_id=model_id,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=max_input_tokens,
        normalize_text=True,
        compute_cohesion=False,
        include_token_metrics=True,
    )

    token_metrics = analysis.get("token_metrics") or {}
    tokens = analysis.get("tokens") or []
    segments = analysis.get("segments") or {}
    sentences = segments.get("sentences") or {}
    sentence_items = sentences.get("items") or []

    surprisal = token_metrics.get("surprisal") or []
    entropy = token_metrics.get("entropy") or []
    rank = token_metrics.get("rank") or []

    if not tokens or not surprisal:
        return []

    # Build sentence boundaries
    sentence_ranges: List[Tuple[int, int]] = []
    for sent in sentence_items:
        s_start = sent.get("start_token", 0)
        s_end = sent.get("end_token", len(tokens))
        sentence_ranges.append((s_start, s_end))

    # Compute local averages for surprisal delta
    window = 5
    local_avg = np.convolve(
        surprisal, np.ones(window) / window, mode="same"
    ).tolist()

    patterns: List[SpikePattern] = []

    for i, tok in enumerate(tokens):
        if i >= len(surprisal):
            break

        s = surprisal[i]
        if s < surprisal_threshold:
            continue

        # This is a spike
        tok_text = tok.get("token", "") if isinstance(tok, dict) else str(tok)
        tok_type = _classify_token(tok_text)

        # Find sentence
        sent_idx = 0
        sent_start, sent_end = 0, len(tokens)
        for si, (ss, se) in enumerate(sentence_ranges):
            if ss <= i < se:
                sent_idx = si
                sent_start, sent_end = ss, se
                break

        pos = _position_in_sentence(i, sent_start, sent_end)

        # Get prev/next context
        prev_tok = None
        prev_type = None
        if i > 0 and i - 1 < len(tokens):
            pt = tokens[i - 1]
            prev_tok = pt.get("token", "") if isinstance(pt, dict) else str(pt)
            prev_type = _classify_token(prev_tok)

        next_tok = None
        next_type = None
        if i + 1 < len(tokens):
            nt = tokens[i + 1]
            next_tok = nt.get("token", "") if isinstance(nt, dict) else str(nt)
            next_type = _classify_token(next_tok)

        # Compute surprisal delta
        delta = s - local_avg[i] if i < len(local_avg) else 0.0

        # Entropy context (average of surrounding)
        ent_ctx = 0.0
        if entropy:
            start = max(0, i - 2)
            end = min(len(entropy), i + 3)
            ent_ctx = float(np.mean(entropy[start:end])) if end > start else 0.0

        # Rank
        rk = rank[i] if i < len(rank) else 0

        # Content word check
        is_content = tok_type == "content"

        # Ends clause check
        ends_clause = next_type == "punctuation" if next_type else False

        # Char positions
        char_start = tok.get("start", 0) if isinstance(tok, dict) else 0
        char_end = tok.get("end", 0) if isinstance(tok, dict) else 0

        patterns.append(
            SpikePattern(
                token_index=i,
                char_start=char_start,
                char_end=char_end,
                position_in_sentence=pos,
                sentence_index=sent_idx,
                token_text=tok_text,
                token_type=tok_type,
                prev_token=prev_tok,
                prev_token_type=prev_type,
                next_token=next_tok,
                next_token_type=next_type,
                surprisal=float(s),
                surprisal_delta=float(delta),
                entropy_context=float(ent_ctx),
                rank=int(rk),
                is_content_word=is_content,
                ends_clause=ends_clause,
            )
        )

    return patterns


def compute_spike_stats(patterns: List[SpikePattern], total_tokens: int) -> SpikeStats:
    """Compute aggregate statistics from a list of spike patterns."""
    n = len(patterns)
    if n == 0:
        return SpikeStats(
            total_spikes=0,
            spike_rate=0.0,
            position_counts={"start": 0, "middle": 0, "end": 0},
            position_rates={"start": 0.0, "middle": 0.0, "end": 0.0},
            type_counts={"content": 0, "function": 0, "punctuation": 0},
            type_rates={"content": 0.0, "function": 0.0, "punctuation": 0.0},
            mean_surprisal=0.0,
            std_surprisal=0.0,
            mean_surprisal_delta=0.0,
            mean_rank=0.0,
            ipi_mean=0.0,
            ipi_std=0.0,
            ipi_median=0.0,
            avg_cooldown_length=0.0,
        )

    # Position counts
    pos_counts = {"start": 0, "middle": 0, "end": 0}
    for p in patterns:
        pos_counts[p.position_in_sentence] = pos_counts.get(p.position_in_sentence, 0) + 1
    pos_rates = {k: v / n for k, v in pos_counts.items()}

    # Type counts
    type_counts = {"content": 0, "function": 0, "punctuation": 0}
    for p in patterns:
        type_counts[p.token_type] = type_counts.get(p.token_type, 0) + 1
    type_rates = {k: v / n for k, v in type_counts.items()}

    # Surprisal stats
    surprisals = [p.surprisal for p in patterns]
    deltas = [p.surprisal_delta for p in patterns]
    ranks = [p.rank for p in patterns]

    # IPIs
    indices = sorted(p.token_index for p in patterns)
    ipis = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)] if len(indices) > 1 else [0]

    return SpikeStats(
        total_spikes=n,
        spike_rate=100.0 * n / max(1, total_tokens),
        position_counts=pos_counts,
        position_rates=pos_rates,
        type_counts=type_counts,
        type_rates=type_rates,
        mean_surprisal=float(np.mean(surprisals)),
        std_surprisal=float(np.std(surprisals)),
        mean_surprisal_delta=float(np.mean(deltas)),
        mean_rank=float(np.mean(ranks)),
        ipi_mean=float(np.mean(ipis)),
        ipi_std=float(np.std(ipis)),
        ipi_median=float(np.median(ipis)),
        avg_cooldown_length=float(np.mean(ipis)) * 0.3,  # Rough estimate
    )


def suggest_spike_positions(
    text: str,
    *,
    model_id: str = "gpt2",
    backend: str = "auto",
    max_input_tokens: int = 512,
    doc_type: str = "prose",
    target_spike_rate: float = 8.0,  # spikes per 100 tokens
) -> List[Dict[str, Any]]:
    """Suggest positions where adding a spike might improve cadence.

    Looks for "valleys" - spans with low surprisal that could benefit from variation.

    Returns list of suggestions with char positions and reasons.
    """
    analysis = analyze_text(
        text,
        model_id=model_id,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=max_input_tokens,
        normalize_text=True,
        compute_cohesion=False,
        include_token_metrics=True,
    )

    token_metrics = analysis.get("token_metrics") or {}
    tokens = analysis.get("tokens") or []
    surprisal = token_metrics.get("surprisal") or []

    if not tokens or not surprisal or len(tokens) < 10:
        return []

    current_spike_rate = sum(1 for s in surprisal if s > 2.0) * 100 / len(surprisal)

    if current_spike_rate >= target_spike_rate:
        return []  # Already has enough spikes

    # Find valleys - runs of low surprisal tokens
    threshold = float(np.percentile(surprisal, 30))  # Bottom 30%
    valleys: List[Tuple[int, int]] = []
    start = None

    for i, s in enumerate(surprisal):
        if s < threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 4:  # At least 4 tokens
                valleys.append((start, i))
            start = None

    if start is not None and len(surprisal) - start >= 4:
        valleys.append((start, len(surprisal)))

    suggestions: List[Dict[str, Any]] = []

    for valley_start, valley_end in valleys[:5]:  # Top 5 valleys
        mid = (valley_start + valley_end) // 2
        if mid < len(tokens):
            tok = tokens[mid]
            char_start = tok.get("start", 0) if isinstance(tok, dict) else 0
            char_end = tok.get("end", 0) if isinstance(tok, dict) else 0

            suggestions.append({
                "token_index": mid,
                "char_start": char_start,
                "char_end": char_end,
                "valley_length": valley_end - valley_start,
                "avg_surprisal": float(np.mean(surprisal[valley_start:valley_end])),
                "reason": "low_variation_span",
                "suggestion": "Consider a more surprising word choice here",
            })

    # Sort by valley length (longer = more need for spike)
    suggestions.sort(key=lambda x: x["valley_length"], reverse=True)

    return suggestions


def save_spike_patterns(patterns: List[SpikePattern], path: Path) -> None:
    """Save patterns to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in patterns:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_spike_patterns(path: Path) -> List[SpikePattern]:
    """Load patterns from JSONL file."""
    patterns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            patterns.append(SpikePattern(**data))
    return patterns
