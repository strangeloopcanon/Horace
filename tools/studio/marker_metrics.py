from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|[A-Za-z]+)?\b")
_URL_RE = re.compile(r"https?://\S+", flags=re.I)
_BRACKET_CITE_RE = re.compile(r"\[\s*\d{1,4}\s*\]")

# Marker vocabularies are intentionally small and high-precision; we prefer stable, learnable
# signals over brittle coverage.
_CONTRASTIVE = (
    "but",
    "however",
    "though",
    "yet",
    "nevertheless",
    "still",
    "instead",
    "whereas",
)
_CAUSAL = ("because", "since", "therefore", "thus", "hence", "so")
_CONDITIONAL = ("if", "unless", "provided", "assuming")
_TEMPORAL = ("today", "yesterday", "tomorrow", "now", "then", "later", "before", "after", "during", "meanwhile")
_EVIDENTIAL = (
    "according",
    "evidence",
    "data",
    "study",
    "studies",
    "research",
    "report",
    "reports",
    "survey",
    "analysis",
    "said",
    "says",
    "wrote",
    "writes",
    "told",
)
_HEDGES = ("maybe", "perhaps", "apparently", "likely", "probably", "seems", "seemed", "roughly", "around")

_SENSORY = {
    # Visual
    "see",
    "saw",
    "seen",
    "look",
    "looked",
    "gaze",
    "glance",
    "shine",
    "shone",
    "bright",
    "dark",
    "shadow",
    "glow",
    "glitter",
    # Sound
    "hear",
    "heard",
    "sound",
    "silence",
    "whisper",
    "whispered",
    "clang",
    "crash",
    "rumble",
    # Touch/temperature
    "touch",
    "touched",
    "warm",
    "cold",
    "heat",
    "chill",
    "rough",
    "smooth",
    "sharp",
    "soft",
    # Smell/taste
    "smell",
    "smelled",
    "scent",
    "stink",
    "taste",
    "tasted",
    "sweet",
    "bitter",
    "salt",
    "sour",
    # Body/texture
    "skin",
    "bone",
    "blood",
    "breath",
    "tongue",
    "hand",
    "hands",
    "eyes",
    "eye",
    "teeth",
}

_COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
    "brown",
    "gold",
    "silver",
}

_CONTRASTIVE_RE = re.compile(rf"\b({'|'.join(map(re.escape, _CONTRASTIVE))})\b", flags=re.I)
_CAUSAL_RE = re.compile(rf"\b({'|'.join(map(re.escape, _CAUSAL))})\b", flags=re.I)
_CONDITIONAL_RE = re.compile(rf"\b({'|'.join(map(re.escape, _CONDITIONAL))})\b", flags=re.I)
_TEMPORAL_RE = re.compile(
    rf"\b({'|'.join(map(re.escape, _TEMPORAL))})\b|\b(1[6-9]\d{{2}}|20\d{{2}})\b",
    flags=re.I,
)
_EVIDENTIAL_RE = re.compile(rf"\b({'|'.join(map(re.escape, _EVIDENTIAL))})\b", flags=re.I)
_HEDGE_RE = re.compile(rf"\b({'|'.join(map(re.escape, _HEDGES))})\b", flags=re.I)
_PASSIVE_RE = re.compile(r"\b(am|is|are|was|were|be|been|being)\b\s+\b\w+(?:ed|en)\b", flags=re.I)


def _proper_noun_count(seg: str) -> int:
    # Proper noun proxy: capitalized words not in sentence-initial position, plus acronyms.
    proper = 0
    first_word = True
    for m in re.finditer(r"\b[A-Za-z][A-Za-z']+\b", seg):
        w = m.group(0)
        low = w.lower()
        if first_word:
            first_word = False
            continue
        if low in {"i"}:
            continue
        if w.isupper() and len(w) >= 2:
            proper += 1
            continue
        if w[0].isupper() and w[1:].islower() and len(w) >= 3:
            proper += 1
            continue
    return int(proper)


def marker_sentence_features(
    text: str,
    *,
    sent_spans: List[Tuple[int, int]],
    max_sentences: int = 64,
    max_sentence_chars: int = 280,
) -> List[Dict[str, object]]:
    """Return lightweight per-sentence marker tags for interpretability.

    The scorer model outputs document-level scores. For sentence-level breakdowns we expose
    deterministic, high-precision markers (turns, evidence, hedges, passive voice, etc.).
    """
    t = text or ""
    out: List[Dict[str, object]] = []
    for idx, (s0, s1) in enumerate(sent_spans):
        if len(out) >= int(max_sentences):
            break
        if not (0 <= int(s0) <= int(s1) <= len(t)):
            continue
        seg = t[int(s0) : int(s1)]
        seg = seg.strip()
        if not seg:
            continue
        if max_sentence_chars and len(seg) > int(max_sentence_chars):
            seg = seg[: int(max_sentence_chars)].rstrip() + "…"

        markers = {
            "contrastive": bool(_CONTRASTIVE_RE.search(seg)),
            "causal": bool(_CAUSAL_RE.search(seg)),
            "conditional": bool(_CONDITIONAL_RE.search(seg)),
            "temporal": bool(_TEMPORAL_RE.search(seg)),
            "evidential": bool(_EVIDENTIAL_RE.search(seg)),
            "hedge": bool(_HEDGE_RE.search(seg)),
            "quote": any(q in seg for q in ("\"", "“", "”", "‘", "’")),
            "passive": bool(_PASSIVE_RE.search(seg)),
        }
        out.append(
            {
                "index": int(idx),
                "start_char": int(s0),
                "end_char": int(s1),
                "text": seg,
                "numbers": int(len(_NUM_RE.findall(seg))),
                "proper_nouns": int(_proper_noun_count(seg)),
                "markers": markers,
            }
        )
    return out


def _cv(counts: List[int]) -> Optional[float]:
    if len(counts) < 2:
        return None
    arr = np.array([float(x) for x in counts], dtype=np.float32)
    mu = float(np.mean(arr))
    if not math.isfinite(mu) or abs(mu) <= 1e-12:
        return None
    sd = float(np.std(arr))
    return float(sd / (abs(mu) + 1e-12))


def marker_metrics(
    text: str,
    *,
    sent_spans: List[Tuple[int, int]],
) -> Dict[str, Optional[float]]:
    """Compute deterministic linguistic marker metrics on raw text.

    These are lightweight, high-precision signals intended to:
    - enrich the rubric beyond token/logit quirks
    - provide learnable auxiliary targets for later multi-task heads
    """
    t = text or ""
    words = [m.group(0).lower() for m in _WORD_RE.finditer(t)]
    n_words = int(len(words))
    n_sents = int(len(sent_spans))

    def per_1000(x: int) -> Optional[float]:
        if n_words <= 0:
            return None
        return float(x) / float(n_words) * 1000.0

    contrastive_sents = 0
    causal_sents = 0
    conditional_sents = 0
    temporal_sents = 0
    evidential_sents = 0
    hedge_sents = 0
    quote_sents = 0
    passive_sents = 0

    numbers_per_sent: List[int] = []
    proper_per_sent: List[int] = []

    total_numbers = 0
    total_proper = 0

    for s0, s1 in sent_spans:
        if not (0 <= int(s0) <= int(s1) <= len(t)):
            continue
        seg = t[int(s0) : int(s1)]
        if not seg.strip():
            continue

        if _CONTRASTIVE_RE.search(seg):
            contrastive_sents += 1
        if _CAUSAL_RE.search(seg):
            causal_sents += 1
        if _CONDITIONAL_RE.search(seg):
            conditional_sents += 1
        if _TEMPORAL_RE.search(seg):
            temporal_sents += 1
        if _EVIDENTIAL_RE.search(seg):
            evidential_sents += 1
        if _HEDGE_RE.search(seg):
            hedge_sents += 1
        if any(q in seg for q in ("\"", "“", "”", "‘", "’")):
            quote_sents += 1
        if _PASSIVE_RE.search(seg):
            passive_sents += 1

        nums = len(_NUM_RE.findall(seg))
        total_numbers += int(nums)
        numbers_per_sent.append(int(nums))

        proper = _proper_noun_count(seg)
        total_proper += int(proper)
        proper_per_sent.append(int(proper))

    denom_s = float(max(1, n_sents))

    url_count = len(_URL_RE.findall(t))
    bracket_cite_count = len(_BRACKET_CITE_RE.findall(t))

    sensory = sum(1 for w in words if w in _SENSORY)
    colors = sum(1 for w in words if w in _COLORS)

    commas = t.count(",")
    semicolons = t.count(";")
    dashes = t.count("—") + t.count("–") + t.count("--")

    return {
        # Rhetorical/structure markers (sentence-level)
        "marker_contrastive_sentence_fraction": float(contrastive_sents) / denom_s if n_sents else None,
        "marker_causal_sentence_fraction": float(causal_sents) / denom_s if n_sents else None,
        "marker_conditional_sentence_fraction": float(conditional_sents) / denom_s if n_sents else None,
        "marker_temporal_sentence_fraction": float(temporal_sents) / denom_s if n_sents else None,
        "marker_evidential_sentence_fraction": float(evidential_sents) / denom_s if n_sents else None,
        "marker_hedge_sentence_fraction": float(hedge_sents) / denom_s if n_sents else None,
        # Quotation / voice
        "marker_quote_sentence_fraction": float(quote_sents) / denom_s if n_sents else None,
        "marker_passive_sentence_fraction": float(passive_sents) / denom_s if n_sents else None,
        # Evidence + specificity (counts + burstiness)
        "marker_numbers_per_1000_words": per_1000(int(total_numbers)),
        "marker_numbers_per_sentence_mean": float(np.mean(np.array(numbers_per_sent, dtype=np.float32)))
        if numbers_per_sent
        else None,
        "marker_numbers_per_sentence_cv": _cv(numbers_per_sent),
        "marker_proper_nouns_per_1000_words": per_1000(int(total_proper)),
        "marker_proper_nouns_per_sentence_mean": float(np.mean(np.array(proper_per_sent, dtype=np.float32)))
        if proper_per_sent
        else None,
        "marker_proper_nouns_per_sentence_cv": _cv(proper_per_sent),
        "marker_url_per_1000_words": per_1000(int(url_count)),
        "marker_bracket_cite_per_1000_words": per_1000(int(bracket_cite_count)),
        # Concreteness / imagery proxies
        "marker_sensory_word_fraction": (float(sensory) / float(n_words)) if n_words else None,
        "marker_color_word_fraction": (float(colors) / float(n_words)) if n_words else None,
        # Syntactic / rhythm proxies
        "marker_commas_per_sentence_mean": float(commas) / denom_s if n_sents else None,
        "marker_semicolons_per_1000_words": per_1000(int(semicolons)),
        "marker_dashes_per_1000_words": per_1000(int(dashes)),
    }
