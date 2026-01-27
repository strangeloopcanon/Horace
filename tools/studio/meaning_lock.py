from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|[A-Za-z]+)?(?![A-Za-z0-9_])")
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']+\b")
_NEG_RE = re.compile(
    r"\b(?:no|not|never|none|nothing|nowhere|neither|nor|without|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't)\b",
    flags=re.I,
)


def _pick_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_EMBED_CACHE: Dict[str, Tuple[Any, Any, str]] = {}


def _get_embedder(model_id: str, *, device: Optional[str] = None):
    ident = (model_id or "").strip() or "distilbert-base-uncased"
    cached = _EMBED_CACHE.get(ident)
    if cached is not None:
        return cached

    from transformers import AutoModel, AutoTokenizer

    dev = device or _pick_device()
    tok = AutoTokenizer.from_pretrained(ident)
    model = AutoModel.from_pretrained(ident)
    model.to(dev)
    model.eval()
    _EMBED_CACHE[ident] = (tok, model, dev)
    return tok, model, dev


def _mean_pool(last_hidden_state, attention_mask) -> np.ndarray:
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-6)
    vec = (summed / denom).squeeze(0)
    arr = vec.detach().float().cpu().numpy()
    return arr.astype(np.float32)


def embed_text(
    text: str,
    *,
    model_id: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    tok, model, dev = _get_embedder(model_id, device=device)
    import torch

    t = (text or "").strip()
    if not t:
        return np.zeros((model.config.hidden_size,), dtype=np.float32)

    inputs = tok(
        t,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        padding=False,
    ).to(dev)
    with torch.no_grad():
        out = model(**inputs)
    return _mean_pool(out.last_hidden_state, inputs["attention_mask"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.array(a, dtype=np.float32).reshape(-1)
    bb = np.array(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def _sentence_start_word_spans(text: str) -> List[Tuple[int, int]]:
    """Return spans for the first word of each (very loosely) detected sentence.

    Used to avoid counting sentence-initial capitalization as a "proper noun".
    """
    t = text or ""
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.!?\n]+", t):
        end = m.end()
        seg = t[start:end]
        wm = _WORD_RE.search(seg)
        if wm:
            spans.append((start + wm.start(), start + wm.end()))
        start = end
    seg = t[start:]
    wm = _WORD_RE.search(seg)
    if wm:
        spans.append((start + wm.start(), start + wm.end()))
    return spans


def extract_numbers(text: str) -> List[str]:
    t = text or ""
    nums = {m.group(0) for m in _NUM_RE.finditer(t)}
    return sorted(nums)


def extract_negations(text: str) -> List[str]:
    t = text or ""
    neg = {m.group(0).lower() for m in _NEG_RE.finditer(t)}
    return sorted(neg)


def extract_proper_nouns(text: str) -> List[str]:
    t = text or ""
    first_word_spans = _sentence_start_word_spans(t)
    first_word_mask = [False] * len(t)
    for s, e in first_word_spans:
        for i in range(max(0, s), min(len(t), e)):
            first_word_mask[i] = True

    proper: set[str] = set()
    for m in _WORD_RE.finditer(t):
        s, e = m.start(), m.end()
        w = m.group(0)
        if w == "I":
            continue
        if any(first_word_mask[i] for i in range(s, min(e, len(t)))):
            continue
        if w.isupper() and len(w) >= 2:
            proper.add(w)
            continue
        if w[0].isupper() and w[1:].islower() and len(w) >= 3:
            proper.add(w)
            continue
    return sorted(proper)


@dataclass(frozen=True)
class MeaningLockConfig:
    # Set to empty string to disable embedding-based semantic check (useful for offline tests).
    embedder_model_id: str = "distilbert-base-uncased"
    embedder_max_length: int = 256
    min_cosine_sim: float = 0.86
    max_length_ratio: float = 1.45
    max_edit_ratio: float = 0.55  # approx char-level change ratio
    allow_new_numbers: bool = False
    allow_new_proper_nouns: bool = False
    allow_negation_change: bool = False


@dataclass(frozen=True)
class MeaningLockReport:
    ok: bool
    cosine_sim: Optional[float]
    length_ratio: float
    edit_ratio: float
    numbers_added: List[str]
    numbers_removed: List[str]
    negations_added: List[str]
    negations_removed: List[str]
    proper_nouns_added: List[str]
    proper_nouns_removed: List[str]
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _edit_ratio(a: str, b: str) -> float:
    # Approximate "how much changed" using a stable character similarity.
    # 0 -> identical, 1 -> completely different.
    import difflib

    aa = (a or "").strip()
    bb = (b or "").strip()
    if not aa and not bb:
        return 0.0
    if not aa or not bb:
        return 1.0
    r = difflib.SequenceMatcher(a=aa, b=bb).ratio()
    return float(1.0 - r)


def check_meaning_lock(
    original: str,
    candidate: str,
    *,
    cfg: Optional[MeaningLockConfig] = None,
    device: Optional[str] = None,
) -> MeaningLockReport:
    c = cfg or MeaningLockConfig()
    a = (original or "").strip()
    b = (candidate or "").strip()
    reasons: List[str] = []

    if not a or not b:
        return MeaningLockReport(
            ok=False,
            cosine_sim=None,
            length_ratio=0.0,
            edit_ratio=1.0,
            numbers_added=[],
            numbers_removed=[],
            negations_added=[],
            negations_removed=[],
            proper_nouns_added=[],
            proper_nouns_removed=[],
            reasons=["empty_text"],
        )

    length_ratio = float(len(b) / max(1, len(a)))
    if length_ratio > float(c.max_length_ratio):
        reasons.append("too_long")

    edit_ratio = _edit_ratio(a, b)
    if edit_ratio > float(c.max_edit_ratio):
        reasons.append("too_much_changed")

    nums_a = set(extract_numbers(a))
    nums_b = set(extract_numbers(b))
    nums_added = sorted(nums_b - nums_a)
    nums_removed = sorted(nums_a - nums_b)
    if (nums_added or nums_removed) and not bool(c.allow_new_numbers):
        reasons.append("numbers_changed")

    neg_a = set(extract_negations(a))
    neg_b = set(extract_negations(b))
    neg_added = sorted(neg_b - neg_a)
    neg_removed = sorted(neg_a - neg_b)
    if (neg_added or neg_removed) and not bool(c.allow_negation_change):
        reasons.append("negation_changed")

    pn_a = set(extract_proper_nouns(a))
    pn_b = set(extract_proper_nouns(b))
    pn_added = sorted(pn_b - pn_a)
    pn_removed = sorted(pn_a - pn_b)
    if (pn_added or pn_removed) and not bool(c.allow_new_proper_nouns):
        reasons.append("proper_nouns_changed")

    sim = None
    embed_id = (c.embedder_model_id or "").strip()
    if embed_id:
        try:
            va = embed_text(a, model_id=embed_id, max_length=c.embedder_max_length, device=device)
            vb = embed_text(b, model_id=embed_id, max_length=c.embedder_max_length, device=device)
            sim = cosine_similarity(va, vb)
            if sim < float(c.min_cosine_sim):
                reasons.append("semantic_drift")
        except Exception:
            reasons.append("embed_failed")

    ok = len(reasons) == 0
    return MeaningLockReport(
        ok=bool(ok),
        cosine_sim=sim,
        length_ratio=length_ratio,
        edit_ratio=edit_ratio,
        numbers_added=nums_added,
        numbers_removed=nums_removed,
        negations_added=neg_added,
        negations_removed=neg_removed,
        proper_nouns_added=pn_added,
        proper_nouns_removed=pn_removed,
        reasons=reasons,
    )
