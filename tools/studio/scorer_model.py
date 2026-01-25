from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.text_normalize import normalize_for_studio
from tools.studio.score import rubric_category_weights


_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}


def _best_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def _cache_key(model_path_or_id: str) -> str:
    s = str(model_path_or_id or "").strip()
    if not s:
        raise ValueError("model_path_or_id is empty")
    p = Path(s)
    if p.exists():
        try:
            return str(p.resolve())
        except Exception:
            return str(p)
    return s


def _needs_trust_remote_code(model_path_or_id: str) -> bool:
    s = str(model_path_or_id or "").strip()
    if not s:
        return False
    p = Path(s)
    if p.exists():
        cfg = p / "config.json"
        if cfg.exists():
            try:
                import json

                obj = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and "auto_map" in obj:
                    return True
            except Exception:
                return False
        return False
    return "qwen" in s.lower()


def _ensure_pad_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        tokenizer.pad_token = eos
        return
    try:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    except Exception:
        return


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"[^.!?\n]+[.!?]?\s*", text):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            spans.append((s, e))
    return spans


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    parts = re.split(r"(\n\s*\n+)", text)
    cursor = 0
    idx = 0
    while idx < len(parts):
        seg = parts[idx]
        s = cursor
        e = cursor + len(seg)
        if seg.strip():
            spans.append((s, e))
        cursor = e
        if idx + 1 < len(parts):
            sep = parts[idx + 1]
            cursor += len(sep)
            idx += 2
        else:
            break
    if not spans:
        spans = _sentence_spans(text)
    return spans


def _head_labels_from_model(model, n_heads: int) -> List[str]:
    try:
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict):
            return [str(id2label.get(i) or f"head_{i}") for i in range(n_heads)]
    except Exception:
        return []
    return [f"head_{i}" for i in range(n_heads)]


def _derive_rubric_overall(
    per_window_probs_by_label: Dict[str, np.ndarray],
    head_probs_by_label: Dict[str, float],
) -> None:
    try:
        cat_w = rubric_category_weights()
        pairs: List[Tuple[str, float]] = []
        for cat, w in cat_w.items():
            lab = f"rubric_{cat}"
            if lab in per_window_probs_by_label:
                wf = float(w)
                if math.isfinite(wf) and wf > 0:
                    pairs.append((lab, wf))
        if not pairs:
            for lab in per_window_probs_by_label:
                if lab.startswith("rubric_") and lab not in ("rubric_overall", "rubric_overall_from_categories"):
                    pairs.append((lab, 1.0))
        if pairs:
            total_w = float(sum(w for _, w in pairs))
            if total_w > 0:
                acc = np.zeros((int(next(iter(per_window_probs_by_label.values())).shape[0]),), dtype=np.float64)
                for lab, w in pairs:
                    acc += float(w) * per_window_probs_by_label[lab].astype(np.float64)
                derived = acc / float(total_w)
                derived = np.clip(derived, 0.0, 1.0)
                derived_key = "rubric_overall" if "rubric_overall" not in head_probs_by_label else "rubric_overall_from_categories"
                per_window_probs_by_label[str(derived_key)] = derived.astype(np.float64)
                head_probs_by_label[str(derived_key)] = float(np.mean(derived).item()) if derived.size else 0.0
    except Exception:
        return


def _primary_from_heads(model, head_probs_by_label: Dict[str, float], head_labels: Sequence[str]) -> Tuple[float, Dict[str, Any]]:
    primary_prob = float(head_probs_by_label.get(head_labels[0], 0.0)) if head_labels else 0.0
    primary_from_heads: Dict[str, Any] = {}
    if len(head_labels) > 1:
        try:
            cfg = getattr(model.config, "horace_primary", None)
            if isinstance(cfg, dict) and cfg.get("kind") == "weighted_sum":
                weights = cfg.get("heads")
                if isinstance(weights, dict):
                    total_w = 0.0
                    acc = 0.0
                    used: Dict[str, float] = {}
                    for lab, w in weights.items():
                        if lab not in head_probs_by_label:
                            continue
                        wf = float(w)
                        if not math.isfinite(wf) or wf <= 0:
                            continue
                        used[str(lab)] = float(wf)
                        total_w += wf
                        acc += wf * float(head_probs_by_label[str(lab)])
                    if total_w > 0:
                        primary_prob = float(acc / total_w)
                        primary_from_heads = {"kind": "weighted_sum", "weights": used}
        except Exception:
            primary_from_heads = {}
    primary_prob = 0.0 if not math.isfinite(primary_prob) else max(0.0, min(1.0, float(primary_prob)))
    return primary_prob, primary_from_heads


def _effective_max_length(tok, max_length: int) -> int:
    try:
        special = int(tok.num_special_tokens_to_add(pair=False))
    except Exception:
        special = 0
    return max(1, int(max_length) - int(special))


def _token_span_for_char_span(offsets: Sequence[Tuple[int, int]], s0: int, s1: int) -> Optional[Tuple[int, int]]:
    i0 = None
    i1 = None
    for i, (a, b) in enumerate(offsets):
        if b <= s0:
            continue
        if a >= s1:
            break
        if i0 is None:
            i0 = i
        i1 = i
    if i0 is None or i1 is None:
        return None
    return int(i0), int(i1)


def _span_excerpt(text: str, s0: int, s1: int, *, max_chars: int = 320) -> Tuple[str, bool]:
    seg = text[int(s0) : int(s1)].strip()
    if max_chars and len(seg) > int(max_chars):
        return seg[: int(max_chars)].rstrip() + "…", True
    return seg, False


def _span_items_from_char_spans(
    text: str,
    *,
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int]],
    kind: str,
    max_items: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, (s0, s1) in enumerate(spans):
        if max_items and len(items) >= int(max_items):
            break
        span = _token_span_for_char_span(offsets, int(s0), int(s1))
        if span is None:
            continue
        i0, i1 = span
        excerpt, truncated = _span_excerpt(text, int(s0), int(s1), max_chars=int(max_chars))
        items.append(
            {
                "index": int(idx),
                "kind": str(kind),
                "char_start": int(s0),
                "char_end": int(s1),
                "span_token_start": int(i0),
                "span_token_end": int(i1),
                "text": excerpt,
                "text_truncated": bool(truncated),
            }
        )
    return items


def _span_items_from_token_spans(
    text: str,
    *,
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int]],
    kind: str,
    max_items: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    text_len = len(text)
    for idx, (i0, i1) in enumerate(spans):
        if max_items and len(items) >= int(max_items):
            break
        if i0 < 0 or i1 < i0:
            continue
        if i0 >= len(offsets):
            continue
        i1 = min(int(i1), len(offsets) - 1)
        cs = int(offsets[i0][0]) if i0 < len(offsets) else 0
        ce = int(offsets[i1][1]) if i1 < len(offsets) else text_len
        excerpt, truncated = _span_excerpt(text, cs, ce, max_chars=int(max_chars))
        items.append(
            {
                "index": int(idx),
                "kind": str(kind),
                "char_start": int(cs),
                "char_end": int(ce),
                "span_token_start": int(i0),
                "span_token_end": int(i1),
                "text": excerpt,
                "text_truncated": bool(truncated),
            }
        )
    return items


def _score_span_items(
    *,
    model,
    tok,
    dev: str,
    text: str,
    input_ids: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    span_items: Sequence[Dict[str, Any]],
    effective_max: int,
    window_tokens: int,
    batch_size: int,
    center_on_span: bool,
    head_labels: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    if not span_items:
        return [], head_labels or [], {}

    total_tokens = int(len(input_ids))
    win_tokens = max(8, min(int(effective_max), int(window_tokens)))
    win_tokens = min(win_tokens, total_tokens) if total_tokens else win_tokens

    encodings: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    for item in span_items:
        i0 = int(item.get("span_token_start") or 0)
        i1 = int(item.get("span_token_end") or 0)
        span_len = int(i1 - i0 + 1)
        if span_len <= 0:
            continue
        win_len = min(int(effective_max), max(span_len, win_tokens))
        if center_on_span:
            win_start = int(max(0, min(i0 - (win_len - span_len) // 2, total_tokens - win_len)))
        else:
            win_start = int(max(0, min(i0, total_tokens - win_len)))
        win_end = int(win_start + win_len)

        window_ids = list(input_ids[win_start:win_end])
        prepared = tok.prepare_for_model(window_ids, truncation=False, padding=False)
        encodings.append(prepared)

        window_char_start = int(offsets[win_start][0]) if win_start < len(offsets) else 0
        window_char_end = int(offsets[win_end - 1][1]) if win_end - 1 < len(offsets) else len(text)

        meta = dict(item)
        meta.update(
            {
                "window_token_start": int(win_start),
                "window_token_end": int(win_end),
                "window_char_start": int(window_char_start),
                "window_char_end": int(window_char_end),
                "segment_truncated": bool(span_len > int(effective_max)),
            }
        )
        if meta.get("kind") == "sentence":
            meta["sentence_truncated"] = bool(meta["segment_truncated"])
        if meta.get("kind") == "paragraph":
            meta["paragraph_truncated"] = bool(meta["segment_truncated"])
        metas.append(meta)

    if not encodings:
        return [], head_labels or [], {}

    results: List[Dict[str, Any]] = []
    labels_out: List[str] = head_labels or []
    primary_from_heads: Dict[str, Any] = {}

    for start in range(0, len(encodings), int(max(1, batch_size))):
        batch_enc = encodings[start : start + int(max(1, batch_size))]
        batch_meta = metas[start : start + int(max(1, batch_size))]

        padded = tok.pad(batch_enc, padding=True, return_tensors="pt")
        padded = {k: v.to(dev) for k, v in padded.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = model(**padded)
            logits = out.logits
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)

        n_heads = int(probs.shape[1]) if probs.size else 1
        if not labels_out:
            labels_out = _head_labels_from_model(model, n_heads)

        for row_idx, row in enumerate(probs.tolist()):
            head_probs_by_label = {lab: float(p) for lab, p in zip(labels_out, row)}
            per_window_probs_by_label = {
                lab: np.asarray([float(p)], dtype=np.float64) for lab, p in zip(labels_out, row)
            }
            _derive_rubric_overall(per_window_probs_by_label, head_probs_by_label)
            primary_prob, primary_from_heads = _primary_from_heads(model, head_probs_by_label, labels_out)

            meta = dict(batch_meta[row_idx])
            meta.update(
                {
                    "score_0_100": float(100.0 * primary_prob),
                    "prob_0_1": float(primary_prob),
                    "head_probs_by_label": dict(head_probs_by_label),
                }
            )
            results.append(meta)

    return results, labels_out, dict(primary_from_heads)


def load_scorer(model_path_or_id: str, *, device: Optional[str] = None) -> Tuple[Any, Any, str]:
    """Load a trained scorer model (HF directory or model id) with a small in-process cache."""
    key = _cache_key(model_path_or_id)
    with _CACHE_LOCK:
        hit = _MODEL_CACHE.get(key)
        if hit is not None:
            tok, model, dev = hit
            # If caller asked for a different device, reload.
            if device is None or str(device) == str(dev):
                return tok, model, dev
    dev = _best_device(device)
    trc = _needs_trust_remote_code(key)
    tok_kwargs = {"trust_remote_code": trc}
    try:
        tok = AutoTokenizer.from_pretrained(key, fix_mistral_regex=True, **tok_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(key, **tok_kwargs)
    _ensure_pad_token(tok)
    model = AutoModelForSequenceClassification.from_pretrained(key, trust_remote_code=trc)
    if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.to(dev)
    model.eval()
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = (tok, model, dev)
    return tok, model, dev


@dataclass(frozen=True)
class ScorerResult:
    score_0_100: float
    prob_0_1: float
    model_path_or_id: str
    device: str
    max_length: int
    doc_type: str
    normalized: bool
    n_windows: int = 1
    window_stride: int = 0
    window_probs: Tuple[float, ...] = ()
    windows_capped: bool = False
    head_labels: Tuple[str, ...] = ()
    head_probs_0_1: Tuple[float, ...] = ()
    head_probs_by_label: Dict[str, float] = field(default_factory=dict)
    primary_from_heads: Dict[str, Any] = field(default_factory=dict)


def score_with_scorer(
    text: str,
    *,
    model_path_or_id: str,
    doc_type: str = "prose",
    normalize_text: bool = True,
    max_length: int = 384,
    stride: Optional[int] = None,
    max_windows: int = 24,
    device: Optional[str] = None,
) -> ScorerResult:
    tok, model, dev = load_scorer(model_path_or_id, device=device)
    dt = str(doc_type)
    t, _ = normalize_for_studio(text, doc_type=dt, enabled=bool(normalize_text))

    # Windowed scoring: for long inputs, score multiple overlapping windows and average.
    # This avoids "first 384 tokens only" truncation for typical paste-in use cases.
    stride_len = int(stride) if stride is not None else max(1, int(max_length) // 2)
    enc = None
    try:
        enc = tok(
            t,
            truncation=True,
            max_length=int(max_length),
            stride=int(stride_len),
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        enc = tok(t, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")

    # Keep only model input tensors (BatchEncoding can include overflow bookkeeping fields).
    allowed_keys = {"input_ids", "attention_mask", "token_type_ids"}
    tensors: Dict[str, torch.Tensor] = {k: v for k, v in dict(enc).items() if k in allowed_keys and isinstance(v, torch.Tensor)}
    n_windows = int(tensors.get("input_ids", torch.empty(0)).shape[0]) if "input_ids" in tensors else 0
    windows_capped = False
    if n_windows > int(max_windows) > 0:
        windows_capped = True
        idx = torch.linspace(0, n_windows - 1, steps=int(max_windows)).round().to(dtype=torch.long)
        idx = torch.unique(idx, sorted=True)
        tensors = {k: v.index_select(0, idx) for k, v in tensors.items()}
        n_windows = int(tensors.get("input_ids", torch.empty(0)).shape[0]) if "input_ids" in tensors else 0

    tensors = {k: v.to(dev) for k, v in tensors.items()}
    with torch.no_grad():
        out = model(**tensors)
        logits = out.logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    n_heads = int(probs.shape[1]) if probs.size else 1
    head_labels: List[str] = _head_labels_from_model(model, n_heads)

    mean_probs = probs.mean(axis=0).astype(float).tolist() if probs.size else [0.0] * n_heads
    mean_probs = [0.0 if not math.isfinite(p) else max(0.0, min(1.0, float(p))) for p in mean_probs]

    per_window_probs_by_label: Dict[str, np.ndarray] = {}
    if probs.size:
        for i, lab in enumerate(head_labels):
            per_window_probs_by_label[str(lab)] = probs[:, i].astype(np.float64)

    head_probs_by_label = {lab: float(p) for lab, p in zip(head_labels, mean_probs)}

    # Derived: rubric_overall from rubric_* category heads.
    # This keeps inference stable even if the model only predicts rubric categories.
    _derive_rubric_overall(per_window_probs_by_label, head_probs_by_label)

    primary_from_heads: Dict[str, Any] = {}
    primary_prob = float(mean_probs[0]) if mean_probs else 0.0

    if n_heads > 1:
        try:
            cfg = getattr(model.config, "horace_primary", None)
            if isinstance(cfg, dict) and cfg.get("kind") == "weighted_sum":
                weights = cfg.get("heads")
                if isinstance(weights, dict):
                    total_w = 0.0
                    acc = 0.0
                    used: Dict[str, float] = {}
                    for lab, w in weights.items():
                        if lab not in head_probs_by_label:
                            continue
                        wf = float(w)
                        if not math.isfinite(wf) or wf <= 0:
                            continue
                        used[str(lab)] = float(wf)
                        total_w += wf
                        acc += wf * float(head_probs_by_label[str(lab)])
                    if total_w > 0:
                        primary_prob = float(acc / total_w)
                        primary_from_heads = {"kind": "weighted_sum", "weights": used}
        except Exception:
            primary_from_heads = {}

    primary_prob = 0.0 if not math.isfinite(primary_prob) else max(0.0, min(1.0, float(primary_prob)))

    # Per-window primary probs (for debugging/plots)
    window_probs: List[float] = []
    if probs.size:
        if n_heads <= 1 or not primary_from_heads:
            key = head_labels[0] if head_labels else "head_0"
            wv = per_window_probs_by_label.get(str(key))
            if wv is None:
                wv = probs[:, 0].astype(np.float64)
            window_probs = [float(x) for x in wv.astype(float).tolist()]
        else:
            weights = primary_from_heads.get("weights") or {}
            if isinstance(weights, dict):
                total_w = float(sum(float(w) for w in weights.values() if isinstance(w, (int, float)) and float(w) > 0))
                if total_w > 0:
                    acc = np.zeros((probs.shape[0],), dtype=np.float64)
                    for lab, w in weights.items():
                        wf = float(w)
                        if not math.isfinite(wf) or wf <= 0:
                            continue
                        wv = per_window_probs_by_label.get(str(lab))
                        if wv is None:
                            continue
                        acc += wf * wv.astype(np.float64)
                    window_probs = (acc / float(total_w)).astype(float).tolist()
                else:
                    window_probs = [float(x) for x in probs[:, 0].astype(float).tolist()]
            else:
                window_probs = [float(x) for x in probs[:, 0].astype(float).tolist()]

    window_probs = [0.0 if not math.isfinite(p) else max(0.0, min(1.0, float(p))) for p in window_probs]
    return ScorerResult(
        score_0_100=float(100.0 * primary_prob),
        prob_0_1=float(primary_prob),
        model_path_or_id=str(model_path_or_id),
        device=str(dev),
        max_length=int(max_length),
        doc_type=dt,
        normalized=bool(normalize_text),
        n_windows=int(len(window_probs)),
        window_stride=int(stride_len),
        window_probs=tuple(window_probs[:64]),
        windows_capped=bool(windows_capped),
        head_labels=tuple(head_labels),
        head_probs_0_1=tuple(float(x) for x in mean_probs),
        head_probs_by_label=head_probs_by_label,
        primary_from_heads=primary_from_heads,
    )


def score_sentence_windows(
    text: str,
    *,
    model_path_or_id: str,
    doc_type: str = "prose",
    normalize_text: bool = True,
    max_length: int = 384,
    window_tokens: Optional[int] = None,
    max_sentences: int = 96,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """So what: provide sentence-level diagnostic scores by scoring per-sentence windows."""
    out = score_window_levels(
        text,
        model_path_or_id=model_path_or_id,
        doc_type=doc_type,
        normalize_text=normalize_text,
        max_length=max_length,
        window_tokens=window_tokens,
        max_sentences=max_sentences,
        batch_size=batch_size,
        include_paragraphs=False,
        include_pages=False,
        device=device,
    )
    return {
        "model_path_or_id": out.get("model_path_or_id"),
        "doc_type": out.get("doc_type"),
        "normalized": out.get("normalized"),
        "max_length": out.get("max_length"),
        "n_sentences": int(len(out.get("sentences") or [])),
        "head_labels": list(out.get("head_labels") or []),
        "primary_from_heads": dict(out.get("primary_from_heads") or {}),
        "text_normalization": out.get("text_normalization"),
        "sentences": list(out.get("sentences") or []),
    }


def score_window_levels(
    text: str,
    *,
    model_path_or_id: str,
    doc_type: str = "prose",
    normalize_text: bool = True,
    max_length: int = 384,
    window_tokens: Optional[int] = None,
    page_window_tokens: Optional[int] = None,
    page_stride_tokens: Optional[int] = None,
    max_sentences: int = 96,
    max_paragraphs: int = 32,
    max_pages: int = 16,
    batch_size: int = 8,
    include_sentences: bool = True,
    include_paragraphs: bool = True,
    include_pages: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Score sentence/paragraph/page windows for diagnostic breakdowns."""
    tok, model, dev = load_scorer(model_path_or_id, device=device)
    dt = str(doc_type)
    t, norm_meta = normalize_for_studio(text, doc_type=dt, enabled=bool(normalize_text))

    if not t.strip():
        return {
            "model_path_or_id": str(model_path_or_id),
            "doc_type": dt,
            "normalized": bool(normalize_text),
            "max_length": int(max_length),
            "head_labels": [],
            "primary_from_heads": {},
            "text_normalization": norm_meta,
            "sentences": [],
            "paragraphs": [],
            "pages": [],
        }

    enc = tok(
        t,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(enc.get("input_ids") or [])
    offsets = enc.get("offset_mapping") or [(0, 0)] * len(input_ids)
    total_tokens = int(len(input_ids))
    if total_tokens <= 0:
        return {
            "model_path_or_id": str(model_path_or_id),
            "doc_type": dt,
            "normalized": bool(normalize_text),
            "max_length": int(max_length),
            "head_labels": [],
            "primary_from_heads": {},
            "text_normalization": norm_meta,
            "sentences": [],
            "paragraphs": [],
            "pages": [],
        }

    effective_max = _effective_max_length(tok, int(max_length))
    seg_window_tokens = int(window_tokens) if window_tokens is not None else int(max_length)
    seg_window_tokens = max(8, min(int(effective_max), int(seg_window_tokens)))

    page_window = int(page_window_tokens) if page_window_tokens is not None else int(max_length)
    page_window = max(8, min(int(effective_max), int(page_window)))
    page_stride = int(page_stride_tokens) if page_stride_tokens is not None else int(page_window)
    page_stride = max(1, int(page_stride))

    head_labels: List[str] = []
    primary_from_heads: Dict[str, Any] = {}

    sentences: List[Dict[str, Any]] = []
    if include_sentences:
        sent_spans = _sentence_spans(t)
        span_items = _span_items_from_char_spans(
            t,
            offsets=offsets,
            spans=sent_spans,
            kind="sentence",
            max_items=int(max_sentences),
            max_chars=280,
        )
        sentences, head_labels, primary_from_heads = _score_span_items(
            model=model,
            tok=tok,
            dev=dev,
            text=t,
            input_ids=input_ids,
            offsets=offsets,
            span_items=span_items,
            effective_max=int(effective_max),
            window_tokens=int(seg_window_tokens),
            batch_size=int(batch_size),
            center_on_span=True,
            head_labels=head_labels,
        )

    paragraphs: List[Dict[str, Any]] = []
    if include_paragraphs:
        para_spans = _paragraph_spans(t)
        span_items = _span_items_from_char_spans(
            t,
            offsets=offsets,
            spans=para_spans,
            kind="paragraph",
            max_items=int(max_paragraphs),
            max_chars=320,
        )
        paragraphs, head_labels, primary_from_heads = _score_span_items(
            model=model,
            tok=tok,
            dev=dev,
            text=t,
            input_ids=input_ids,
            offsets=offsets,
            span_items=span_items,
            effective_max=int(effective_max),
            window_tokens=int(seg_window_tokens),
            batch_size=int(batch_size),
            center_on_span=True,
            head_labels=head_labels,
        )

    pages: List[Dict[str, Any]] = []
    if include_pages:
        page_spans: List[Tuple[int, int]] = []
        start = 0
        while start < total_tokens:
            end = min(total_tokens, start + int(page_window))
            page_spans.append((int(start), int(end - 1)))
            start += int(page_stride)
            if max_pages and len(page_spans) >= int(max_pages):
                break
        span_items = _span_items_from_token_spans(
            t,
            offsets=offsets,
            spans=page_spans,
            kind="page",
            max_items=int(max_pages),
            max_chars=360,
        )
        pages, head_labels, primary_from_heads = _score_span_items(
            model=model,
            tok=tok,
            dev=dev,
            text=t,
            input_ids=input_ids,
            offsets=offsets,
            span_items=span_items,
            effective_max=int(effective_max),
            window_tokens=int(page_window),
            batch_size=int(batch_size),
            center_on_span=False,
            head_labels=head_labels,
        )

    return {
        "model_path_or_id": str(model_path_or_id),
        "doc_type": dt,
        "normalized": bool(normalize_text),
        "max_length": int(max_length),
        "head_labels": list(head_labels),
        "primary_from_heads": dict(primary_from_heads),
        "text_normalization": norm_meta,
        "sentences": sentences,
        "paragraphs": paragraphs,
        "pages": pages,
        "window_tokens": int(seg_window_tokens),
        "page_window_tokens": int(page_window),
        "page_stride_tokens": int(page_stride),
    }
