from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.text_normalize import normalize_for_studio


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
    head_labels: List[str] = []
    try:
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict):
            head_labels = [str(id2label.get(i) or f"head_{i}") for i in range(n_heads)]
    except Exception:
        head_labels = []
    if not head_labels:
        head_labels = [f"head_{i}" for i in range(n_heads)]

    mean_probs = probs.mean(axis=0).astype(float).tolist() if probs.size else [0.0] * n_heads
    mean_probs = [0.0 if not math.isfinite(p) else max(0.0, min(1.0, float(p))) for p in mean_probs]
    head_probs_by_label = {lab: float(p) for lab, p in zip(head_labels, mean_probs)}

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
            window_probs = [float(x) for x in probs[:, 0].astype(float).tolist()]
        else:
            weights = primary_from_heads.get("weights") or {}
            if isinstance(weights, dict):
                total_w = float(sum(float(w) for w in weights.values() if isinstance(w, (int, float)) and float(w) > 0))
                if total_w > 0:
                    acc = np.zeros((probs.shape[0],), dtype=np.float64)
                    for lab, w in weights.items():
                        if lab not in head_probs_by_label:
                            continue
                        try:
                            idx = head_labels.index(str(lab))
                        except ValueError:
                            continue
                        wf = float(w)
                        if not math.isfinite(wf) or wf <= 0:
                            continue
                        acc += wf * probs[:, idx].astype(np.float64)
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
