from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.model_security import resolve_model_source, resolve_trust_remote_code
from tools.studio.text_normalize import normalize_for_studio


_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}


def _best_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
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
    source = resolve_model_source(model_path_or_id, purpose="scorer model")
    key = _cache_key(source.source_id)
    if source.revision is not None:
        key = f"{key}@{source.revision}"
    with _CACHE_LOCK:
        hit = _MODEL_CACHE.get(key)
        if hit is not None:
            tok, model, dev = hit
            # If caller asked for a different device, reload.
            if device is None or str(device) == str(dev):
                return tok, model, dev
    dev = _best_device(device)
    trc_requested = _needs_trust_remote_code(source.source_id)
    trc = resolve_trust_remote_code(source, requested=trc_requested, purpose="scorer model")
    tok_kwargs: Dict[str, Any] = {"trust_remote_code": trc}
    model_kwargs: Dict[str, Any] = {"trust_remote_code": trc}
    if source.revision is not None:
        tok_kwargs["revision"] = source.revision
        model_kwargs["revision"] = source.revision
    try:
        tok = AutoTokenizer.from_pretrained(source.source_id, fix_mistral_regex=True, **tok_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(source.source_id, **tok_kwargs)
    _ensure_pad_token(tok)
    model = AutoModelForSequenceClassification.from_pretrained(source.source_id, **model_kwargs)
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
        if logits.ndim != 2:
            raise ValueError(f"unexpected scorer logits shape: {tuple(logits.shape)}")
        head_dim = int(logits.shape[-1])
        if head_dim == 1:
            probs_t = torch.sigmoid(logits[:, 0])
        elif head_dim == 2:
            # Binary classifier with two logits: use positive class probability.
            probs_t = torch.softmax(logits, dim=-1)[:, 1]
        else:
            raise ValueError(
                "score_with_scorer supports only binary classifier heads (1 or 2 logits); "
                f"got {head_dim} logits"
            )
        probs = probs_t.detach().cpu().numpy()

    ps = [float(x) for x in probs.reshape(-1).tolist()] if probs.size else []
    ps = [0.0 if not math.isfinite(p) else max(0.0, min(1.0, float(p))) for p in ps]
    p = float(sum(ps) / len(ps)) if ps else 0.0
    return ScorerResult(
        score_0_100=float(100.0 * p),
        prob_0_1=float(p),
        model_path_or_id=str(model_path_or_id),
        device=str(dev),
        max_length=int(max_length),
        doc_type=dt,
        normalized=bool(normalize_text),
        n_windows=int(len(ps)),
        window_stride=int(stride_len),
        window_probs=tuple(ps[:64]),
        windows_capped=bool(windows_capped),
    )
