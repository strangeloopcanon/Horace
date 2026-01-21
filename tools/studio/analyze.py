from __future__ import annotations

import hashlib
import math
import random
import re
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from tools.analyze import (
    ModelBackend,
    pick_backend,
    paragraph_units,
    permutation_entropy,
    hurst_rs,
)
from tools.studio.text_normalize import normalize_for_studio


DocType = Literal["poem", "prose", "shortstory", "novel", "all"]

_BACKEND_CACHE: Dict[Tuple[str, str], ModelBackend] = {}
_BACKEND_LOCK = threading.Lock()


@dataclass(frozen=True)
class Spike:
    token_index: int
    char_start: int
    char_end: int
    token_str: str
    surface: str
    surprisal: float
    entropy: float
    is_content: bool
    is_punct: bool
    line_pos: str
    context: str


def _line_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r".*(?:\n|$)", text):
        s, e = m.start(), m.end()
        if s >= e:
            continue
        chunk = text[s:e]
        if chunk.endswith("\n"):
            e -= 1
        spans.append((s, e))
    return spans


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"[^.!?\n]+[.!?]?\s*", text):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            spans.append((s, e))
    return spans


def _detect_punct(surface: str) -> bool:
    s = (surface or "").strip()
    if not s:
        return True
    for ch in s:
        if ch.isalnum():
            return False
    return True


def _detect_newline(surface: str) -> bool:
    return "\n" in (surface or "")


def _detect_content(surface: str, is_punct: bool) -> bool:
    if is_punct:
        return False
    s = (surface or "").strip()
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SYMBOL_CHARS = set("{}[]();<>/\\=+-*#@")
_PUNCT_CHARS = set(",.;:!?…—–-()[]{}\"'“”‘’")


def _slice_text_for_offsets(text: str, offsets: List[Tuple[int, int]]) -> str:
    if not text or not offsets:
        return text or ""
    end = 0
    for _, ce in offsets:
        try:
            e = int(ce)
        except Exception:
            continue
        if e > end:
            end = e
    if 0 < end <= len(text):
        return text[:end]
    return text


def _estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ye")):
        count -= 1
    return max(1, int(count))


def _surface_metrics(text: str, *, sent_spans: List[Tuple[int, int]]) -> Dict[str, Optional[float]]:
    t = text or ""
    chars = len(t)
    if chars <= 0:
        return {
            "chars_count": 0.0,
            "alpha_char_fraction": None,
            "digit_char_fraction": None,
            "symbol_char_fraction": None,
            "whitespace_char_fraction": None,
            "punct_charset_size": None,
            "punct_variety_per_1000_chars": None,
            "word_count": 0.0,
            "word_unique": 0.0,
            "word_ttr": None,
            "word_hapax_ratio": None,
            "word_top5_frac": None,
            "adjacent_word_repeat_rate": None,
            "bigram_repeat_rate": None,
            "trigram_repeat_rate": None,
            "word_len_mean": None,
            "word_len_p90": None,
            "word_len_cv": None,
            "syllables_per_word_mean": None,
            "syllables_per_word_p90": None,
            "syllables_per_word_cv": None,
            "sent_words_mean": None,
            "sent_words_p90": None,
            "sent_words_cv": None,
            "line_code_fraction": None,
            "line_list_fraction": None,
            "line_heading_fraction": None,
            "line_quote_fraction": None,
            "line_indented_fraction": None,
        }

    alpha = sum(1 for ch in t if ch.isalpha())
    digit = sum(1 for ch in t if ch.isdigit())
    whitespace = sum(1 for ch in t if ch.isspace())
    symbol = sum(1 for ch in t if ch in _SYMBOL_CHARS)
    punct_chars = {ch for ch in t if ch in _PUNCT_CHARS}
    punct_size = len(punct_chars)
    punct_var = float(punct_size) / (float(chars) / 1000.0) if chars >= 200 else None

    words = [m.group(0).lower() for m in _WORD_RE.finditer(t)]
    if len(words) > 6000:
        words = words[:6000]
    n_words = len(words)
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    uniq = len(freq)
    hapax = sum(1 for c in freq.values() if c == 1)
    top5 = 0
    if freq:
        top5 = sum(c for _, c in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5])

    adjacent_rep = None
    if n_words >= 2:
        adjacent = sum(1 for i in range(1, n_words) if words[i] == words[i - 1])
        adjacent_rep = float(adjacent) / float(n_words - 1)

    bigram_rep = None
    trigram_rep = None
    if n_words >= 3:
        bigrams = [(words[i], words[i + 1]) for i in range(n_words - 1)]
        bigram_rep = 1.0 - (float(len(set(bigrams))) / float(len(bigrams))) if bigrams else None
        trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(n_words - 2)]
        trigram_rep = 1.0 - (float(len(set(trigrams))) / float(len(trigrams))) if trigrams else None

    word_lens = np.array([len(w) for w in words], dtype=np.float32) if words else np.array([], dtype=np.float32)
    word_len_mean = float(np.mean(word_lens)) if word_lens.size else None
    word_len_p90 = float(np.percentile(word_lens, 90)) if word_lens.size else None
    word_len_cv = None
    if word_lens.size >= 2 and word_len_mean and abs(word_len_mean) > 1e-12:
        word_len_cv = float(np.std(word_lens) / float(abs(word_len_mean)))

    syll = np.array([_estimate_syllables(w) for w in words], dtype=np.float32) if words else np.array([], dtype=np.float32)
    syll_mean = float(np.mean(syll)) if syll.size else None
    syll_p90 = float(np.percentile(syll, 90)) if syll.size else None
    syll_cv = None
    if syll.size >= 2 and syll_mean and abs(syll_mean) > 1e-12:
        syll_cv = float(np.std(syll) / float(abs(syll_mean)))

    sent_counts: List[int] = []
    for s0, s1 in sent_spans:
        if not (0 <= int(s0) < int(s1) <= chars):
            continue
        seg = t[int(s0) : int(s1)]
        sent_counts.append(len(_WORD_RE.findall(seg)))
    sent_words_mean = None
    sent_words_p90 = None
    sent_words_cv = None
    if sent_counts:
        arr = np.array(sent_counts, dtype=np.float32)
        sent_words_mean = float(np.mean(arr))
        sent_words_p90 = float(np.percentile(arr, 90))
        if arr.size >= 2 and sent_words_mean and abs(sent_words_mean) > 1e-12:
            sent_words_cv = float(np.std(arr) / float(abs(sent_words_mean)))

    lines = t.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    n_lines = len(nonempty)
    code_pat = re.compile(
        r"^\s*(?:def |class |from |import |package |#include|using |public |private |func |let |var )",
        flags=re.I,
    )
    bullet_pat = re.compile(r"^\s*(?:[-*+]|•|\d+[.)])\s+\S")
    heading_pat = re.compile(r"^\s*#{1,6}\s+\S")
    quote_pat = re.compile(r"^\s*>\s+\S")
    indented_pat = re.compile(r"^(?:\t+|\s{4,})\S")

    code_lines = 0
    list_lines = 0
    heading_lines = 0
    quote_lines = 0
    indented_lines = 0
    for ln in nonempty:
        s = ln.rstrip()
        if indented_pat.match(s):
            indented_lines += 1
        if quote_pat.match(s):
            quote_lines += 1
        if bullet_pat.match(s):
            list_lines += 1
        if heading_pat.match(s):
            heading_lines += 1
        sym = sum(1 for ch in s if ch in _SYMBOL_CHARS)
        a = sum(1 for ch in s if ch.isalpha())
        sym_ratio = float(sym) / float(max(1, a + sym))
        if code_pat.match(s) or sym_ratio >= 0.33:
            code_lines += 1

    denom_lines = float(max(1, n_lines))
    return {
        "chars_count": float(chars),
        "alpha_char_fraction": float(alpha) / float(chars),
        "digit_char_fraction": float(digit) / float(chars),
        "symbol_char_fraction": float(symbol) / float(chars),
        "whitespace_char_fraction": float(whitespace) / float(chars),
        "punct_charset_size": float(punct_size),
        "punct_variety_per_1000_chars": float(punct_var) if punct_var is not None else None,
        "word_count": float(n_words),
        "word_unique": float(uniq),
        "word_ttr": (float(uniq) / float(n_words)) if n_words else None,
        "word_hapax_ratio": (float(hapax) / float(n_words)) if n_words else None,
        "word_top5_frac": (float(top5) / float(n_words)) if n_words else None,
        "adjacent_word_repeat_rate": adjacent_rep,
        "bigram_repeat_rate": float(bigram_rep) if bigram_rep is not None else None,
        "trigram_repeat_rate": float(trigram_rep) if trigram_rep is not None else None,
        "word_len_mean": word_len_mean,
        "word_len_p90": word_len_p90,
        "word_len_cv": word_len_cv,
        "syllables_per_word_mean": syll_mean,
        "syllables_per_word_p90": syll_p90,
        "syllables_per_word_cv": syll_cv,
        "sent_words_mean": sent_words_mean,
        "sent_words_p90": sent_words_p90,
        "sent_words_cv": sent_words_cv,
        "line_code_fraction": float(code_lines) / denom_lines if n_lines else None,
        "line_list_fraction": float(list_lines) / denom_lines if n_lines else None,
        "line_heading_fraction": float(heading_lines) / denom_lines if n_lines else None,
        "line_quote_fraction": float(quote_lines) / denom_lines if n_lines else None,
        "line_indented_fraction": float(indented_lines) / denom_lines if n_lines else None,
    }


def _line_pos_for_char(line_spans: List[Tuple[int, int]], cs: int, ce: int) -> str:
    if not line_spans:
        return "middle"
    import bisect

    starts = [s for s, _ in line_spans]
    idx = bisect.bisect_right(starts, cs) - 1
    if not (0 <= idx < len(line_spans)):
        return "middle"
    ls, le = line_spans[idx]
    if cs == ls:
        return "start"
    if ce == le:
        return "end"
    return "middle"


def _excerpt(text: str, cs: int, ce: int, *, window: int = 48) -> str:
    n = len(text)
    a = max(0, cs - window)
    b = min(n, ce + window)
    prefix = "…" if a > 0 else ""
    suffix = "…" if b < n else ""
    return prefix + text[a:b] + suffix


def _coerce_doc_type(doc_type: str) -> str:
    dt = (doc_type or "").strip().lower()
    if dt in ("poem", "poetry"):
        return "poem"
    if dt in ("shortstory", "short_story", "short-story", "story"):
        return "shortstory"
    if dt in ("novel",):
        return "novel"
    if dt in ("prose", ""):
        return "prose"
    return dt


def _get_backend(model_id: str, *, backend: str) -> ModelBackend:
    """Return a cached backend instance for interactive use (UI/API)."""
    key = (str(backend or "auto").lower(), str(model_id))
    with _BACKEND_LOCK:
        cached = _BACKEND_CACHE.get(key)
        if cached is not None:
            return cached
        inst: ModelBackend = pick_backend(model_id, prefer_mlx=True, backend=backend)
        _BACKEND_CACHE[key] = inst
        return inst


def analyze_text(
    text: str,
    *,
    model_id: str = "gpt2",
    doc_type: str = "prose",
    backend: str = "auto",
    normalize_text: bool = True,
    k: int = 10,
    p: float = 0.9,
    context: int = 1024,
    stride: Optional[int] = None,
    max_input_tokens: int = 1024,
    compute_cohesion: bool = True,
    max_spikes: int = 12,
) -> Dict[str, Any]:
    """Analyze arbitrary input text and return doc-level metrics + spike highlights.

    This is the Studio-oriented, in-memory counterpart to `tools/analyze.py`, optimized for:
    - predictable runtime (token cap)
    - returning lightweight artifacts suitable for a web UI
    """
    doc_type_norm = _coerce_doc_type(doc_type)
    model: ModelBackend = _get_backend(model_id, backend=backend)

    raw_text = text or ""
    text, norm_meta = normalize_for_studio(raw_text, doc_type=doc_type_norm, enabled=bool(normalize_text))
    doc_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

    enc = model.tokenize(text)
    input_ids = list(enc.get("input_ids") or [])
    offsets = enc.get("offset_mapping") or [(0, 0)] * len(input_ids)
    if not input_ids:
        return {
            "doc_metrics": {
                "doc_id": f"user_{doc_hash}",
                "doc_type": doc_type_norm,
                "author": "(user)",
                "title": "input",
                "model_id": model.model_id,
                "tokens_count": 0,
            },
            "spikes": [],
            "segments": {},
            "truncated": False,
            "model_id": model.model_id,
            "text_normalization": norm_meta,
        }

    truncated = False
    if max_input_tokens and len(input_ids) > max_input_tokens:
        truncated = True
        input_ids = input_ids[:max_input_tokens]
        offsets = offsets[:max_input_tokens]

    text_used = _slice_text_for_offsets(text, offsets)

    # Precompute spans (align to the analyzed prefix if truncated)
    line_spans = _line_spans(text_used)
    sent_spans = _sentence_spans(text_used)
    surface_metrics = _surface_metrics(text_used, sent_spans=sent_spans)

    # Sliding-window settings (kept similar to tools/analyze.py)
    max_ctx = min(int(context), int(getattr(model, "max_context")() or context), 2048)
    window = min(max_ctx, len(input_ids))
    if window < 2:
        return {
            "doc_metrics": {
                "doc_id": f"user_{doc_hash}",
                "doc_type": doc_type_norm,
                "author": "(user)",
                "title": "input",
                "model_id": model.model_id,
                "tokens_count": 0,
            },
            "spikes": [],
            "segments": {},
            "truncated": truncated,
            "model_id": model.model_id,
            "text_normalization": norm_meta,
        }
    emit_overlap = 128 if window > 256 else max(0, window // 4)
    stride_len = int(stride) if stride is not None else max(1, window - emit_overlap)

    # Per-token series (aligned to emitted tokens)
    token_indices: List[int] = []
    char_starts: List[int] = []
    char_ends: List[int] = []
    token_strs: List[str] = []
    surfaces: List[str] = []

    p_true_list: List[float] = []
    logp_list: List[float] = []
    H_list: List[float] = []
    eff_list: List[float] = []
    rk_list: List[int] = []
    w_list: List[int] = []

    is_punct_list: List[int] = []
    is_newline_list: List[int] = []
    is_content_list: List[int] = []
    line_pos_list: List[str] = []

    seen: set[int] = set()
    for start in range(0, len(input_ids), stride_len):
        end = min(len(input_ids), start + window)
        ids = input_ids[start:end]
        if len(ids) < 2:
            break
        m = model.metrics_for_input_ids(ids, k=int(k), nucleus_p=float(p))
        labels = ids[1:]
        emit_from = 0 if start == 0 else min(len(labels) - 1, emit_overlap)

        for i in range(len(labels)):
            if start > 0 and i < emit_from:
                continue
            global_tok = start + i + 1
            if global_tok in seen:
                continue
            seen.add(global_tok)

            # token metadata
            off = offsets[global_tok] if global_tok < len(offsets) else (0, 0)
            cs, ce = int(off[0]), int(off[1])
            surface = ""
            if 0 <= cs < len(text) and 0 <= ce <= len(text) and cs < ce:
                surface = text[cs:ce]
            tok_str = model.token_str(int(labels[i]))
            if tok_str is None:
                tok_str = ""

            punct = 1 if _detect_punct(surface) else 0
            nl = 1 if _detect_newline(surface) else 0
            content = 1 if _detect_content(surface, bool(punct or nl)) else 0

            token_indices.append(int(global_tok))
            char_starts.append(cs)
            char_ends.append(ce)
            token_strs.append(tok_str)
            surfaces.append(surface)

            p_true_list.append(float(m["p_true"][i]))
            logp_list.append(float(m["logp"][i]))
            H_list.append(float(m["H"][i]))
            eff_list.append(float(m["eff"][i]))
            rk_list.append(int(m["rk"][i]))
            w_list.append(int(m["w"][i]))

            is_punct_list.append(punct)
            is_newline_list.append(nl)
            is_content_list.append(content)
            line_pos_list.append(_line_pos_for_char(line_spans, cs, ce))

        if end >= len(input_ids):
            break

    if not logp_list:
        return {
            "doc_metrics": {
                "doc_id": f"user_{doc_hash}",
                "doc_type": doc_type_norm,
                "author": "(user)",
                "title": "input",
                "model_id": model.model_id,
                "tokens_count": 0,
            },
            "spikes": [],
            "segments": {},
            "truncated": truncated,
            "model_id": model.model_id,
            "text_normalization": norm_meta,
        }

    p_true_arr = np.array(p_true_list, dtype=np.float32)
    logp_arr = np.array(logp_list, dtype=np.float32)
    H_arr = np.array(H_list, dtype=np.float32)
    eff_arr = np.array(eff_list, dtype=np.float32)
    rk_arr = np.array(rk_list, dtype=np.float32)
    w_arr = np.array(w_list, dtype=np.float32)

    surprisal = (-logp_arr).astype(np.float32)
    norm_surprisal = surprisal / np.clip(H_arr, 1e-12, None)
    norm_surprisal_clipped = surprisal / np.clip(H_arr, 1.0, None)
    ratio_of_means = float(np.mean(surprisal) / max(float(np.mean(H_arr)), 1e-12))
    norm_surprisal_median = float(np.median(norm_surprisal_clipped))
    LN2 = math.log(2.0)
    bpt_mean = float(np.mean(surprisal) / LN2)
    entropy_bits_mean = float(np.mean(H_arr) / LN2)

    # Cadence metrics (token-level)
    def cadence_metrics(s: np.ndarray) -> Dict[str, Optional[float]]:
        n = int(s.shape[0])
        if n < 5:
            return {
                "surprisal_cv": None,
                "surprisal_masd": None,
                "surprisal_acf1": None,
                "surprisal_acf2": None,
                "surprisal_peak_period_tokens": None,
                "high_surprise_rate_per_100": None,
                "high_surprise_ipi_mean": None,
                "high_surprise_ipi_cv": None,
                "run_low_mean_len": None,
                "run_high_mean_len": None,
            }

        mu = float(np.mean(s))
        sd = float(np.std(s))
        out: Dict[str, Optional[float]] = {}
        out["surprisal_cv"] = (sd / mu) if mu > 1e-12 else None
        out["surprisal_masd"] = float(np.mean(np.abs(np.diff(s))))
        z = s - mu
        denom = float(np.sum(z * z) + 1e-12)

        def _acf(k: int) -> Optional[float]:
            if n <= k:
                return None
            num = float(np.sum(z[k:] * z[:-k]))
            return num / denom

        out["surprisal_acf1"] = _acf(1)
        out["surprisal_acf2"] = _acf(2)
        try:
            ps = np.abs(np.fft.rfft(z)) ** 2
            if ps.shape[0] > 2:
                idx = int(np.argmax(ps[1:]) + 1)
                freq = idx / n
                out["surprisal_peak_period_tokens"] = (1.0 / freq) if freq > 0 else None
            else:
                out["surprisal_peak_period_tokens"] = None
        except Exception:
            out["surprisal_peak_period_tokens"] = None

        thr = mu + sd
        peaks: List[int] = []
        for i in range(1, n - 1):
            if s[i] >= thr and s[i] > s[i - 1] and s[i] >= s[i + 1]:
                peaks.append(i)
        out["high_surprise_rate_per_100"] = float((len(peaks) / n) * 100.0)
        if len(peaks) >= 2:
            ipi = np.diff(np.array(peaks))
            out["high_surprise_ipi_mean"] = float(np.mean(ipi))
            mipi = float(np.mean(ipi))
            out["high_surprise_ipi_cv"] = float(np.std(ipi) / (mipi + 1e-12)) if ipi.size > 1 else None
        else:
            out["high_surprise_ipi_mean"] = None
            out["high_surprise_ipi_cv"] = None

        med = float(np.median(s))
        is_low = s <= med

        def runlens(mask: np.ndarray, val: bool) -> List[int]:
            res: List[int] = []
            cur = 0
            for m in mask:
                if bool(m) == val:
                    cur += 1
                else:
                    if cur > 0:
                        res.append(cur)
                        cur = 0
            if cur > 0:
                res.append(cur)
            return res

        low_lens = runlens(is_low, True)
        high_lens = runlens(is_low, False)
        out["run_low_mean_len"] = float(np.mean(low_lens)) if low_lens else None
        out["run_high_mean_len"] = float(np.mean(high_lens)) if high_lens else None
        return out

    cad = cadence_metrics(surprisal)

    surprisal_p10 = float(np.percentile(surprisal, 10))
    entropy_p10 = float(np.percentile(H_arr, 10))
    entropy_p90 = float(np.percentile(H_arr, 90))
    pe = permutation_entropy(surprisal, m=3, tau=1)
    hurst = hurst_rs(surprisal)

    # Spikes and neighborhoods (threshold = mean + std)
    mu_s = float(np.mean(surprisal))
    sd_s = float(np.std(surprisal))
    thr = mu_s + sd_s
    spike_pos = np.where(surprisal >= thr)[0]
    prev_idx = spike_pos - 1
    next_idx = spike_pos + 1
    prev_mask = prev_idx >= 0
    next_mask = next_idx < len(surprisal)
    is_content_arr = np.array(is_content_list, dtype=np.float32)
    is_punct_arr = np.array(is_punct_list, dtype=np.float32)
    prev_content_rate = float(np.mean(is_content_arr[prev_idx[prev_mask]]) if np.any(prev_mask) else 0.0)
    prev_punct_rate = float(np.mean(is_punct_arr[prev_idx[prev_mask]]) if np.any(prev_mask) else 0.0)
    next_content_rate = float(np.mean(is_content_arr[next_idx[next_mask]]) if np.any(next_mask) else 0.0)
    next_punct_rate = float(np.mean(is_punct_arr[next_idx[next_mask]]) if np.any(next_mask) else 0.0)
    ipi_mean = None
    ipi_cv = None
    ipi_p50 = None
    if len(spike_pos) >= 2:
        ipi = np.diff(spike_pos)
        ipi_mean = float(np.mean(ipi))
        ipi_p50 = float(np.median(ipi))
        ipi_cv = float(np.std(ipi) / (ipi_mean + 1e-12)) if ipi.size > 1 else None

    cooldown = []
    for idx in spike_pos:
        if idx + 3 < len(H_arr):
            cooldown.append(float(H_arr[idx] - np.mean(H_arr[idx + 1 : idx + 4])))
    cooldown_drop_3 = float(np.mean(cooldown)) if cooldown else None

    content_fraction = float(np.mean(is_content_arr)) if is_content_list else None
    punct_rate = float(np.mean(is_punct_arr)) if is_punct_list else None
    newline_rate = float(np.mean(np.array(is_newline_list, dtype=np.float32))) if is_newline_list else None

    doc_metrics: Dict[str, Any] = {
        "doc_id": f"user_{doc_hash}",
        "doc_type": doc_type_norm,
        "author": "(user)",
        "title": "input",
        "model_id": model.model_id,
        "tokens_count": int(len(surprisal)),
        "p_true_mean": float(np.mean(p_true_arr)),
        "surprisal_mean": float(np.mean(surprisal)),
        "surprisal_median": float(np.median(surprisal)),
        "surprisal_p90": float(np.percentile(surprisal, 90)),
        "entropy_mean": float(np.mean(H_arr)),
        "entropy_median": float(np.median(H_arr)),
        "eff_support_mean": float(np.mean(eff_arr)),
        "rank_percentile_mean": float(np.mean(rk_arr / float(model.vocab_size()))),
        "nucleus_w_mean": float(np.mean(w_arr)),
        "norm_surprisal_mean": float(np.mean(norm_surprisal)),
        "norm_surprisal_mean_clipped": float(np.mean(norm_surprisal_clipped)),
        "norm_surprisal_median": norm_surprisal_median,
        "norm_surprisal_ratio_of_means": ratio_of_means,
        "bpt_mean": bpt_mean,
        "entropy_bits_mean": entropy_bits_mean,
        **cad,
        "surprisal_p10": surprisal_p10,
        "entropy_p10": entropy_p10,
        "entropy_p90": entropy_p90,
        "perm_entropy": pe,
        "hurst_rs": hurst,
        "content_fraction": content_fraction,
        "punct_rate": punct_rate,
        "newline_rate": newline_rate,
        "spike_prev_content_rate": prev_content_rate,
        "spike_prev_punct_rate": prev_punct_rate,
        "spike_next_content_rate": next_content_rate,
        "spike_next_punct_rate": next_punct_rate,
        "ipi_mean": ipi_mean,
        "ipi_cv": ipi_cv,
        "ipi_p50": ipi_p50,
        "cooldown_entropy_drop_3": cooldown_drop_3,
    }
    doc_metrics.update(surface_metrics)

    # Cohesion delta (shuffled units) – expensive, but useful; run once if possible.
    if compute_cohesion:
        try:
            kind = "poem" if doc_type_norm == "poem" else "prose"
            units = paragraph_units(text_used, kind)
            if len(units) >= 3:
                shuffled = units[:]
                random.Random(0).shuffle(shuffled)
                shuf_text = "".join(text_used[s:e] for s, e in shuffled)
                enc_s = model.tokenize(shuf_text)
                ids_s = list(enc_s.get("input_ids") or [])
                if max_input_tokens and len(ids_s) > max_input_tokens:
                    ids_s = ids_s[:max_input_tokens]
                if len(ids_s) >= 2:
                    ms = model.metrics_for_input_ids(ids_s, k=int(k), nucleus_p=float(p))
                    shuf_logp = ms.get("logp")
                    if shuf_logp is not None and len(shuf_logp) > 0:
                        shuffle_ll = float(np.mean(np.array(shuf_logp, dtype=np.float32)))
                        doc_metrics["cohesion_shuffle_logp_per_token"] = shuffle_ll
                        logp_orig = float(np.mean(logp_arr))
                        doc_metrics["logp_per_token_original"] = logp_orig
                        doc_metrics["cohesion_delta"] = shuffle_ll - logp_orig
        except Exception:
            pass

    # Spike highlights (top-N by surprisal)
    spikes: List[Spike] = []
    if len(spike_pos) > 0:
        order = sorted(spike_pos.tolist(), key=lambda i: float(surprisal[i]), reverse=True)
        for idx in order[: max(0, int(max_spikes))]:
            tok_i = token_indices[idx]
            cs = char_starts[idx]
            ce = char_ends[idx]
            spikes.append(
                Spike(
                    token_index=int(tok_i),
                    char_start=int(cs),
                    char_end=int(ce),
                    token_str=token_strs[idx],
                    surface=surfaces[idx],
                    surprisal=float(surprisal[idx]),
                    entropy=float(H_arr[idx]),
                    is_content=bool(is_content_list[idx]),
                    is_punct=bool(is_punct_list[idx]),
                    line_pos=str(line_pos_list[idx]),
                    context=_excerpt(text, cs, ce),
                )
            )

    # Sentence-level burstiness (broad cadence)
    char_starts_arr = np.array(char_starts, dtype=np.int32)

    def _segment_series(spans: List[Tuple[int, int]]):
        means: List[float] = []
        token_counts: List[int] = []
        for s0, s1 in spans:
            mask = (char_starts_arr >= int(s0)) & (char_starts_arr < int(s1))
            c = int(mask.sum())
            if c <= 0:
                continue
            token_counts.append(c)
            means.append(float(np.mean(surprisal[mask])))
        cv = None
        if len(means) >= 2:
            mu = float(np.mean(means))
            sd = float(np.std(means))
            cv = float(sd / (abs(mu) + 1e-12))
        len_cv = None
        if len(token_counts) >= 2:
            mu = float(np.mean(token_counts))
            sd = float(np.std(token_counts))
            len_cv = float(sd / (abs(mu) + 1e-12))
        return means, token_counts, cv, len_cv

    sent_means, sent_counts, burst_cv_sent, len_cv_sent = _segment_series(sent_spans)
    para_spans = paragraph_units(text_used, "prose")
    para_means, para_counts, burst_cv_para, len_cv_para = _segment_series(para_spans)
    line_means, line_counts, burst_cv_line, len_cv_line = _segment_series(line_spans)

    def _tokens_mean_p90(counts: List[int]) -> Tuple[Optional[float], Optional[float]]:
        if not counts:
            return None, None
        arr = np.array(counts, dtype=np.float32)
        return float(np.mean(arr)), float(np.percentile(arr, 90))

    sent_tokens_mean, sent_tokens_p90 = _tokens_mean_p90(sent_counts)
    para_tokens_mean, para_tokens_p90 = _tokens_mean_p90(para_counts)
    line_tokens_mean, line_tokens_p90 = _tokens_mean_p90(line_counts)

    # Promote broad cadence into doc-level metrics so they can be baselined and learned.
    doc_metrics.update(
        {
            "sent_count": int(len(sent_spans)),
            "sent_tokens_mean": sent_tokens_mean,
            "sent_tokens_p90": sent_tokens_p90,
            "sent_burst_cv": burst_cv_sent,
            "sent_len_cv": len_cv_sent,
            "para_count": int(len(para_spans)),
            "para_tokens_mean": para_tokens_mean,
            "para_tokens_p90": para_tokens_p90,
            "para_burst_cv": burst_cv_para,
            "para_len_cv": len_cv_para,
            "line_count": int(len(line_spans)),
            "line_tokens_mean": line_tokens_mean,
            "line_tokens_p90": line_tokens_p90,
            "line_burst_cv": burst_cv_line,
            "line_len_cv": len_cv_line,
        }
    )

    segments = {
        "sentences": {
            "count": int(len(sent_spans)),
            "mean_surprisal": sent_means[:64],
            "token_counts": sent_counts[:64],
            "burst_cv": burst_cv_sent,
            "len_cv": len_cv_sent,
        },
        "paragraphs": {
            "count": int(len(para_spans)),
            "mean_surprisal": para_means[:64],
            "token_counts": para_counts[:64],
            "burst_cv": burst_cv_para,
            "len_cv": len_cv_para,
        },
        "lines": {
            "count": int(len(line_spans)),
            "mean_surprisal": line_means[:64],
            "token_counts": line_counts[:64],
            "burst_cv": burst_cv_line,
            "len_cv": len_cv_line,
        },
    }

    series_max = 800
    return {
        "doc_metrics": doc_metrics,
        "spikes": [asdict(s) for s in spikes],
        "segments": segments,
        "series": {
            "token_index": token_indices[:series_max],
            "char_start": char_starts[:series_max],
            "surprisal": [float(x) for x in surprisal[:series_max].tolist()],
            "entropy": [float(x) for x in H_arr[:series_max].tolist()],
            "threshold_surprisal": float(thr),
        },
        "truncated": truncated,
        "model_id": model.model_id,
        "text_normalization": norm_meta,
    }
