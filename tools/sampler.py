#!/usr/bin/env python3
import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse model backends from analyze.py
from tools.analyze import pick_backend, ModelBackend


# ---- Utilities --------------------------------------------------------------

_EN_STOPWORDS = set(
    """
    a an the and or but if then else when while for nor so yet at by for from in of on to up with as is are was were be been being do does did done doing have has had having i me my we us our you your he him his she her it its they them their this that these those who whom which what where when why how not no only also very just too into over under again further once here there about between through during before after above below off out down up same other more most some such own than ever never always often sometimes usually rather quite perhaps maybe almost nearly either neither both each few many much less least lot lots plenty rather without within cannot can't don't doesn't didn't won't wouldn't shouldn't isn't aren't wasn't weren't i'm you're we're they're i've you've we've they've it's that's there's here's who's what's let's mustn't mightn't
    """
    .split()
)


def is_punct_token(tok: str) -> bool:
    s = tok.strip()
    if not s:
        return True
    # Heuristics to handle BPE artifacts (leading space tokens)
    # Keep only punctuation characters
    for ch in s:
        if ch.isalnum():
            return False
    # All non-alnum → treat as punctuation
    return True


def is_newline_token(tok: str) -> bool:
    return "\n" in tok


def is_content_token(tok: str) -> bool:
    s = tok.strip()
    # Drop leading BPE space markers (e.g., ' Ġ' style handled by tok itself)
    # Heuristic: keep letters only for length
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    low = ''.join(letters).lower()
    if low in _EN_STOPWORDS:
        return False
    return True


def sample_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> int:
    x = logits.astype(np.float32)
    if temperature is None or temperature <= 0:
        temperature = 1.0
    x = x / float(temperature)

    # Top-k filter
    if top_k is not None and top_k > 0 and top_k < x.shape[-1]:
        idx = np.argpartition(-x, kth=top_k - 1)[-top_k:]
        mask = np.full_like(x, -np.inf, dtype=np.float32)
        mask[idx] = x[idx]
        x = mask

    # Top-p (nucleus) filter
    if top_p is not None and 0 < top_p < 1.0:
        # sort by prob
        y = x - np.max(x)
        p = np.exp(y)
        p = p / (p.sum() + 1e-12)
        order = np.argsort(-p)
        p_sorted = p[order]
        cdf = np.cumsum(p_sorted)
        k = int(np.searchsorted(cdf, top_p, side='left')) + 1
        keep = order[:k]
        mask = np.full_like(x, -np.inf, dtype=np.float32)
        mask[keep] = x[keep]
        x = mask

    # Softmax + sample
    y = x - np.max(x)
    p = np.exp(y)
    p = p / (p.sum() + 1e-12)
    return int(np.random.choice(np.arange(p.shape[-1]), p=p))


# ---- Cadence Sampler --------------------------------------------------------

@dataclass
class PhaseParams:
    top_p: float
    temperature: float
    top_k: Optional[int] = None
    # optional logit nudges
    content_boost: float = 0.0
    stop_punct_penalty: float = 0.0


@dataclass
class PoetryConfig:
    base: PhaseParams
    spike: PhaseParams
    cool: PhaseParams
    interval_range: Tuple[int, int] = (10, 16)
    cooldown_range: Tuple[int, int] = (3, 6)
    # punctuation handling
    defer_spike_on_punct: bool = True
    # sustained shift (line-level): open entropy for a short span periodically
    shift_every_lines: Tuple[int, int] = (1, 3)
    shift_span_tokens: Tuple[int, int] = (8, 16)
    # Optional rhyme/line controls (present in some presets)
    rhyme_enabled: bool = False
    rhyme_scheme: Optional[str] = None
    rhyme_boost: float = 0.0
    rhyme_min_line_tokens: int = 6
    line_tokens_target: Optional[Tuple[int, int]] = None
    newline_penalty_until_target: float = 0.0
    newline_penalty_until_rhyme: float = 0.0


class CadenceSampler:
    def __init__(self, backend: ModelBackend, config: PoetryConfig, seed: Optional[int] = None, debug: bool = False):
        self.backend = backend
        self.cfg = config
        self.debug = debug
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Precompute vocab gating masks
        V = backend.vocab_size()
        self._is_punct = np.zeros(V, dtype=bool)
        self._is_newline = np.zeros(V, dtype=bool)
        self._is_content = np.zeros(V, dtype=bool)
        self._forbidden = np.zeros(V, dtype=bool)
        # Build masks by inspecting token strings
        for tid in range(V):
            tok = backend.token_str(tid)
            tok = tok or ''
            nl = is_newline_token(tok)
            self._is_newline[tid] = nl
            self._is_punct[tid] = is_punct_token(tok) or nl
            self._is_content[tid] = (not self._is_punct[tid]) and is_content_token(tok)
        # Mark special/unknown tokens as forbidden (avoid <unk>, control placeholders)
        try:
            tok_obj = getattr(self.backend, 'hf_tok', None)
            if tok_obj is not None:
                for sid in (getattr(tok_obj, 'all_special_ids', None) or []):
                    if isinstance(sid, int) and 0 <= sid < V:
                        self._forbidden[sid] = True
                unk_id = getattr(tok_obj, 'unk_token_id', None)
                if isinstance(unk_id, int) and 0 <= unk_id < V:
                    self._forbidden[unk_id] = True
        except Exception:
            pass

        # State
        self.to_next_spike = self._sample_between(self.cfg.interval_range)
        self.cooldown_left = 0
        self.line_tokens = 0
        self.lines_since_shift = 0
        self.shift_tokens_left = 0
        self.line_index = 0
        self.current_text = ""
        self._rhyme_memory: Dict[str, str] = {}

    @staticmethod
    def _sample_between(rr: Tuple[int, int]) -> int:
        a, b = rr
        if a == b:
            return a
        return random.randint(min(a, b), max(a, b))

    def _choose_phase(self) -> str:
        if self.cooldown_left > 0:
            return 'cool'
        if self.shift_tokens_left > 0:
            return 'spike'  # sustained shift uses spike-like openness
        if self.to_next_spike <= 0:
            return 'spike'
        return 'base'

    def _apply_bias(self, logits: np.ndarray, phase: str) -> np.ndarray:
        params = getattr(self.cfg, phase)
        # Ensure masks align with current vocab dimension
        self._ensure_masks_len(logits.shape[-1])
        if params.stop_punct_penalty != 0.0:
            logits = logits.copy()
            logits[self._is_punct] -= float(params.stop_punct_penalty)
        if params.content_boost != 0.0:
            if isinstance(logits, np.ndarray):
                logits[self._is_content] += float(params.content_boost)
        # Always discourage forbidden tokens (e.g., <unk>, specials)
        try:
            if self._forbidden.shape[0] == logits.shape[-1] and self._forbidden.any():
                logits[self._forbidden] -= 50.0
        except Exception:
            pass
        # Rhyme-aware nudging and line-length control
        logits = self._apply_rhyme_and_line_bias(logits)
        return logits

    def _ensure_masks_len(self, V: int):
        if self._is_punct.shape[0] == V:
            return
        self._is_punct = np.zeros(V, dtype=bool)
        self._is_newline = np.zeros(V, dtype=bool)
        self._is_content = np.zeros(V, dtype=bool)
        self._forbidden = np.zeros(V, dtype=bool)
        for tid in range(V):
            tok = self.backend.token_str(tid)
            tok = tok or ''
            nl = is_newline_token(tok)
            self._is_newline[tid] = nl
            self._is_punct[tid] = is_punct_token(tok) or nl
            self._is_content[tid] = (not self._is_punct[tid]) and is_content_token(tok)
        try:
            tok_obj = getattr(self.backend, 'hf_tok', None)
            if tok_obj is not None:
                for sid in (getattr(tok_obj, 'all_special_ids', None) or []):
                    if isinstance(sid, int) and 0 <= sid < V:
                        self._forbidden[sid] = True
                unk_id = getattr(tok_obj, 'unk_token_id', None)
                if isinstance(unk_id, int) and 0 <= unk_id < V:
                    self._forbidden[unk_id] = True
        except Exception:
            pass

    # ---- Rhyme helpers ----
    def _scheme_letters(self) -> Optional[List[str]]:
        scheme = getattr(self.cfg, 'rhyme_scheme', None)
        if not scheme:
            return None
        letters = [ch for ch in scheme if ch.isalpha()]
        return letters if letters else None

    @staticmethod
    def _rhyme_key(word: str) -> str:
        w = ''.join([c for c in word.lower() if c.isalpha()])
        if not w:
            return ''
        vowels = set('aeiouy')
        # take last vowel-run + trailing consonants
        i = len(w) - 1
        # skip trailing non-letters already removed
        # back to last vowel
        end = len(w)
        start = end - 1
        while start >= 0 and w[start] not in vowels:
            start -= 1
        while start >= 0 and w[start] in vowels:
            start -= 1
        start += 1
        key = w[start:end]
        if len(key) < 3:
            key = w[-3:] if len(w) >= 3 else w
        return key

    @staticmethod
    def _last_word(text: str) -> str:
        import re as _re
        words = _re.findall(r"[A-Za-z']+", text)
        return words[-1] if words else ''

    def _target_rhyme_for_line(self) -> Optional[str]:
        letters = self._scheme_letters()
        if not letters:
            return None
        L = len(letters)
        lbl = letters[self.line_index % L]
        return self._rhyme_memory.get(lbl)

    def _record_rhyme_for_line(self, line_text: str):
        letters = self._scheme_letters()
        if not letters:
            return
        L = len(letters)
        lbl = letters[self.line_index % L]
        last = self._last_word(line_text)
        key = self._rhyme_key(last)
        if key:
            self._rhyme_memory[lbl] = key

    def _apply_rhyme_and_line_bias(self, logits: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        logits = logits.copy()
        # Control early newlines (line length target)
        if hasattr(cfg, 'line_tokens_target') and cfg.line_tokens_target:
            min_len, max_len = cfg.line_tokens_target
            if self.line_tokens < min_len:
                # discourage newline until minimum reached
                pen = getattr(cfg, 'newline_penalty_until_target', 0.0) or 0.0
                if pen > 0:
                    logits[self._is_newline] -= float(pen)
        # Rhyme nudging near line end
        rhyme_enabled = getattr(cfg, 'rhyme_enabled', False)
        if rhyme_enabled and self.line_tokens >= getattr(cfg, 'rhyme_min_line_tokens', 6):
            target = self._target_rhyme_for_line()
            if target:
                frag = target[-3:]
                boost = float(getattr(cfg, 'rhyme_boost', 0.0) or 0.0)
                if boost > 0:
                    # Boost top candidates that contain the fragment
                    # Limit to a candidate pool for compute efficiency
                    K = min(256, logits.shape[-1])
                    idx = np.argpartition(-logits, kth=K - 1)[-K:]
                    for j in idx:
                        ts = self.backend.token_str(int(j)).lower()
                        if frag and frag in ts and ts.strip():
                            logits[j] += boost
                # Allow newline when current line already satisfies rhyme key
                last = self._last_word(self.current_text.split('\n')[-1])
                if self._rhyme_key(last) != target:
                    pen2 = float(getattr(cfg, 'newline_penalty_until_rhyme', 0.0) or 0.0)
                    if pen2 > 0:
                        logits[self._is_newline] -= pen2
        return logits

    def _step_logits(self, last_logits: Optional[np.ndarray], out_ids: List[int], cache_state) -> Tuple[np.ndarray, object]:
        # Use KV cache if available; otherwise recompute
        if cache_state is not None and last_logits is not None:
            return last_logits, cache_state
        # Fallback full pass
        # Ensure integer ids, with debug
        non_ints = [x for x in out_ids if not isinstance(x, (int, np.integer))]
        if non_ints:
            if self.debug:
                print(f"[DEBUG] Found non-int ids before forward: types={[type(x).__name__ for x in non_ints[:5]]}; tail={repr(out_ids[-8:])}")
            try:
                out_ids = [int(x) for x in out_ids]
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Failed to coerce ids to int: {e}")
                raise
        logits = self.backend.logits_for_input_ids(out_ids)[-1]
        return logits, None

    def _step(self, last_logits: Optional[np.ndarray], out_ids: List[int], cache_state) -> Tuple[int, str, np.ndarray, object]:
        # Get next-token logits for current context
        logits, cache_state = self._step_logits(last_logits, out_ids, cache_state)

        # Decide phase and parameters
        phase = self._choose_phase()
        params: PhaseParams = getattr(self.cfg, phase)
        # Apply content/punct bias for spike phase (and others if configured)
        biased = self._apply_bias(logits, phase)
        # Sample
        tid = sample_from_logits(
            biased,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
        )
        return tid, phase, logits, cache_state

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        enc = self.backend.tokenize(prompt)
        if isinstance(enc, dict) or ('input_ids' in enc):
            ids = list(enc['input_ids'])
        else:
            ids = list(enc)
        # Ensure ints
        try:
            ids = [int(x) for x in ids]
        except Exception:
            pass
        if self.debug:
            bad = [type(x).__name__ for x in ids if not isinstance(x, (int, np.integer))]
            if bad:
                print(f"[DEBUG] Non-int ids in prompt encoding: {bad[:5]} ... total {len(bad)}")
        # Prepend BOS if tokenizer defines one and it's not present
        bos_id = None
        try:
            if hasattr(self.backend, 'hf_tok') and self.backend.hf_tok is not None:
                bos_id = getattr(self.backend.hf_tok, 'bos_token_id', None)
            if bos_id is None and hasattr(self.backend, 'tok'):
                bos_id = getattr(self.backend.tok, 'bos_token_id', None)
        except Exception:
            bos_id = None
        out_ids = ids.copy()
        if bos_id is not None and (len(out_ids) == 0 or out_ids[0] != bos_id):
            out_ids = [int(bos_id)] + out_ids

        # Track lines
        self.line_tokens = 0
        self.lines_since_shift = 0
        self.shift_tokens_left = 0
        self.to_next_spike = self._sample_between(self.cfg.interval_range)
        self.cooldown_left = 0
        self.line_index = 0
        self.current_text = prompt
        self._rhyme_memory.clear()
        # Cache state
        cache = None
        last_logits = None
        if self.backend.supports_kv():
            cache, last_logits = self.backend.prefill_cache(out_ids)

        for step in range(max_new_tokens):
            tid, phase, used_logits, cache = self._step(last_logits, out_ids, cache)
            tok_str = self.backend.token_str(tid)
            if tok_str is None:
                tok_str = ''

            # Spike handling rules
            if phase == 'spike':
                if self.cfg.defer_spike_on_punct and (is_punct_token(tok_str) or is_newline_token(tok_str)):
                    # don't spend spike; retry soon
                    self.to_next_spike = 1
                else:
                    # valid spike: start cooldown and reschedule next spike
                    self.cooldown_left = self._sample_between(self.cfg.cooldown_range)
                    self.to_next_spike = self._sample_between(self.cfg.interval_range)
                if self.shift_tokens_left > 0:
                    self.shift_tokens_left -= 1

            elif phase == 'cool':
                self.cooldown_left = max(0, self.cooldown_left - 1)
                self.to_next_spike = max(0, self.to_next_spike - 1)
                if self.shift_tokens_left > 0:
                    self.shift_tokens_left -= 1

            else:  # base
                self.to_next_spike = max(0, self.to_next_spike - 1)
                if self.shift_tokens_left > 0:
                    self.shift_tokens_left -= 1

            out_ids.append(tid)

            # Line accounting and sustained shifts
            if is_newline_token(tok_str):
                # record rhyme for finished line
                line_text = self.current_text.split('\n')[-1]
                self._record_rhyme_for_line(line_text)
                self.line_tokens = 0
                self.lines_since_shift += 1
                # occasionally open a sustained shift next line
                if self.lines_since_shift >= self._sample_between(self.cfg.shift_every_lines):
                    self.shift_tokens_left = self._sample_between(self.cfg.shift_span_tokens)
                    self.lines_since_shift = 0
                self.line_index += 1
            else:
                self.line_tokens += 1

            # Update incremental decode string for rhyme tracking
            try:
                if hasattr(self.backend, 'hf_tok') and self.backend.hf_tok is not None:
                    piece = self.backend.hf_tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                elif hasattr(self.backend, 'tok') and hasattr(self.backend.tok, 'decode'):
                    piece = self.backend.tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                else:
                    piece = tok_str
                self.current_text += piece
            except Exception:
                self.current_text += tok_str

            # Advance KV cache if present
            if self.backend.supports_kv():
                last_logits, cache = self.backend.logits_with_cache(tid, cache)
            else:
                last_logits = None

        # Decode
        # Use tokenizer to decode; prefer HF fast tokenizers
        try:
            # Prefer HF fast tokenizer for stable decoding
            if hasattr(self.backend, 'hf_tok') and self.backend.hf_tok is not None:
                text = self.backend.hf_tok.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            elif hasattr(self.backend, 'tok') and hasattr(self.backend.tok, 'decode'):
                try:
                    text = self.backend.tok.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                except TypeError:
                    text = self.backend.tok.decode(out_ids)
            else:
                # naive: join token strings
                text = ''.join(self.backend.token_str(i) for i in out_ids)
        except Exception:
            text = ''.join(self.backend.token_str(i) for i in out_ids)
        return text


# ---- Author-guided tuning and baseline generation --------------------------

def _load_author_stats(model_id: str) -> List[Dict]:
    """Load per-author aggregates for a given model id.

    Tries several plausible locations under data/analysis/ to accommodate
    nested vendors (e.g., Qwen/Qwen2.5-1.5B).
    """
    # Exact nested path
    p = Path('data/analysis') / model_id / 'authors.jsonl'
    if not p.exists():
        parts = model_id.split('/')
        if len(parts) >= 2:
            p = Path('data/analysis') / parts[0] / parts[1] / 'authors.jsonl'
    if not p.exists():
        p = Path('data/analysis') / (model_id.split('/')[-1]) / 'authors.jsonl'
    rows: List[Dict] = []
    try:
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []
    return rows


def _match_author(rows: List[Dict], name: str) -> Optional[Dict]:
    if not rows:
        return None
    want = name.strip().lower()
    # Exact case-insensitive
    for r in rows:
        if str(r.get('author', '')).strip().lower() == want:
            return r
    # Contains match
    for r in rows:
        a = str(r.get('author', '')).strip().lower()
        if want in a or a in want:
            return r
    # Fallback: last token (e.g., "shakespeare")
    want_tok = want.split()[-1]
    for r in rows:
        a = str(r.get('author', '')).strip().lower()
        if want_tok in a.split():
            return r
    return None


def _merge_author_stats(primary: Optional[Dict], secondary: Optional[Dict], weight: float) -> Optional[Dict]:
    if primary is None and secondary is None:
        return None
    if secondary is None or weight <= 0.0:
        return primary
    if primary is None:
        return secondary
    w = min(max(weight, 0.0), 1.0)
    merged = {}
    keys = set(primary.keys()) | set(secondary.keys())
    for k in keys:
        v1 = primary.get(k)
        v2 = secondary.get(k)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            merged[k] = (1.0 - w) * float(v1) + w * float(v2)
        else:
            merged[k] = v2 if w >= 0.5 and v2 is not None else v1
    return merged


def _blend_tuple(lo_hi_base: Tuple[int, int], lo_hi_new: Tuple[int, int], alpha: float) -> Tuple[int, int]:
    a = max(0.0, alpha)
    b0, b1 = lo_hi_base
    n0, n1 = lo_hi_new
    return (
        int(round(b0 + (n0 - b0) * a)),
        int(round(b1 + (n1 - b1) * a)),
    )


def _blend_scalar(base_val: float, new_val: float, alpha: float) -> float:
    a = max(0.0, alpha)
    return float(base_val + (new_val - base_val) * a)


def _blend_poetry_config(base_cfg: PoetryConfig, new_cfg: PoetryConfig, alpha: float) -> PoetryConfig:
    if alpha == 1.0:
        return new_cfg
    a = max(0.0, alpha)
    new_cfg.interval_range = _blend_tuple(base_cfg.interval_range, new_cfg.interval_range, a)
    new_cfg.cooldown_range = _blend_tuple(base_cfg.cooldown_range, new_cfg.cooldown_range, a)
    new_cfg.shift_every_lines = _blend_tuple(base_cfg.shift_every_lines, new_cfg.shift_every_lines, a)
    new_cfg.shift_span_tokens = _blend_tuple(base_cfg.shift_span_tokens, new_cfg.shift_span_tokens, a)
    if base_cfg.line_tokens_target and new_cfg.line_tokens_target:
        new_cfg.line_tokens_target = _blend_tuple(
            base_cfg.line_tokens_target,
            new_cfg.line_tokens_target,
            a,
        )
    new_cfg.newline_penalty_until_target = _blend_scalar(
        base_cfg.newline_penalty_until_target,
        new_cfg.newline_penalty_until_target,
        a,
    )
    new_cfg.newline_penalty_until_rhyme = _blend_scalar(
        base_cfg.newline_penalty_until_rhyme,
        new_cfg.newline_penalty_until_rhyme,
        a,
    )

    for phase_name in ('base', 'spike', 'cool'):
        base_phase: PhaseParams = getattr(base_cfg, phase_name)
        new_phase: PhaseParams = getattr(new_cfg, phase_name)
        new_phase.top_p = _blend_scalar(base_phase.top_p, new_phase.top_p, a)
        new_phase.temperature = _blend_scalar(base_phase.temperature, new_phase.temperature, a)
        if base_phase.top_k is not None and new_phase.top_k is not None:
            new_phase.top_k = int(round(base_phase.top_k + (new_phase.top_k - base_phase.top_k) * a))
        new_phase.content_boost = _blend_scalar(base_phase.content_boost, new_phase.content_boost, a)
        new_phase.stop_punct_penalty = _blend_scalar(base_phase.stop_punct_penalty, new_phase.stop_punct_penalty, a)
    return new_cfg


def _adjust_config_from_author(cfg: PoetryConfig, stats: Dict, *, alpha: float = 1.0) -> PoetryConfig:
    base_cfg = copy.deepcopy(cfg)

    # Inter-peak interval → cadence interval
    ipi = stats.get('ipi_mean_mean') or stats.get('ipi_mean')
    try:
        if ipi and ipi == ipi and float(ipi) > 0:
            m = float(ipi)
            lo = max(3, int(round(0.8 * m)))
            hi = max(lo + 1, int(round(1.2 * m)))
            cfg.interval_range = (lo, hi)
    except Exception:
        pass

    # Cooldown tokens from cooldown entropy drop (heuristic mapping)
    cd = stats.get('cooldown_entropy_drop_3_mean')
    try:
        if cd and cd == cd:
            v = float(cd)
            if v >= 1.4:
                cfg.cooldown_range = (5, 8)
            elif v >= 1.1:
                cfg.cooldown_range = (3, 6)
            else:
                cfg.cooldown_range = (2, 4)
    except Exception:
        pass

    # Content fraction tunes spike content boost
    cf = stats.get('content_fraction_mean')
    try:
        if cf and cf == cf:
            c = float(cf)
            if c >= 0.40:
                cfg.spike.content_boost = 0.30
            elif c >= 0.35:
                cfg.spike.content_boost = 0.25
            else:
                cfg.spike.content_boost = max(0.18, cfg.spike.content_boost)
    except Exception:
        pass

    # Punctuation near spikes → avoid spending spikes there
    sp_prev_punct = stats.get('spike_prev_punct_rate_mean')
    try:
        if sp_prev_punct and sp_prev_punct == sp_prev_punct:
            s = float(sp_prev_punct)
            if s >= 0.25:
                cfg.spike.stop_punct_penalty = 1.8
            elif s >= 0.18:
                cfg.spike.stop_punct_penalty = 1.4
            else:
                cfg.spike.stop_punct_penalty = max(1.0, cfg.spike.stop_punct_penalty)
    except Exception:
        pass

    # Nucleus width → mild focus tuning
    nw = stats.get('nucleus_w_mean_mean')
    try:
        if nw and nw == nw:
            w = float(nw)
            if w <= 100:
                cfg.base.top_p, cfg.cool.top_p = 0.88, 0.82
            elif w <= 180:
                cfg.base.top_p, cfg.cool.top_p = 0.90, 0.84
            else:
                cfg.base.top_p, cfg.cool.top_p = 0.92, 0.86
    except Exception:
        pass

    return _blend_poetry_config(base_cfg, cfg, alpha)


def _generate_baseline(
    backend: ModelBackend,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.85,
    top_p: float = 0.92,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    enc = backend.tokenize(prompt)
    ids = list(enc['input_ids']) if isinstance(enc, dict) or ('input_ids' in enc) else list(enc)
    try:
        ids = [int(x) for x in ids]
    except Exception:
        pass
    cache, last_logits = (None, None)
    if backend.supports_kv():
        cache, last_logits = backend.prefill_cache(ids)
    else:
        last_logits = backend.logits_for_input_ids(ids)[-1]
    out_ids: List[int] = []
    for _ in range(max_new_tokens):
        logits = last_logits
        tid = sample_from_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        out_ids.append(int(tid))
        if backend.supports_kv():
            last_logits, cache = backend.logits_with_cache(tid, cache)
        else:
            last_logits = backend.logits_for_input_ids(ids + out_ids)[-1]
    # Decode
    try:
        if hasattr(backend, 'hf_tok') and backend.hf_tok is not None:
            return backend.hf_tok.decode(ids + out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif hasattr(backend, 'tok') and hasattr(backend.tok, 'decode'):
            return backend.tok.decode(ids + out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception:
        pass
    return ''.join(backend.token_str(i) for i in (ids + out_ids))
# ---- Presets ---------------------------------------------------------------

def poetry_default_preset() -> PoetryConfig:
    return PoetryConfig(
        base=PhaseParams(top_p=0.90, temperature=0.78, top_k=120),
        spike=PhaseParams(top_p=0.97, temperature=1.02, top_k=200, content_boost=0.2, stop_punct_penalty=1.6),
        cool=PhaseParams(top_p=0.84, temperature=0.68, top_k=60),
        interval_range=(8, 16),
        cooldown_range=(3, 8),
        shift_every_lines=(1, 3),
        shift_span_tokens=(8, 16),
        # line targets for poetry
        # these are soft; newline discouraged until min
        # rhyme disabled by default
        # optional nudges present in sonnet/couplets presets
    )


def sonnet_preset() -> PoetryConfig:
    return PoetryConfig(
        base=PhaseParams(top_p=0.90, temperature=0.76, top_k=120),
        spike=PhaseParams(top_p=0.975, temperature=1.05, top_k=220, content_boost=0.25, stop_punct_penalty=1.8),
        cool=PhaseParams(top_p=0.83, temperature=0.66, top_k=60),
        interval_range=(10, 16),
        cooldown_range=(3, 6),
        shift_every_lines=(2, 3),
        shift_span_tokens=(10, 16),
        line_tokens_target=(8, 12),
        newline_penalty_until_target=2.0,
        rhyme_enabled=True,
        rhyme_scheme="ABAB CDCD EFEF GG",
        rhyme_boost=0.8,
        rhyme_min_line_tokens=6,
        newline_penalty_until_rhyme=1.2,
    )


def dickinson_preset() -> PoetryConfig:
    return PoetryConfig(
        base=PhaseParams(top_p=0.90, temperature=0.78, top_k=120),
        spike=PhaseParams(top_p=0.98, temperature=1.08, top_k=240, content_boost=0.3, stop_punct_penalty=1.2),
        cool=PhaseParams(top_p=0.85, temperature=0.64, top_k=50),
        interval_range=(6, 12),
        cooldown_range=(3, 5),
        shift_every_lines=(1, 2),
        shift_span_tokens=(6, 12),
        line_tokens_target=(6, 10),
        newline_penalty_until_target=1.2,
        rhyme_enabled=False,
    )


def freeverse_preset() -> PoetryConfig:
    return PoetryConfig(
        base=PhaseParams(top_p=0.90, temperature=0.80, top_k=140),
        spike=PhaseParams(top_p=0.97, temperature=1.04, top_k=220, content_boost=0.22, stop_punct_penalty=1.4),
        cool=PhaseParams(top_p=0.85, temperature=0.70, top_k=70),
        interval_range=(9, 14),
        cooldown_range=(4, 8),
        shift_every_lines=(1, 2),
        shift_span_tokens=(10, 18),
        line_tokens_target=(8, 14),
        newline_penalty_until_target=1.0,
        rhyme_enabled=False,
    )


def couplets_preset() -> PoetryConfig:
    return PoetryConfig(
        base=PhaseParams(top_p=0.90, temperature=0.78, top_k=120),
        spike=PhaseParams(top_p=0.97, temperature=1.02, top_k=220, content_boost=0.24, stop_punct_penalty=1.6),
        cool=PhaseParams(top_p=0.84, temperature=0.68, top_k=60),
        interval_range=(8, 14),
        cooldown_range=(3, 7),
        shift_every_lines=(1, 2),
        shift_span_tokens=(8, 14),
        line_tokens_target=(8, 12),
        newline_penalty_until_target=1.4,
        rhyme_enabled=True,
        rhyme_scheme="AA",
        rhyme_boost=0.9,
        rhyme_min_line_tokens=6,
        newline_penalty_until_rhyme=1.2,
    )


PRESETS = {
    'poetry_default': poetry_default_preset,
    'sonnet': sonnet_preset,
    'dickinson': dickinson_preset,
    'freeverse': freeverse_preset,
    'couplets': couplets_preset,
}


def prose_preset() -> PoetryConfig:
    # Longer intervals, gentle spikes, no rhyme; prose-like cadence
    return PoetryConfig(
        base=PhaseParams(top_p=0.90, temperature=0.78, top_k=120),
        spike=PhaseParams(top_p=0.95, temperature=1.03, top_k=180, content_boost=0.15, stop_punct_penalty=0.6),
        cool=PhaseParams(top_p=0.86, temperature=0.66, top_k=80),
        interval_range=(14, 24),
        cooldown_range=(5, 10),
        shift_every_lines=(2, 4),
        shift_span_tokens=(16, 28),
        rhyme_enabled=False,
    )


# Register after definition to avoid forward reference issues
PRESETS['prose'] = prose_preset


# ---- CLI -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Cadence-based poetry sampler (MLX/HF)')
    ap.add_argument('model', type=str, help='Model id (HF or MLX format)')
    ap.add_argument('--backend', default='auto', choices=['auto', 'mlx', 'hf'], help='Backend preference')
    ap.add_argument('--preset', default='poetry_default', choices=sorted(PRESETS.keys()))
    ap.add_argument('--prompt', type=str, default='', help='Prompt text')
    ap.add_argument('--max-new-tokens', type=int, default=120)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--debug', action='store_true')
    # Rhyme overrides
    ap.add_argument('--rhyme', action='store_true', help='Enable rhyme nudging (overrides preset off)')
    ap.add_argument('--rhyme-scheme', type=str, default=None, help='Rhyme scheme letters (e.g., "ABAB CDCD EFEF GG" or "AA" for couplets)')
    ap.add_argument('--rhyme-boost', type=float, default=None, help='Boost for rhyme-matching tokens')
    ap.add_argument('--line-target', type=str, default=None, help='Target line token range, e.g., "8,12"')
    # Author-guided tuning
    ap.add_argument('--author-seed', type=str, default=None, help='Author name to guide cadence from analysis stats (case-insensitive)')
    ap.add_argument('--secondary-author-seed', type=str, default=None, help='Optional second author to blend stats from')
    ap.add_argument('--author-mix', type=float, default=0.0, help='Blend weight for secondary author (0-1)')
    ap.add_argument('--author-strength', type=float, default=1.0, help='Strength multiplier for author stats influence (0 disables)')
    ap.add_argument('--author-stats-model', type=str, default=None, help='Model id to read author stats from (defaults to --model)')
    # Baseline comparison
    ap.add_argument('--also-baseline', action='store_true', help='Also generate a plain baseline sample alongside the cadence-controlled one')
    ap.add_argument('--baseline-temp', type=float, default=0.85)
    ap.add_argument('--baseline-top-p', type=float, default=0.92)
    ap.add_argument('--baseline-top-k', type=int, default=None)
    ap.add_argument('--save', action='store_true', default=True, help='Save output under data/generated')
    args = ap.parse_args()

    backend = pick_backend(args.model, prefer_mlx=True, backend=args.backend)
    cfg = PRESETS[args.preset]()
    # Apply overrides
    if args.rhyme:
        setattr(cfg, 'rhyme_enabled', True)
    if args.rhyme_scheme is not None:
        setattr(cfg, 'rhyme_scheme', args.rhyme_scheme)
        setattr(cfg, 'rhyme_enabled', True)
    if args.rhyme_boost is not None:
        setattr(cfg, 'rhyme_boost', float(args.rhyme_boost))
    if args.line_target:
        try:
            a, b = args.line_target.split(',')
            setattr(cfg, 'line_tokens_target', (int(a), int(b)))
        except Exception:
            pass
    # Author-guided tuning from analysis stats
    if args.author_seed or args.secondary_author_seed:
        stats_model = args.author_stats_model or args.model
        rows = _load_author_stats(stats_model)
        primary = _match_author(rows, args.author_seed) if args.author_seed else None
        secondary = _match_author(rows, args.secondary_author_seed) if args.secondary_author_seed else None
        mix = float(args.author_mix or 0.0)
        stats = _merge_author_stats(primary, secondary, mix)
        if stats:
            cfg = _adjust_config_from_author(cfg, stats, alpha=float(max(0.0, args.author_strength)))
        elif args.debug:
            who = args.author_seed or args.secondary_author_seed
            print(f"[DEBUG] No author stats found for '{who}' in {stats_model}")

    sampler = CadenceSampler(backend, cfg, seed=args.seed, debug=args.debug)

    prompt = args.prompt
    if not prompt:
        prompt = "Write a short poem about dawn in the city.\n"

    fixed_text = sampler.generate(prompt, max_new_tokens=args.max_new_tokens)
    # Optional baseline
    base_text: Optional[str] = None
    if args.also_baseline:
        base_text = _generate_baseline(
            backend,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.baseline_temp,
            top_p=args.baseline_top_p,
            top_k=args.baseline_top_k,
            seed=args.seed,
        )
    # Print
    if args.also_baseline and base_text is not None:
        print('--- Normal ---')
        print(base_text)
        print('\n--- Fixed-Up ---')
        print(fixed_text)
    else:
        print(fixed_text)
    # Save
    if args.save:
        from pathlib import Path
        import json, hashlib
        def _slugify(s: str) -> str:
            s = s.strip().lower().replace('\n', ' ')
            s = ' '.join(s.split())
            keep = []
            for ch in s:
                if ch.isalnum(): keep.append(ch)
                elif ch in (' ','-','_'): keep.append('_')
            slug = ''.join(keep).strip('_') or 'prompt'
            return slug[:64]
        def _safe_model_id(mid: str) -> str:
            return '_'.join([p for p in mid.replace('/', '_').split('_') if p])
        model_dir = _safe_model_id(args.model)
        prompt_slug = _slugify(prompt)
        h = hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:8]
        tag = (args.author_seed or '').strip().lower().replace(' ', '_')
        preset_leaf = f'sampler_{args.preset}' + (f'_author_{tag}' if tag else '')
        base_dir = Path('data/generated') / model_dir / preset_leaf / f'{prompt_slug}_{h}'
        base_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            'model': args.model,
            'preset': f'sampler_{args.preset}',
            'seed': args.seed,
            'max_new_tokens': args.max_new_tokens,
            'prompt': args.prompt,
        }
        if args.also_baseline and base_text is not None:
            (base_dir / 'baseline.txt').write_text(base_text, encoding='utf-8')
            (base_dir / 'fixed.txt').write_text(fixed_text, encoding='utf-8')
            meta['paths'] = {
                'baseline': str(base_dir / 'baseline.txt'),
                'fixed': str(base_dir / 'fixed.txt'),
            }
        else:
            (base_dir / 'output.txt').write_text(fixed_text, encoding='utf-8')
            meta['paths'] = {'text': str(base_dir / 'output.txt')}
        if args.author_seed:
            meta['author_seed'] = args.author_seed
            if args.author_stats_model:
                meta['author_stats_model'] = args.author_stats_model
        (base_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        idx = Path('data/generated/index.jsonl')
        idx.parent.mkdir(parents=True, exist_ok=True)
        with idx.open('a', encoding='utf-8') as f:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        # Update unified page
        try:
            from tools.show_generated import load_index, make_report  # type: ignore
            entries = load_index(Path('data/generated/index.jsonl'))
            report = make_report(entries)
            out_path = Path('reports/generated/README.md')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report, encoding='utf-8')
        except Exception:
            pass


if __name__ == '__main__':
    main()
