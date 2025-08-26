#!/usr/bin/env python3
import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)


def is_punct_token(tok: str) -> bool:
    s = tok.strip()
    if not s:
        return True
    for ch in s:
        if ch.isalnum():
            return False
    return True


def is_newline_token(tok: str) -> bool:
    return "\n" in tok


def is_content_token(tok: str) -> bool:
    s = tok.strip()
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    low = ''.join(letters).lower()
    # small stoplist
    stop = {
        'a','an','the','and','or','but','if','then','else','when','while','for','of','in','on','to','with','as','is','are','was','were','be','been','being','do','does','did','have','has','had','i','me','my','we','us','our','you','your','he','him','his','she','her','it','its','they','them','their','this','that','these','those','who','whom','which','what','where','why','how','not','no','only','also','very','just','too'
    }
    return low not in stop


def rhyme_key(word: str) -> str:
    w = ''.join([c for c in word.lower() if c.isalpha()])
    if not w:
        return ''
    vowels = set('aeiouy')
    end = len(w)
    i = end - 1
    while i >= 0 and w[i] not in vowels:
        i -= 1
    while i >= 0 and w[i] in vowels:
        i -= 1
    start = i + 1
    key = w[start:end]
    if len(key) < 3:
        key = w[-3:] if len(w) >= 3 else w
    return key


def last_word_from_ids(tokenizer, ids: List[int]) -> str:
    # decode last ~6 tokens to get a stable last word
    tail = ids[-6:]
    try:
        s = tokenizer.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except TypeError:
        s = tokenizer.decode(tail)
    import re
    m = re.findall(r"[A-Za-z']+", s)
    return m[-1] if m else ''


@dataclass
class PhaseParams:
    temperature: float
    top_p: float
    content_boost: float = 0.0
    stop_punct_penalty: float = 0.0


@dataclass
class CadenceConfig:
    base: PhaseParams
    spike: PhaseParams
    cool: PhaseParams
    interval_range: Tuple[int, int] = (8, 16)
    cooldown_range: Tuple[int, int] = (3, 8)
    defer_spike_on_punct: bool = True
    # rhyme/line control
    rhyme_enabled: bool = False
    rhyme_scheme: Optional[str] = None
    rhyme_boost: float = 0.0
    rhyme_min_line_tokens: int = 6
    line_tokens_target: Tuple[int, int] = (8, 12)
    newline_penalty_until_target: float = 0.0
    newline_penalty_until_rhyme: float = 0.0
    # anti-meta bias
    anti_meta_penalty: float = 0.0
    # repetition controls
    rep_window: int = 32
    rep_penalty: float = 0.6
    no_repeat_ngram: int = 3
    # diversity bonus on spikes (implemented as extra penalty to recently seen tokens)
    div_window: int = 40
    div_bonus: float = 0.15


class CadenceProcessor(LogitsProcessor):
    def __init__(self, tokenizer, cfg: CadenceConfig):
        self.tok = tokenizer
        self.cfg = cfg
        self.is_punct = np.zeros(0, dtype=bool)
        self.is_newline = np.zeros(0, dtype=bool)
        self.is_content = np.zeros(0, dtype=bool)
        self.anti_meta = np.zeros(0, dtype=bool)
        # state
        self.to_next_spike = random.randint(*cfg.interval_range)
        self.cooldown_left = 0
        self.line_tokens = 0
        self.line_index = 0
        self.lines_since_shift = 0
        self.shift_tokens_left = 0
        self.rhyme_memory: Dict[str, str] = {}
        self.prev_phase = 'base'

        self.scheme_letters = None
        if cfg.rhyme_scheme:
            letters = [ch for ch in cfg.rhyme_scheme if ch.isalpha()]
            self.scheme_letters = letters if letters else None

    def _phase(self) -> str:
        if self.cooldown_left > 0:
            return 'cool'
        if self.shift_tokens_left > 0:
            return 'spike'
        if self.to_next_spike <= 0:
            return 'spike'
        return 'base'

    def _letter_for_line(self) -> Optional[str]:
        if not self.scheme_letters:
            return None
        L = len(self.scheme_letters)
        return self.scheme_letters[self.line_index % L]

    def _target_rhyme(self) -> Optional[str]:
        key = self._letter_for_line()
        if not key:
            return None
        return self.rhyme_memory.get(key)

    def _record_rhyme(self, last_word: str):
        key = self._letter_for_line()
        if not key:
            return
        rk = rhyme_key(last_word)
        if rk:
            self.rhyme_memory[key] = rk

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # ensure masks match current vocab dimension
        V = scores.shape[-1]
        if self.is_punct.shape[0] != V:
            self.is_punct = np.zeros(V, dtype=bool)
            self.is_newline = np.zeros(V, dtype=bool)
            self.is_content = np.zeros(V, dtype=bool)
            self.anti_meta = np.zeros(V, dtype=bool)
            for tid in range(V):
                try:
                    t = self.tok.convert_ids_to_tokens([tid])[0]
                except Exception:
                    t = ''
                t = t or ''
                nl = is_newline_token(t)
                self.is_newline[tid] = nl
                self.is_punct[tid] = nl or is_punct_token(t)
                self.is_content[tid] = (not self.is_punct[tid]) and is_content_token(t)
                # anti-meta terms
                low = t.lower()
                for frag in (
                    'poem','poet','line','stanza','rhyme','metaphor','example','great','love','prompt','instruction',
                    'answer','question','which','following','correct','choose','option','multiple choice','true or false'
                ):
                    if frag in low:
                        self.anti_meta[tid] = True
                        break

        # schedule update based on last generated token and prior phase
        last_id = int(input_ids[0, -1].item()) if input_ids.dim() == 2 else int(input_ids[-1].item())
        last_is_nl = last_id < len(self.is_newline) and self.is_newline[last_id]
        last_is_punct = last_id < len(self.is_punct) and self.is_punct[last_id]

        # Handle line accounting first
        if last_is_nl:
            lw = last_word_from_ids(self.tok, input_ids[0].tolist())
            self._record_rhyme(lw)
            self.line_index += 1
            self.line_tokens = 0
            self.lines_since_shift += 1
            if self.lines_since_shift >= random.randint(1, 3):
                self.shift_tokens_left = random.randint(8, 16)
                self.lines_since_shift = 0
        else:
            self.line_tokens += 1

        # Update cadence counters using the actual last token and previous phase
        if self.prev_phase == 'spike':
            if last_is_punct:
                # defer spike; try again almost immediately
                self.to_next_spike = 1
                # do not start cooldown yet
            else:
                self.cooldown_left = random.randint(*self.cfg.cooldown_range)
                self.to_next_spike = random.randint(*self.cfg.interval_range)
                if self.shift_tokens_left > 0:
                    self.shift_tokens_left -= 1
        elif self.prev_phase == 'cool':
            self.cooldown_left = max(0, self.cooldown_left - 1)
            self.to_next_spike = max(0, self.to_next_spike - 1)
            if self.shift_tokens_left > 0:
                self.shift_tokens_left -= 1
        else:  # base
            self.to_next_spike = max(0, self.to_next_spike - 1)
            if self.shift_tokens_left > 0:
                self.shift_tokens_left -= 1

        # schedule
        phase = self._phase()
        par = getattr(self.cfg, phase)

        # make a copy to mutate
        logits = scores
        # per-phase temperature scaling (divide by temp)
        if par.temperature and abs(par.temperature - 1.0) > 1e-6:
            logits = logits / float(par.temperature)
        # bias punctuation/content
        if par.stop_punct_penalty:
            mask = torch.from_numpy(self.is_punct).to(logits.device).unsqueeze(0)
            logits = logits + (mask.float() * (-float(par.stop_punct_penalty)))
        if par.content_boost:
            mask = torch.from_numpy(self.is_content).to(logits.device).unsqueeze(0)
            logits = logits + (mask.float() * float(par.content_boost))

        # per-phase nucleus masking (enforce top_p per phase)
        if par.top_p and 0.0 < par.top_p < 1.0:
            # compute softmax probs along vocab dim
            # subtract max for stability
            l = logits - logits.max(dim=-1, keepdim=True).values
            p = torch.exp(l)
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
            # sort descending
            p_sorted, idx = torch.sort(p, dim=-1, descending=True)
            cdf = torch.cumsum(p_sorted, dim=-1)
            # find cut where cdf >= top_p
            cutoff = (cdf >= par.top_p).float().argmax(dim=-1)
            # build keep mask
            keep = torch.zeros_like(p, dtype=torch.bool)
            for b in range(p.shape[0]):
                k = int(cutoff[b].item()) + 1
                keep_idx = idx[b, :k]
                keep[b, keep_idx] = True
            # mask tail
            big_neg = torch.finfo(logits.dtype).min / 4
            logits = torch.where(keep, logits, torch.full_like(logits, big_neg))

        # line/newline control + rhyme bias near line end
        if self.cfg.rhyme_enabled and self.line_tokens >= self.cfg.rhyme_min_line_tokens:
            target = self._target_rhyme()
            if target:
                frag = target[-3:]
                # only nudge near line end to improve end-rhyme
                near_line_end = self.line_tokens >= max(1, self.cfg.line_tokens_target[0] - 2)
                if frag and self.cfg.rhyme_boost and near_line_end:
                    # boost top-K candidates that contain the fragment
                    K = min(256, logits.shape[-1])
                    topk = torch.topk(logits, K)
                    ids = topk.indices.tolist()
                    for j in ids:
                        ts = self.tok.convert_ids_to_tokens([j])[0].lower()
                        if frag in ts and ts.strip():
                            logits[0, j] = logits[0, j] + float(self.cfg.rhyme_boost)
                # discourage newline until rhyme satisfied
                lw = last_word_from_ids(self.tok, input_ids[0].tolist())
                if rhyme_key(lw) != target and self.cfg.newline_penalty_until_rhyme:
                    nl_mask = torch.from_numpy(self.is_newline).to(logits.device).unsqueeze(0)
                    logits = logits + (nl_mask.float() * (-float(self.cfg.newline_penalty_until_rhyme)))

        # discourage newline until minimum line length
        if self.cfg.newline_penalty_until_target and self.line_tokens < self.cfg.line_tokens_target[0]:
            nl_mask = torch.from_numpy(self.is_newline).to(logits.device).unsqueeze(0)
            logits = logits + (nl_mask.float() * (-float(self.cfg.newline_penalty_until_target)))

        # remember phase used for this scoring step; we will finalize schedule on next call using the sampled token
        self.prev_phase = phase
        # anti-meta nudging (apply after other adjustments; avoid during spikes)
        if self.cfg.anti_meta_penalty and phase != 'spike':
            mask = torch.from_numpy(self.anti_meta).to(logits.device).unsqueeze(0)
            logits = logits + (mask.float() * (-float(self.cfg.anti_meta_penalty)))
        return logits


def default_cadence() -> CadenceConfig:
    return CadenceConfig(
        base=PhaseParams(temperature=0.80, top_p=0.90, content_boost=0.0, stop_punct_penalty=0.0),
        spike=PhaseParams(temperature=1.05, top_p=0.97, content_boost=0.25, stop_punct_penalty=1.2),
        cool=PhaseParams(temperature=0.68, top_p=0.84, content_boost=0.0, stop_punct_penalty=0.0),
        interval_range=(8, 16),
        cooldown_range=(3, 7),
        rhyme_enabled=False,
        anti_meta_penalty=0.8,
        rep_window=40,
        rep_penalty=0.8,
        no_repeat_ngram=3,
        div_window=40,
        div_bonus=0.15,
    )


def sonnet_cadence() -> CadenceConfig:
    cfg = default_cadence()
    cfg.rhyme_enabled = True
    cfg.rhyme_scheme = "ABAB CDCD EFEF GG"
    cfg.rhyme_boost = 0.8
    cfg.newline_penalty_until_target = 2.0
    cfg.newline_penalty_until_rhyme = 1.2
    cfg.line_tokens_target = (8, 12)
    return cfg


def couplets_cadence() -> CadenceConfig:
    cfg = default_cadence()
    cfg.rhyme_enabled = True
    cfg.rhyme_scheme = "AA"
    cfg.rhyme_boost = 0.9
    cfg.newline_penalty_until_target = 1.4
    cfg.newline_penalty_until_rhyme = 1.2
    cfg.line_tokens_target = (8, 12)
    return cfg


def imagist_cadence() -> CadenceConfig:
    cfg = default_cadence()
    cfg.interval_range = (9, 14)
    cfg.cooldown_range = (4, 8)
    cfg.base = PhaseParams(temperature=0.80, top_p=0.90, content_boost=0.05, stop_punct_penalty=0.0)
    cfg.spike = PhaseParams(temperature=1.06, top_p=0.97, content_boost=0.3, stop_punct_penalty=1.0)
    cfg.cool = PhaseParams(temperature=0.66, top_p=0.84, content_boost=0.0, stop_punct_penalty=0.0)
    cfg.rhyme_enabled = False
    cfg.line_tokens_target = (7, 14)
    cfg.rep_window = 40
    cfg.rep_penalty = 0.9
    cfg.no_repeat_ngram = 3
    cfg.div_window = 40
    cfg.div_bonus = 0.2
    return cfg


def prose_cadence() -> CadenceConfig:
    cfg = default_cadence()
    cfg.interval_range = (14, 24)
    cfg.cooldown_range = (5, 10)
    cfg.base = PhaseParams(temperature=0.78, top_p=0.90, content_boost=0.0, stop_punct_penalty=0.0)
    cfg.spike = PhaseParams(temperature=1.03, top_p=0.95, content_boost=0.2, stop_punct_penalty=0.8)
    cfg.cool = PhaseParams(temperature=0.66, top_p=0.86, content_boost=0.0, stop_punct_penalty=0.0)
    cfg.rhyme_enabled = False
    cfg.newline_penalty_until_target = 0.0
    cfg.newline_penalty_until_rhyme = 0.0
    cfg.line_tokens_target = (0, 0)
    cfg.anti_meta_penalty = 1.0
    cfg.rep_window = 40
    cfg.rep_penalty = 0.6
    cfg.no_repeat_ngram = 3
    cfg.div_window = 40
    cfg.div_bonus = 0.1
    return cfg


def run_compare(model_id: str, prompt: str, max_new_tokens: int, seed: Optional[int], preset: str = 'default'):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    inputs = tok(prompt, return_tensors='pt')

    # Baseline
    out_base = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_p=0.92,
        temperature=0.85,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    base_text = tok.decode(out_base[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Fixed-up with cadence (logits-processor path)
    if preset == 'default':
        cfg = default_cadence()
    elif preset == 'sonnet':
        cfg = sonnet_cadence()
    elif preset == 'couplets':
        cfg = couplets_cadence()
    elif preset == 'imagist':
        cfg = imagist_cadence()
    elif preset == 'prose':
        cfg = prose_cadence()
    else:
        cfg = default_cadence()
    proc = CadenceProcessor(tok, cfg)
    processors = LogitsProcessorList([proc])
    out_fix = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_p=0.92,
        temperature=1.0,  # per-step temp handled in processor
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
        logits_processor=processors,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    fix_text = tok.decode(out_fix[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return base_text, fix_text


# ---- Manual token-aware fixed-up loop -------------------------------------

class ManualCadence:
    def __init__(self, tok, cfg: CadenceConfig):
        self.tok = tok
        self.cfg = cfg
        self.is_punct = None
        self.is_newline = None
        self.is_content = None
        self.anti_meta = None
        # state
        self.to_next_spike = random.randint(*cfg.interval_range)
        self.cooldown_left = 0
        self.line_tokens = 0
        self.line_index = 0
        self.lines_since_shift = 0
        self.shift_tokens_left = 0
        self.prev_phase = 'base'
        self.scheme_letters = None
        if cfg.rhyme_scheme:
            letters = [ch for ch in cfg.rhyme_scheme if ch.isalpha()]
            self.scheme_letters = letters if letters else None
        self.rhyme_memory: Dict[str, str] = {}

    def _ensure_masks(self, V: int):
        if self.is_punct is not None and len(self.is_punct) == V:
            return
        self.is_punct = np.zeros(V, dtype=bool)
        self.is_newline = np.zeros(V, dtype=bool)
        self.is_content = np.zeros(V, dtype=bool)
        self.anti_meta = np.zeros(V, dtype=bool)
        for tid in range(V):
            try:
                t = self.tok.convert_ids_to_tokens([tid])[0]
            except Exception:
                t = ''
            t = t or ''
            nl = is_newline_token(t)
            self.is_newline[tid] = nl
            self.is_punct[tid] = nl or is_punct_token(t)
            self.is_content[tid] = (not self.is_punct[tid]) and is_content_token(t)
            low = str(t).lower()
            for frag in (
                'poem','poet','line','stanza','rhyme','metaphor','example','great','love','prompt','instruction',
                'answer','question','which','following','correct','choose','option','multiple choice','true or false'
            ):
                if frag in low:
                    self.anti_meta[tid] = True
                    break

    def _letter_for_line(self) -> Optional[str]:
        if not self.scheme_letters:
            return None
        L = len(self.scheme_letters)
        return self.scheme_letters[self.line_index % L]

    def _target_rhyme(self) -> Optional[str]:
        key = self._letter_for_line()
        return self.rhyme_memory.get(key) if key else None

    def _record_rhyme(self, last_word: str):
        key = self._letter_for_line()
        if not key:
            return
        rk = rhyme_key(last_word)
        if rk:
            self.rhyme_memory[key] = rk

    def _phase(self) -> str:
        if self.cooldown_left > 0:
            return 'cool'
        if self.shift_tokens_left > 0:
            return 'spike'
        if self.to_next_spike <= 0:
            return 'spike'
        return 'base'

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if not (0.0 < top_p < 1.0):
            return logits
        l = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.exp(l)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
        p_sorted, idx = torch.sort(p, dim=-1, descending=True)
        cdf = torch.cumsum(p_sorted, dim=-1)
        cutoff = (cdf >= top_p).float().argmax(dim=-1)
        keep = torch.zeros_like(p, dtype=torch.bool)
        for b in range(p.shape[0]):
            k = int(cutoff[b].item()) + 1
            keep_idx = idx[b, :k]
            keep[b, keep_idx] = True
        big_neg = torch.finfo(logits.dtype).min / 4
        return torch.where(keep, logits, torch.full_like(logits, big_neg))

    def step(self, logits: torch.Tensor, input_ids: List[int]) -> int:
        V = logits.shape[-1]
        self._ensure_masks(V)
        phase = self._phase()
        par = getattr(self.cfg, phase)

        # temperature
        if par.temperature and abs(par.temperature - 1.0) > 1e-6:
            logits = logits / float(par.temperature)
        # punctuation/content bias
        if par.stop_punct_penalty:
            mask = torch.from_numpy(self.is_punct).to(logits.device).unsqueeze(0)
            logits = logits + (mask.float() * (-float(par.stop_punct_penalty)))
        if par.content_boost:
            mask = torch.from_numpy(self.is_content).to(logits.device).unsqueeze(0)
            logits = logits + (mask.float() * float(par.content_boost))
        # rhyme near line end
        if self.cfg.rhyme_enabled and self.line_tokens >= self.cfg.rhyme_min_line_tokens:
            target = self._target_rhyme()
            near_line_end = self.line_tokens >= max(1, self.cfg.line_tokens_target[0] - 2)
            if target and near_line_end:
                frag = target[-3:]
                if frag and self.cfg.rhyme_boost:
                    K = min(256, V)
                    topk = torch.topk(logits, K)
                    ids = topk.indices.tolist()
                    for j in ids:
                        ts = self.tok.convert_ids_to_tokens([j])[0].lower()
                        if frag in ts and ts.strip():
                            logits[0, j] = logits[0, j] + float(self.cfg.rhyme_boost)
                lw = last_word_from_ids(self.tok, input_ids)
                if rhyme_key(lw) != target and self.cfg.newline_penalty_until_rhyme:
                    nl_mask = torch.from_numpy(self.is_newline).to(logits.device).unsqueeze(0)
                    logits = logits + (nl_mask.float() * (-float(self.cfg.newline_penalty_until_rhyme)))
        # discourage early newline
        if self.cfg.newline_penalty_until_target and self.line_tokens < self.cfg.line_tokens_target[0]:
            nl_mask = torch.from_numpy(self.is_newline).to(logits.device).unsqueeze(0)
            logits = logits + (nl_mask.float() * (-float(self.cfg.newline_penalty_until_target)))
        # anti-meta (not during spikes)
        if self.cfg.anti_meta_penalty and phase != 'spike':
            mask = torch.from_numpy(self.anti_meta).to(logits.device).unsqueeze(0)
            logits = logits + (mask.float() * (-float(self.cfg.anti_meta_penalty)))
        # per-phase top_p
        logits = self._apply_top_p(logits, par.top_p)
        # repetition penalty (windowed)
        if self.cfg.rep_window and self.cfg.rep_penalty:
            window = input_ids[-self.cfg.rep_window:] if self.cfg.rep_window > 0 else []
            if window:
                mask_ids = list(set(window))
                mask = torch.zeros_like(logits)
                for mid in mask_ids:
                    if 0 <= mid < V:
                        mask[0, mid] = 1.0
                logits = logits + (mask * (-float(self.cfg.rep_penalty)))
        # no-repeat n-gram ban
        n = int(self.cfg.no_repeat_ngram or 0)
        if n >= 2 and len(input_ids) >= n - 1:
            hist = input_ids
            # build map of (n-1)-gram -> set(next)
            next_map = {}
            for i in range(len(hist) - n + 1):
                key = tuple(hist[i:i + n - 1])
                nxt = hist[i + n - 1]
                s = next_map.get(key)
                if s is None:
                    s = set()
                    next_map[key] = s
                s.add(nxt)
            key = tuple(hist[-(n - 1):])
            if key in next_map:
                banned = list(next_map[key])
                if banned:
                    big_neg = torch.finfo(logits.dtype).min / 4
                    for bid in banned:
                        if 0 <= bid < V:
                            logits[0, bid] = big_neg
        # sample
        probs = torch.softmax(logits, dim=-1)
        tid = int(torch.multinomial(probs[0], num_samples=1).item())
        return tid

    def update_after(self, tid: int):
        # Update schedule based on chosen token
        last_is_nl = bool(self.is_newline[tid]) if self.is_newline is not None and tid < len(self.is_newline) else False
        last_is_punct = bool(self.is_punct[tid]) if self.is_punct is not None and tid < len(self.is_punct) else False
        if last_is_nl:
            self.line_index += 1
            self.line_tokens = 0
            self.lines_since_shift += 1
            if self.lines_since_shift >= random.randint(1, 3):
                self.shift_tokens_left = random.randint(8, 16)
                self.lines_since_shift = 0
        else:
            self.line_tokens += 1

        if self.prev_phase == 'spike':
            if last_is_punct:
                self.to_next_spike = 1
            else:
                self.cooldown_left = random.randint(*self.cfg.cooldown_range)
                self.to_next_spike = random.randint(*self.cfg.interval_range)
                if self.shift_tokens_left > 0:
                    self.shift_tokens_left -= 1
        elif self.prev_phase == 'cool':
            self.cooldown_left = max(0, self.cooldown_left - 1)
            self.to_next_spike = max(0, self.to_next_spike - 1)
            if self.shift_tokens_left > 0:
                self.shift_tokens_left -= 1
        else:
            self.to_next_spike = max(0, self.to_next_spike - 1)
            if self.shift_tokens_left > 0:
                self.shift_tokens_left -= 1

    def remember_phase(self):
        self.prev_phase = self._phase()


def run_fixed_manual(model_id: str, prompt: str, max_new_tokens: int, seed: Optional[int], preset: str) -> str:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    if preset == 'default':
        cfg = default_cadence()
    elif preset == 'sonnet':
        cfg = sonnet_cadence()
    elif preset == 'couplets':
        cfg = couplets_cadence()
    elif preset == 'imagist':
        cfg = imagist_cadence()
    elif preset == 'prose':
        cfg = prose_cadence()
    else:
        cfg = default_cadence()

    enc = tok(prompt, return_tensors='pt')
    input_ids = enc['input_ids']
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values if hasattr(out, 'past_key_values') else out.past
        logits = out.logits[:, -1, :]
    ids = input_ids[0].tolist()
    mc = ManualCadence(tok, cfg)
    text_ids = ids[:]

    for _ in range(max_new_tokens):
        # Remember phase used to score current step
        mc.remember_phase()
        tid = mc.step(logits, text_ids)
        text_ids.append(tid)
        # Update rhyme memory on newline (use decoded last word)
        if tid == tok.eos_token_id:
            break
        # advance model with KV cache
        with torch.no_grad():
            next_ids = torch.tensor([[tid]], dtype=torch.long)
            out = model(input_ids=next_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values if hasattr(out, 'past_key_values') else out.past
            logits = out.logits[:, -1, :]
        mc.update_after(tid)

    text = tok.decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text


def _slugify(s: str) -> str:
    s = s.strip().lower().replace('\n', ' ')
    s = ' '.join(s.split())
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in (' ', '-', '_'):
            keep.append('_')
    slug = ''.join(keep).strip('_')
    if not slug:
        slug = 'prompt'
    return slug[:64]


def _safe_model_id(mid: str) -> str:
    parts = [p for p in mid.replace('/', '_').split('_') if p]
    return '_'.join(parts)


def main():
    ap = argparse.ArgumentParser(description='Compare normal vs cadence-controlled generation (HF)')
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    ap.add_argument('--prompt', default='At dawn, the city leans into light:\n')
    ap.add_argument('--max-new-tokens', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--preset', default='default', choices=['default','sonnet','couplets','imagist','prose'])
    ap.add_argument('--save', action='store_true', default=True, help='Save outputs to data/generated')
    ap.add_argument('--out-dir', default=None, help='Optional output dir override')
    ap.add_argument('--manual-fixed', action='store_true', help='Use manual token-aware loop for fixed-up output')
    ap.add_argument('--variants', type=int, default=1, help='Number of fixed-up variants to sample (manual mode only)')
    ap.add_argument('--task', default=None, help='Optional directive to prepend (e.g., "Write an imagist poem about ...")')
    args = ap.parse_args()

    prompt = args.prompt
    if args.task:
        prompt = f"{args.task.strip()}\n\n{prompt}"

    if args.manual_fixed:
        base_text, _ = run_compare(args.model, prompt, args.max_new_tokens, args.seed, args.preset)
        if args.variants and args.variants > 1:
            fixed_variants = []
            for i in range(args.variants):
                vseed = (args.seed or 0) + i + 1
                fx = run_fixed_manual(args.model, prompt, args.max_new_tokens, vseed, args.preset)
                fixed_variants.append(fx)
            # display first variant in console, but save all
            fix_text = fixed_variants[0]
        else:
            fixed_variants = None
            fix_text = run_fixed_manual(args.model, prompt, args.max_new_tokens, args.seed, args.preset)
    else:
        base_text, fix_text = run_compare(args.model, prompt, args.max_new_tokens, args.seed, args.preset)

    print('--- Normal ---')
    print(base_text)
    print('\n--- Fixed-Up ---')
    print(fix_text)

    if args.save:
        model_dir = _safe_model_id(args.model)
        prompt_slug = _slugify(prompt)
        h = hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:8]
        leaf = f"{prompt_slug}_{args.preset}_{h}"
        base_dir = Path(args.out_dir) if args.out_dir else Path('data/generated') / model_dir / args.preset / leaf
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / 'baseline.txt').write_text(base_text, encoding='utf-8')
        (base_dir / 'fixed.txt').write_text(fix_text, encoding='utf-8')
        fixed_list = None
        if args.manual_fixed and args.variants and args.variants > 1:
            fixed_list = []
            for i in range(args.variants):
                fn = base_dir / f'fixed_{i+1:03d}.txt'
                fx = fixed_variants[i] if i < len(fixed_variants) else fix_text
                fn.write_text(fx, encoding='utf-8')
                fixed_list.append(str(fn))
        meta = {
            'model': args.model,
            'preset': args.preset,
            'seed': args.seed,
            'max_new_tokens': args.max_new_tokens,
            'prompt': prompt,
            'paths': {
                'baseline': str(base_dir / 'baseline.txt'),
                'fixed': str(base_dir / 'fixed.txt'),
            },
        }
        if fixed_list:
            meta['paths']['fixed_variants'] = fixed_list
        (base_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        # Append to global index
        idx = Path('data/generated/index.jsonl')
        idx.parent.mkdir(parents=True, exist_ok=True)
        with idx.open('a', encoding='utf-8') as f:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        # Auto-update the unified README
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
