#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gzip
import itertools
import numpy as np
from tqdm import tqdm

from tools.studio.model_security import split_model_revision


# Backend selection: try MLX later, default to HF Transformers now.
class ModelBackend:
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device

    def max_context(self) -> int:
        raise NotImplementedError

    def tokenize(self, text: str) -> Dict:
        raise NotImplementedError

    def logits_for_input_ids(self, input_ids: List[int]) -> np.ndarray:
        """Return logits array [T, V] for given input ids (teacher-forced)."""
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError

    def token_str(self, token_id: int) -> str:
        raise NotImplementedError

    def tokenizer_id(self) -> str:
        raise NotImplementedError

    # Incremental decoding (KV cache) — optional
    def supports_kv(self) -> bool:
        return False

    def prefill_cache(self, input_ids: List[int]):
        """Optional: Return a backend-specific cache after prefill and the last-step logits.
        Should return (cache, last_logits: np.ndarray). Default: None, None
        """
        return None, None

    def logits_with_cache(self, next_token_id: int, cache) -> Tuple[np.ndarray, object]:
        """Optional: Given a cache and next token id, return (last_logits, new_cache).
        Default: raises NotImplementedError.
        """
        raise NotImplementedError


class HFBackend(ModelBackend):
    def __init__(self, model_id: str, device: Optional[str] = None):
        super().__init__(model_id, device)
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            print("Please install transformers and torch: pip install transformers torch", file=sys.stderr)
            raise

        source_id, revision = split_model_revision(model_id)
        self.torch = torch
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        tok_kwargs = {"use_fast": True, "trust_remote_code": True}
        model_kwargs = {}
        if revision is not None:
            tok_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        try:
            self.tok = self.AutoTokenizer.from_pretrained(source_id, **tok_kwargs)
        except TypeError:
            tok_kwargs.pop("trust_remote_code", None)
            self.tok = self.AutoTokenizer.from_pretrained(source_id, **tok_kwargs)
        # Silence long-seq warning by bumping max length if available
        try:
            if hasattr(self.tok, 'model_max_length') and isinstance(self.tok.model_max_length, int):
                if self.tok.model_max_length < 10**8:
                    self.tok.model_max_length = 10**8
        except Exception:
            pass
        self.model = self.AutoModelForCausalLM.from_pretrained(source_id, **model_kwargs)
        self.model.eval()
        # Device selection
        dev = device
        if dev is None:
            if torch.backends.mps.is_available():
                dev = 'mps'
            elif torch.cuda.is_available():
                dev = 'cuda'
            else:
                dev = 'cpu'
        self.dev = dev
        self.model.to(dev)

    def max_context(self) -> int:
        try:
            return int(self.model.config.max_position_embeddings)
        except Exception:
            return 1024

    def tokenize(self, text: str) -> Dict:
        enc = self.tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        # Normalize to plain Python lists of ints
        try:
            ids = enc.get('input_ids') if isinstance(enc, dict) else None
            if ids is not None:
                enc['input_ids'] = [int(x) for x in list(ids)]
        except Exception:
            pass
        return enc

    def logits_for_input_ids(self, input_ids: List[int]) -> np.ndarray:
        # input_ids: list of ints length L
        with self.torch.no_grad():
            ids = self.torch.tensor([input_ids], dtype=self.torch.long, device=self.dev)
            out = self.model(input_ids=ids)
            logits = out.logits[0]  # [L, V]
            # Some backends (e.g., MPS) yield bfloat16/float16 which NumPy can't view directly.
            # Cast to float32 on CPU to avoid PEP 3118 buffer format errors.
            return logits.detach().to('cpu', dtype=self.torch.float32).numpy()

    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def token_str(self, token_id: int) -> str:
        return self.tok.convert_ids_to_tokens([token_id])[0]

    def tokenizer_id(self) -> str:
        return getattr(self.tok, 'name_or_path', 'tokenizer')

    # ---- KV cache incremental path ----
    def supports_kv(self) -> bool:
        return True

    def prefill_cache(self, input_ids: List[int]):
        with self.torch.no_grad():
            ids = self.torch.tensor([input_ids], dtype=self.torch.long, device=self.dev)
            out = self.model(input_ids=ids, use_cache=True)
            logits = out.logits[0, -1, :].detach().to('cpu', dtype=self.torch.float32).numpy()
            cache = out.past_key_values if hasattr(out, 'past_key_values') else out.past
            return cache, logits

    def logits_with_cache(self, next_token_id: int, cache) -> Tuple[np.ndarray, object]:
        with self.torch.no_grad():
            ids = self.torch.tensor([[next_token_id]], dtype=self.torch.long, device=self.dev)
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
            logits = out.logits[0, -1, :].detach().to('cpu', dtype=self.torch.float32).numpy()
            new_cache = out.past_key_values if hasattr(out, 'past_key_values') else out.past
            return logits, new_cache

    def metrics_for_input_ids(self, input_ids: List[int], k: int, nucleus_p: float, nucleus_topk_k: int = 2048):
        """Compute per-token metrics for a teacher-forced input.

        Returns arrays aligned to labels input_ids[1:], i.e. length (L-1).

        Implementation notes:
        - Uses KV-cache incremental inference to avoid materializing [T, V] logits,
          which can blow up memory for interactive workloads (UI/API, Modal).
        - Computes exact logp_true and entropy per step.
        """
        use_full_logits = str(os.environ.get("HORACE_HF_FULL_LOGITS") or "").strip().lower() in ("1", "true", "yes")
        if use_full_logits:
            return self._metrics_for_input_ids_full_logits(
                input_ids=input_ids,
                k=k,
                nucleus_p=nucleus_p,
                nucleus_topk_k=nucleus_topk_k,
            )
        torch = self.torch
        # Ensure python ints
        try:
            input_ids = [int(x) for x in input_ids]
        except Exception:
            pass
        if len(input_ids) < 2:
            return {
                'p_true': np.zeros((0,), dtype=np.float32),
                'logp': np.zeros((0,), dtype=np.float32),
                'H': np.zeros((0,), dtype=np.float32),
                'eff': np.zeros((0,), dtype=np.float32),
                'rk': np.zeros((0,), dtype=np.int32),
                'tk_vals': np.zeros((0, 0), dtype=np.float32),
                'tk_idx': np.zeros((0, 0), dtype=np.int32),
                'cum_topk': np.zeros((0,), dtype=np.float32),
                'tail_mass': np.zeros((0,), dtype=np.float32),
                'w': np.zeros((0,), dtype=np.int32),
            }

        T = len(input_ids) - 1
        tk_k = int(max(1, min(int(k), int(self.vocab_size()))))
        topk_k = int(max(tk_k, min(int(nucleus_topk_k), int(self.vocab_size()))))

        p_true = np.zeros((T,), dtype=np.float32)
        logp = np.zeros((T,), dtype=np.float32)
        H = np.zeros((T,), dtype=np.float32)
        eff = np.zeros((T,), dtype=np.float32)
        rk = np.zeros((T,), dtype=np.int32)
        tk_vals = np.zeros((T, tk_k), dtype=np.float32)
        tk_idx = np.zeros((T, tk_k), dtype=np.int32)
        cum_topk = np.zeros((T,), dtype=np.float32)
        tail_mass = np.zeros((T,), dtype=np.float32)
        w = np.zeros((T,), dtype=np.int32)

        nuc_p = float(nucleus_p)

        with torch.no_grad():
            # Prime cache with the first token to get logits for the 2nd token.
            first = torch.tensor([[input_ids[0]]], dtype=torch.long, device=self.dev)
            out = self.model(input_ids=first, use_cache=True)
            cache = out.past_key_values if hasattr(out, 'past_key_values') else out.past
            logits = out.logits[0, -1, :]  # predicts token 1

            for t in range(T):
                label = int(input_ids[t + 1])
                lf = logits.float()
                logZ = torch.logsumexp(lf, dim=-1)
                logit_true = lf[label]
                lp = (logit_true - logZ).to(dtype=torch.float32)
                logp[t] = float(lp.item())
                pt = torch.exp(lp)
                p_true[t] = float(pt.item())

                # Entropy: H = logZ - E[logits] under softmax
                probs = torch.softmax(lf, dim=-1)
                El = torch.sum(probs * lf)
                Ht = (logZ - El).to(dtype=torch.float32)
                H[t] = float(Ht.item())
                eff[t] = float(torch.exp(Ht).item())

                rk[t] = int(torch.sum(lf >= logit_true).to(dtype=torch.int32).item())

                # Top-k probs under full normalization
                tk_logit, tk_id = torch.topk(lf, k=tk_k, dim=-1)
                tk_p = torch.exp((tk_logit - logZ).to(dtype=torch.float32))
                tk_vals[t, :] = tk_p.detach().to('cpu').numpy().astype(np.float32, copy=False)
                tk_idx[t, :] = tk_id.to(dtype=torch.int32).detach().to('cpu').numpy().astype(np.int32, copy=False)
                ctk = float(torch.sum(tk_p).item())
                cum_topk[t] = ctk
                tail_mass[t] = float(1.0 - ctk)

                # Approx nucleus width using top-k sorted probs (k large enough to cover p mass)
                if topk_k == tk_k:
                    nuc_pvals = tk_p
                else:
                    nuc_logit, _ = torch.topk(lf, k=topk_k, dim=-1)
                    nuc_pvals = torch.exp((nuc_logit - logZ).to(dtype=torch.float32))
                csum = torch.cumsum(nuc_pvals, dim=-1)
                hit = csum >= nuc_p
                if bool(torch.any(hit).item()):
                    wi = int(torch.argmax(hit.to(dtype=torch.int32)).item()) + 1
                else:
                    wi = int(topk_k)
                w[t] = wi

                # Advance cache except after the last label
                if t < T - 1:
                    nxt = torch.tensor([[label]], dtype=torch.long, device=self.dev)
                    out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True)
                    cache = out.past_key_values if hasattr(out, 'past_key_values') else out.past
                    logits = out.logits[0, -1, :]

        return {
            'p_true': p_true,
            'logp': logp,
            'H': H,
            'eff': eff,
            'rk': rk,
            'tk_vals': tk_vals,
            'tk_idx': tk_idx,
            'cum_topk': cum_topk,
            'tail_mass': tail_mass,
            'w': w,
        }

    def _metrics_for_input_ids_full_logits(
        self,
        *,
        input_ids: List[int],
        k: int,
        nucleus_p: float,
        nucleus_topk_k: int,
    ) -> Dict[str, np.ndarray]:
        """Vectorized metrics path using a single forward pass for the full sequence.

        So what: speeds up offline labeling/training on GPU by avoiding per-token Python loops.
        """
        torch = self.torch

        # Ensure python ints
        try:
            input_ids = [int(x) for x in input_ids]
        except Exception:
            pass
        if len(input_ids) < 2:
            return {
                'p_true': np.zeros((0,), dtype=np.float32),
                'logp': np.zeros((0,), dtype=np.float32),
                'H': np.zeros((0,), dtype=np.float32),
                'eff': np.zeros((0,), dtype=np.float32),
                'rk': np.zeros((0,), dtype=np.int32),
                'tk_vals': np.zeros((0, 0), dtype=np.float32),
                'tk_idx': np.zeros((0, 0), dtype=np.int32),
                'cum_topk': np.zeros((0,), dtype=np.float32),
                'tail_mass': np.zeros((0,), dtype=np.float32),
                'w': np.zeros((0,), dtype=np.int32),
            }

        T = len(input_ids) - 1
        vocab = int(self.vocab_size())
        tk_k = int(max(1, min(int(k), vocab)))
        topk_k = int(max(tk_k, min(int(nucleus_topk_k), vocab)))
        nuc_p = float(nucleus_p)

        with torch.no_grad():
            ids = torch.tensor([input_ids], dtype=torch.long, device=self.dev)
            out = self.model(input_ids=ids)
            logits_full = out.logits[0]  # [L, V]
            pred = logits_full[:-1, :]  # [T, V]
            labels = torch.tensor(input_ids[1:], dtype=torch.long, device=self.dev)  # [T]

            pred_f = pred.float()
            log_probs = torch.log_softmax(pred_f, dim=-1)  # [T, V]
            logp_true = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [T]

            probs = torch.exp(log_probs)
            H = -(probs * log_probs).sum(dim=-1)  # [T]
            eff = torch.exp(H)

            logit_true = pred_f.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [T]
            rk = torch.sum(pred_f >= logit_true.unsqueeze(-1), dim=-1).to(dtype=torch.int32)  # [T]

            top_logit, top_id = torch.topk(pred_f, k=topk_k, dim=-1)  # [T, topk_k]
            top_logp = log_probs.gather(-1, top_id)  # [T, topk_k]
            top_p = torch.exp(top_logp).to(dtype=torch.float32)  # [T, topk_k]

            tk_vals_t = top_p[:, :tk_k]
            tk_idx_t = top_id[:, :tk_k].to(dtype=torch.int32)
            cum_topk_t = tk_vals_t.sum(dim=-1).to(dtype=torch.float32)
            tail_mass_t = (1.0 - cum_topk_t).to(dtype=torch.float32)

            csum = torch.cumsum(top_p, dim=-1)
            hit = csum >= nuc_p
            wi = torch.argmax(hit.to(dtype=torch.int32), dim=-1) + 1
            has_hit = torch.any(hit, dim=-1)
            wi = torch.where(has_hit, wi, torch.full_like(wi, int(topk_k)))
            w = wi.to(dtype=torch.int32)

            return {
                'p_true': torch.exp(logp_true).to(dtype=torch.float32).detach().to('cpu').numpy().astype(np.float32, copy=False),
                'logp': logp_true.to(dtype=torch.float32).detach().to('cpu').numpy().astype(np.float32, copy=False),
                'H': H.to(dtype=torch.float32).detach().to('cpu').numpy().astype(np.float32, copy=False),
                'eff': eff.to(dtype=torch.float32).detach().to('cpu').numpy().astype(np.float32, copy=False),
                'rk': rk.detach().to('cpu').numpy().astype(np.int32, copy=False),
                'tk_vals': tk_vals_t.detach().to('cpu').numpy().astype(np.float32, copy=False),
                'tk_idx': tk_idx_t.detach().to('cpu').numpy().astype(np.int32, copy=False),
                'cum_topk': cum_topk_t.detach().to('cpu').numpy().astype(np.float32, copy=False),
                'tail_mass': tail_mass_t.detach().to('cpu').numpy().astype(np.float32, copy=False),
                'w': w.detach().to('cpu').numpy().astype(np.int32, copy=False),
            }


def pick_backend(model_id: str, prefer_mlx: bool = True, device: Optional[str] = None, backend: Optional[str] = None) -> ModelBackend:
    # backend: 'auto'|'mlx'|'hf' (None treated as 'auto')
    choice = (backend or 'auto').lower()
    if choice == 'hf':
        return HFBackend(model_id=model_id, device=device)
    if choice == 'mlx' or (choice == 'auto' and prefer_mlx):
        try:
            return MLXBackend(model_id=model_id)
        except Exception as e:
            if choice == 'mlx':
                raise
            print(f"MLX backend unavailable ({e}); falling back to HF/PyTorch.")
    return HFBackend(model_id=model_id, device=device)


class MLXBackend(ModelBackend):
    def __init__(self, model_id: str):
        super().__init__(model_id, device=None)
        try:
            import mlx.core as mx  # type: ignore
            from mlx_lm import load  # type: ignore
        except Exception as e:
            raise RuntimeError("Install MLX packages: pip install mlx mlx-lm") from e
        self.mx = mx
        self._load = load
        # load returns a torch-like tokenizer (HF) and an MLX model
        try:
            self.model, self.tok = self._load(model_id, dtype=mx.float32)
        except Exception:
            self.model, self.tok = self._load(model_id)
        # Ensure eval mode if available
        if hasattr(self.model, 'eval'):
            self.model.eval()
        # Use a fast Hugging Face tokenizer for robust encoding + offsets
        # This avoids calling mlx-lm's TokenizerWrapper directly, which may not be callable
        # or support return_offsets_mapping.
        try:
            from transformers import AutoTokenizer  # type: ignore
            try:
                self.hf_tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
            except TypeError:
                self.hf_tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            # Silence long-seq warning by bumping max length if available
            try:
                if hasattr(self.hf_tok, 'model_max_length') and isinstance(self.hf_tok.model_max_length, int):
                    if self.hf_tok.model_max_length < 10**8:
                        self.hf_tok.model_max_length = 10**8
            except Exception:
                pass
        except Exception:
            self.hf_tok = None

        # Attempt to repair tokenizer/vocab mismatch (common with some mlx-community ports)
        try:
            model_vocab = int(getattr(self.model.config, 'vocab_size'))
        except Exception:
            model_vocab = None
        tok_vocab = None
        try:
            if self.hf_tok is not None:
                tv = getattr(self.hf_tok, 'vocab_size', None)
                if tv is None and hasattr(self.hf_tok, 'get_vocab'):
                    tv = len(self.hf_tok.get_vocab())
                tok_vocab = int(tv) if tv is not None else None
        except Exception:
            tok_vocab = None

        def _try_set_tokenizer(tid: str) -> bool:
            try:
                from transformers import AutoTokenizer  # type: ignore
                try:
                    self.hf_tok = AutoTokenizer.from_pretrained(tid, use_fast=True, trust_remote_code=True)
                except TypeError:
                    self.hf_tok = AutoTokenizer.from_pretrained(tid, use_fast=True)
                return True
            except Exception:
                return False

        if model_vocab is not None and (tok_vocab is None or tok_vocab != model_vocab):
            mid = (model_id or '').lower()
            tried = False
            if 'qwen2.5-1.5b' in mid:
                tried = _try_set_tokenizer('Qwen/Qwen2.5-1.5B-Instruct') or _try_set_tokenizer('Qwen/Qwen2.5-1.5B')
            elif 'qwen2-1.5b' in mid:
                tried = _try_set_tokenizer('Qwen/Qwen2-1.5B-Instruct') or _try_set_tokenizer('Qwen/Qwen2-1.5B')
            if tried:
                try:
                    tv = getattr(self.hf_tok, 'vocab_size', None)
                    if tv is None and hasattr(self.hf_tok, 'get_vocab'):
                        tv = len(self.hf_tok.get_vocab())
                    tok_vocab = int(tv) if tv is not None else None
                except Exception:
                    pass

    def max_context(self) -> int:
        try:
            return int(getattr(self.model.config, 'max_position_embeddings', 1024))
        except Exception:
            return 1024

    def tokenize(self, text: str) -> Dict:
        # Prefer HF fast tokenizer for offsets and stable behavior
        if self.hf_tok is not None:
            enc = self.hf_tok(text, return_offsets_mapping=True, add_special_tokens=False)
            try:
                ids = enc.get('input_ids') if isinstance(enc, dict) else None
                if ids is not None:
                    enc['input_ids'] = [int(x) for x in list(ids)]
            except Exception:
                pass
            return enc
        # Fallbacks: attempt to use the wrapped tokenizer in mlx-lm
        tok = self.tok
        try:
            return tok(text, return_offsets_mapping=True, add_special_tokens=False)  # type: ignore
        except Exception:
            # Last resort: encode ids and synthesize empty offsets
            ids = None
            if hasattr(tok, 'encode'):
                try:
                    ids = tok.encode(text, add_special_tokens=False)  # type: ignore
                except Exception:
                    pass
            if ids is None and hasattr(tok, '__call__'):
                try:
                    enc = tok(text, add_special_tokens=False)  # type: ignore
                    ids = enc.get('input_ids') if isinstance(enc, dict) else None
                except Exception:
                    pass
            if ids is None:
                raise RuntimeError("Failed to tokenize input text with MLX tokenizer")
            return {'input_ids': ids, 'offset_mapping': [(0, 0)] * len(ids)}

    def logits_for_input_ids(self, input_ids: List[int]) -> np.ndarray:
        # MLX models typically accept int32 arrays
        # Ensure ids are ints
        try:
            input_ids = [int(x) for x in input_ids]
        except Exception:
            pass
        x = self.mx.array([input_ids], dtype=self.mx.int32)
        out = self.model(x)
        # out can be a tuple (logits, cache) or logits
        if isinstance(out, tuple) or isinstance(out, list):
            logits = out[0]
        else:
            logits = out
        # logits: [1, L, V]
        # Ensure float32 to avoid buffer format/dtype issues
        try:
            logits = self.mx.astype(logits, self.mx.float32)
        except Exception:
            pass
        # Convert via MLX helper if available (more robust than NumPy view)
        try:
            logits_np = self.mx.to_numpy(logits)[0]
        except Exception:
            try:
                # Fallback: go through Python lists to avoid buffer protocol
                logits_np = np.array(logits.tolist(), dtype=np.float32)[0]
            except Exception:
                logits_np = np.asarray(logits, dtype=np.float32)[0]
        return logits_np

    def vocab_size(self) -> int:
        try:
            return int(self.model.config.vocab_size)
        except Exception:
            return int(self.tok.vocab_size)

    def token_str(self, token_id: int) -> str:
        # Prefer HF tokenizer for consistent decoding
        tok = getattr(self, 'hf_tok', None) or self.tok
        if hasattr(tok, 'convert_ids_to_tokens'):
            return tok.convert_ids_to_tokens([token_id])[0]
        # Fallback if only decode is available
        if hasattr(tok, 'decode'):
            try:
                return tok.decode([token_id])
            except Exception:
                pass
        return str(token_id)

    def tokenizer_id(self) -> str:
        tok = getattr(self, 'hf_tok', None) or self.tok
        return getattr(tok, 'name_or_path', 'tokenizer')

    def metrics_for_input_ids(self, input_ids: List[int], k: int, nucleus_p: float):
        mx = self.mx
        # Forward pass
        x = mx.array([input_ids], dtype=mx.int32)
        out = self.model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if len(logits.shape) == 3:
            logits = logits[0]
        # Align to predict t from prefix up to t-1
        logits = logits[:-1, :]  # [T, V]
        # softmax stable
        m = mx.max(logits, axis=1, keepdims=True)
        z = logits - m
        e = mx.exp(z)
        s = mx.sum(e, axis=1, keepdims=True)
        probs = e / s
        # labels
        labels = mx.array(input_ids[1:], dtype=mx.int32)  # [T]
        # gather true probs
        try:
            idx = labels.reshape((-1, 1))
            p_true = mx.take_along_axis(probs, idx, axis=1).reshape((-1,))
        except Exception:
            ar = mx.arange(probs.shape[0], dtype=mx.int32)
            p_true = probs[ar, labels]
        p_true = mx.maximum(p_true, mx.array(1e-12, dtype=probs.dtype))
        logp = mx.log(p_true)
        # entropy and effective support
        probs_clip = mx.maximum(probs, mx.array(1e-12, dtype=probs.dtype))
        H = -mx.sum(probs_clip * mx.log(probs_clip), axis=1)
        eff = mx.exp(H)
        # rank of true
        comp = (probs >= p_true.reshape((-1, 1))).astype(mx.int32)
        rk = mx.sum(comp, axis=1)
        # top-k
        try:
            tk_vals, tk_idx = mx.topk(probs, k=k, axis=1, largest=True, sorted=True)
        except Exception:
            order = mx.argsort(probs, axis=1)[:, ::-1]
            tk_idx = order[:, :k]
            tk_vals = mx.take_along_axis(probs, tk_idx, axis=1)
        cum_topk = mx.sum(tk_vals, axis=1)
        tail_mass = mx.array(1.0, dtype=probs.dtype) - cum_topk
        # nucleus width
        order = mx.argsort(probs, axis=1)[:, ::-1]
        sorted_p = mx.take_along_axis(probs, order, axis=1)
        csum = mx.cumsum(sorted_p, axis=1)
        mask = (csum >= mx.array(nucleus_p, dtype=probs.dtype)).astype(mx.int32)
        w = mx.argmax(mask, axis=1) + 1

        def to_np_safe(arr, dtype):
            try:
                return np.asarray(arr, dtype=dtype)
            except Exception:
                try:
                    return np.array(arr.tolist(), dtype=dtype)
                except Exception:
                    return np.array(arr, dtype=dtype)
        return {
            'p_true': to_np_safe(p_true, np.float32),
            'logp': to_np_safe(logp, np.float32),
            'H': to_np_safe(H, np.float32),
            'eff': to_np_safe(eff, np.float32),
            'rk': to_np_safe(rk, np.int32),
            'tk_vals': to_np_safe(tk_vals, np.float32),
            'tk_idx': to_np_safe(tk_idx, np.int32),
            'cum_topk': to_np_safe(cum_topk, np.float32),
            'tail_mass': to_np_safe(tail_mass, np.float32),
            'w': to_np_safe(w, np.int32),
        }

    # ---- KV cache incremental path (best-effort) ----
    def supports_kv(self) -> bool:
        # Try a small probe to see if the model returns a cache
        try:
            mx = self.mx
            x = mx.array([[1]], dtype=mx.int32)
            out = self.model(x)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                return True
        except Exception:
            pass
        return False

    def prefill_cache(self, input_ids: List[int]):
        mx = self.mx
        x = mx.array([input_ids], dtype=mx.int32)
        out = self.model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, cache = out[0], out[1]
            try:
                logits = mx.astype(logits, mx.float32)
            except Exception:
                pass
            try:
                last = self.mx.to_numpy(logits[0, -1, :])
            except Exception:
                last = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            return cache, last
        # Fallback: no cache support
        return None, None

    def logits_with_cache(self, next_token_id: int, cache) -> Tuple[np.ndarray, object]:
        mx = self.mx
        x = mx.array([[next_token_id]], dtype=mx.int32)
        # Try common signatures: (x, cache=cache) then (x, cache)
        try:
            out = self.model(x, cache=cache)
        except Exception:
            try:
                out = self.model(x, cache)
            except Exception:
                # As a last resort, just run without cache
                out = self.model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, new_cache = out[0], out[1]
        else:
            logits, new_cache = out, cache
        try:
            logits = mx.astype(logits, mx.float32)
        except Exception:
            pass
        try:
            last = self.mx.to_numpy(logits[0, -1, :])
        except Exception:
            last = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        return last, new_cache


def slugify(*parts: str) -> str:
    s = '_'.join(parts)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def read_text(path: Path) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


_SPACY = None
def _get_spacy():
    global _SPACY
    if _SPACY is not None:
        return _SPACY
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load('en_core_web_sm')
        except Exception:
            # Try a blank English model as fallback (no POS/NER)
            nlp = spacy.blank('en')
        _SPACY = nlp
    except Exception:
        _SPACY = None
    return _SPACY


def jsonl_write(path: Path, rows: Iterable[dict], compress: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> Optional[float]:
    n = len(x)
    if n < (m - 1) * tau + 1 or m < 2:
        return None
    patterns = {}
    for i in range(n - (m - 1) * tau):
        # ordinal pattern of length m
        seg = x[i:i + m * tau:tau]
        ranks = tuple(np.argsort(seg))
        patterns[ranks] = patterns.get(ranks, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    Hmax = np.log(math.factorial(m))
    return float(H / (Hmax + 1e-12))


def hurst_rs(x: np.ndarray) -> Optional[float]:
    # Simple R/S estimator; requires reasonable length
    n = len(x)
    if n < 64:
        return None
    try:
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)
        # window sizes roughly logarithmic
        sizes = [8, 16, 32, 64, 128, 256, 512]
        sizes = [s for s in sizes if s * 2 <= n]
        if not sizes:
            return None
        RS = []
        N = []
        for s in sizes:
            k = n // s
            if k < 2:
                continue
            rs_vals = []
            for j in range(k):
                seg = x[j * s:(j + 1) * s]
                y = np.cumsum(seg - np.mean(seg))
                R = np.max(y) - np.min(y)
                S = np.std(seg) + 1e-12
                rs_vals.append(R / S)
            if rs_vals:
                RS.append(np.mean(rs_vals))
                N.append(s)
        if len(RS) < 2:
            return None
        logN = np.log(N)
        logRS = np.log(RS)
        H = np.polyfit(logN, logRS, 1)[0]
        return float(H)
    except Exception:
        return None


def topk(probs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # probs: [T, V]
    idx = np.argpartition(-probs, kth=min(k, probs.shape[1]-1), axis=1)[:, :k]
    vals = np.take_along_axis(probs, idx, axis=1)
    # sort within top-k
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return vals, idx


def nucleus_width(probs: np.ndarray, p: float) -> np.ndarray:
    # probs: [T, V]
    sort_idx = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, sort_idx, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    w = np.argmax(cum >= p, axis=1) + 1
    return w


def rank_of_true(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # probs: [T, V]; labels: [T]
    # Rank = 1 + number of tokens with prob > true_prob (ties: include equals)
    true_p = probs[np.arange(probs.shape[0]), labels]
    # Count how many >= true_p
    comp = (probs >= true_p[:, None]).sum(axis=1)
    return comp


def paragraph_units(text: str, kind: str) -> List[Tuple[int, int]]:
    # Return (start,end) char spans for units to shuffle for cohesion
    if kind == 'poem':
        # Use lines (non-empty)
        lines = []
        pos = 0
        for m in re.finditer(r".*(?:\n|$)", text):
            s, e = m.start(), m.end()
            chunk = text[s:e]
            if chunk.strip():
                lines.append((s, e))
        return lines
    # Default: paragraphs split by blank lines
    spans = []
    start = 0
    parts = re.split(r"(\n\s*\n+)", text)
    idx = 0
    cursor = 0
    while idx < len(parts):
        seg = parts[idx]
        s = cursor
        e = cursor + len(seg)
        if seg.strip():
            spans.append((s, e))
        cursor = e
        if idx + 1 < len(parts):
            sep = parts[idx+1]
            cursor += len(sep)
            idx += 2
        else:
            break
    if not spans:
        # Fallback to sentences
        spans = []
        for m in re.finditer(r"[^.!?\n]+[.!?]?\s*", text):
            s, e = m.start(), m.end()
            if text[s:e].strip():
                spans.append((s, e))
    return spans


def compute_doc_signature(model: ModelBackend, doc: dict, k: int, p: float, ctx: Optional[int], stride: Optional[int], out_dir: Path, write_tokens: bool = True):
    path = Path(doc['path'])
    text = read_text(path)
    nlp = _get_spacy()
    spacy_doc = None
    if nlp is not None:
        try:
            spacy_doc = nlp(text)
        except Exception:
            spacy_doc = None
    enc = model.tokenize(text)
    input_ids = enc['input_ids']
    offsets = enc.get('offset_mapping', None)
    if offsets is None:
        # best-effort offsets
        offsets = [(0, 0)] * len(input_ids)

    max_ctx = ctx or min(model.max_context(), 1024)
    # leave 0 for BOS since we use no special tokens; teacher forcing uses token t from prefix up to t-1
    window = min(max_ctx, len(input_ids))
    emit_overlap = 128 if window > 256 else max(0, window // 4)
    stride_len = stride or max(1, window - emit_overlap)

    all_rows = []
    doc_stats = []
    # Token-class accumulators
    is_punct_arr: List[int] = []
    is_newline_arr: List[int] = []
    is_content_arr: List[int] = []
    seen = set()  # global token indices emitted

    # Line boundaries for quick line position checks
    line_spans = []
    for m in re.finditer(r".*(?:\n|$)", text):
        s, e = m.start(), m.end()
        if s < e:
            # trim trailing newline for end calc
            t = text[s:e]
            if t.endswith('\n'):
                e = e - 1
            line_spans.append((s, e))
    # spaCy token index by char start (optional)
    spacy_starts = []
    if spacy_doc is not None and hasattr(spacy_doc, 'to_array'):
        spacy_starts = [t.idx for t in spacy_doc]

    chunk_idx = 0
    for start in range(0, len(input_ids), stride_len):
        end = min(len(input_ids), start + window)
        ids = input_ids[start:end]
        if len(ids) < 2:
            break
        # Compute per-token metrics
        if isinstance(model, MLXBackend):
            m = model.metrics_for_input_ids(ids, k=k, nucleus_p=p)
            true_p = m['p_true']
            logp = m['logp']
            H = m['H']
            eff = m['eff']
            rk = m['rk']
            tk_vals = m['tk_vals']
            tk_ids = m['tk_idx']
            cum_topk = m['cum_topk']
            tail_mass = m['tail_mass']
            w_p = m['w']
            labels = np.array(ids[1:], dtype=np.int64)
        else:
            logits = model.logits_for_input_ids(ids)  # [L, V]
            # Align to predict t from prefix up to t-1
            logits = logits[:-1, :]
            labels = np.array(ids[1:], dtype=np.int64)
            probs = softmax(logits, axis=-1)
            true_p = probs[np.arange(probs.shape[0]), labels]
            logp = np.log(np.clip(true_p, 1e-12, 1.0))
            H = entropy(probs, axis=1)
            eff = np.exp(H)
            rk = rank_of_true(probs, labels)
            tk_vals, tk_ids = topk(probs, k)
            cum_topk = np.sum(tk_vals, axis=1)
            tail_mass = 1.0 - cum_topk
            w_p = nucleus_width(probs, p)

        # Determine emission range to avoid double counting
        emit_from = 0 if start == 0 else max(0, (start + len(ids) - end) + (window - stride_len))
        # Fallback: emit only last (len(ids)-1 - emit_overlap)
        if start > 0:
            emit_from = min(len(labels) - 1, emit_overlap)

        for i in range(len(labels)):
            global_tok = start + i + 1  # label position index in the full tokenization
            if start > 0 and i < emit_from:
                continue
            if global_tok in seen:
                continue
            seen.add(global_tok)
            off = offsets[global_tok] if global_tok < len(offsets) else (0, 0)
            cs, ce = int(off[0]), int(off[1])
            substr = text[cs:ce] if 0 <= cs < len(text) and 0 <= ce <= len(text) and cs < ce else ''
            is_punct = 1 if substr and all((not ch.isalnum()) and not ch.isspace() for ch in substr) else 0
            is_nl = 1 if substr and ('\n' in substr) else 0
            # line position
            line_pos = 'middle'
            if line_spans:
                # binary search by start char
                import bisect
                idx = bisect.bisect_right([s for s, _ in line_spans], cs) - 1
                if 0 <= idx < len(line_spans):
                    ls, le = line_spans[idx]
                    if cs == ls:
                        line_pos = 'start'
                    elif ce == le:
                        line_pos = 'end'
            pos_tag = None
            ent_type = None
            is_content = 0
            if spacy_doc is not None and spacy_starts:
                import bisect
                j = bisect.bisect_right(spacy_starts, cs) - 1
                if 0 <= j < len(spacy_starts):
                    try:
                        tok = spacy_doc[j]
                        pos_tag = tok.pos_ if hasattr(tok, 'pos_') else None
                        ent_type = tok.ent_type_ if hasattr(tok, 'ent_type_') else None
                        if pos_tag in {'NOUN','VERB','ADJ','ADV','PROPN','NUM'}:
                            is_content = 1
                    except Exception:
                        pass
            else:
                # Approximate content via simple heuristic
                if substr and any(ch.isalpha() for ch in substr) and len(substr) > 2 and not is_punct:
                    is_content = 1
            is_punct_arr.append(is_punct)
            is_newline_arr.append(is_nl)
            is_content_arr.append(is_content)
            row = {
                'doc_id': slugify(doc['type'], doc['author'], doc['title']),
                'doc_type': doc['type'],
                'author': doc['author'],
                'title': doc['title'],
                'model_id': model.model_id,
                'token_index': int(global_tok),
                'char_start': cs,
                'char_end': ce,
                'token_id': int(labels[i]),
                'token_str': model.token_str(int(labels[i])),
                'p_true': float(true_p[i]),
                'logp_true': float(logp[i]),
                'rank': int(rk[i]),
                'entropy': float(H[i]),
                'effective_support': float(eff[i]),
                'nucleus_width': int(w_p[i]),
                'cum_mass_topk': float(cum_topk[i]),
                'tail_mass': float(tail_mass[i]),
                'topk_ids': [int(x) for x in tk_ids[i].tolist()],
                'topk_probs': [float(x) for x in tk_vals[i].tolist()],
                'is_punct': is_punct,
                'is_newline': is_nl,
                'line_pos': line_pos,
                'pos': pos_tag,
                'ent_type': ent_type,
            }
            doc_stats.append((true_p[i], logp[i], H[i], eff[i], rk[i], w_p[i]))
            if write_tokens:
                all_rows.append(row)
        chunk_idx += 1

    # Aggregate per-doc
    if not doc_stats:
        agg = None
    else:
        arr = np.array(doc_stats)
        p_true_arr = arr[:, 0]
        logp_arr = arr[:, 1]
        H_arr = arr[:, 2]
        eff_arr = arr[:, 3]
        rk_arr = arr[:, 4]
        w_arr = arr[:, 5]
        # derived
        surprisal = -logp_arr
        norm_surprisal = surprisal / np.clip(H_arr, 1e-12, None)
        # Robust variants
        norm_surprisal_clipped = surprisal / np.clip(H_arr, 1.0, None)
        ratio_of_means = float(np.mean(surprisal) / max(np.mean(H_arr), 1e-12))
        norm_surprisal_median = float(np.median(norm_surprisal_clipped))
        # Bits per token (base-2) for readability
        LN2 = math.log(2.0)
        bpt_mean = float(np.mean(surprisal) / LN2)
        entropy_bits_mean = float(np.mean(H_arr) / LN2)

        # Cadence metrics over surprisal time series
        def cadence_metrics(s: np.ndarray) -> Dict[str, Optional[float]]:
            out: Dict[str, Optional[float]] = {}
            n = int(s.shape[0])
            if n < 5:
                return {
                    'surprisal_cv': None,
                    'surprisal_masd': None,
                    'surprisal_acf1': None,
                    'surprisal_acf2': None,
                    'surprisal_peak_period_tokens': None,
                    'high_surprise_rate_per_100': None,
                    'high_surprise_ipi_mean': None,
                    'high_surprise_ipi_cv': None,
                    'run_low_mean_len': None,
                    'run_high_mean_len': None,
                }
            mu = float(np.mean(s))
            sd = float(np.std(s))
            out['surprisal_cv'] = (sd / mu) if mu > 1e-12 else None
            dif = np.diff(s)
            out['surprisal_masd'] = float(np.mean(np.abs(dif)))
            z = s - mu
            denom = float(np.sum(z * z) + 1e-12)
            def _acf(k: int) -> Optional[float]:
                if n <= k:
                    return None
                num = float(np.sum(z[k:] * z[:-k]))
                return num / denom
            out['surprisal_acf1'] = _acf(1)
            out['surprisal_acf2'] = _acf(2)
            # Spectral peak (ignore DC)
            try:
                ps = np.abs(np.fft.rfft(z)) ** 2
                if ps.shape[0] > 2:
                    idx = int(np.argmax(ps[1:]) + 1)
                    freq = idx / n
                    out['surprisal_peak_period_tokens'] = (1.0 / freq) if freq > 0 else None
                else:
                    out['surprisal_peak_period_tokens'] = None
            except Exception:
                out['surprisal_peak_period_tokens'] = None
            # High-surprise peaks and inter-peak intervals
            thr = mu + sd
            peaks: List[int] = []
            for i in range(1, n - 1):
                if s[i] >= thr and s[i] > s[i - 1] and s[i] >= s[i + 1]:
                    peaks.append(i)
            rate = (len(peaks) / n) * 100.0
            out['high_surprise_rate_per_100'] = float(rate)
            if len(peaks) >= 2:
                ipi = np.diff(np.array(peaks))
                out['high_surprise_ipi_mean'] = float(np.mean(ipi))
                mipi = float(np.mean(ipi))
                out['high_surprise_ipi_cv'] = float(np.std(ipi) / (mipi + 1e-12)) if ipi.size > 1 else None
            else:
                out['high_surprise_ipi_mean'] = None
                out['high_surprise_ipi_cv'] = None
            # Run lengths below/above median
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
            out['run_low_mean_len'] = float(np.mean(low_lens)) if low_lens else None
            out['run_high_mean_len'] = float(np.mean(high_lens)) if high_lens else None
            return out

        cad = cadence_metrics(surprisal)
        # Additional distributional metrics
        surprisal_p10 = float(np.percentile(surprisal, 10))
        entropy_p10 = float(np.percentile(H_arr, 10))
        entropy_p90 = float(np.percentile(H_arr, 90))
        pe = permutation_entropy(surprisal, m=3, tau=1)
        hurst = hurst_rs(surprisal)
        # Spikes and neighborhoods
        mu = float(np.mean(surprisal)); sd = float(np.std(surprisal))
        thr = mu + sd
        spikes = np.where(surprisal >= thr)[0]
        prev_idx = spikes - 1
        next_idx = spikes + 1
        prev_mask = prev_idx >= 0
        next_mask = next_idx < len(surprisal)
        prev_content_rate = float(np.mean(np.array(is_content_arr)[prev_idx[prev_mask]]) if np.any(prev_mask) else 0.0)
        prev_punct_rate = float(np.mean(np.array(is_punct_arr)[prev_idx[prev_mask]]) if np.any(prev_mask) else 0.0)
        next_content_rate = float(np.mean(np.array(is_content_arr)[next_idx[next_mask]]) if np.any(next_mask) else 0.0)
        next_punct_rate = float(np.mean(np.array(is_punct_arr)[next_idx[next_mask]]) if np.any(next_mask) else 0.0)
        # IPI stats
        ipi_mean = None; ipi_cv = None; ipi_p50 = None
        if len(spikes) >= 2:
            ipi = np.diff(spikes)
            ipi_mean = float(np.mean(ipi))
            ipi_p50 = float(np.median(ipi))
            ipi_cv = float(np.std(ipi) / (ipi_mean + 1e-12)) if ipi.size > 1 else None
        # Cooldown: average entropy drop over next 3 tokens after spikes
        cooldown = []
        for idx in spikes:
            if idx + 3 < len(H_arr):
                cooldown.append(float(H_arr[idx] - np.mean(H_arr[idx+1:idx+4])))
        cooldown_drop_3 = float(np.mean(cooldown)) if cooldown else None
        content_fraction = float(np.mean(is_content_arr)) if is_content_arr else None
        punct_rate = float(np.mean(is_punct_arr)) if is_punct_arr else None
        newline_rate = float(np.mean(is_newline_arr)) if is_newline_arr else None
        doc_len = len(doc_stats)
        agg = {
            'doc_id': slugify(doc['type'], doc['author'], doc['title']),
            'doc_type': doc['type'],
            'author': doc['author'],
            'title': doc['title'],
            'model_id': model.model_id,
            'tokens_count': int(doc_len),
            'p_true_mean': float(np.mean(p_true_arr)),
            'surprisal_mean': float(np.mean(surprisal)),
            'surprisal_median': float(np.median(surprisal)),
            'surprisal_p90': float(np.percentile(surprisal, 90)),
            'entropy_mean': float(np.mean(H_arr)),
            'entropy_median': float(np.median(H_arr)),
            'eff_support_mean': float(np.mean(eff_arr)),
            'rank_percentile_mean': float(np.mean(rk_arr / model.vocab_size())),
            'nucleus_w_mean': float(np.mean(w_arr)),
            'norm_surprisal_mean': float(np.mean(norm_surprisal)),
            'norm_surprisal_mean_clipped': float(np.mean(norm_surprisal_clipped)),
            'norm_surprisal_median': norm_surprisal_median,
            'norm_surprisal_ratio_of_means': ratio_of_means,
            'bpt_mean': bpt_mean,
            'entropy_bits_mean': entropy_bits_mean,
            # Cadence metrics
            **cad,
            # Distributional additions
            'surprisal_p10': surprisal_p10,
            'entropy_p10': entropy_p10,
            'entropy_p90': entropy_p90,
            'perm_entropy': pe,
            'hurst_rs': hurst,
            # Token-class/cadence neighborhood
            'content_fraction': content_fraction,
            'punct_rate': punct_rate,
            'newline_rate': newline_rate,
            'spike_prev_content_rate': prev_content_rate,
            'spike_prev_punct_rate': prev_punct_rate,
            'spike_next_content_rate': next_content_rate,
            'spike_next_punct_rate': next_punct_rate,
            'ipi_mean': ipi_mean,
            'ipi_cv': ipi_cv,
            'ipi_p50': ipi_p50,
            'cooldown_entropy_drop_3': cooldown_drop_3,
        }

    # Cohesion: shuffle units and compute delta in total log-likelihood per token
    try:
        units = paragraph_units(text, doc['type'])
        if len(units) >= 3:
            shuffled = units[:]
            random.Random(0).shuffle(shuffled)
            shuf_text = ''.join(text[s:e] for s, e in shuffled)
            enc_s = model.tokenize(shuf_text)
            ids_s = enc_s['input_ids']
            # One pass over shuffled text
            ll_sum = 0.0
            tok_count = 0
            for st in range(0, len(ids_s), max(len(ids_s), 1)):
                ed = min(len(ids_s), st + min(model.max_context(), 1024))
                if ed - st < 2:
                    break
                logits_s = model.logits_for_input_ids(ids_s[st:ed])[:-1, :]
                labels_s = np.array(ids_s[st+1:ed], dtype=np.int64)
                probs_s = softmax(logits_s, axis=-1)
                true_p_s = np.clip(probs_s[np.arange(probs_s.shape[0]), labels_s], 1e-12, 1.0)
                ll_sum += float(np.sum(np.log(true_p_s)))
                tok_count += int(len(true_p_s))
            if agg and tok_count > 0:
                shuffle_ll = float(ll_sum / tok_count)
                agg['cohesion_shuffle_logp_per_token'] = shuffle_ll
                # Original logp per token is the negative of surprisal_mean
                logp_orig = -agg['surprisal_mean']
                agg['logp_per_token_original'] = logp_orig
                agg['cohesion_delta'] = shuffle_ll - logp_orig
    except Exception:
        # Cohesion optional
        pass

    # Write outputs
    base = out_dir / model.model_id
    base.mkdir(parents=True, exist_ok=True)
    if write_tokens and all_rows:
        tok_dir = base / 'tokens'
        tok_dir.mkdir(parents=True, exist_ok=True)
        tok_path = tok_dir / f"{slugify(doc['type'], doc['author'], doc['title'])}.jsonl.gz"
        jsonl_write(tok_path, all_rows, compress=True)
    return agg


def cmd_init(args):
    model = pick_backend(args.model, prefer_mlx=True, backend=getattr(args, 'backend', None))
    out = Path('data/analysis') / model.model_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        'model_id': model.model_id,
        'tokenizer_id': model.tokenizer_id(),
        'vocab_size': model.vocab_size(),
        'max_context': model.max_context(),
        'params': {
            'k': args.k,
            'p': args.p,
            'context': args.context,
            'stride': args.stride,
        },
        'timestamp': time.time(),
    }
    with open(out / 'run_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Initialized analysis folder: {out}")


def load_index() -> List[dict]:
    idx_path = Path('data/index.json')
    if not idx_path.exists():
        print("data/index.json not found.", file=sys.stderr)
        sys.exit(1)
    with open(idx_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _parse_types_arg(val: Optional[str]) -> Optional[List[str]]:
    if not val:
        return None
    if isinstance(val, list):
        return val
    parts = [s.strip().lower() for s in str(val).split(',') if s.strip()]
    valid = {'poem', 'shortstory', 'novel'}
    out = [p for p in parts if p in valid]
    return out or None


def cmd_tokens(args):
    model = pick_backend(args.model, prefer_mlx=True, backend=getattr(args, 'backend', None))
    out_dir = Path('data/analysis')
    items = load_index()
    # filter by types if requested
    types = _parse_types_arg(getattr(args, 'types', None))
    if types:
        items = [it for it in items if it.get('type', '').lower() in types]
    if args.limit:
        items = items[: args.limit]
    # If resuming, precompute existing docs and token files
    base = out_dir / model.model_id
    tok_dir = base / 'tokens'
    tok_dir.mkdir(parents=True, exist_ok=True)
    existing_docs = set()
    if getattr(args, 'resume', False):
        docs_path = base / 'docs.jsonl'
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        existing_docs.add(r.get('doc_id'))
                    except Exception:
                        pass
    docs_out = []
    for doc in tqdm(items, desc=f"tokens:{model.model_id}"):
        doc_id = slugify(doc['type'], doc['author'], doc['title'])
        tok_path = (out_dir / model.model_id / 'tokens' / f"{doc_id}.jsonl.gz")
        # Resume behavior: skip if token file exists and not forcing
        if getattr(args, 'resume', False) and tok_path.exists() and not getattr(args, 'force', False):
            # Also avoid re-appending doc aggregate if it already exists
            continue
        try:
            agg = compute_doc_signature(
                model=model,
                doc=doc,
                k=args.k,
                p=args.p,
                ctx=args.context,
                stride=args.stride,
                out_dir=out_dir,
                write_tokens=True,
            )
            if agg:
                # If resuming and doc already aggregated, skip appending duplicate
                if not (getattr(args, 'resume', False) and agg['doc_id'] in existing_docs):
                    docs_out.append(agg)
        except Exception as e:
            print(f"Error processing {doc['path']}: {e}", file=sys.stderr)
    # append per-doc signatures
    docs_path = base / 'docs.jsonl'
    if docs_out:
        with open(docs_path, 'a', encoding='utf-8') as f:
            for r in docs_out:
                f.write(json.dumps(r) + "\n")
    print(f"Wrote per-doc signatures: {docs_path}")


def cmd_docs(args):
    # Re-aggregate docs.jsonl into a clean JSONL (de-duplicated by doc_id)
    base = Path('data/analysis') / args.model
    docs_path = base / 'docs.jsonl'
    if not docs_path.exists():
        print(f"No docs.jsonl at {docs_path}", file=sys.stderr)
        return
    seen = {}
    types = _parse_types_arg(getattr(args, 'types', None))
    with open(docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                if not types or r.get('doc_type','').lower() in types:
                    seen[r['doc_id']] = r
            except Exception:
                pass
    clean_path = base / 'docs_clean.jsonl'
    with open(clean_path, 'w', encoding='utf-8') as f:
        for r in seen.values():
            f.write(json.dumps(r) + "\n")
    print(f"Docs aggregated at {clean_path} ({len(seen)} docs)")


def cmd_authors(args):
    base = Path('data/analysis') / args.model
    docs_path = base / 'docs_clean.jsonl'
    if not docs_path.exists():
        print(f"No docs_clean.jsonl at {docs_path}. Run 'docs' first.", file=sys.stderr)
        return
    # Aggregate by author
    per_author: Dict[str, List[dict]] = {}
    types = _parse_types_arg(getattr(args, 'types', None))
    with open(docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if not types or r.get('doc_type','').lower() in types:
                per_author.setdefault(r['author'], []).append(r)
    rows = []
    for author, docs in per_author.items():
        def agg_mean(key):
            vals = [d.get(key) for d in docs if d.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        keys_mean = [
            'surprisal_mean',
            'entropy_mean',
            'eff_support_mean',
            'rank_percentile_mean',
            'nucleus_w_mean',
            'norm_surprisal_mean',
            'norm_surprisal_mean_clipped',
            'norm_surprisal_median',
            'norm_surprisal_ratio_of_means',
            'bpt_mean',
            'entropy_bits_mean',
            'cohesion_delta',
            'cohesion_shuffle_logp_per_token',
            'logp_per_token_original',
            # cadence
            'surprisal_cv',
            'surprisal_masd',
            'surprisal_acf1',
            'surprisal_acf2',
            'surprisal_peak_period_tokens',
            'high_surprise_rate_per_100',
            'high_surprise_ipi_mean',
            'high_surprise_ipi_cv',
            'run_low_mean_len',
            'run_high_mean_len',
            # distributional
            'surprisal_p10',
            'entropy_p10',
            'entropy_p90',
            'perm_entropy',
            'hurst_rs',
            # token-class & neighborhoods
            'content_fraction',
            'punct_rate',
            'newline_rate',
            'spike_prev_content_rate',
            'spike_prev_punct_rate',
            'spike_next_content_rate',
            'spike_next_punct_rate',
            'ipi_mean',
            'ipi_cv',
            'ipi_p50',
            'cooldown_entropy_drop_3',
        ]
        row = {
            'author': author,
            'model_id': args.model,
            'docs_count': len(docs),
            'tokens_total': int(np.sum([d.get('tokens_count', 0) for d in docs])),
        }
        for k in keys_mean:
            row[f"{k}_mean"] = agg_mean(k)
        rows.append(row)
    out = base / 'authors.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Authors aggregated at {out} ({len(rows)} authors)")


def build_parser():
    p = argparse.ArgumentParser(description="Per-token and per-doc analysis pipeline")
    sub = p.add_subparsers(dest='cmd', required=True)

    def add_common(sp):
        sp.add_argument('--model', default='mlx-community/Llama-3.2-3B-Instruct', help='HuggingFace model id (e.g., gpt2)')
        sp.add_argument('--backend', default='auto', choices=['auto','mlx','hf'], help='Backend: auto (prefer MLX), mlx, or hf')
        sp.add_argument('--k', type=int, default=10, help='top-k size')
        sp.add_argument('--p', type=float, default=0.9, help='nucleus p')
        sp.add_argument('--context', type=int, default=1024, help='context window tokens')
        sp.add_argument('--stride', type=int, default=None, help='stride tokens (default: context-128)')

    sp = sub.add_parser('init', help='Initialize analysis run metadata')
    add_common(sp)
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser('tokens', help='Compute per-token features and per-doc aggregates')
    add_common(sp)
    sp.add_argument('--types', help='comma-separated subset: poem,shortstory,novel')
    sp.add_argument('--limit', type=int, default=None, help='limit docs for a quick run')
    sp.add_argument('--resume', action='store_true', help='skip docs with existing token files and avoid duplicate doc aggregates')
    sp.add_argument('--force', action='store_true', help='recompute even if token file exists (overwrites)')
    sp.set_defaults(func=cmd_tokens)

    sp = sub.add_parser('docs', help='Aggregate docs into a clean unique set')
    sp.add_argument('--model', default='mlx-community/Llama-3.2-3B-Instruct')
    sp.add_argument('--types', help='comma-separated subset: poem,shortstory,novel')
    sp.set_defaults(func=cmd_docs)

    sp = sub.add_parser('authors', help='Aggregate per-author signatures from docs')
    sp.add_argument('--model', default='mlx-community/Llama-3.2-3B-Instruct')
    sp.add_argument('--types', help='comma-separated subset: poem,shortstory,novel')
    sp.set_defaults(func=cmd_authors)

    sp = sub.add_parser('run', help='Run init -> tokens -> docs -> authors from a JSON config')
    sp.add_argument('--config', required=True, help='Path to JSON config with fields: model,k,p,context,stride,limit(optional)')
    def _run(args):
        cfg_path = Path(args.config)
        cfg = json.loads(cfg_path.read_text())
        model = cfg.get('model', 'gpt2')
        backend = cfg.get('backend', 'auto')
        k = int(cfg.get('k', 10))
        pval = float(cfg.get('p', 0.9))
        ctx = int(cfg.get('context', 1024))
        stride = cfg.get('stride')
        stride = int(stride) if stride is not None else None
        limit = cfg.get('limit')
        limit = int(limit) if limit is not None else None
        types = _parse_types_arg(cfg.get('types'))
        steps = cfg.get('steps') or ['init','tokens','docs','authors']
        # init
        if 'init' in steps:
            cmd_init(argparse.Namespace(model=model, backend=backend, k=k, p=pval, context=ctx, stride=stride))
        # tokens
        if 'tokens' in steps:
            cmd_tokens(argparse.Namespace(model=model, backend=backend, k=k, p=pval, context=ctx, stride=stride, limit=limit, types=','.join(types) if types else None, resume=False, force=False))
        # docs
        if 'docs' in steps:
            cmd_docs(argparse.Namespace(model=model, types=','.join(types) if types else None))
        # authors
        if 'authors' in steps:
            cmd_authors(argparse.Namespace(model=model, types=','.join(types) if types else None))
    sp.set_defaults(func=_run)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
