"""
GRPO runner (MLX-only rollouts) with simple diverse sampler and reward scoring.

It does NOT implement parameter updates; it prepares rollouts, scores them
with Horace rewards, and computes group-relative advantages to plug into
your PPO/GRPO training loop. Outputs are saved for inspection.

Config JSON example (see configs/grpo_default.json):
{
  "model": "Qwen/Qwen3-1.7B",
  "backend": "mlx",
  "seed": 42,
  "group_size": 6,
  "max_new_tokens": 120,
  "sampler": {"temperature": 0.9, "top_p": 0.92, "top_k": 120, "jitter": {"temperature": 0.05, "top_p": 0.03}},
  "reward": {"preset": "freeverse", "presets_file": "configs/reward_presets.json"},
  "prompts": ["Write a free-verse poem about tidal flats."],
  "prompts_file": null,
  "out_dir": "data/grpo_runs/qwen3_1p7b_freeverse_debug",
  "steps": 1
}
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.analyze import pick_backend, ModelBackend
from tools.reward import load_presets, compute_reward


@dataclass
class SimpleSamplerCfg:
    temperature: float = 0.9
    top_p: float = 0.92
    top_k: int = 120
    jitter_temperature: float = 0.0
    jitter_top_p: float = 0.0


class MLXSimpleSampler:
    """Simple top-p sampler for MLX backend with KV cache and diversity jitter.

    Records logits_per_step for reward/kl; tracks line boundaries for cadence.
    """
    def __init__(self, backend: ModelBackend, cfg: SimpleSamplerCfg, seed: Optional[int] = None):
        self.backend = backend
        self.cfg = cfg
        if seed is not None:
            np.random.seed(seed)
        # Optional logit bias adapter (e.g., trained by GRPO)
        self.logit_bias: Optional[np.ndarray] = None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        y = x - np.max(x)
        e = np.exp(y)
        return e / (e.sum() + 1e-12)

    def _sample(self, logits: np.ndarray, temperature: float, top_p: float, top_k: Optional[int]) -> int:
        x = logits.astype(np.float32)
        x = x / max(1e-6, float(temperature))
        # top-k
        if top_k is not None and top_k > 0 and top_k < x.shape[-1]:
            idx = np.argpartition(-x, kth=top_k - 1)[-top_k:]
            mask = np.full_like(x, -np.inf, dtype=np.float32)
            mask[idx] = x[idx]
            x = mask
        # top-p
        if 0.0 < top_p < 1.0:
            y = x - np.max(x)
            p = np.exp(y)
            p = p / (p.sum() + 1e-12)
            order = np.argsort(-p)
            cdf = np.cumsum(p[order])
            k = int(np.searchsorted(cdf, top_p, side="left")) + 1
            keep = order[:k]
            mask = np.full_like(x, -np.inf, dtype=np.float32)
            mask[keep] = x[keep]
            x = mask
        p = self._softmax(x)
        return int(np.random.choice(np.arange(p.shape[-1]), p=p))

    def _decode_piece(self, tid: int) -> str:
        # Prefer HF tokenizer if available through backend; else token_str
        try:
            if hasattr(self.backend, 'hf_tok') and self.backend.hf_tok is not None:
                return self.backend.hf_tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            pass
        try:
            return self.backend.token_str(tid) or ''
        except Exception:
            return ''

    def generate(self, prompt: str, max_new_tokens: int, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
        enc = self.backend.tokenize(prompt)
        ids = list(enc['input_ids']) if isinstance(enc, dict) else list(enc)
        # Attempt to add BOS if present in tokenizer
        bos_id = None
        try:
            if hasattr(self.backend, 'hf_tok') and self.backend.hf_tok is not None:
                bos_id = getattr(self.backend.hf_tok, 'bos_token_id', None)
        except Exception:
            bos_id = None
        if bos_id is not None and (len(ids) == 0 or ids[0] != bos_id):
            ids = [int(bos_id)] + ids

        # KV cache if supported
        cache, last_logits = (None, None)
        if self.backend.supports_kv():
            cache, last_logits = self.backend.prefill_cache(ids)
        else:
            # compute last logits via full pass
            last_logits = self.backend.logits_for_input_ids(ids)[-1]

        out_ids: List[int] = []
        logits_per_step: List[np.ndarray] = []
        line_steps: List[Tuple[int, int]] = []  # (start, end) in gen steps
        current_line_start = 0
        gen_text = ""

        # Diversity jitter per generation
        t0 = max(1e-6, float(self.cfg.temperature + np.random.uniform(-self.cfg.jitter_temperature, self.cfg.jitter_temperature)))
        p0 = float(np.clip(self.cfg.top_p + np.random.uniform(-self.cfg.jitter_top_p, self.cfg.jitter_top_p), 0.05, 0.99))
        k0 = int(self.cfg.top_k) if self.cfg.top_k and self.cfg.top_k > 0 else None

        for step in range(max_new_tokens):
            # Apply optional logit bias
            logits_step = last_logits
            if self.logit_bias is not None and self.logit_bias.shape[-1] == logits_step.shape[-1]:
                logits_step = logits_step + self.logit_bias.astype(np.float32)
            logits_per_step.append(logits_step)
            tid = self._sample(logits_step, temperature=t0, top_p=p0, top_k=k0)
            out_ids.append(int(tid))
            piece = self._decode_piece(tid)
            gen_text += piece
            if "\n" in piece:
                # Close current line at this gen step+1
                line_steps.append((current_line_start, step + 1))
                current_line_start = step + 1
            # advance cache
            if self.backend.supports_kv():
                last_logits, cache = self.backend.logits_with_cache(tid, cache)
            else:
                last_logits = self.backend.logits_for_input_ids(ids + out_ids)[-1]

        # Close trailing line if non-empty
        if current_line_start < len(out_ids):
            line_steps.append((current_line_start, len(out_ids)))

        # Final decode of prompt+gen for completeness
        try:
            if hasattr(self.backend, 'hf_tok') and self.backend.hf_tok is not None:
                full_text = self.backend.hf_tok.decode(ids + out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                full_text = gen_text
        except Exception:
            full_text = gen_text

        lines = [ln for ln in gen_text.splitlines()]
        # Also compute per-step old logprobs under the sampling policy (with bias)
        old_logprobs: List[float] = []
        for ls, t in zip(logits_per_step, out_ids):
            y = ls - np.max(ls)
            p = np.exp(y) / (np.sum(np.exp(y)) + 1e-12)
            old_logprobs.append(float(np.log(max(1e-12, p[int(t)]))))

        return {
            "prompt": prompt,
            "gen_text": gen_text,
            "full_text": full_text,
            "lines": lines,
            "gen_token_ids": out_ids,
            "logits_per_step": logits_per_step,
            "line_steps": line_steps,
            "old_logprobs": old_logprobs,
        }


def run_grpo(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text())
    model_id = cfg.get("model", "Qwen/Qwen3-1.7B")
    backend_choice = cfg.get("backend", "mlx")
    out_dir = Path(cfg.get("out_dir", f"data/grpo_runs/{int(time.time())}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    # Backend and sampler
    backend = pick_backend(model_id, prefer_mlx=True, backend=backend_choice)
    s_cfg = cfg.get("sampler", {})
    sampler = MLXSimpleSampler(
        backend,
        SimpleSamplerCfg(
            temperature=float(s_cfg.get("temperature", 0.9)),
            top_p=float(s_cfg.get("top_p", 0.92)),
            top_k=int(s_cfg.get("top_k", 120)),
            jitter_temperature=float(s_cfg.get("jitter", {}).get("temperature", 0.0)),
            jitter_top_p=float(s_cfg.get("jitter", {}).get("top_p", 0.0)),
        ),
        seed=cfg.get("seed"),
    )
    # Optional: load a saved logit-bias adapter
    bias_path = cfg.get("adapter_bias")
    if bias_path and Path(bias_path).exists():
        try:
            sampler.logit_bias = np.load(bias_path)
            print(f"Loaded adapter bias from {bias_path}")
        except Exception:
            pass

    # Prompts
    prompts: List[str] = cfg.get("prompts") or []
    pfile = cfg.get("prompts_file")
    if not prompts and pfile:
        with Path(pfile).open() as f:
            prompts = [json.loads(line)["prompt"] for line in f]
    if not prompts:
        prompts = ["Write a free-verse poem about tidal flats."]

    # Reward preset
    presets_file = cfg.get("reward", {}).get("presets_file", "configs/reward_presets.json")
    presets = load_presets(presets_file)
    preset_name = cfg.get("reward", {}).get("preset", "freeverse")
    preset = presets[preset_name]
    # Weight overrides
    wov = cfg.get("reward", {}).get("weights_override")
    if wov:
        for k, v in wov.items():
            if hasattr(preset.weights, k):
                setattr(preset.weights, k, float(v))
    # Format/rhyme/meter toggles
    if "format_lines" in cfg.get("reward", {}):
        preset.format_lines = cfg["reward"]["format_lines"]
    preset.enable_rhyme = bool(cfg.get("reward", {}).get("enable_rhyme", preset.enable_rhyme))
    preset.enable_meter = bool(cfg.get("reward", {}).get("enable_meter", preset.enable_meter))

    K = int(cfg.get("group_size", 6))
    T = int(cfg.get("max_new_tokens", 120))
    steps = int(cfg.get("steps", 1))

    idx_path = out_dir / "rollouts.jsonl"
    with idx_path.open("w", encoding="utf-8") as fout:
        for step in range(steps):
            for pr in prompts:
                samples: List[Dict[str, Any]] = []
                # Generate group
                for g in range(K):
                    smp = sampler.generate(pr, max_new_tokens=T, seed=(cfg.get("seed") or 0) + 1337 * g + 97 * step)
                    samples.append(smp)
                # Score rewards
                rewards: List[float] = []
                comps: List[Dict[str, float]] = []
                for smp in samples:
                    r, c = compute_reward({**smp, "prompt": pr}, preset)
                    rewards.append(float(r))
                    comps.append(c)
                # Group-relative baseline
                baseline = float(np.mean(rewards))
                advantages = [float(r - baseline) for r in rewards]
                # Save
                rec = {
                    "step": step,
                    "prompt": pr,
                    "preset": preset.name,
                    "rewards": rewards,
                    "advantages": advantages,
                    "group": [
                        {
                            "text": smp["gen_text"],
                            "lines": smp["lines"],
                            "tokens": smp["gen_token_ids"],
                            # do not dump logits (huge); keep small per-token stats optional later
                        }
                        for smp in samples
                    ],
                    "components": comps,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote rollouts + rewards to {idx_path}")
    return idx_path


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO rollouts and reward scoring (MLX)")
    ap.add_argument("--config", type=Path, default=Path("configs/grpo_default.json"))
    args = ap.parse_args()
    run_grpo(args.config)


if __name__ == "__main__":
    main()
