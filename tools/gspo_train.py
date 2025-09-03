"""
GSPO training (MLX, bias adapter) — Grouped Preference Optimization using rewards.

Implements a pairwise preference loss over K sampled responses per prompt.
We optimize a lightweight logit-bias adapter b in R^V (vocab size) that is
added to the model logits before softmax during generation and likelihood eval.

Loss (DPO/BT-style on rewards): for a pair (i, j) with rewards r_i and r_j,
we prefer i over j with probability sigma(beta * (L_i - L_j)),
where L_i is the sequence log-likelihood under the current policy.

We minimize:  L = - sum_{pairs} w_ij * log sigma(beta * (L_i - L_j)) + lambda * ||b||^2
where w_ij is a weight derived from (r_i - r_j), default w_ij = sigmoid(k * (r_i - r_j)).

Notes:
- MLX rollouts with tools.analyze.pick_backend (prefer MLX) and MLXSimpleSampler.
- Updates only the bias adapter; base weights stay fixed.
- For speed, we reuse logits_per_step collected during sampling to compute L_i.

Config (example): see configs/gspo_default.json
{
  "model": "Qwen/Qwen3-1.7B",
  "backend": "mlx",
  "group_size": 6,
  "max_new_tokens": 120,
  "sampler": {"temperature": 0.9, "top_p": 0.92, "top_k": 120, "jitter": {"temperature": 0.05, "top_p": 0.03}},
  "reward": {"preset": "freeverse", "presets_file": "configs/reward_presets.json"},
  "prompts": ["Write a free-verse poem about tidal flats."],
  "out_dir": "data/grpo_runs/qwen3_1p7b_gspo_train",
  "gspo": {"beta": 0.05, "pair_k": 2.0, "l2": 1e-4, "lr": 0.05, "clipnorm": 5.0, "steps": 50, "save_every": 10}
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.analyze import pick_backend, ModelBackend
from tools.reward import load_presets, compute_reward
from tools.grpo_runner import MLXSimpleSampler, SimpleSamplerCfg


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def seq_loglik(logits_steps: List[np.ndarray], token_ids: List[int], bias: np.ndarray) -> float:
    """Compute total log-likelihood of the generated sequence under logits+bias."""
    if not logits_steps or not token_ids:
        return 0.0
    T = min(len(logits_steps), len(token_ids))
    total = 0.0
    V = bias.shape[0]
    for t in range(T):
        ls = logits_steps[t]
        if ls.shape[-1] != V:
            # Skip if vocab mismatch
            continue
        x = (ls.astype(np.float32) + bias)
        x = x - np.max(x)
        p = np.exp(x)
        p = p / (np.sum(p) + 1e-12)
        tid = int(token_ids[t])
        if 0 <= tid < V:
            total += float(np.log(max(1e-12, p[tid])))
    return total


def seq_grad_loglik(logits_steps: List[np.ndarray], token_ids: List[int], bias: np.ndarray) -> np.ndarray:
    """Gradient of sequence log-likelihood wrt bias vector (sum over steps of p - onehot)."""
    V = bias.shape[0]
    g = np.zeros((V,), dtype=np.float32)
    if not logits_steps or not token_ids:
        return g
    T = min(len(logits_steps), len(token_ids))
    for t in range(T):
        ls = logits_steps[t]
        if ls.shape[-1] != V:
            continue
        x = (ls.astype(np.float32) + bias)
        x = x - np.max(x)
        e = np.exp(x)
        p = e / (np.sum(e) + 1e-12)
        tid = int(token_ids[t])
        g += p
        if 0 <= tid < V:
            g[tid] -= 1.0
    return g


def train_gspo(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text())
    out_dir = Path(cfg.get("out_dir", f"data/grpo_runs/_gspo_{config_path.stem}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    # Backend and vocab
    model_id = cfg.get("model", "Qwen/Qwen3-1.7B")
    backend_choice = cfg.get("backend", "mlx")
    backend = pick_backend(model_id, prefer_mlx=True, backend=backend_choice)
    V = int(backend.vocab_size())

    # Sampler
    s_cfg_raw = cfg.get("sampler", {})
    sampler = MLXSimpleSampler(
        backend,
        SimpleSamplerCfg(
            temperature=float(s_cfg_raw.get("temperature", 0.9)),
            top_p=float(s_cfg_raw.get("top_p", 0.92)),
            top_k=int(s_cfg_raw.get("top_k", 120)),
            jitter_temperature=float(s_cfg_raw.get("jitter", {}).get("temperature", 0.0)),
            jitter_top_p=float(s_cfg_raw.get("jitter", {}).get("top_p", 0.0)),
        ),
        seed=cfg.get("seed"),
    )

    # Reward preset
    presets_file = cfg.get("reward", {}).get("presets_file", "configs/reward_presets.json")
    presets = load_presets(presets_file)
    preset_name = cfg.get("reward", {}).get("preset", "freeverse")
    preset = presets[preset_name]
    wov = cfg.get("reward", {}).get("weights_override")
    if wov:
        for k, v in wov.items():
            if hasattr(preset.weights, k):
                setattr(preset.weights, k, float(v))
    if "format_lines" in cfg.get("reward", {}):
        preset.format_lines = cfg["reward"]["format_lines"]
    preset.enable_rhyme = bool(cfg.get("reward", {}).get("enable_rhyme", preset.enable_rhyme))
    preset.enable_meter = bool(cfg.get("reward", {}).get("enable_meter", preset.enable_meter))
    preset.enable_grammar = bool(cfg.get("reward", {}).get("enable_grammar", getattr(preset, "enable_grammar", False)))

    # Prompts
    prompts: List[str] = cfg.get("prompts") or []
    pfile = cfg.get("prompts_file")
    if not prompts and pfile:
        with Path(pfile).open() as f:
            prompts = [json.loads(line).get("prompt", "") for line in f]
    if not prompts:
        prompts = ["Write a free-verse poem about tidal flats."]

    # GSPO hyperparams
    gspo = cfg.get("gspo", {})
    beta = float(gspo.get("beta", 0.05))
    pair_k = float(gspo.get("pair_k", 2.0))  # weight = sigmoid(pair_k * (r_i - r_j))
    lr = float(gspo.get("lr", 0.05))
    l2 = float(gspo.get("l2", 1e-4))
    clipnorm = float(gspo.get("clipnorm", 5.0))
    steps = int(gspo.get("steps", 50))
    save_every = int(gspo.get("save_every", 10))
    K = int(cfg.get("group_size", 6))
    T = int(cfg.get("max_new_tokens", 120))

    # Bias adapter params
    bias_path = out_dir / "logit_bias.npy"
    if bias_path.exists():
        bias = np.load(bias_path)
        if bias.shape[0] != V:
            bias = np.zeros((V,), dtype=np.float32)
    else:
        bias = np.zeros((V,), dtype=np.float32)
    sampler.logit_bias = bias

    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as flog:
        for step_idx in range(1, steps + 1):
            grad = np.zeros_like(bias)
            total_pairs, total_groups = 0, 0

            for pr in prompts:
                # Sample K responses
                samples = [
                    sampler.generate(pr, max_new_tokens=T, seed=(cfg.get("seed") or 0) + 1337 * g + 19 * step_idx)
                    for g in range(K)
                ]
                # Rewards
                rewards: List[float] = []
                for smp in samples:
                    r, _ = compute_reward({**smp, "prompt": pr}, preset)
                    rewards.append(float(r))

                # Precompute seq loglik and its gradient wrt bias
                seq_ll: List[float] = []
                seq_dL: List[np.ndarray] = []
                for smp in samples:
                    ll = seq_loglik(smp["logits_per_step"], smp["gen_token_ids"], bias)
                    dL = seq_grad_loglik(smp["logits_per_step"], smp["gen_token_ids"], bias)
                    seq_ll.append(ll)
                    seq_dL.append(dL)

                # Pairwise GSPO gradient accumulation
                n = len(samples)
                if n >= 2:
                    total_groups += 1
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                continue
                            ri, rj = rewards[i], rewards[j]
                            # weight from reward diff
                            w_ij = 1.0 / (1.0 + float(np.exp(-pair_k * (ri - rj))))
                            # logistic on (L_i - L_j)
                            z = beta * (seq_ll[i] - seq_ll[j])
                            sig = 1.0 / (1.0 + float(np.exp(-z)))
                            # d/d b of -w*log(sig) = -w*(1 - sig)*beta*(dL_i - dL_j)
                            coeff = -w_ij * (1.0 - sig) * beta
                            grad += coeff * (seq_dL[i] - seq_dL[j])
                            total_pairs += 1

            # Regularization
            if l2 > 0:
                grad += l2 * bias

            # Clip and update
            gnorm = float(np.linalg.norm(grad) + 1e-9)
            scale = 1.0
            if clipnorm > 0 and gnorm > clipnorm:
                scale = clipnorm / gnorm
            bias -= lr * scale * grad
            sampler.logit_bias = bias

            rec = {
                "step": step_idx,
                "pairs": total_pairs,
                "groups": total_groups,
                "grad_norm": gnorm,
                "bias_norm": float(np.linalg.norm(bias)),
                "lr": lr,
                "K": K,
                "T": T,
            }
            flog.write(json.dumps(rec) + "\n")
            if step_idx % save_every == 0 or step_idx == steps:
                np.save(bias_path, bias)
                (out_dir / f"bias_step_{step_idx:05d}.npy").write_bytes(b"")
            print(f"[GSPO {step_idx}/{steps}] pairs={total_pairs} grad||={gnorm:.3f} bias||={np.linalg.norm(bias):.3f}")

    print(f"GSPO training complete. Bias saved to {bias_path}")
    return bias_path


def main() -> None:
    ap = argparse.ArgumentParser(description="GSPO training (MLX bias adapter)")
    ap.add_argument("--config", type=Path, default=Path("configs/gspo_default.json"))
    args = ap.parse_args()
    train_gspo(args.config)


if __name__ == "__main__":
    main()

