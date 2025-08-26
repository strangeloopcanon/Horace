"""
GRPO training with a lightweight logit-bias adapter on MLX.

This trains a learnable bias vector b in R^V (vocab size), added to model
logits before softmax. We use group-relative rewards (GRPO) to update b via
policy gradient with baseline. This avoids full LLM finetuning while staying
in MLX for rollouts and providing a true on-policy update.

Config: reuse `configs/grpo_default.json` and add a `train` section, e.g.:
{
  "model": "Qwen/Qwen3-1.7B",
  "backend": "mlx",
  "group_size": 6,
  "max_new_tokens": 120,
  "sampler": {"temperature": 0.9, "top_p": 0.92, "top_k": 120, "jitter": {"temperature": 0.05, "top_p": 0.03}},
  "reward": {"preset": "freeverse", "presets_file": "configs/reward_presets.json"},
  "prompts": ["Write a free-verse poem about tidal flats."],
  "out_dir": "data/grpo_runs/qwen3_1p7b_freeverse_train",
  "train": {"steps": 50, "lr": 0.05, "l2": 1e-4, "clipnorm": 5.0, "save_every": 10}
}

Notes:
- This updates only the bias adapter, not the base model weights.
- The bias can be applied in generation by attaching it to the sampler.
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


def train_grpo(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text())
    out_dir = Path(cfg.get("out_dir", f"data/grpo_runs/_train_{config_path.stem}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    # Backend and vocab
    model_id = cfg.get("model", "Qwen/Qwen3-1.7B")
    backend_choice = cfg.get("backend", "mlx")
    backend = pick_backend(model_id, prefer_mlx=True, backend=backend_choice)
    V = int(backend.vocab_size())

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

    # Prompts
    prompts: List[str] = cfg.get("prompts") or []
    pfile = cfg.get("prompts_file")
    if not prompts and pfile:
        with Path(pfile).open() as f:
            prompts = [json.loads(line)["prompt"] for line in f]
    if not prompts:
        prompts = ["Write a free-verse poem about tidal flats."]

    # Training hyperparams
    train = cfg.get("train", {})
    steps = int(train.get("steps", 50))
    lr = float(train.get("lr", 0.05))
    l2 = float(train.get("l2", 1e-4))
    clipnorm = float(train.get("clipnorm", 5.0))
    save_every = int(train.get("save_every", 10))
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

    # Training loop
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as flog:
        for step_idx in range(1, steps + 1):
            total_reward = 0.0
            total_samples = 0
            grad = np.zeros_like(bias)

            for pr in prompts:
                # Sample a group
                samples = [sampler.generate(pr, max_new_tokens=T, seed=(cfg.get("seed") or 0) + 1337 * g + 17 * step_idx) for g in range(K)]
                # Rewards and advantages
                rewards: List[float] = []
                for smp in samples:
                    r, _ = compute_reward({**smp, "prompt": pr}, preset)
                    rewards.append(float(r))
                baseline = float(np.mean(rewards))
                adv = [float(r - baseline) for r in rewards]
                total_reward += float(np.sum(rewards))
                total_samples += len(rewards)

                # Policy gradient w.r.t. bias: sum_t A*(p - onehot)
                for a, smp in zip(adv, samples):
                    if abs(a) < 1e-8:
                        continue
                    for ls, tid in zip(smp["logits_per_step"], smp["gen_token_ids"]):
                        p = softmax(ls)
                        grad += float(a) * p
                        grad[int(tid)] -= float(a)

            # Regularization (acts like KL to base by keeping bias small)
            if l2 > 0:
                grad += l2 * bias

            # Clip gradient norm
            gnorm = float(np.linalg.norm(grad) + 1e-8)
            scale = 1.0
            if clipnorm > 0 and gnorm > clipnorm:
                scale = clipnorm / gnorm
            bias -= lr * scale * grad

            # Update sampler view
            sampler.logit_bias = bias

            avg_reward = total_reward / max(1, total_samples)
            rec = {
                "step": step_idx,
                "avg_reward": avg_reward,
                "grad_norm": gnorm,
                "bias_norm": float(np.linalg.norm(bias)),
                "lr": lr,
                "K": K,
                "T": T,
            }
            flog.write(json.dumps(rec) + "\n")
            if step_idx % save_every == 0 or step_idx == steps:
                np.save(bias_path, bias)
                (out_dir / f"bias_step_{step_idx:05d}.npy").write_bytes(b"")  # marker file
            print(f"[step {step_idx}/{steps}] avgR={avg_reward:.3f} grad||={gnorm:.3f} bias||={np.linalg.norm(bias):.3f}")

    print(f"Training complete. Bias saved to {bias_path}")
    return bias_path


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO training (logit-bias adapter, MLX rollouts)")
    ap.add_argument("--config", type=Path, default=Path("configs/grpo_default.json"))
    args = ap.parse_args()
    train_grpo(args.config)


if __name__ == "__main__":
    main()

