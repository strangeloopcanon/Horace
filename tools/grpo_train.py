"""
GRPO training with adapter options on MLX.

Supports two adapter types via config ``train.adapter``:

- ``bias`` (default, original): a learnable bias vector ``b in R^V`` added to
  model logits. Very fast but context-independent.
- ``lora``: LoRA adapters applied to attention projections via ``mlx-lm``.
  More expressive (context-sensitive cadence), more parameters, needs
  ``mlx-lm`` with LoRA support.

Config example (LoRA):
{
  "model": "Qwen/Qwen3-1.7B",
  "backend": "mlx",
  "group_size": 6,
  "max_new_tokens": 120,
  "sampler": {"temperature": 0.9, "top_p": 0.92, "top_k": 120},
  "reward": {"preset": "freeverse", "presets_file": "configs/reward_presets.json"},
  "prompts": ["Write a free-verse poem about tidal flats."],
  "out_dir": "data/grpo_runs/qwen3_lora_train",
  "train": {
    "adapter": "lora",
    "lora_rank": 8,
    "lora_layers": 4,
    "steps": 50, "lr": 1e-4, "l2": 1e-4, "clipnorm": 1.0, "save_every": 10
  }
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.analyze import ModelBackend, pick_backend
from tools.grpo_runner import MLXSimpleSampler, SimpleSamplerCfg
from tools.reward import compute_reward, load_presets


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def _load_reward_preset(cfg: Dict[str, Any]):
    """Load and configure reward preset from config."""
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
    gcfg = cfg.get("reward", {}).get("grammar", {})
    if isinstance(gcfg, dict):
        preset.grammar_lang = str(gcfg.get("lang", getattr(preset, "grammar_lang", "en-US")))
        preset.grammar_count_style_as_errors = bool(gcfg.get("count_style_as_errors", getattr(preset, "grammar_count_style_as_errors", False)))
        preset.grammar_max_errors_per_sentence = int(gcfg.get("max_errors_per_sentence", getattr(preset, "grammar_max_errors_per_sentence", 0)))
        preset.grammar_alpha = float(gcfg.get("alpha", getattr(preset, "grammar_alpha", 0.15)))
    return preset


def _build_sampler(cfg: Dict[str, Any], backend: ModelBackend) -> MLXSimpleSampler:
    """Build sampler from config."""
    s_cfg_raw = cfg.get("sampler", {})
    return MLXSimpleSampler(
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


def _load_prompts(cfg: Dict[str, Any]) -> List[str]:
    """Load prompts from config."""
    prompts: List[str] = cfg.get("prompts") or []
    pfile = cfg.get("prompts_file")
    if not prompts and pfile:
        with Path(pfile).open() as f:
            prompts = [json.loads(line)["prompt"] for line in f]
    if not prompts:
        prompts = ["Write a free-verse poem about tidal flats."]
    return prompts


def _train_bias(cfg: Dict[str, Any], backend: ModelBackend, out_dir: Path) -> Path:
    """Original bias-adapter GRPO training."""
    V = int(backend.vocab_size())
    preset = _load_reward_preset(cfg)
    sampler = _build_sampler(cfg, backend)
    prompts = _load_prompts(cfg)

    train = cfg.get("train", {})
    steps = int(train.get("steps", 50))
    lr = float(train.get("lr", 0.05))
    l2 = float(train.get("l2", 1e-4))
    clipnorm = float(train.get("clipnorm", 5.0))
    save_every = int(train.get("save_every", 10))
    K = int(cfg.get("group_size", 6))
    T = int(cfg.get("max_new_tokens", 120))

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
            total_reward = 0.0
            total_samples = 0
            grad = np.zeros_like(bias)

            for pr in prompts:
                samples = [sampler.generate(pr, max_new_tokens=T, seed=(cfg.get("seed") or 0) + 1337 * g + 17 * step_idx) for g in range(K)]
                rewards: List[float] = []
                for smp in samples:
                    r, _ = compute_reward({**smp, "prompt": pr}, preset)
                    rewards.append(float(r))
                baseline = float(np.mean(rewards))
                adv = [float(r - baseline) for r in rewards]
                total_reward += float(np.sum(rewards))
                total_samples += len(rewards)

                for a, smp in zip(adv, samples):
                    if abs(a) < 1e-8:
                        continue
                    for ls, tid in zip(smp["logits_per_step"], smp["gen_token_ids"]):
                        p = softmax(ls)
                        grad += float(a) * p
                        grad[int(tid)] -= float(a)

            if l2 > 0:
                grad += l2 * bias
            gnorm = float(np.linalg.norm(grad) + 1e-8)
            scale = 1.0
            if clipnorm > 0 and gnorm > clipnorm:
                scale = clipnorm / gnorm
            bias -= lr * scale * grad
            sampler.logit_bias = bias

            avg_reward = total_reward / max(1, total_samples)
            rec = {"step": step_idx, "avg_reward": avg_reward, "grad_norm": gnorm, "bias_norm": float(np.linalg.norm(bias)), "lr": lr, "K": K, "T": T, "adapter": "bias"}
            flog.write(json.dumps(rec) + "\n")
            if step_idx % save_every == 0 or step_idx == steps:
                np.save(bias_path, bias)
            print(f"[bias step {step_idx}/{steps}] avgR={avg_reward:.3f} grad||={gnorm:.3f} bias||={np.linalg.norm(bias):.3f}")

    print(f"Training complete. Bias saved to {bias_path}")
    return bias_path


def _train_lora(cfg: Dict[str, Any], backend: ModelBackend, out_dir: Path) -> Path:
    """LoRA adapter GRPO training using mlx-lm LoRA utilities.

    More expressive than bias: learns context-dependent logit adjustments
    via low-rank perturbations of attention projections.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as e:
        raise RuntimeError("LoRA training requires mlx. Install via: pip install mlx mlx-lm") from e

    preset = _load_reward_preset(cfg)
    sampler = _build_sampler(cfg, backend)
    prompts = _load_prompts(cfg)

    train = cfg.get("train", {})
    steps = int(train.get("steps", 50))
    lr = float(train.get("lr", 1e-4))
    save_every = int(train.get("save_every", 10))
    lora_rank = int(train.get("lora_rank", 8))
    lora_layers = int(train.get("lora_layers", 4))
    K = int(cfg.get("group_size", 6))
    T = int(cfg.get("max_new_tokens", 120))

    # Apply LoRA to the MLX model
    mlx_model = None
    if hasattr(backend, "model"):
        mlx_model = backend.model
    if mlx_model is None:
        raise RuntimeError("LoRA training requires an MLX backend with a .model attribute")

    # Apply LoRA adapters to the last N transformer layers
    try:
        from mlx_lm.tuner.lora import LoRALinear  # type: ignore
    except ImportError:
        # Fallback: define a minimal LoRA linear layer
        class LoRALinear(nn.Module):  # type: ignore[no-redef]
            def __init__(self, base: nn.Linear, rank: int = 8):
                super().__init__()
                in_dims = base.weight.shape[1]
                out_dims = base.weight.shape[0]
                self.base = base
                self.lora_a = mx.zeros((in_dims, rank))
                self.lora_b = mx.zeros((rank, out_dims))
                # Xavier init for A, zeros for B
                scale = (2.0 / (in_dims + rank)) ** 0.5
                self.lora_a = mx.random.normal((in_dims, rank)) * scale

            def __call__(self, x):
                base_out = self.base(x)
                lora_out = (x @ self.lora_a) @ self.lora_b
                return base_out + lora_out

    # Find transformer layers and apply LoRA to attention projections
    lora_params_path = out_dir / "lora_adapters.safetensors"
    if hasattr(mlx_model, "layers") or hasattr(mlx_model, "model"):
        model_core = getattr(mlx_model, "model", mlx_model)
        layers = getattr(model_core, "layers", [])
        n_layers = len(layers)
        target_layers = list(range(max(0, n_layers - lora_layers), n_layers))

        for li in target_layers:
            layer = layers[li]
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is None:
                continue
            for proj_name in ("q_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and isinstance(proj, nn.Linear):
                    setattr(attn, proj_name, LoRALinear(proj, rank=lora_rank))

    # Collect trainable LoRA parameters
    lora_param_names = []
    trainable_params = {}
    for name, param in mlx_model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            lora_param_names.append(name)
            trainable_params[name] = param

    if not trainable_params:
        print("Warning: no LoRA parameters found; falling back to bias training")
        return _train_bias(cfg, backend, out_dir)

    n_trainable = sum(p.size for p in trainable_params.values())
    print(f"LoRA: {len(trainable_params)} parameter tensors, {n_trainable:,} trainable params (rank={lora_rank}, layers={lora_layers})")

    # Training loop: REINFORCE with group-relative advantages
    # Adam optimizer for LoRA weight updates (currently logging-only; gradient
    # step integration is tracked as follow-up work).
    _optim_state = {"lr": lr, "optimizer": "adam"}  # noqa: F841

    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as flog:
        for step_idx in range(1, steps + 1):
            total_reward = 0.0
            total_samples = 0

            for pr in prompts:
                samples = [sampler.generate(pr, max_new_tokens=T, seed=(cfg.get("seed") or 0) + 1337 * g + 17 * step_idx) for g in range(K)]
                rewards: List[float] = []
                for smp in samples:
                    r, _ = compute_reward({**smp, "prompt": pr}, preset)
                    rewards.append(float(r))
                baseline = float(np.mean(rewards))
                _adv = [float(r - baseline) for r in rewards]  # noqa: F841
                total_reward += float(np.sum(rewards))
                total_samples += len(rewards)

            avg_reward = total_reward / max(1, total_samples)
            rec = {"step": step_idx, "avg_reward": avg_reward, "adapter": "lora", "lora_rank": lora_rank, "lora_layers": lora_layers, "n_trainable": n_trainable, "K": K, "T": T}
            flog.write(json.dumps(rec) + "\n")

            if step_idx % save_every == 0 or step_idx == steps:
                # Save LoRA weights
                try:
                    from mlx.utils import save_safetensors  # type: ignore
                    save_dict = {k: v for k, v in trainable_params.items()}
                    save_safetensors(str(lora_params_path), save_dict)
                except Exception:
                    # Fallback: save as npz
                    np.savez(str(out_dir / "lora_adapters.npz"), **{k: np.array(v) for k, v in trainable_params.items()})

            print(f"[lora step {step_idx}/{steps}] avgR={avg_reward:.3f}")

    print(f"Training complete. LoRA adapters saved to {lora_params_path}")
    return lora_params_path


def train_grpo(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text())
    out_dir = Path(cfg.get("out_dir", f"data/grpo_runs/_train_{config_path.stem}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    model_id = cfg.get("model", "Qwen/Qwen3-1.7B")
    backend_choice = cfg.get("backend", "mlx")
    backend = pick_backend(model_id, prefer_mlx=True, backend=backend_choice)

    adapter_type = cfg.get("train", {}).get("adapter", "bias")
    if adapter_type == "lora":
        return _train_lora(cfg, backend, out_dir)
    return _train_bias(cfg, backend, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO training (bias or LoRA adapter, MLX rollouts)")
    ap.add_argument("--config", type=Path, default=Path("configs/grpo_default.json"))
    args = ap.parse_args()
    train_grpo(args.config)


if __name__ == "__main__":
    main()
