"""
Full GRPO training (HF/PyTorch) with PPO-style clipped objective.

- Samples K responses per prompt on-policy from the current model (manual loop with KV cache).
- Computes Horace rewards and group-relative advantages.
- Optimizes the base model weights directly (no LoRA/adapters).

Config: reuse configs/grpo_default.json (model, sampler, reward, prompts, out_dir) and add a `train` block:
{
  "model": "Qwen/Qwen3-1.7B",
  "backend": "hf",
  "group_size": 6,
  "max_new_tokens": 120,
  "sampler": {"temperature": 0.9, "top_p": 0.92, "top_k": 120, "jitter": {"temperature": 0.05, "top_p": 0.03}},
  "reward": {"preset": "freeverse", "presets_file": "configs/reward_presets.json"},
  "prompts": ["Write a free-verse poem about tidal flats."],
  "out_dir": "data/grpo_runs/qwen3_1p7b_fulltrain",
  "train": {"steps": 50, "lr": 5e-6, "clip": 0.2, "wd": 0.0, "grad_accum": 1, "save_every": 10}
}

Notes:
- Uses HF/PyTorch for training; for Apple Silicon this runs on MPS if available.
- Keeps reward flexible via tools/reward.py. Uses simple on-policy sampling (no cadence sampler).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.reward import load_presets, compute_reward


def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: Optional[int]) -> int:
    x = logits.float()
    if temperature and temperature > 0:
        x = x / temperature
    # top-k
    if top_k is not None and top_k > 0 and top_k < x.shape[-1]:
        v, idx = torch.topk(x, k=top_k)
        mask = torch.full_like(x, float('-inf'))
        mask.scatter_(0, idx, v)
        x = mask
    # top-p
    if 0.0 < top_p < 1.0:
        probs = torch.softmax(x, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        k = int(torch.searchsorted(cum, torch.tensor(top_p, device=cum.device)).item()) + 1
        keep_idx = sorted_idx[:k]
        mask = torch.full_like(x, float('-inf'))
        mask[keep_idx] = x[keep_idx]
        x = mask
    p = torch.softmax(x, dim=-1)
    tid = torch.multinomial(p, num_samples=1).item()
    return int(tid)


def generate_group(model, tok, prompt: str, K: int, T: int, sampler_cfg: Dict[str, Any], device: str, seed: Optional[int] = None):
    torch.manual_seed(seed or 0)
    group = []
    for g in range(K):
        # jitter per sample
        t = float(sampler_cfg.get('temperature', 0.9)) + np.random.uniform(-float(sampler_cfg.get('jitter', {}).get('temperature', 0.0)), float(sampler_cfg.get('jitter', {}).get('temperature', 0.0)))
        p = float(sampler_cfg.get('top_p', 0.92)) + np.random.uniform(-float(sampler_cfg.get('jitter', {}).get('top_p', 0.0)), float(sampler_cfg.get('jitter', {}).get('top_p', 0.0)))
        t = max(1e-6, t)
        p = float(np.clip(p, 0.05, 0.99))
        k = sampler_cfg.get('top_k')

        enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids = enc['input_ids'][0].tolist()
        with torch.no_grad():
            out = model(input_ids=enc['input_ids'], use_cache=True)
            past = out.past_key_values if hasattr(out, 'past_key_values') else out.past
            last_logits = out.logits[0, -1, :].detach()

        gen_ids: List[int] = []
        logits_steps: List[np.ndarray] = []
        old_logprobs: List[float] = []
        gen_text = ''
        line_steps: List[Tuple[int, int]] = []
        curr_line = 0

        for step in range(T):
            logits_steps.append(last_logits.float().cpu().numpy())
            tid = top_p_sample(last_logits, t, p, k)
            gen_ids.append(tid)
            with torch.no_grad():
                piece = tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            gen_text += piece
            if '\n' in piece:
                line_steps.append((curr_line, step + 1))
                curr_line = step + 1
            # logprob under sampling policy
            lp = torch.log_softmax(last_logits.float(), dim=-1)[tid].item()
            old_logprobs.append(lp)
            # advance cache
            with torch.no_grad():
                nxt = torch.tensor([[tid]], dtype=torch.long, device=device)
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values if hasattr(out, 'past_key_values') else out.past
                last_logits = out.logits[0, -1, :].detach()

        if curr_line < len(gen_ids):
            line_steps.append((curr_line, len(gen_ids)))

        lines = gen_text.splitlines()
        group.append({
            'prompt': prompt,
            'gen_text': gen_text,
            'lines': lines,
            'gen_token_ids': gen_ids,
            'logits_per_step': logits_steps,
            'line_steps': line_steps,
            'old_logprobs': old_logprobs,
            'input_ids': input_ids,
        })
    return group


def ppo_loss_for_group(
    model,
    tok,
    batch: List[Dict[str, Any]],
    advantages: List[float],
    clip_eps: float,
    device: str,
    ref_model=None,
) -> Tuple[torch.Tensor, float]:
    """Compute PPO loss for a group and optional KL(new||ref).

    Returns (loss, mean_kl) where mean_kl is a float (0.0 if ref_model is None).
    """
    losses = []
    kl_vals: List[float] = []
    for smp, A in zip(batch, advantages):
        if abs(A) < 1e-8:
            continue
        # Build sequence: prompt + gen
        ids_list = smp['input_ids'] + smp['gen_token_ids']
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        out = model(input_ids=ids)
        logits = out.logits[0]  # [L, V]
        # Shift for next-token prediction
        Lp = len(smp['input_ids'])
        Tg = len(smp['gen_token_ids'])
        # next positions for generated tokens are indices Lp-1 .. Lp+Tg-2 in logits
        logps_new = []
        pos_indices: List[int] = []
        for i in range(Tg):
            pos = Lp - 1 + i
            if pos < 0 or pos >= logits.shape[0]:
                continue
            lp = torch.log_softmax(logits[pos], dim=-1)[smp['gen_token_ids'][i]]
            logps_new.append(lp)
            pos_indices.append(pos)
        if not logps_new:
            continue
        logps_new = torch.stack(logps_new)
        logps_old = torch.tensor(smp['old_logprobs'][: len(logps_new)], dtype=torch.float32, device=device)
        ratio = torch.exp(logps_new - logps_old)
        a = torch.tensor(float(A), dtype=torch.float32, device=device)
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a
        # negative sign because we minimize
        loss_seq = -torch.mean(torch.minimum(unclipped, clipped))
        losses.append(loss_seq)

        # KL(new || ref) over the same positions
        if ref_model is not None and len(pos_indices) > 0:
            with torch.no_grad():
                ref_out = ref_model(input_ids=ids)
                ref_logits = ref_out.logits[0]
            # Compute KL per position
            kl_sum = 0.0
            for pos in pos_indices:
                p = torch.softmax(logits[pos].detach(), dim=-1)
                q = torch.softmax(ref_logits[pos].detach(), dim=-1)
                kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)))
                kl_sum += float(kl.item())
            kl_vals.append(kl_sum / len(pos_indices))

    mean_kl = float(np.mean(kl_vals)) if kl_vals else 0.0
    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=device), mean_kl
    return torch.stack(losses).mean(), mean_kl


def train_full_grpo(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text())
    out_dir = Path(cfg.get('out_dir', f"data/grpo_runs/_full_{config_path.stem}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    model_id = cfg.get('model', 'Qwen/Qwen3-1.7B')
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.train()

    # Reference model for KL penalty (frozen)
    ref_model = None
    ref_kl_coef = float(cfg.get('train', {}).get('kl_coef', 0.0))
    if ref_kl_coef > 0.0:
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model.to(device)
        ref_model.eval()

    presets = load_presets(cfg.get('reward', {}).get('presets_file', 'configs/reward_presets.json'))
    preset = presets[cfg.get('reward', {}).get('preset', 'freeverse')]
    wov = cfg.get("reward", {}).get("weights_override")
    if wov:
        for k, v in wov.items():
            if hasattr(preset.weights, k):
                setattr(preset.weights, k, float(v))

    prompts: List[str] = cfg.get('prompts') or []
    pfile = cfg.get('prompts_file')
    if not prompts and pfile:
        with Path(pfile).open() as f:
            prompts = [json.loads(line)['prompt'] for line in f]
    if not prompts:
        prompts = ["Write a free-verse poem about tidal flats."]

    sampler_cfg = cfg.get('sampler', {})
    train_cfg = cfg.get('train', {})
    steps = int(train_cfg.get('steps', 50))
    lr = float(train_cfg.get('lr', 5e-6))
    wd = float(train_cfg.get('wd', 0.0))
    clip_eps = float(train_cfg.get('clip', 0.2))
    grad_accum = int(train_cfg.get('grad_accum', 1))
    save_every = int(train_cfg.get('save_every', 10))
    use_amp = bool(train_cfg.get('amp', False))
    amp_dtype = str(train_cfg.get('amp_dtype', 'bf16')).lower()
    amp_torch_dtype = torch.bfloat16 if amp_dtype in ('bf16', 'bfloat16') else torch.float16
    K = int(cfg.get('group_size', 6))
    T = int(cfg.get('max_new_tokens', 120))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device == 'cuda')

    log_path = out_dir / 'train_log.jsonl'
    with log_path.open('w', encoding='utf-8') as flog:
        step_idx = 0
        while step_idx < steps:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for acc in range(grad_accum):
                # On-policy rollouts per prompt
                total_reward, total_samples = 0.0, 0
                all_batches: List[Tuple[List[Dict[str, Any]], List[float]]] = []
                for pr in prompts:
                    group = generate_group(model, tok, pr, K=K, T=T, sampler_cfg=sampler_cfg, device=device, seed=cfg.get('seed'))
                    rewards: List[float] = []
                    for smp in group:
                        r, _ = compute_reward({**smp, 'prompt': pr}, preset)
                        rewards.append(float(r))
                    baseline = float(np.mean(rewards))
                    adv = [float(r - baseline) for r in rewards]
                    total_reward += float(np.sum(rewards))
                    total_samples += len(rewards)
                    all_batches.append((group, adv))

                # Compute PPO loss (+ KL) and backprop
                loss_val = torch.tensor(0.0, device=device)
                total_kl = 0.0
                autocast_ctx = (
                    torch.autocast(device_type='cuda', dtype=amp_torch_dtype) if (use_amp and device == 'cuda') else
                    torch.autocast(device_type='cpu', dtype=amp_torch_dtype) if (use_amp and device == 'cpu') else
                    torch.cuda.amp.autocast(enabled=False)
                )
                with autocast_ctx:
                    for group, adv in all_batches:
                        l, mk = ppo_loss_for_group(model, tok, group, adv, clip_eps=clip_eps, device=device, ref_model=ref_model)
                        loss_val = loss_val + l
                        total_kl += mk
                    loss_val = loss_val / max(1, len(all_batches))
                    if ref_model is not None and ref_kl_coef > 0.0:
                        loss_val = loss_val + float(ref_kl_coef) * torch.tensor(total_kl / max(1, len(all_batches)), device=device)
                if scaler.is_enabled():
                    scaler.scale(loss_val).backward()
                else:
                    loss_val.backward()
                accum_loss += float(loss_val.item())

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            step_idx += 1

            avg_r = total_reward / max(1, total_samples)
            rec = {"step": step_idx, "avg_reward": avg_r, "loss": accum_loss / max(1, grad_accum), "K": K, "T": T, "kl_coef": ref_kl_coef}
            flog.write(json.dumps(rec) + "\n")
            print(f"[full-GRPO {step_idx}/{steps}] avgR={avg_r:.3f} loss={rec['loss']:.4f}")

            if step_idx % save_every == 0 or step_idx == steps:
                save_dir = out_dir / f"checkpoint_step_{step_idx:05d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tok.save_pretrained(save_dir)

    print(f"Training complete. Final checkpoint in {save_dir}")
    return save_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Full GRPO training (HF/PyTorch)")
    ap.add_argument("--config", type=Path, default=Path("configs/grpo_default.json"))
    args = ap.parse_args()
    train_full_grpo(args.config)


if __name__ == "__main__":
    main()
