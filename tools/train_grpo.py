"""
Skeleton GRPO loop for Horace rewards.

This is a non-training stub to illustrate how to:
- sample K responses per prompt
- compute the v0 composite reward via tools.reward
- aggregate group-relative advantages

Wire this into your PPO/GRPO trainer of choice (e.g., TRL/OpenRLHF).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from pathlib import Path as _Path
import sys as _sys
_sys.path.append(str(_Path(__file__).resolve().parents[1]))
from tools.reward import load_presets, compute_reward


def sample_group(prompt: str, k: int) -> List[Dict[str, Any]]:
    """
    Placeholder sampler. Replace with actual model inference returning
    tokens, logits_per_step, decoded text, and line splits.
    """
    samples: List[Dict[str, Any]] = []
    for i in range(k):
        samples.append({
            "prompt": prompt,
            "text": f"[baseline placeholder output {i}]\n",
            "lines": [f"baseline placeholder line {i}"],
            "tokens": [],
            "logits_per_step": None,
            "line_embeddings": None,
        })
    return samples


def compute_group_rewards(samples: List[Dict[str, Any]], preset_cfg, refs=None, norm_stats=None) -> Tuple[List[float], List[Dict[str, float]]]:
    rewards: List[float] = []
    parts: List[Dict[str, float]] = []
    for s in samples:
        r, p = compute_reward(s, preset_cfg, refs=refs, norm_stats=norm_stats)
        rewards.append(r)
        parts.append(p)
    return rewards, parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=False, help="Path to a JSONL of prompts")
    ap.add_argument("--preset", type=str, default="freeverse", choices=["sonnet", "dickinson", "freeverse", "prose"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--presets_file", type=Path, default=Path("configs/reward_presets.json"))
    ap.add_argument("--out", type=Path, default=Path("data/grpo_debug.jsonl"))
    args = ap.parse_args()

    presets = load_presets(str(args.presets_file))
    preset_cfg = presets[args.preset]

    prompts: List[str] = [
        "Write a free-verse poem about tidal flats.",
        "Compose a sonnet about the first frost.",
    ]
    if args.prompts and args.prompts.exists():
        with args.prompts.open() as f:
            prompts = [json.loads(line)["prompt"] for line in f]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for pr in prompts:
            group = sample_group(pr, args.k)
            rewards, parts = compute_group_rewards(group, preset_cfg)
            baseline = sum(rewards) / max(1, len(rewards))
            advantages = [r - baseline for r in rewards]
            rec = {
                "prompt": pr,
                "preset": preset_cfg.name,
                "rewards": rewards,
                "advantages": advantages,
                "components": parts,
            }
            fout.write(json.dumps(rec) + "\n")
    print(f"Wrote debug rewards to {args.out}")


if __name__ == "__main__":
    main()
