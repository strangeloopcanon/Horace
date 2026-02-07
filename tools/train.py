"""Unified training entry point for Horace.

Dispatches to the appropriate trainer based on config:
  - method=grpo, adapter=bias  -> tools.grpo_train (MLX bias adapter)
  - method=grpo, adapter=lora  -> tools.grpo_train (MLX LoRA adapter)
  - method=grpo, adapter=full  -> tools.grpo_full_train (HF full weights)
  - method=gspo               -> tools.gspo_train (pairwise preference)
  - method=scorer              -> tools.studio.train_scorer (encoder fine-tune)
  - method=preference_scorer   -> tools.studio.train_preference_scorer

Usage:
  python -m tools.train --config configs/grpo_default.json
  python -m tools.train --config configs/scorer_v4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _detect_method(cfg: dict) -> str:
    """Infer training method from config if not explicitly set."""
    explicit = cfg.get("method") or cfg.get("train", {}).get("method", "")
    if explicit:
        return str(explicit).lower()

    # Heuristic detection
    if cfg.get("train", {}).get("adapter") in ("lora", "bias"):
        return "grpo"
    if "base-model" in cfg or "base_model" in cfg:
        return "scorer"
    if "chosen_text" in str(cfg) or "rejected_text" in str(cfg):
        return "preference_scorer"
    if cfg.get("reward"):
        adapter = cfg.get("train", {}).get("adapter", "bias")
        if adapter == "full":
            return "grpo"
        return "grpo"
    return "grpo"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Horace unified trainer: dispatches to GRPO, GSPO, or scorer training",
    )
    ap.add_argument("--config", type=Path, required=True, help="Training config JSON")
    ap.add_argument(
        "--method",
        choices=["grpo", "gspo", "scorer", "preference_scorer"],
        default=None,
        help="Override auto-detected training method",
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    method = args.method or _detect_method(cfg)

    if method == "grpo":
        adapter = cfg.get("train", {}).get("adapter", "bias")
        if adapter == "full":
            from tools.grpo_full_train import train_grpo_full
            train_grpo_full(args.config)
        else:
            from tools.grpo_train import train_grpo
            train_grpo(args.config)

    elif method == "gspo":
        from tools.gspo_train import train_gspo
        train_gspo(args.config)

    elif method == "scorer":
        # Re-invoke with the same config through its CLI
        import sys

        from tools.studio.train_scorer import main as scorer_main
        sys.argv = ["train_scorer", "--config", str(args.config)]
        scorer_main()

    elif method == "preference_scorer":
        import sys

        from tools.studio.train_preference_scorer import main as pref_main
        sys.argv = ["train_preference_scorer", "--config", str(args.config)]
        pref_main()

    else:
        raise ValueError(f"Unknown training method: {method}")


if __name__ == "__main__":
    main()
