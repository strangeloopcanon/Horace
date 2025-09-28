#!/usr/bin/env python3
"""Generic GEPA runner configurable via module hooks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from experiments.gepa_gemma3.lm_clients import AIStudioGemmaLM, LocalGemmaLM


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA with a configurable dataset/metric module.")
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Python module path providing GEPA hooks (default: experiments.gepa_gemma3.configs.writing).",
    )
    parser.add_argument("--train", type=int, default=12, help="Number of training tasks")
    parser.add_argument("--val", type=int, default=6, help="Number of validation tasks")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for task split")
    parser.add_argument("--max-metric-calls", type=int, default=90, help="GEPA metric budget")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/gepa_gemma3/outputs"))
    parser.add_argument("--task-model", type=str, default=os.getenv("TASK_MODEL", "gemma-3-4b-it"))
    parser.add_argument("--reflect-model", type=str, default=os.getenv("REFLECT_MODEL", "openai/gpt-4.1-mini"))
    parser.add_argument("--reflection-minibatch", type=int, default=4)
    parser.add_argument(
        "--candidate-strategy",
        default="pareto",
        choices=["pareto", "topk", "top1"],
    )
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = os.getenv("GEPA_CONFIG", "experiments.gepa_gemma3.configs.writing")
    return args


def load_config(path: str):
    module = importlib.import_module(path)
    required = [
        "build_datasets",
        "build_student",
        "get_seed_prompt",
        "metric",
        "prediction_to_dict",
    ]
    for attr in required:
        if not hasattr(module, attr):
            raise AttributeError(f"Config module {path} missing required attribute '{attr}'")
    return module


def ensure_lm(task_model: str) -> Tuple[str, dspy.BaseLM]:
    resolved_task_model = task_model
    if task_model.startswith("local:"):
        local_name = task_model.split("local:", 1)[1]
        lm = LocalGemmaLM(local_name)
        resolved_task_model = f"local/{local_name}"
    elif task_model.startswith("aistudio/") or task_model in {"gemma-3-4b-it", "google/gemma-3-4b-it"}:
        model_name = task_model.split("/", 1)[1] if "/" in task_model else task_model
        lm = AIStudioGemmaLM(model_name)
        resolved_task_model = f"aistudio/{model_name}"
    elif task_model.startswith("openrouter/"):
        local_name = task_model.split("/", 1)[1]
        if os.getenv("OPENROUTER_API_KEY"):
            lm = dspy.LM(task_model)
        elif os.getenv("GEMINI_API_KEY"):
            lm = AIStudioGemmaLM(local_name)
            resolved_task_model = f"aistudio/{local_name}"
        else:
            lm = LocalGemmaLM(local_name)
            resolved_task_model = f"local/{local_name}"
    else:
        lm = dspy.LM(task_model)
    dspy.configure(lm=lm)
    return resolved_task_model, lm


def summarise_scores(entries: list[dict[str, Any]]) -> dict[str, float]:
    if not entries:
        return {}
    scores = [e["score"] for e in entries]
    return {
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def run_eval(model: dspy.Module, dataset: list[dspy.Example], metric: Callable, pred_to_dict: Callable) -> list[dict[str, Any]]:
    rows = []
    for example in dataset:
        pred = model(instruction=example.instruction, context=example.context)
        score = metric(example, pred)
        rows.append(
            {
                "id": example.get("id"),
                "score": score["score"],
                "feedback": score["feedback"],
                "prediction": pred_to_dict(pred),
                "reference": getattr(example, "answer", None),
            }
        )
    return rows


def write_outputs(
    run_dir: Path,
    seed_instruction: str,
    optimized_prompt: str,
    train_eval: list[dict[str, Any]],
    val_eval: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "seed_prompt.txt").write_text(seed_instruction, encoding="utf-8")
    (run_dir / "optimized_prompt.txt").write_text(optimized_prompt, encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    config = load_config(args.config)

    trainset, valset = config.build_datasets(args.train, args.val, args.seed)
    metric_fn = config.metric()
    seed_instruction = config.get_seed_prompt()
    student = config.build_student(seed_instruction)
    pred_to_dict = getattr(config, "prediction_to_dict")

    resolved_model, _ = ensure_lm(args.task_model)
    reflect_model = args.reflect_model
    reflect_lm = dspy.LM(reflect_model)

    gepa = dspy.GEPA(
        metric=metric_fn,
        reflection_minibatch_size=args.reflection_minibatch,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflect_lm,
        candidate_selection_strategy=args.candidate_strategy,
        track_stats=True,
        track_best_outputs=True,
    )

    optimized = gepa.compile(student, trainset=trainset, valset=valset)
    optimized_prompt = optimized.step.signature.instructions

    train_eval = run_eval(optimized, trainset, metric_fn, pred_to_dict)
    val_eval = run_eval(optimized, valset, metric_fn, pred_to_dict)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = args.config.split(".")[-1]
    run_dir = args.output_dir / config_name / timestamp

    summary = {
        "timestamp": timestamp,
        "config": args.config,
        "task_model": resolved_model,
        "reflect_model": reflect_model,
        "train_task_count": len(trainset),
        "val_task_count": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "train_scores": summarise_scores(train_eval),
        "val_scores": summarise_scores(val_eval),
        "train_details": train_eval,
        "val_details": val_eval,
        "seed_instruction": seed_instruction,
        "optimized_instruction": optimized_prompt,
    }

    write_outputs(run_dir, seed_instruction, optimized_prompt, train_eval, val_eval, summary)
    print("=== Optimized Prompt ===\n")
    print(optimized_prompt)
    print(f"\nResults stored in {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
