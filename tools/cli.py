"""Horace CLI — quick text scoring without starting a server.

Usage:
  horace score "Your text here"
  echo "text" | horace score --stdin
  horace score --file passage.txt
  horace analyze "Your text here" --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _score_cmd(args: argparse.Namespace) -> None:
    from tools.studio.analyze import DEFAULT_BASELINE_MODEL, DEFAULT_SCORING_MODEL, analyze_text
    from tools.studio.baselines import build_baseline, load_baseline_cached
    from tools.studio.score import score_text

    # Resolve input text
    if args.stdin:
        text = sys.stdin.read()
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = " ".join(args.text)
    else:
        print("Error: provide text as argument, --file, or --stdin", file=sys.stderr)
        sys.exit(1)

    text = text.strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    model_id = args.model or DEFAULT_SCORING_MODEL
    baseline_id = args.baseline or DEFAULT_BASELINE_MODEL
    doc_type = args.doc_type or "prose"

    # Run analysis
    analysis = analyze_text(
        text,
        model_id=model_id,
        doc_type=doc_type,
        backend="auto",
        max_input_tokens=int(args.max_tokens),
        normalize_text=True,
        compute_cohesion=bool(args.cohesion),
    )

    # Load baseline and score
    try:
        baseline = load_baseline_cached(baseline_id)
    except Exception:
        build_baseline(baseline_id)
        baseline = load_baseline_cached(baseline_id)

    score = score_text(analysis["doc_metrics"], baseline, doc_type=doc_type)

    if args.json_output:
        out = {
            "score": score.overall_0_100,
            "categories": score.categories,
            "top_improvements": [
                {"category": h.category, "metric": h.metric, "direction": h.direction, "potential_gain": h.potential_gain}
                for h in (score.top_improvements or [])
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"\n  Score: {score.overall_0_100:.1f}/100\n")
        for cat, val in score.categories.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {cat:<16s} {bar} {val*100:.0f}")
        if score.top_improvements:
            print("\n  Top improvements:")
            for i, h in enumerate(score.top_improvements, 1):
                print(f"    {i}. {h.category}/{h.metric} ({h.direction}, gain: {h.potential_gain:.3f})")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="horace", description="Horace — literary cadence tools")
    sub = parser.add_subparsers(dest="command")

    # score subcommand
    score_p = sub.add_parser("score", help="Score text quality (0-100)")
    score_p.add_argument("text", nargs="*", default=None, help="Text to score")
    score_p.add_argument("--stdin", action="store_true", help="Read text from stdin")
    score_p.add_argument("--file", type=str, default=None, help="Read text from file")
    score_p.add_argument("--model", type=str, default=None, help="Scoring model (default: Qwen/Qwen3-0.6B)")
    score_p.add_argument("--baseline", type=str, default=None, help="Baseline model ID or path")
    score_p.add_argument("--doc-type", type=str, default="prose", choices=["prose", "poem", "novel", "shortstory"])
    score_p.add_argument("--max-tokens", type=int, default=512, help="Max input tokens")
    score_p.add_argument("--cohesion", action="store_true", help="Compute cohesion (slower)")
    score_p.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")

    # analyze subcommand (alias for score with --json)
    analyze_p = sub.add_parser("analyze", help="Full analysis with JSON output")
    analyze_p.add_argument("text", nargs="*", default=None, help="Text to analyze")
    analyze_p.add_argument("--stdin", action="store_true", help="Read text from stdin")
    analyze_p.add_argument("--file", type=str, default=None, help="Read text from file")
    analyze_p.add_argument("--model", type=str, default=None)
    analyze_p.add_argument("--baseline", type=str, default=None)
    analyze_p.add_argument("--doc-type", type=str, default="prose")
    analyze_p.add_argument("--max-tokens", type=int, default=512)
    analyze_p.add_argument("--cohesion", action="store_true")
    analyze_p.add_argument("--json", dest="json_output", action="store_true", default=True)

    # train subcommand
    train_p = sub.add_parser("train", help="Train a model (GRPO/GSPO/scorer)")
    train_p.add_argument("--config", type=str, required=True, help="Training config JSON")
    train_p.add_argument("--method", type=str, default=None, choices=["grpo", "gspo", "scorer", "preference_scorer"])

    args = parser.parse_args()

    if args.command in ("score", "analyze"):
        _score_cmd(args)
    elif args.command == "train":
        from tools.train import main as train_main
        sys.argv = ["horace-train", "--config", args.config]
        if args.method:
            sys.argv += ["--method", args.method]
        train_main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
