"""Human evaluation framework for measuring correlation between Horace scores and human judgment.

Provides tools to:
1. Build an annotated test set from passages
2. Collect human quality ratings (1-10 scale)
3. Compute Horace scores for the same passages
4. Measure Pearson/Spearman correlation

Usage:
  # Create test set from text files:
  python -m tools.studio.human_eval create --input-dir data/eval_passages --output data/eval/test_set.jsonl

  # Score a test set with Horace:
  python -m tools.studio.human_eval score --test-set data/eval/test_set.jsonl --output data/eval/scored.jsonl

  # Compute correlation between human and Horace scores:
  python -m tools.studio.human_eval correlate --scored data/eval/scored.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EvalPassage:
    """A single passage for evaluation."""
    id: str
    text: str
    source: str = ""
    genre: str = "prose"
    # Human annotations (list of ratings from different raters)
    human_ratings: List[float] = None  # type: ignore[assignment]
    human_mean: Optional[float] = None
    # Horace scores
    horace_score: Optional[float] = None
    horace_categories: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.human_ratings is None:
            self.human_ratings = []


def create_test_set(
    input_dir: Path,
    output_path: Path,
    *,
    min_words: int = 50,
    max_words: int = 500,
    passages_per_file: int = 1,
) -> List[EvalPassage]:
    """Create evaluation passages from text files in a directory."""
    passages: List[EvalPassage] = []
    idx = 0

    for fpath in sorted(input_dir.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8").strip()
        if not text:
            continue

        # Split into chunks if the file is long
        words = text.split()
        if len(words) < min_words:
            continue

        for chunk_start in range(0, len(words), max_words):
            chunk_words = words[chunk_start : chunk_start + max_words]
            if len(chunk_words) < min_words:
                continue
            chunk_text = " ".join(chunk_words)
            idx += 1
            passages.append(EvalPassage(
                id=f"eval_{idx:04d}",
                text=chunk_text,
                source=fpath.stem,
                genre="prose",
            ))
            if len(passages) >= passages_per_file * len(list(input_dir.glob("*.txt"))):
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    return passages


def add_human_rating(
    test_set_path: Path,
    passage_id: str,
    rating: float,
    rater_id: str = "anonymous",
) -> None:
    """Add a human rating to a passage in the test set."""
    rows = []
    with test_set_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))

    for row in rows:
        if row["id"] == passage_id:
            if row.get("human_ratings") is None:
                row["human_ratings"] = []
            row["human_ratings"].append(float(rating))
            row["human_mean"] = float(np.mean(row["human_ratings"]))
            break

    with test_set_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_test_set(
    test_set_path: Path,
    output_path: Path,
    *,
    model_id: str = "Qwen/Qwen3-0.6B",
    baseline_model: str = "gpt2_gutenberg_512",
    max_input_tokens: int = 512,
) -> List[EvalPassage]:
    """Score all passages in a test set with Horace."""
    from tools.studio.analyze import analyze_text
    from tools.studio.baselines import build_baseline, load_baseline_cached
    from tools.studio.score import score_text

    try:
        baseline = load_baseline_cached(baseline_model)
    except Exception:
        build_baseline(baseline_model)
        baseline = load_baseline_cached(baseline_model)

    scored: List[EvalPassage] = []
    with test_set_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            passage = EvalPassage(**{k: v for k, v in row.items() if k in EvalPassage.__dataclass_fields__})

            analysis = analyze_text(
                passage.text,
                model_id=model_id,
                doc_type=passage.genre,
                max_input_tokens=max_input_tokens,
                normalize_text=True,
                compute_cohesion=True,
            )
            score = score_text(analysis["doc_metrics"], baseline, doc_type=passage.genre)
            passage.horace_score = score.overall_0_100
            passage.horace_categories = score.categories
            scored.append(passage)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in scored:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    return scored


def compute_correlation(scored_path: Path) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation between human and Horace scores."""
    human_scores: List[float] = []
    horace_scores: List[float] = []

    with scored_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            human_mean = row.get("human_mean")
            horace = row.get("horace_score")
            if human_mean is not None and horace is not None:
                if math.isfinite(float(human_mean)) and math.isfinite(float(horace)):
                    human_scores.append(float(human_mean))
                    horace_scores.append(float(horace))

    if len(human_scores) < 3:
        return {"error": "Need at least 3 scored passages with human ratings", "n": len(human_scores)}

    h = np.array(human_scores)
    s = np.array(horace_scores)

    # Pearson correlation
    h_z = h - h.mean()
    s_z = s - s.mean()
    pearson = float(np.sum(h_z * s_z) / (np.sqrt(np.sum(h_z ** 2)) * np.sqrt(np.sum(s_z ** 2)) + 1e-12))

    # Spearman rank correlation
    h_rank = np.argsort(np.argsort(h)).astype(np.float64)
    s_rank = np.argsort(np.argsort(s)).astype(np.float64)
    h_rz = h_rank - h_rank.mean()
    s_rz = s_rank - s_rank.mean()
    spearman = float(np.sum(h_rz * s_rz) / (np.sqrt(np.sum(h_rz ** 2)) * np.sqrt(np.sum(s_rz ** 2)) + 1e-12))

    return {
        "n": len(human_scores),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "human_mean": float(np.mean(h)),
        "human_std": float(np.std(h)),
        "horace_mean": float(np.mean(s)),
        "horace_std": float(np.std(s)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Human evaluation framework for Horace")
    sub = parser.add_subparsers(dest="command")

    create_p = sub.add_parser("create", help="Create test set from text files")
    create_p.add_argument("--input-dir", type=Path, required=True)
    create_p.add_argument("--output", type=Path, default=Path("data/eval/test_set.jsonl"))
    create_p.add_argument("--min-words", type=int, default=50)
    create_p.add_argument("--max-words", type=int, default=500)

    score_p = sub.add_parser("score", help="Score test set with Horace")
    score_p.add_argument("--test-set", type=Path, required=True)
    score_p.add_argument("--output", type=Path, default=Path("data/eval/scored.jsonl"))
    score_p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    score_p.add_argument("--baseline", type=str, default="gpt2_gutenberg_512")

    corr_p = sub.add_parser("correlate", help="Compute human-Horace correlation")
    corr_p.add_argument("--scored", type=Path, required=True)

    rate_p = sub.add_parser("rate", help="Add a human rating")
    rate_p.add_argument("--test-set", type=Path, required=True)
    rate_p.add_argument("--id", type=str, required=True)
    rate_p.add_argument("--rating", type=float, required=True)
    rate_p.add_argument("--rater", type=str, default="anonymous")

    args = parser.parse_args()

    if args.command == "create":
        passages = create_test_set(args.input_dir, args.output, min_words=args.min_words, max_words=args.max_words)
        print(f"Created {len(passages)} passages in {args.output}")

    elif args.command == "score":
        scored = score_test_set(args.test_set, args.output, model_id=args.model, baseline_model=args.baseline)
        print(f"Scored {len(scored)} passages, saved to {args.output}")

    elif args.command == "correlate":
        result = compute_correlation(args.scored)
        print(json.dumps(result, indent=2))

    elif args.command == "rate":
        add_human_rating(args.test_set, args.id, args.rating, args.rater)
        print(f"Added rating {args.rating} for passage {args.id}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
