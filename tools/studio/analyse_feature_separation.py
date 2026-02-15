"""Analyse feature separation between good (human) and LLM texts.

Reads featurized pair JSONL files, unpacks into individual labeled texts,
and computes effect sizes + overlap for each feature. Outputs a ranked table
showing which features actually discriminate good from slop.

Usage:
    python -m tools.studio.analyse_feature_separation \
        --data data/pairs_v7/train_featurized.jsonl \
              data/pairs_v7/val_featurized.jsonl \
              data/pairs_v7/test_featurized.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_features(paths: List[Path]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Load featurized pairs and unpack into (good_features, llm_features)."""
    good: List[Dict[str, float]] = []
    llm: List[Dict[str, float]] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                chosen = row.get("chosen_features")
                rejected = row.get("rejected_features")
                if chosen and rejected:
                    good.append(chosen)
                    llm.append(rejected)

    return good, llm


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: standardised mean difference. Positive = a > b."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std < 1e-15:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, n_bins: int = 100) -> float:
    """Estimate overlap between two distributions using histogram intersection.

    Returns a value between 0 (no overlap) and 1 (identical distributions).
    """
    combined = np.concatenate([a, b])
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if hi - lo < 1e-15:
        return 1.0  # identical constant values

    bins = np.linspace(lo, hi, n_bins + 1)
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)

    bin_width = (hi - lo) / n_bins
    overlap = float(np.sum(np.minimum(hist_a, hist_b)) * bin_width)
    return min(overlap, 1.0)


def _auc_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Quick AUC: P(random good > random LLM). Measures separability."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.5
    # Mann-Whitney U approach
    count = 0
    ties = 0
    for va in a:
        gt = np.sum(b < va)
        eq = np.sum(b == va)
        count += gt
        ties += eq
    return float(count + 0.5 * ties) / (na * nb)


def analyse(paths: List[Path]) -> List[dict]:
    """Run the full analysis. Returns a list of per-feature results, sorted by |Cohen's d|."""
    good_dicts, llm_dicts = _load_features(paths)

    if not good_dicts or not llm_dicts:
        print("ERROR: No data loaded.", file=sys.stderr)
        return []

    # Get all feature names from the first entry
    feature_names = sorted(good_dicts[0].keys())
    n_good = len(good_dicts)
    n_llm = len(llm_dicts)

    print(f"Loaded {n_good} good texts, {n_llm} LLM texts, {len(feature_names)} features\n")

    results = []
    for feat in feature_names:
        good_vals = np.array([d.get(feat, 0.0) for d in good_dicts], dtype=np.float64)
        llm_vals = np.array([d.get(feat, 0.0) for d in llm_dicts], dtype=np.float64)

        # Handle NaN/inf
        good_vals = np.nan_to_num(good_vals, nan=0.0, posinf=0.0, neginf=0.0)
        llm_vals = np.nan_to_num(llm_vals, nan=0.0, posinf=0.0, neginf=0.0)

        d = _cohens_d(good_vals, llm_vals)
        overlap = _overlap_coefficient(good_vals, llm_vals)
        auc = _auc_simple(good_vals, llm_vals)

        good_mean = float(np.mean(good_vals))
        good_std = float(np.std(good_vals, ddof=1)) if len(good_vals) > 1 else 0.0
        llm_mean = float(np.mean(llm_vals))
        llm_std = float(np.std(llm_vals, ddof=1)) if len(llm_vals) > 1 else 0.0

        # Direction: which group has higher values?
        if abs(d) < 0.05:
            direction = "~same"
        elif d > 0:
            direction = "good > LLM"
        else:
            direction = "LLM > good"

        # Effect size interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            strength = "negligible"
        elif abs_d < 0.5:
            strength = "small"
        elif abs_d < 0.8:
            strength = "medium"
        else:
            strength = "large"

        results.append({
            "feature": feat,
            "cohens_d": round(d, 3),
            "abs_d": round(abs_d, 3),
            "strength": strength,
            "overlap": round(overlap, 3),
            "auc": round(auc, 3),
            "direction": direction,
            "good_mean": round(good_mean, 4),
            "good_std": round(good_std, 4),
            "llm_mean": round(llm_mean, 4),
            "llm_std": round(llm_std, 4),
        })

    # Sort by absolute effect size, descending
    results.sort(key=lambda r: r["abs_d"], reverse=True)
    return results


def print_table(results: List[dict]) -> None:
    """Print a human-readable ranked table."""
    # Header
    print(f"{'Rank':>4}  {'Feature':<35} {'Cohen_d':>8} {'Strength':<12} "
          f"{'Overlap':>8} {'AUC':>6} {'Direction':<14} "
          f"{'Good_mean':>10} {'LLM_mean':>10}")
    print("-" * 140)

    for i, r in enumerate(results, 1):
        print(f"{i:>4}  {r['feature']:<35} {r['cohens_d']:>+8.3f} {r['strength']:<12} "
              f"{r['overlap']:>8.3f} {r['auc']:>6.3f} {r['direction']:<14} "
              f"{r['good_mean']:>10.4f} {r['llm_mean']:>10.4f}")

    # Summary
    print(f"\n--- Summary ---")
    large = [r for r in results if r["strength"] == "large"]
    medium = [r for r in results if r["strength"] == "medium"]
    small = [r for r in results if r["strength"] == "small"]
    negl = [r for r in results if r["strength"] == "negligible"]
    print(f"Large effect (|d| >= 0.8):    {len(large)} features")
    print(f"Medium effect (0.5 <= |d| < 0.8): {len(medium)} features")
    print(f"Small effect (0.2 <= |d| < 0.5):  {len(small)} features")
    print(f"Negligible (|d| < 0.2):       {len(negl)} features")

    if large:
        print(f"\nTop discriminators:")
        for r in large:
            print(f"  {r['feature']}: d={r['cohens_d']:+.3f}, overlap={r['overlap']:.3f}, "
                  f"AUC={r['auc']:.3f} ({r['direction']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse feature separation between good and LLM texts")
    parser.add_argument(
        "--data", nargs="+", required=True, type=Path,
        help="Featurized JSONL files (train, val, test)",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional path to write results as JSON",
    )
    args = parser.parse_args()

    results = analyse(args.data)
    if not results:
        sys.exit(1)

    print_table(results)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
