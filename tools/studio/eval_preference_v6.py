"""Evaluate the v6 feature preference scorer on featurized pair data.

Computes pairwise accuracy, score distributions, feature importance,
and actionable diagnostics. Can also compare against rubric scores
if doc_metrics are available.

Usage:
    python -m tools.studio.eval_preference_v6 \
        --test data/pairs_v6/test_featurized.jsonl \
        --model models/preference_v6/preference_model.json \
        --report-out reports/preference_v6_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.studio.dataset_utils import iter_jsonl
from tools.studio.preference_features import (
    FEATURE_SCHEMA,
    FeaturePreferenceModel,
    generate_feedback,
)

logger = logging.getLogger(__name__)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    pos = (len(vs) - 1) * q
    i = int(math.floor(pos))
    j = min(len(vs) - 1, i + 1)
    frac = pos - i
    return float((1.0 - frac) * vs[i] + frac * vs[j])


def _summarize(values: List[float]) -> Dict[str, Any]:
    vs = [float(v) for v in values if math.isfinite(float(v))]
    if not vs:
        return {"n": 0}
    return {
        "n": len(vs),
        "mean": round(float(np.mean(vs)), 2),
        "std": round(float(np.std(vs)), 2),
        "min": round(min(vs), 2),
        "p10": round(_quantile(vs, 0.10), 2),
        "p25": round(_quantile(vs, 0.25), 2),
        "median": round(_quantile(vs, 0.50), 2),
        "p75": round(_quantile(vs, 0.75), 2),
        "p90": round(_quantile(vs, 0.90), 2),
        "max": round(max(vs), 2),
    }


def _load_featurized_pairs(
    path: Path,
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """Load featurized pairs as (chosen_vec, rejected_vec, row) tuples."""
    pairs: List[Tuple[np.ndarray, np.ndarray, dict]] = []
    for row in iter_jsonl(path):
        chosen_feat = row.get("chosen_features")
        rejected_feat = row.get("rejected_features")
        if not chosen_feat or not rejected_feat:
            continue
        chosen_vec = np.array(
            [float(chosen_feat.get(k, 0.0)) for k in FEATURE_SCHEMA],
            dtype=np.float64,
        )
        rejected_vec = np.array(
            [float(rejected_feat.get(k, 0.0)) for k in FEATURE_SCHEMA],
            dtype=np.float64,
        )
        pairs.append((chosen_vec, rejected_vec, row))
    return pairs


def _auc_roc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    """Compute AUC-ROC from labels and scores."""
    n = len(y_true)
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: y_score[i])
    ranks = [0] * n
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    pos_ranks = [ranks[i] for i, y in enumerate(y_true) if y == 1]
    auc = (sum(pos_ranks) - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return round(float(auc), 4)


def evaluate(
    *,
    model_path: Path,
    test_path: Path,
    val_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full evaluation of the v6 preference model."""
    model = FeaturePreferenceModel.load(model_path)

    report: Dict[str, Any] = {
        "model_path": str(model_path),
        "n_features": len(model.feature_names),
        "score_calibration": {
            "center": round(model.score_center, 3),
            "scale": round(model.score_scale, 3),
        },
    }

    for split_name, split_path in [("test", test_path), ("val", val_path)]:
        if split_path is None or not split_path.exists():
            continue

        pairs = _load_featurized_pairs(split_path)
        if not pairs:
            report[split_name] = {"error": "no pairs loaded"}
            continue

        chosen_raw: List[float] = []
        rejected_raw: List[float] = []
        chosen_scores: List[float] = []
        rejected_scores: List[float] = []
        correct = 0
        margin_correct = 0
        close_calls: List[Dict[str, Any]] = []

        for chosen_vec, rejected_vec, row in pairs:
            cr = model.score(chosen_vec)
            rr = model.score(rejected_vec)
            cs = model.score_0_100(chosen_vec)
            rs = model.score_0_100(rejected_vec)

            chosen_raw.append(cr)
            rejected_raw.append(rr)
            chosen_scores.append(float(cs))
            rejected_scores.append(float(rs))

            if cr > rr:
                correct += 1
            if cr > rr + 0.5:
                margin_correct += 1

            gap = cr - rr
            if abs(gap) < 2.0:
                close_calls.append({
                    "pair_id": str(row.get("pair_id", "")),
                    "group_id": str(row.get("group_id", "")),
                    "raw_gap": round(gap, 3),
                    "chosen_score": cs,
                    "rejected_score": rs,
                    "method": str(row.get("method", "")),
                })

        n = len(pairs)
        gaps = [c - r for c, r in zip(chosen_scores, rejected_scores)]

        # AUC: treat chosen as positive (label 1), rejected as negative (label 0)
        y_true = [1] * n + [0] * n
        y_score = chosen_scores + rejected_scores
        auc = _auc_roc(y_true, y_score)

        split_report: Dict[str, Any] = {
            "n_pairs": n,
            "pairwise_accuracy": round(correct / max(1, n), 4),
            "margin_accuracy_0.5": round(margin_correct / max(1, n), 4),
            "auc": auc,
            "chosen_scores": _summarize(chosen_scores),
            "rejected_scores": _summarize(rejected_scores),
            "score_gap": _summarize(gaps),
            "chosen_raw": _summarize(chosen_raw),
            "rejected_raw": _summarize(rejected_raw),
        }

        if close_calls:
            split_report["close_calls"] = close_calls[:10]

        # Feature importance breakdown by contribution
        top_contributions: Dict[str, List[float]] = {name: [] for name in FEATURE_SCHEMA}
        for chosen_vec, rejected_vec, _ in pairs:
            contribs = model.feature_contributions(chosen_vec)
            for name, val in contribs.items():
                top_contributions[name].append(val)

        feature_stats = []
        for name in FEATURE_SCHEMA:
            vals = top_contributions[name]
            weight = float(model.weights[FEATURE_SCHEMA.index(name)])
            feature_stats.append({
                "feature": name,
                "weight": round(weight, 4),
                "mean_contribution": round(float(np.mean(vals)), 4),
                "std_contribution": round(float(np.std(vals)), 4),
            })
        feature_stats.sort(key=lambda x: abs(x["mean_contribution"]), reverse=True)
        split_report["feature_contributions"] = feature_stats[:15]

        report[split_name] = split_report

    # Sample feedback for a few test pairs
    if test_path.exists():
        pairs = _load_featurized_pairs(test_path)
        if pairs:
            feedback_examples: List[Dict[str, Any]] = []
            # Pick 3 examples: best chosen, worst chosen, typical rejected
            chosen_with_scores = [
                (model.score(cv), cv, rv, row) for cv, rv, row in pairs
            ]
            chosen_with_scores.sort(key=lambda x: x[0], reverse=True)

            for label, idx in [("best_chosen", 0), ("worst_chosen", -1), ("typical_rejected", len(pairs) // 2)]:
                raw, cv, rv, row = chosen_with_scores[min(idx, len(chosen_with_scores) - 1)]
                target_vec = cv if "chosen" in label else rv
                fb = generate_feedback(model, target_vec, max_suggestions=3)
                feedback_examples.append({
                    "label": label,
                    "pair_id": str(row.get("pair_id", "")),
                    "score_0_100": model.score_0_100(target_vec),
                    "raw_score": round(model.score(target_vec), 3),
                    "suggestions": fb,
                })
            report["feedback_examples"] = feedback_examples

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate v6 feature preference scorer.")
    ap.add_argument("--model", default="models/preference_v6/preference_model.json")
    ap.add_argument("--test", required=True, help="Featurized test JSONL")
    ap.add_argument("--val", default=None, help="Featurized val JSONL (optional)")
    ap.add_argument("--report-out", default="reports/preference_v6_eval.json")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = evaluate(
        model_path=Path(args.model),
        test_path=Path(args.test),
        val_path=Path(args.val) if args.val else None,
    )

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    for split in ("test", "val"):
        if split not in report:
            continue
        s = report[split]
        print(f"\n=== {split} ({s['n_pairs']} pairs) ===")
        print(f"Pairwise accuracy: {s['pairwise_accuracy']:.1%}")
        print(f"AUC: {s['auc']}")
        cs = s["chosen_scores"]
        rs = s["rejected_scores"]
        gs = s["score_gap"]
        print(f"Chosen 0-100:   mean={cs['mean']}, median={cs['median']}, p10={cs['p10']}, p90={cs['p90']}")
        print(f"Rejected 0-100: mean={rs['mean']}, median={rs['median']}, p10={rs['p10']}, p90={rs['p90']}")
        print(f"Score gap:      mean={gs['mean']}, median={gs['median']}")
        if s.get("close_calls"):
            print(f"Close calls:    {len(s['close_calls'])}")

    print(f"\nReport written to {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
