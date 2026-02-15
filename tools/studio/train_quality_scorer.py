"""Train the v8 direct quality scorer from featurized pair data.

Reads featurized JSONL pairs (with chosen_features / rejected_features dicts),
splits them into individual texts with labels (chosen=1, rejected=0), selects
only the features in the current FEATURE_SCHEMA, and trains a logistic
regression P(good | features).

Usage:
    python -m tools.studio.train_quality_scorer \
        --train data/pairs_v7/train_featurized.jsonl \
        --val data/pairs_v7/val_featurized.jsonl \
        --test data/pairs_v7/test_featurized.jsonl \
        --out-dir models/preference_v8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.studio.dataset_utils import iter_jsonl
from tools.studio.preference_features import (
    FEATURE_SCHEMA,
    FeaturePreferenceModel,
)

logger = logging.getLogger(__name__)


def _load_individual_texts(
    path: Path,
    feature_names: List[str],
) -> Tuple[List[np.ndarray], List[int]]:
    """Load featurized pairs as individual (features, label) samples.

    chosen_features -> label 1 (good)
    rejected_features -> label 0 (slop)
    """
    features: List[np.ndarray] = []
    labels: List[int] = []

    for row in iter_jsonl(path):
        chosen_feat = row.get("chosen_features")
        rejected_feat = row.get("rejected_features")
        if not chosen_feat or not rejected_feat:
            continue

        chosen_vec = np.array(
            [float(chosen_feat.get(k, 0.0)) for k in feature_names],
            dtype=np.float64,
        )
        rejected_vec = np.array(
            [float(rejected_feat.get(k, 0.0)) for k in feature_names],
            dtype=np.float64,
        )

        features.append(chosen_vec)
        labels.append(1)
        features.append(rejected_vec)
        labels.append(0)

    return features, labels


def train(
    *,
    train_path: Path,
    val_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    out_dir: Path,
    l2_reg: float = 1.0,
) -> Dict[str, Any]:
    """Train, evaluate, and save the quality scorer."""
    feature_names = list(FEATURE_SCHEMA)

    train_feats, train_labels = _load_individual_texts(train_path, feature_names)
    if not train_feats:
        logger.error("no_train_data", extra={"path": str(train_path)})
        return {"error": "no training data"}

    model = FeaturePreferenceModel()
    train_report = model.train_direct(
        train_feats,
        train_labels,
        l2_reg=l2_reg,
        feature_names=feature_names,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "preference_model.json")

    results: Dict[str, Any] = {
        "version": "v8",
        "mode": "direct_classification",
        "n_features": len(feature_names),
        "train": train_report,
    }

    # Evaluate on val/test sets
    for split_name, split_path in [("val", val_path), ("test", test_path)]:
        if split_path is None or not split_path.exists():
            continue
        feats, labels = _load_individual_texts(split_path, feature_names)
        if not feats:
            continue

        # Load saved model for clean evaluation
        loaded = FeaturePreferenceModel.load(out_dir / "preference_model.json")
        y = np.array(labels, dtype=np.int32)

        # Direct classification metrics
        scores = np.array([loaded.score_0_100(f) for f in feats])
        preds = (scores >= 50).astype(int)
        accuracy = float(np.mean(preds == y))

        from sklearn.metrics import roc_auc_score

        raw_scores = np.array([loaded.score(f) for f in feats])
        auc = float(roc_auc_score(y, raw_scores))

        # Also compute pairwise accuracy (pairs from same file)
        pair_feats, _ = _load_individual_texts(split_path, feature_names)
        n_pairs = len(pair_feats) // 2
        pairwise_correct = 0
        for i in range(n_pairs):
            chosen_score = loaded.score(pair_feats[i * 2])
            rejected_score = loaded.score(pair_feats[i * 2 + 1])
            if chosen_score > rejected_score:
                pairwise_correct += 1
        pairwise_acc = float(pairwise_correct) / max(1, n_pairs)

        # Score distribution
        good_scores = scores[y == 1]
        slop_scores = scores[y == 0]

        results[split_name] = {
            "n_samples": int(len(y)),
            "accuracy": accuracy,
            "auc": auc,
            "pairwise_accuracy": pairwise_acc,
            "n_pairs": n_pairs,
            "good_score_mean": float(np.mean(good_scores)),
            "good_score_std": float(np.std(good_scores)),
            "slop_score_mean": float(np.mean(slop_scores)),
            "slop_score_std": float(np.std(slop_scores)),
        }

    # Feature importance (|weight|, sorted)
    importance = sorted(
        zip(train_report["feature_names"], train_report["weights"]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    results["feature_importance"] = [
        {"feature": name, "weight": round(w, 6)} for name, w in importance
    ]

    report_path = out_dir / "train_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train v8 direct quality scorer.")
    ap.add_argument("--train", required=True, help="Featurized train JSONL")
    ap.add_argument("--val", help="Featurized val JSONL")
    ap.add_argument("--test", help="Featurized test JSONL")
    ap.add_argument("--out-dir", required=True, help="Output directory for model + report")
    ap.add_argument("--l2-reg", type=float, default=1.0, help="L2 regularization (C parameter)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = train(
        train_path=Path(args.train),
        val_path=Path(args.val) if args.val else None,
        test_path=Path(args.test) if args.test else None,
        out_dir=Path(args.out_dir),
        l2_reg=float(args.l2_reg),
    )
    print(json.dumps(results, indent=2))
    return 0 if "error" not in results else 1


if __name__ == "__main__":
    raise SystemExit(main())
