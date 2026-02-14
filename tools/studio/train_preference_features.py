"""Train the v6 feature preference model from featurized pair data.

Usage:
    python -m tools.studio.train_preference_features \
        --train data/pairs_v6/train_featurized.jsonl \
        --val data/pairs_v6/val_featurized.jsonl \
        --out-dir models/preference_v6
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


def _load_featurized_pairs(
    path: Path,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load featurized pairs as (chosen_vec, rejected_vec) tuples."""
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
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
        pairs.append((chosen_vec, rejected_vec))
    return pairs


def train(
    *,
    train_path: Path,
    val_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    out_dir: Path,
    l2_reg: float = 1.0,
) -> Dict[str, Any]:
    """Train, evaluate, and save the preference model."""
    train_pairs = _load_featurized_pairs(train_path)
    if not train_pairs:
        logger.error("no_train_pairs", extra={"path": str(train_path)})
        return {"error": "no training pairs"}

    val_pairs = _load_featurized_pairs(val_path) if val_path and val_path.exists() else None
    test_pairs = _load_featurized_pairs(test_path) if test_path and test_path.exists() else None

    model = FeaturePreferenceModel()
    report = model.train(train_pairs, val_pairs, l2_reg=l2_reg)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "preference_model.json")

    results: Dict[str, Any] = {
        "n_features": len(FEATURE_SCHEMA),
        "n_train": report.n_train,
        "n_val": report.n_val,
        "train_accuracy": report.train_accuracy,
        "val_accuracy": report.val_accuracy,
        "l2_reg": report.l2_reg,
        "bias": report.bias,
    }

    # Test accuracy
    if test_pairs:
        model_loaded = FeaturePreferenceModel.load(out_dir / "preference_model.json")
        test_acc = model_loaded._pairwise_accuracy(test_pairs)
        results["test_accuracy"] = test_acc
        results["n_test"] = len(test_pairs)

    # Feature importance (|weight|, sorted)
    importance = sorted(
        zip(report.feature_names, report.weights),
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
    ap = argparse.ArgumentParser(description="Train v6 feature preference model.")
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
