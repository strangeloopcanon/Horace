from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tools.studio.calibrator import (
    LogisticCalibrator,
    featurize_from_report_row,
    iter_default_feature_names,
    save_logistic_calibrator,
)


@dataclass(frozen=True)
class TrainResult:
    out_path: Path
    n_rows: int
    n_pos: int
    n_neg: int
    train_acc: float
    train_auc: Optional[float]
    feature_dim: int


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64, copy=False)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y = y_true.astype(np.int64, copy=False)
    s = y_score.astype(np.float64, copy=False)
    if y.size == 0:
        return None
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, y.size + 1)
    pos_ranks = ranks[y == 1]
    auc = (float(np.sum(pos_ranks)) - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-2,
    lr: float = 0.5,
    steps: int = 600,
    seed: int = 1337,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(int(seed))
    n, d = X.shape
    w = rng.normal(0.0, 0.01, size=(d,)).astype(np.float64)
    b = 0.0

    for _ in range(max(1, int(steps))):
        logits = X @ w + b
        p = _sigmoid_np(logits)
        err = p - y
        grad_w = (X.T @ err) / max(1, n) + float(l2) * w
        grad_b = float(np.mean(err)) if n > 0 else 0.0
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w, float(b)


def train_from_eval_report(
    report_path: Path,
    *,
    out_path: Path,
    positive_sources: Sequence[str],
    negative_sources: Sequence[str],
    missing_value: float = 0.5,
    l2: float = 1e-2,
    lr: float = 0.5,
    steps: int = 600,
    seed: int = 1337,
) -> TrainResult:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    run_meta = rep.get("run_meta") if isinstance(rep.get("run_meta"), dict) else {}
    max_input_tokens = run_meta.get("max_input_tokens")
    max_input_tokens_i = int(max_input_tokens) if isinstance(max_input_tokens, (int, float)) else None
    rows = rep.get("scored") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("report has no scored rows")

    pos_set = {str(s) for s in positive_sources}
    neg_set = {str(s) for s in negative_sources}

    feats = iter_default_feature_names(rows)

    X_list: List[List[float]] = []
    y_list: List[int] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source") or "")
        if src not in pos_set and src not in neg_set:
            continue
        cats = r.get("categories") or {}
        rms = r.get("rubric_metrics") or {}
        dm = r.get("doc_metrics") or {}
        if not isinstance(cats, dict) or not isinstance(rms, dict):
            continue
        if not isinstance(dm, dict):
            dm = {}
        x = featurize_from_report_row(
            feature_names=feats,
            categories=cats,
            rubric_metrics=rms,
            doc_metrics=dm,
            max_input_tokens=max_input_tokens_i,
            missing_value=float(missing_value),
        )
        X_list.append(x)
        y_list.append(1 if src in pos_set else 0)

    if not X_list:
        raise ValueError("no training rows after filtering by source")

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)

    w, b = train_logistic(X, y, l2=float(l2), lr=float(lr), steps=int(steps), seed=int(seed))

    probs = _sigmoid_np(X @ w + b)
    preds = (probs >= 0.5).astype(np.int64)
    acc = float(np.mean(preds == y.astype(np.int64)))
    auc = _auc_roc(y.astype(np.int64), probs)

    cal = LogisticCalibrator(
        feature_names=tuple(feats),
        weights=tuple(float(x) for x in w.tolist()),
        bias=float(b),
        meta={
            "trained_from": str(report_path),
            "report_run_meta": run_meta,
            "seed": int(seed),
            "positive_sources": sorted(pos_set),
            "negative_sources": sorted(neg_set),
            "missing_value": float(missing_value),
            "l2": float(l2),
            "lr": float(lr),
            "steps": int(steps),
            "train_acc": float(acc),
            "train_auc": float(auc) if auc is not None and math.isfinite(float(auc)) else None,
            "feature_dim": int(X.shape[1]),
        },
    )
    save_logistic_calibrator(cal, out_path)
    return TrainResult(
        out_path=out_path,
        n_rows=int(len(X_list)),
        n_pos=int(np.sum(y == 1.0)),
        n_neg=int(np.sum(y == 0.0)),
        train_acc=float(acc),
        train_auc=float(auc) if auc is not None and math.isfinite(float(auc)) else None,
        feature_dim=int(X.shape[1]),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a tiny learned calibrator from a Studio eval report JSON.")
    ap.add_argument("--report", required=True, help="Path to studio_eval_web_report.json")
    ap.add_argument("--out", required=True, help="Output calibrator JSON path")
    ap.add_argument("--pos", action="append", default=["gutenberg_excerpt"], help="Positive source label(s)")
    ap.add_argument(
        "--neg",
        action="append",
        default=["wikipedia_random_summary", "rfc_excerpt", "gibberish_control"],
        help="Negative source label(s)",
    )
    ap.add_argument("--missing", type=float, default=0.5, help="Fill value for missing features")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    res = train_from_eval_report(
        Path(str(args.report)),
        out_path=Path(str(args.out)),
        positive_sources=tuple(args.pos or []),
        negative_sources=tuple(args.neg or []),
        missing_value=float(args.missing),
        l2=float(args.l2),
        lr=float(args.lr),
        steps=int(args.steps),
        seed=int(args.seed),
    )
    payload = {
        **asdict(res),
        "out_path": str(res.out_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
