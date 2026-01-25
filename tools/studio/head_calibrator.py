from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class HeadCalibrator:
    """Logistic calibrator over scorer head probabilities."""

    feature_names: Tuple[str, ...]
    weights: Tuple[float, ...]
    bias: float
    meta: Dict[str, Any]

    def predict_proba(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature length mismatch")
        s = float(self.bias)
        for w, x in zip(self.weights, features):
            s += float(w) * float(x)
        return _sigmoid(s)

    def score_0_100(self, features: Sequence[float]) -> float:
        return 100.0 * self.predict_proba(features)


def load_head_calibrator(path: Path) -> HeadCalibrator:
    obj = json.loads(path.read_text(encoding="utf-8"))
    feature_names = tuple(str(x) for x in (obj.get("feature_names") or []))
    weights = tuple(float(x) for x in (obj.get("weights") or []))
    bias = float(obj.get("bias") or 0.0)
    meta = obj.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    if len(feature_names) != len(weights):
        raise ValueError("Invalid calibrator: feature_names/weights length mismatch")
    return HeadCalibrator(feature_names=feature_names, weights=weights, bias=bias, meta=meta)


def save_head_calibrator(cal: HeadCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "type": "head_logistic_v1",
        "feature_names": list(cal.feature_names),
        "weights": list(cal.weights),
        "bias": float(cal.bias),
        "meta": cal.meta,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

