from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _get_nested_score_0_1(obj: Mapping[str, Any], key: str) -> Optional[float]:
    v = obj.get(key)
    if isinstance(v, dict):
        s = v.get("score_0_1")
        if isinstance(s, (int, float)) and math.isfinite(float(s)):
            return float(s)
    return None


@dataclass(frozen=True)
class LogisticCalibrator:
    """A tiny learned scorer on top of rubric outputs (fully deterministic at inference).

    Features are named:
      - "m:<metric_key>" uses rubric metric `score_0_1`
      - "c:<category_key>" uses category score (0..1)
      - "d:<doc_metric_key>" uses raw `doc_metrics` numeric values
      - "d:tokens_frac" uses `tokens_count / (max_input_tokens-1)` (0..1)
    """

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


def featurize_from_report_row(
    *,
    feature_names: Sequence[str],
    categories: Mapping[str, Any],
    rubric_metrics: Mapping[str, Any],
    doc_metrics: Optional[Mapping[str, Any]] = None,
    report_row: Optional[Mapping[str, Any]] = None,
    max_input_tokens: Optional[int] = None,
    missing_value: float = 0.5,
) -> List[float]:
    out: List[float] = []
    for name in feature_names:
        if name.startswith("m:"):
            mkey = name[2:]
            v = _get_nested_score_0_1(rubric_metrics, mkey)
            out.append(float(v) if v is not None else float(missing_value))
            continue
        if name.startswith("c:"):
            ckey = name[2:]
            v = categories.get(ckey)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(float(v))
            else:
                out.append(float(missing_value))
            continue
        if name == "d:tokens_frac":
            tc = None if doc_metrics is None else doc_metrics.get("tokens_count")
            if isinstance(tc, (int, float)) and math.isfinite(float(tc)) and max_input_tokens is not None:
                denom = max(1.0, float(int(max_input_tokens) - 1))
                out.append(max(0.0, min(1.0, float(tc) / denom)))
            else:
                out.append(float(missing_value))
            continue
        if name.startswith("d:"):
            dkey = name[2:]
            v = None if doc_metrics is None else doc_metrics.get(dkey)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(float(v))
            else:
                out.append(float(missing_value))
            continue
        if name.startswith("s:"):
            skey = name[2:]
            sval = None if report_row is None else report_row.get(skey)
            if isinstance(sval, Mapping):
                score_candidate = sval.get("score_0_1")
                if score_candidate is None:
                    score_candidate = sval.get("score_0_100")
                if score_candidate is None:
                    score_candidate = sval.get("overall_0_1")
                sval = score_candidate
            if isinstance(sval, (int, float)) and math.isfinite(float(sval)):
                out.append(float(sval))
            else:
                out.append(float(missing_value))
            continue
        raise ValueError(f"Unknown feature name: {name}")
    return out


def load_logistic_calibrator(path: Path) -> LogisticCalibrator:
    obj = json.loads(path.read_text(encoding="utf-8"))
    feature_names = tuple(str(x) for x in (obj.get("feature_names") or []))
    weights = tuple(float(x) for x in (obj.get("weights") or []))
    bias = float(obj.get("bias") or 0.0)
    meta = obj.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    if len(feature_names) != len(weights):
        raise ValueError("Invalid calibrator: feature_names/weights length mismatch")
    return LogisticCalibrator(feature_names=feature_names, weights=weights, bias=bias, meta=meta)


def save_logistic_calibrator(cal: LogisticCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "type": "logistic_v1",
        "feature_names": list(cal.feature_names),
        "weights": list(cal.weights),
        "bias": float(cal.bias),
        "meta": cal.meta,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_default_feature_names(scored_rows: Iterable[Mapping[str, Any]]) -> List[str]:
    metric_keys: set[str] = set()
    cat_keys: set[str] = set()
    doc_metric_keys: set[str] = set()
    for r in scored_rows:
        cats = r.get("categories") or {}
        if isinstance(cats, dict):
            cat_keys.update(str(k) for k in cats.keys())
        rms = r.get("rubric_metrics") or {}
        if isinstance(rms, dict):
            metric_keys.update(str(k) for k in rms.keys())
        dm = r.get("doc_metrics") or {}
        if isinstance(dm, dict):
            doc_metric_keys.update(str(k) for k in dm.keys())
    feats: List[str] = []
    feats.append("d:tokens_frac")
    # Broad cadence (doc-level): stable, scale-invariant-ish features.
    for k in ("sent_burst_cv", "sent_len_cv", "para_burst_cv", "para_len_cv", "line_burst_cv", "line_len_cv"):
        if k in doc_metric_keys:
            feats.append(f"d:{k}")
    feats.extend([f"c:{k}" for k in sorted(cat_keys)])
    feats.extend([f"m:{k}" for k in sorted(metric_keys)])

    for r in scored_rows:
        for k in sorted(r.keys()):
            if not isinstance(r.get(k), (int, float)):
                continue
            if not str(k).endswith("_0_1"):
                continue
            if not str(k).startswith(("quality_", "antipattern_", "authenticity_", "trained_", "truncated_", "score_")):
                continue
            feats.append(f"s:{k}")
    # Keep deterministic ordering and remove duplicates.
    return list(dict.fromkeys(feats))
