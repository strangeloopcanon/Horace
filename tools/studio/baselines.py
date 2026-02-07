from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean: float
    sd: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    values: List[float]


@dataclass(frozen=True)
class BaselineSlice:
    name: str
    metrics: Dict[str, MetricSummary]


@dataclass(frozen=True)
class Baseline:
    model_id: str
    doc_types: Dict[str, BaselineSlice]


_LOAD_CACHE: Dict[str, Baseline] = {}
_LOAD_LOCK = threading.Lock()


def safe_model_id(model_id: str) -> str:
    parts = [p for p in (model_id or "").replace("/", "_").split("_") if p]
    return "_".join(parts) or "model"


def resolve_analysis_dir(model_id: str) -> Path:
    base = Path("data/analysis")
    direct = base / model_id
    if direct.exists():
        return direct
    parts = (model_id or "").split("/")
    if len(parts) >= 2:
        nested = base / parts[0] / parts[1]
        if nested.exists():
            return nested
    leaf = base / (parts[-1] if parts else model_id)
    return leaf


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    # linear interpolation
    pos = (len(sorted_vals) - 1) * q
    i = int(math.floor(pos))
    j = min(len(sorted_vals) - 1, i + 1)
    frac = pos - i
    return float((1.0 - frac) * sorted_vals[i] + frac * sorted_vals[j])


def _summary(values: List[float]) -> MetricSummary:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    vals.sort()
    n = len(vals)
    if n == 0:
        nan = float("nan")
        return MetricSummary(n=0, mean=nan, sd=nan, p10=nan, p25=nan, p50=nan, p75=nan, p90=nan, values=[])
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, n - 1))
    return MetricSummary(
        n=n,
        mean=float(mean),
        sd=float(sd),
        p10=_quantile(vals, 0.10),
        p25=_quantile(vals, 0.25),
        p50=_quantile(vals, 0.50),
        p75=_quantile(vals, 0.75),
        p90=_quantile(vals, 0.90),
        values=vals,
    )


def build_baseline(
    model_id: str,
    *,
    out_path: Optional[Path] = None,
    docs_path: Optional[Path] = None,
) -> Path:
    """Build a baseline distribution from `data/analysis/<model>/docs_clean.jsonl`.

    Adds synthetic slices:
      - prose = novel + shortstory
      - all = all doc types
    """
    analysis_dir = resolve_analysis_dir(model_id)
    docs_clean = docs_path or (analysis_dir / "docs_clean.jsonl")
    if not docs_clean.exists():
        raise FileNotFoundError(f"Missing docs_clean.jsonl: {docs_clean}")

    rows = list(_iter_jsonl(docs_clean))
    out = out_path or (Path("data/baselines") / f"{safe_model_id(model_id)}_docs.json")
    return build_baseline_from_rows(model_id, rows, out_path=out)


def build_baseline_from_rows(model_id: str, rows: List[dict], *, out_path: Path) -> Path:
    """Build a baseline distribution from already-collected metric dicts."""
    by_type: Dict[str, List[dict]] = {}
    all_docs: List[dict] = []
    for row in rows:
        dtype = str(row.get("doc_type") or row.get("type") or "unknown").lower()
        by_type.setdefault(dtype, []).append(row)
        all_docs.append(row)

    # Synthetic prose bucket
    prose_docs = (by_type.get("shortstory") or []) + (by_type.get("novel") or [])
    if prose_docs:
        by_type["prose"] = prose_docs
    by_type["all"] = all_docs

    # Collect metric keys (numeric) across the "all" slice
    metric_keys: List[str] = []
    seen: set[str] = set()
    for row in all_docs:
        for k, v in row.items():
            if k in seen:
                continue
            if isinstance(v, (int, float)) and v is not None:
                seen.add(k)
                metric_keys.append(k)

    baseline: Dict[str, Any] = {
        "model_id": model_id,
        "doc_types": {},
    }

    for dtype, rows_t in by_type.items():
        metrics: Dict[str, Any] = {}
        for k in metric_keys:
            vals = []
            for r in rows_t:
                v = r.get(k)
                if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v)):
                    vals.append(float(v))
            if not vals:
                continue
            s = _summary(vals)
            metrics[k] = {
                "n": s.n,
                "mean": s.mean,
                "sd": s.sd,
                "p10": s.p10,
                "p25": s.p25,
                "p50": s.p50,
                "p75": s.p75,
                "p90": s.p90,
                "values": s.values,
            }
        baseline["doc_types"][dtype] = {"name": dtype, "metrics": metrics}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _parse_metric_summary(obj: dict) -> MetricSummary:
    return MetricSummary(
        n=int(obj["n"]),
        mean=float(obj["mean"]),
        sd=float(obj["sd"]),
        p10=float(obj["p10"]),
        p25=float(obj["p25"]),
        p50=float(obj["p50"]),
        p75=float(obj["p75"]),
        p90=float(obj["p90"]),
        values=[float(x) for x in obj.get("values") or []],
    )


def load_baseline(model_id: str, *, path: Optional[Path] = None) -> Baseline:
    default = Path("data/baselines") / f"{safe_model_id(model_id)}_docs.json"
    p = path or default
    raw = json.loads(p.read_text(encoding="utf-8"))
    doc_types: Dict[str, BaselineSlice] = {}
    for dtype, slice_obj in (raw.get("doc_types") or {}).items():
        metrics = {k: _parse_metric_summary(v) for k, v in (slice_obj.get("metrics") or {}).items()}
        doc_types[dtype] = BaselineSlice(name=str(slice_obj.get("name") or dtype), metrics=metrics)
    return Baseline(model_id=str(raw.get("model_id") or model_id), doc_types=doc_types)


def load_baseline_cached(model_id: str, *, path: Optional[Path] = None) -> Baseline:
    """Load a baseline JSON with a simple in-process cache."""
    default = Path("data/baselines") / f"{safe_model_id(model_id)}_docs.json"
    p = path or default
    try:
        key = str(p.resolve())
    except Exception:
        key = str(p)
    with _LOAD_LOCK:
        cached = _LOAD_CACHE.get(key)
        if cached is not None:
            return cached
    b = load_baseline(model_id, path=p)
    with _LOAD_LOCK:
        _LOAD_CACHE[key] = b
    return b


def percentile(sorted_values: List[float], x: float) -> Optional[float]:
    """Empirical percentile in [0,100]."""
    if x is None or not math.isfinite(float(x)):
        return None
    if not sorted_values:
        return None
    lo = 0
    hi = len(sorted_values)
    xv = float(x)
    # bisect_right
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= xv:
            lo = mid + 1
        else:
            hi = mid
    return 100.0 * (lo / len(sorted_values))


def get_slice(baseline: Baseline, doc_type: str) -> BaselineSlice:
    dt = (doc_type or "").strip().lower()
    if dt in baseline.doc_types:
        return baseline.doc_types[dt]
    if dt in ("story", "short_story", "short-story"):
        return baseline.doc_types.get("shortstory") or baseline.doc_types["all"]
    if dt in ("poetry",):
        return baseline.doc_types.get("poem") or baseline.doc_types["all"]
    return baseline.doc_types.get("all") or next(iter(baseline.doc_types.values()))


# Genre-specific baseline identifiers for different corpus types.
# Maps a genre label to the baseline file stem expected under data/baselines/.
GENRE_BASELINES: Dict[str, str] = {
    "classical": "gpt2_gutenberg_512",          # 19th-century literary prose
    "modern_literary": "gpt2_standardebooks_512",  # curated modern literary
    "web_prose": "gpt2_rss_512",                # contemporary web writing
    "mixed": "gpt2_mixed_512",                  # blended corpus
}


def load_genre_baseline(genre: str, *, model_id: str = "gpt2") -> Baseline:
    """Load a baseline for a specific genre, falling back to the default if unavailable."""
    stem = GENRE_BASELINES.get(genre)
    if stem:
        p = Path("data/baselines") / f"{stem}_docs.json"
        if p.exists():
            return load_baseline_cached(stem, path=p)
    # Fallback to default
    return load_baseline_cached(f"{safe_model_id(model_id)}_gutenberg_512")
