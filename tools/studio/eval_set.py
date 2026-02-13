from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.studio.eval_web import TextSample, score_samples
from tools.studio.scorer_model import score_with_scorer


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    pos = (len(sorted_vals) - 1) * q
    i = int(math.floor(pos))
    j = min(len(sorted_vals) - 1, i + 1)
    frac = pos - i
    return float((1.0 - frac) * sorted_vals[i] + frac * sorted_vals[j])


def _summarize_scores(scores: Sequence[float]) -> Dict[str, Any]:
    vals = sorted([float(x) for x in scores if x is not None and math.isfinite(float(x))])
    if not vals:
        return {"n": 0, "mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "n": int(len(vals)),
        "mean": float(sum(vals) / len(vals)),
        "p10": _quantile(vals, 0.10),
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }


def _score_from_model(
    *,
    text: str,
    model_path_or_id: Optional[str],
    doc_type: str,
    normalize_text: bool,
    max_length: int,
    invert: bool = False,
) -> Optional[Dict[str, float]]:
    if not model_path_or_id:
        return None
    try:
        result = score_with_scorer(
            text,
            model_path_or_id=str(model_path_or_id),
            doc_type=str(doc_type),
            normalize_text=bool(normalize_text),
            max_length=int(max_length),
        )
    except Exception:
        return None
    score_0_1 = float(result.prob_0_1)
    if not math.isfinite(score_0_1):
        return None
    if invert:
        score_0_1 = 1.0 - score_0_1
    score_0_1 = max(0.0, min(1.0, score_0_1))
    return {"score_0_1": float(score_0_1), "score_0_100": float(100.0 * score_0_1)}


def _auc_roc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    if not y_true or len(y_true) != len(y_score):
        return None
    n = len(y_true)
    n_pos = int(sum(1 for y in y_true if int(y) == 1))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: float(y_score[i]))
    ranks = [0] * n
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    pos_ranks = [ranks[i] for i, y in enumerate(y_true) if int(y) == 1]
    auc = (sum(pos_ranks) - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _load_samples_jsonl(path: Path) -> List[TextSample]:
    rows: List[TextSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        text = str(obj.get("text") or "")
        if not text.strip():
            continue
        rows.append(
            TextSample(
                sample_id=str(obj.get("sample_id") or ""),
                source=str(obj.get("source") or "unknown"),
                title=str(obj.get("title") or ""),
                url=str(obj.get("url") or ""),
                text=text,
                fetched_at_unix=int(obj.get("fetched_at_unix") or 0),
            )
        )
    return rows


def run_eval_set(
    *,
    samples_path: Path,
    model_id: str,
    baseline_model: str,
    analysis_model: Optional[str] = None,
    doc_type: str,
    backend: str,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
    quality_scorer_model: Optional[str] = None,
    quality_scorer_max_length: int = 512,
    authenticity_scorer_model: Optional[str] = None,
    authenticity_scorer_max_length: int = 512,
    authenticity_invert: bool = False,
    positive_sources: Sequence[str],
    negative_sources: Optional[Sequence[str]],
    calibrator_path: Optional[Path] = None,
    report_out: Path,
) -> dict:
    samples = _load_samples_jsonl(samples_path)
    if not samples:
        raise RuntimeError(f"No samples loaded from {samples_path}")

    scored = score_samples(
        samples,
        model_id=str(model_id),
        baseline_model=str(baseline_model),
        analysis_model=analysis_model,
        doc_type=str(doc_type),
        backend=str(backend),
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        compute_cohesion=bool(compute_cohesion),
    )

    by_source: Dict[str, List[Any]] = {}
    scored_rows: List[Dict[str, Any]] = []
    quality_request_count = len(samples) if quality_scorer_model and str(quality_scorer_model).strip() else 0
    authenticity_request_count = len(samples) if authenticity_scorer_model and str(authenticity_scorer_model).strip() else 0
    quality_success_count = 0
    authenticity_success_count = 0
    for s in scored:
        by_source.setdefault(s.source, []).append(s)

    for s, sample in zip(scored, samples):
        row = asdict(s)
        q_row = _score_from_model(
            text=str(sample.text),
            model_path_or_id=quality_scorer_model,
            doc_type=str(doc_type),
            normalize_text=bool(normalize_text),
            max_length=int(quality_scorer_max_length),
            invert=False,
        )
        if q_row is not None:
            row["quality_score_0_1"] = q_row.get("score_0_1")
            row["quality_score_0_100"] = q_row.get("score_0_100")
            quality_success_count += 1

        a_row = _score_from_model(
            text=str(sample.text),
            model_path_or_id=authenticity_scorer_model,
            doc_type=str(doc_type),
            normalize_text=bool(normalize_text),
            max_length=int(authenticity_scorer_max_length),
            invert=bool(authenticity_invert),
        )
        if a_row is not None:
            row["authenticity_score_0_1"] = a_row.get("score_0_1")
            row["authenticity_score_0_100"] = a_row.get("score_0_100")
            authenticity_success_count += 1
        scored_rows.append(row)

    report: Dict[str, Any] = {
        "run_meta": {
            "samples_path": str(samples_path),
            "model_id": str(model_id),
            "analysis_model": str(analysis_model or ""),
            "baseline_model": str(baseline_model),
            "doc_type": str(doc_type),
            "backend": str(backend),
            "max_input_tokens": int(max_input_tokens),
            "normalize_text": bool(normalize_text),
            "compute_cohesion": bool(compute_cohesion),
            "quality_scorer_model": str(quality_scorer_model or ""),
            "quality_scorer_max_length": int(quality_scorer_max_length),
            "quality_scores_requested": int(quality_request_count),
            "quality_scores_success": int(quality_success_count),
            "authenticity_scorer_model": str(authenticity_scorer_model or ""),
            "authenticity_scorer_max_length": int(authenticity_scorer_max_length),
            "authenticity_scores_requested": int(authenticity_request_count),
            "authenticity_scores_success": int(authenticity_success_count),
            "authenticity_invert": bool(authenticity_invert),
            "n_samples": int(len(samples)),
            "sources": {k: int(len(v)) for k, v in sorted(by_source.items())},
        },
        "sources": {},
        "scored": scored_rows,
    }

    for src, rows in sorted(by_source.items(), key=lambda kv: kv[0]):
        scores = [r.overall_0_100 for r in rows]
        cat_means: Dict[str, float] = {}
        for r in rows:
            for k, v in (r.categories or {}).items():
                cat_means[k] = cat_means.get(k, 0.0) + float(v)
        for k in list(cat_means.keys()):
            cat_means[k] = float((cat_means[k] / max(1, len(rows))) * 100.0)
        report["sources"][src] = {
            "summary": _summarize_scores(scores),
            "category_means_0_100": cat_means,
            "top": [asdict(r) for r in sorted(rows, key=lambda x: x.overall_0_100, reverse=True)[:3]],
            "bottom": [asdict(r) for r in sorted(rows, key=lambda x: x.overall_0_100)[:3]],
        }

    # Global ranking
    sorted_all = sorted(scored, key=lambda x: float(x.overall_0_100), reverse=True)
    report["overall_top"] = [asdict(s) for s in sorted_all[:10]]
    report["overall_bottom"] = [asdict(s) for s in reversed(sorted_all[-10:])]

    # Literary/non-literary diagnostic: AUC over sources
    pos_set = {str(x) for x in positive_sources if str(x).strip()}
    neg_set = {str(x) for x in (negative_sources or []) if str(x).strip()}
    y_true: List[int] = []
    y_score: List[float] = []
    y_quality: List[float] = []
    y_authenticity: List[float] = []
    yq_true: List[int] = []
    ya_true: List[int] = []
    quality_enabled = bool(quality_scorer_model and str(quality_scorer_model).strip())
    auth_enabled = bool(authenticity_scorer_model and str(authenticity_scorer_model).strip())
    for row in scored_rows:
        src = str(row.get("source") or "")
        overall = row.get("overall_0_100")
        if src in pos_set:
            y_true.append(1)
            y_score.append(float(overall) if isinstance(overall, (int, float)) else 0.0)
        elif neg_set and src in neg_set:
            y_true.append(0)
            y_score.append(float(overall) if isinstance(overall, (int, float)) else 0.0)
        elif not neg_set and src not in pos_set:
            y_true.append(0)
            y_score.append(float(overall) if isinstance(overall, (int, float)) else 0.0)

        in_scored_set = src in pos_set or (neg_set and src in neg_set) or (not neg_set and src not in pos_set)
        if in_scored_set and quality_enabled:
            qs = row.get("quality_score_0_100")
            if isinstance(qs, (int, float)):
                y_quality.append(float(qs))
                yq_true.append(1 if src in pos_set else 0)

        if in_scored_set and auth_enabled:
            as_ = row.get("authenticity_score_0_100")
            if isinstance(as_, (int, float)):
                y_authenticity.append(float(as_))
                ya_true.append(1 if src in pos_set else 0)
    report["literary_test"] = {
        "positive_sources": sorted(pos_set),
        "negative_sources": sorted(neg_set) if neg_set else "(all non-positive sources)",
        "n": int(len(y_true)),
        "auc": _auc_roc(y_true, y_score),
    }
    report["literary_test_quality"] = {
        "positive_sources": sorted(pos_set),
        "negative_sources": sorted(neg_set) if neg_set else "(all non-positive sources)",
        "n": int(len(yq_true)) if quality_enabled else 0,
        "auc": _auc_roc(yq_true, y_quality) if quality_enabled else None,
        "enabled": bool(quality_enabled),
    }
    report["literary_test_authenticity"] = {
        "positive_sources": sorted(pos_set),
        "negative_sources": sorted(neg_set) if neg_set else "(all non-positive sources)",
        "n": int(len(ya_true)) if auth_enabled else 0,
        "auc": _auc_roc(ya_true, y_authenticity) if auth_enabled else None,
        "enabled": bool(auth_enabled),
    }

    if calibrator_path is not None and Path(calibrator_path).exists():
        try:
            from tools.studio.calibrator import featurize_from_report_row, load_logistic_calibrator

            cal = load_logistic_calibrator(Path(calibrator_path))
            missing_value = float((cal.meta or {}).get("missing_value", 0.5))
            for row in scored_rows:
                cats = row.get("categories") or {}
                rms = row.get("rubric_metrics") or {}
                dm = row.get("doc_metrics") or {}
                if not isinstance(cats, dict) or not isinstance(rms, dict) or not isinstance(dm, dict):
                    row["calibrated_overall_0_100"] = None
                    continue
                feats = featurize_from_report_row(
                    feature_names=cal.feature_names,
                    categories=cats,
                    rubric_metrics=rms,
                    doc_metrics=dm,
                    report_row=row,
                    max_input_tokens=int(max_input_tokens),
                    missing_value=missing_value,
                )
                row["calibrated_overall_0_100"] = float(cal.score_0_100(feats))

            y_true_c: List[int] = []
            y_score_c: List[float] = []
            for row in scored_rows:
                src = str(row.get("source") or "")
                cs = row.get("calibrated_overall_0_100")
                if not isinstance(cs, (int, float)):
                    continue
                if src in pos_set:
                    y_true_c.append(1)
                    y_score_c.append(float(cs))
                elif neg_set and src in neg_set:
                    y_true_c.append(0)
                    y_score_c.append(float(cs))
                elif not neg_set and src not in pos_set:
                    y_true_c.append(0)
                    y_score_c.append(float(cs))

            report["calibrator"] = {
                "path": str(calibrator_path),
                "meta": cal.meta,
                "feature_dim": int(len(cal.feature_names)),
            }
            report["literary_test_calibrated"] = {
                "positive_sources": sorted(pos_set),
                "negative_sources": sorted(neg_set) if neg_set else "(all non-positive sources)",
                "n": int(len(y_true_c)),
                "auc": _auc_roc(y_true_c, y_score_c),
            }
        except Exception as e:
            report["calibrator_error"] = f"{type(e).__name__}: {e}"

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate Horace Studio scoring on a fixed sample set (JSONL).")
    ap.add_argument("--samples", default="data/eval_sets/studio_fixed_v1.jsonl")
    ap.add_argument("--model-id", default="gpt2")
    ap.add_argument("--baseline-model", default="gpt2_gutenberg_512")
    ap.add_argument(
        "--analysis-model",
        default=None,
        help="Optional causal model used for cadence/rubric analysis. Defaults to --model-id.",
    )
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "hf"])
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--compute-cohesion", action="store_true")
    ap.add_argument("--pos-source", action="append", default=["gutenberg_excerpt"])
    ap.add_argument("--neg-source", action="append", default=None)
    ap.add_argument("--quality-scorer-model", default=None, help="Optional quality scorer model/id for dual-score reporting")
    ap.add_argument("--quality-scorer-max-length", type=int, default=512)
    ap.add_argument(
        "--authenticity-scorer-model",
        default=None,
        help="Optional anti-pattern/authenticity scorer model/id for dual-score reporting",
    )
    ap.add_argument("--authenticity-scorer-max-length", type=int, default=512)
    ap.add_argument(
        "--authenticity-invert",
        action="store_true",
        help="Invert authenticity score (0 means human, 1 means AI-like)",
    )
    ap.add_argument("--calibrator", default=None, help="Optional calibrator JSON (adds calibrated AUC + scores)")
    ap.add_argument("--report-out", default="reports/studio_eval_set_report.json")
    args = ap.parse_args(argv)

    report = run_eval_set(
        samples_path=Path(str(args.samples)),
        model_id=str(args.model_id),
        baseline_model=str(args.baseline_model),
        analysis_model=str(args.analysis_model) if args.analysis_model else None,
        doc_type=str(args.doc_type),
        backend=str(args.backend),
        max_input_tokens=int(args.max_input_tokens),
        normalize_text=not bool(args.no_normalize_text),
        compute_cohesion=bool(args.compute_cohesion),
        quality_scorer_model=str(args.quality_scorer_model) if args.quality_scorer_model else None,
        quality_scorer_max_length=int(args.quality_scorer_max_length),
        authenticity_scorer_model=str(args.authenticity_scorer_model) if args.authenticity_scorer_model else None,
        authenticity_scorer_max_length=int(args.authenticity_scorer_max_length),
        authenticity_invert=bool(args.authenticity_invert),
        positive_sources=tuple(args.pos_source or []),
        negative_sources=tuple(args.neg_source) if args.neg_source else None,
        calibrator_path=Path(str(args.calibrator)) if args.calibrator else None,
        report_out=Path(str(args.report_out)),
    )

    print("== Horace Studio fixed-set eval ==")
    print(json.dumps(report.get("run_meta") or {}, indent=2))
    q_requested = report.get("run_meta", {}).get("quality_scores_requested") or 0
    q_success = report.get("run_meta", {}).get("quality_scores_success") or 0
    a_requested = report.get("run_meta", {}).get("authenticity_scores_requested") or 0
    a_success = report.get("run_meta", {}).get("authenticity_scores_success") or 0
    if int(q_requested) > 0 or int(a_requested) > 0:
        print(
            f"Model scorer rows: quality {q_success}/{q_requested}, "
            f"authenticity {a_success}/{a_requested}"
        )
    for src, info in (report.get("sources") or {}).items():
        summ = info.get("summary") or {}
        print(
            f"- {src}: n={summ.get('n')}, mean={summ.get('mean'):.1f}, "
            f"p10={summ.get('p10'):.1f}, p50={summ.get('p50'):.1f}, p90={summ.get('p90'):.1f}"
        )
    lt = report.get("literary_test") or {}
    print(f"\nAUC (literary test): {lt.get('auc')}")
    lq = report.get("literary_test_quality") or {}
    print(f"AUC (quality): {lq.get('auc')}")
    la = report.get("literary_test_authenticity") or {}
    print(f"AUC (authenticity): {la.get('auc')}")
    ltc = report.get("literary_test_calibrated") or {}
    if ltc.get("auc") is not None:
        print(f"AUC (calibrated): {ltc.get('auc')}")
    print(f"Wrote report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
