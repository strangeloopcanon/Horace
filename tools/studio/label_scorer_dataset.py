from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm import tqdm

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline, load_baseline_cached
from tools.studio.dataset_utils import iter_jsonl, make_sample_id
from tools.studio.score import score_text


def _pick_rows(rows: List[dict], *, seed: int, max_samples: Optional[int]) -> List[dict]:
    if max_samples is None:
        return rows
    n = int(max_samples)
    if n <= 0:
        return []
    if len(rows) <= n:
        return rows
    rng = random.Random(int(seed))
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    keep = set(idxs[:n])
    return [r for i, r in enumerate(rows) if i in keep]


def label_jsonl(
    *,
    in_path: Path,
    out_path: Path,
    max_samples: Optional[int],
    seed: int,
    teacher_model_id: str,
    baseline_model: str,
    doc_type: str,
    backend: str,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
) -> int:
    rows = list(iter_jsonl(in_path))
    picked = _pick_rows(rows, seed=int(seed), max_samples=max_samples)

    baseline = _ensure_baseline(str(baseline_model))
    teacher_meta = {
        "teacher_model_id": str(teacher_model_id),
        "baseline_model": str(baseline_model),
        "doc_type": str(doc_type),
        "backend": str(backend),
        "max_input_tokens": int(max_input_tokens),
        "normalize_text": bool(normalize_text),
        "compute_cohesion": bool(compute_cohesion),
        "seed": int(seed),
    }

    now = int(time.time())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in tqdm(picked, desc=f"label {in_path.name}", unit="sample"):
            text = str(r.get("text") or "")
            if not text.strip():
                continue
            src = str(r.get("source") or "unknown")
            title = str(r.get("title") or "")
            url = str(r.get("url") or "")
            sid = str(r.get("sample_id") or "").strip() or make_sample_id(src, title, url, text)
            analysis = analyze_text(
                text,
                model_id=str(teacher_model_id),
                doc_type=str(doc_type),
                backend=str(backend),
                max_input_tokens=int(max_input_tokens),
                normalize_text=bool(normalize_text),
                compute_cohesion=bool(compute_cohesion),
            )
            score = score_text((analysis.get("doc_metrics") or {}), baseline, doc_type=str(doc_type))
            y0_100 = float(score.overall_0_100)
            y = y0_100 / 100.0
            y = max(0.0, min(1.0, y)) if math.isfinite(y) else 0.0

            out_row = dict(r)
            out_row["sample_id"] = sid
            out_row["fetched_at_unix"] = int(r.get("fetched_at_unix") or now)
            out_row["label"] = float(y)
            out_row["teacher_overall_0_100"] = float(y0_100)
            out_row["teacher_categories_0_1"] = {k: float(v) for k, v in (score.categories or {}).items()}
            out_row["teacher_meta"] = dict(teacher_meta)
            out_row["teacher_tokens_count"] = int((analysis.get("doc_metrics") or {}).get("tokens_count") or 0)
            out_row["teacher_truncated"] = bool(analysis.get("truncated"))

            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _ensure_baseline(baseline_model_or_path: str):
    ident = (baseline_model_or_path or "").strip() or "gpt2"
    p = Path(ident)
    if p.exists():
        return load_baseline_cached(ident, path=p)
    try:
        return load_baseline_cached(ident)
    except Exception:
        build_baseline(ident)
        return load_baseline_cached(ident)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Label a JSONL dataset with a teacher score (rubric overall_0_100) so we can distill a fast text→score model."
        )
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL (rows must include `text`)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL (adds `label` in [0,1])")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)

    # Teacher scoring config (same as eval_web / eval_set)
    ap.add_argument("--teacher-model", default="gpt2", help="Causal LM id used to compute rubric metrics")
    ap.add_argument("--baseline-model", default="gpt2_gutenberg_512", help="Baseline id or baseline JSON path")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--backend", default="hf", choices=["hf", "mlx"], help="Teacher backend")
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--compute-cohesion", action="store_true", help="Include cohesion metric (slower)")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else None

    in_path = Path(str(args.in_path))
    out_path = Path(str(args.out_path))
    n = label_jsonl(
        in_path=in_path,
        out_path=out_path,
        max_samples=max_samples,
        seed=int(args.seed),
        teacher_model_id=str(args.teacher_model),
        baseline_model=str(args.baseline_model),
        doc_type=str(args.doc_type),
        backend=str(args.backend),
        max_input_tokens=int(args.max_input_tokens),
        normalize_text=bool(normalize_text),
        compute_cohesion=bool(args.compute_cohesion),
    )

    print(str(out_path))
    print(f"n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
