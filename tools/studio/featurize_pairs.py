"""Extract feature vectors from pair data using Qwen3 + cheap taxonomy features.

Usage:
    python -m tools.studio.featurize_pairs \
        --pairs data/pairs_v6/train.jsonl \
        --out data/pairs_v6/train_featurized.jsonl \
        --model-id Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from tools.studio.analyze import analyze_text
from tools.studio.dataset_utils import iter_jsonl, write_jsonl
from tools.studio.preference_features import FEATURE_SCHEMA, extract_features

logger = logging.getLogger(__name__)


def featurize_text(
    text: str,
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    doc_type: str = "prose",
    backend: str = "auto",
    max_input_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run analyze_text and extract the feature vector as a dict."""
    result = analyze_text(
        text,
        model_id=model_id,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=max_input_tokens,
        compute_cohesion=True,
        include_token_metrics=False,
    )
    doc_metrics = result.get("doc_metrics") or {}
    vec = extract_features(doc_metrics, text)
    return {name: float(vec[i]) for i, name in enumerate(FEATURE_SCHEMA)}


def featurize_pairs(
    *,
    pairs_path: Path,
    out_path: Path,
    model_id: str = "Qwen/Qwen3-1.7B",
    backend: str = "auto",
    max_input_tokens: int = 1024,
    resume: bool = True,
) -> Dict[str, Any]:
    """Featurize all pairs in a JSONL file.

    Each output row contains the original pair fields plus:
        chosen_features: {feature_name: value, ...}
        rejected_features: {feature_name: value, ...}
    """
    # Resume support: skip already-processed pair_ids
    processed_ids: set = set()
    if resume and out_path.exists():
        for row in iter_jsonl(out_path):
            pid = str(row.get("pair_id") or "")
            if pid:
                processed_ids.add(pid)
        logger.info("resume_featurize", extra={"already_processed": len(processed_ids)})

    pairs = list(iter_jsonl(pairs_path))
    n_total = len(pairs)
    n_skipped = 0
    n_done = 0
    n_errors = 0
    t0 = time.monotonic()

    # Append mode for resume
    mode = "a" if processed_ids else "w"
    with open(out_path, mode, encoding="utf-8") as fout:
        for i, row in enumerate(pairs):
            pid = str(row.get("pair_id") or "")
            if pid in processed_ids:
                n_skipped += 1
                continue

            chosen_text = str(row.get("chosen_text") or "")
            rejected_text = str(row.get("rejected_text") or "")

            try:
                chosen_feat = featurize_text(
                    chosen_text,
                    model_id=model_id,
                    backend=backend,
                    max_input_tokens=max_input_tokens,
                )
                rejected_feat = featurize_text(
                    rejected_text,
                    model_id=model_id,
                    backend=backend,
                    max_input_tokens=max_input_tokens,
                )
                out_row = dict(row)
                out_row["chosen_features"] = chosen_feat
                out_row["rejected_features"] = rejected_feat
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
            except Exception:
                logger.exception("featurize_error", extra={"pair_id": pid, "index": i})
                n_errors += 1

            if (n_done + n_errors) % 10 == 0:
                elapsed = time.monotonic() - t0
                rate = float(n_done + n_errors) / max(elapsed, 0.01)
                remaining = (n_total - n_skipped - n_done - n_errors) / max(rate, 0.001)
                logger.info(
                    "featurize_progress",
                    extra={
                        "done": n_done,
                        "errors": n_errors,
                        "total": n_total,
                        "skipped": n_skipped,
                        "rate_per_sec": round(rate, 2),
                        "eta_min": round(remaining / 60, 1),
                    },
                )

    elapsed = time.monotonic() - t0
    stats = {
        "total_pairs": n_total,
        "skipped_resume": n_skipped,
        "featurized": n_done,
        "errors": n_errors,
        "elapsed_sec": round(elapsed, 1),
        "model_id": model_id,
    }
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract feature vectors from pair data.")
    ap.add_argument("--pairs", required=True, help="Input pairs JSONL")
    ap.add_argument("--out", required=True, help="Output featurized JSONL")
    ap.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stats = featurize_pairs(
        pairs_path=Path(args.pairs),
        out_path=Path(args.out),
        model_id=args.model_id,
        backend=args.backend,
        max_input_tokens=args.max_input_tokens,
        resume=not args.no_resume,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
