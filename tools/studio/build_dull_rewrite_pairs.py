from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from tools.studio.dataset_utils import group_id as make_group_id, iter_jsonl, stable_sha1_hex, write_jsonl
from tools.studio.rewrite import generate_dulled_rewrites
from tools.studio.text_corrupt import corrupt_text


def _pair_id(*, group_id: str, sample_id: str, kind: str, seed: int) -> str:
    return stable_sha1_hex([str(group_id), str(sample_id), str(kind), str(int(seed))])[:12]


def build_dull_rewrite_pairs(
    *,
    in_path: Path,
    out_path: Path,
    seed: int,
    rewrite_model_id: str,
    doc_type: str,
    strength: str,
    rewrites_per_sample: int,
    max_samples: int,
    corruption_kinds: Sequence[str],
    corruptions_per_sample: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    rows = list(iter_jsonl(Path(in_path)))
    rng.shuffle(rows)
    if int(max_samples) > 0:
        rows = rows[: int(max_samples)]

    out_rows: List[dict] = []
    stats: Dict[str, Any] = {
        "seed": int(seed),
        "in_path": str(in_path),
        "out_path": str(out_path),
        "rewrite_model_id": str(rewrite_model_id),
        "doc_type": str(doc_type),
        "strength": str(strength),
        "rewrites_per_sample": int(rewrites_per_sample),
        "max_samples": int(max_samples),
        "corruption_kinds": list(corruption_kinds),
        "corruptions_per_sample": int(corruptions_per_sample),
        "rows_in": int(len(rows)),
        "pairs_out": 0,
        "skipped_empty_text": 0,
        "rewrite_failures": 0,
        "corruption_failures": 0,
    }

    now = int(time.time())
    for r in tqdm(rows, desc=f"dull_pairs {Path(in_path).name}", unit="sample"):
        text = str(r.get("text") or "")
        if not text.strip():
            stats["skipped_empty_text"] += 1
            continue
        gid = str(r.get("group_id") or "").strip()
        if not gid:
            gid = make_group_id("sample", stable=str(r.get("url") or r.get("title") or r.get("sample_id") or ""))
        sid = str(r.get("sample_id") or "").strip() or stable_sha1_hex([gid, text])[:12]

        # 1) LLM “dull rewrites”: within-content negatives (optional).
        rewrites: List[str] = []
        rw_n = max(0, int(rewrites_per_sample))
        if rw_n > 0:
            try:
                rewrites = generate_dulled_rewrites(
                    text,
                    doc_type=str(doc_type),
                    rewrite_model_id=str(rewrite_model_id),
                    strength=str(strength),
                    n=int(rw_n),
                    seed=int(seed) ^ (int(stable_sha1_hex([sid])[:8], 16) & 0xFFFF_FFFF),
                )
            except Exception:
                rewrites = []
            if not rewrites:
                stats["rewrite_failures"] += 1
        for j, rw in enumerate(rewrites):
            if not rw.strip() or rw.strip() == text.strip():
                continue
            kind = f"dull_rewrite_{str(strength).strip().lower() or 'mild'}"
            out_rows.append(
                {
                    "pair_id": _pair_id(group_id=gid, sample_id=sid, kind=f"{kind}:{j}", seed=int(seed)),
                    "group_id": gid,
                    "chosen_text": text,
                    "rejected_text": rw,
                    "meta": {
                        "created_at_unix": now,
                        "rewrite_kind": kind,
                        "rewrite_model_id": str(rewrite_model_id),
                        "input_source": r.get("source"),
                        "input_sample_id": sid,
                    },
                }
            )
            stats["pairs_out"] += 1

        # 2) Deterministic “cadence breaks”: cheaper negatives (optional).
        kinds = [k for k in (corruption_kinds or []) if str(k).strip()]
        rng.shuffle(kinds)
        kinds = kinds[: max(0, int(corruptions_per_sample))]
        for k in kinds:
            try:
                corr = corrupt_text(text, rng=rng, kind=str(k))
            except Exception:
                stats["corruption_failures"] += 1
                continue
            if not corr.strip() or corr.strip() == text.strip():
                continue
            out_rows.append(
                {
                    "pair_id": _pair_id(group_id=gid, sample_id=sid, kind=f"corrupt:{k}", seed=int(seed)),
                    "group_id": gid,
                    "chosen_text": text,
                    "rejected_text": corr,
                    "meta": {
                        "created_at_unix": now,
                        "rewrite_kind": "deterministic_corruption",
                        "corruption_kind": str(k),
                        "input_source": r.get("source"),
                        "input_sample_id": sid,
                    },
                }
            )
            stats["pairs_out"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out_rows)
    return {"pairs_path": str(out_path), "stats": stats}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build preference pairs (chosen > rejected) using LLM 'dull rewrites' plus optional deterministic cadence corruptions."
        )
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL split with `text` (e.g. standardebooks splits/train.jsonl)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL pairs: {chosen_text,rejected_text,...}")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--rewrite-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--strength", default="mild", choices=["mild", "strong"])
    ap.add_argument("--rewrites-per-sample", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0, help="Cap input samples (0 = all)")

    ap.add_argument("--corruption-kind", action="append", default=[], help="Add deterministic corruption kind (repeatable)")
    ap.add_argument("--corruptions-per-sample", type=int, default=0, help="Apply up to N corruptions per sample (0 = none)")
    args = ap.parse_args(argv)

    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else 0

    res = build_dull_rewrite_pairs(
        in_path=Path(str(args.in_path)),
        out_path=Path(str(args.out_path)),
        seed=int(args.seed),
        rewrite_model_id=str(args.rewrite_model),
        doc_type=str(args.doc_type),
        strength=str(args.strength),
        rewrites_per_sample=int(args.rewrites_per_sample),
        max_samples=int(max_samples),
        corruption_kinds=tuple(args.corruption_kind or []),
        corruptions_per_sample=int(args.corruptions_per_sample),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
