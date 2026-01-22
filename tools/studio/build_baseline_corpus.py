from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline_from_rows, safe_model_id
from tools.studio.dataset_utils import iter_jsonl


@dataclass(frozen=True)
class BuildBaselineCorpusResult:
    baseline_path: Path
    rows_path: Optional[Path]
    n_rows: int
    model_id: str
    doc_type: str
    max_input_tokens: int


def _pick_unique_groups(rows: List[dict], *, seed: int, max_groups: Optional[int]) -> List[dict]:
    if not rows:
        return []
    seen: Set[str] = set()
    unique: List[dict] = []
    for r in rows:
        gid = str(r.get("group_id") or "").strip() or str(r.get("sample_id") or "").strip()
        if not gid:
            continue
        if gid in seen:
            continue
        seen.add(gid)
        unique.append(r)

    if max_groups is None:
        return unique
    n = int(max_groups)
    if n <= 0:
        return []
    if len(unique) <= n:
        return unique
    rng = random.Random(int(seed))
    rng.shuffle(unique)
    return unique[:n]


def build_baseline_from_corpus(
    *,
    in_path: Path,
    model_id: str,
    backend: str,
    doc_type: str,
    max_input_tokens: int,
    sources: Sequence[str],
    seed: int,
    max_groups: Optional[int],
    normalize_text: bool,
    compute_cohesion: bool,
    out_path: Path,
    rows_out: Optional[Path],
) -> BuildBaselineCorpusResult:
    src_set = {str(s) for s in sources if str(s).strip()}
    rows: List[dict] = []
    for r in iter_jsonl(in_path):
        if not isinstance(r, dict):
            continue
        if src_set:
            src = str(r.get("source") or "").strip()
            if src not in src_set:
                continue
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        rows.append(r)

    picked = _pick_unique_groups(rows, seed=int(seed), max_groups=max_groups)
    if not picked:
        raise RuntimeError("No baseline candidates found (check --in and --source filters)")

    metric_rows: List[Dict[str, Any]] = []
    docs_meta: List[Dict[str, Any]] = []
    now = int(time.time())
    for r in picked:
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        res = analyze_text(
            text,
            model_id=str(model_id),
            doc_type=str(doc_type),
            backend=str(backend),
            max_input_tokens=int(max_input_tokens),
            normalize_text=bool(normalize_text),
            compute_cohesion=bool(compute_cohesion),
        )
        dm = res.get("doc_metrics") or {}
        if not isinstance(dm, dict) or not dm:
            continue
        metric_rows.append(dm)
        docs_meta.append(
            {
                "sample_id": r.get("sample_id"),
                "group_id": r.get("group_id"),
                "source": r.get("source"),
                "title": r.get("title"),
                "url": r.get("url"),
                "tokens_count": int(dm.get("tokens_count") or 0),
                "truncated": bool(res.get("truncated")),
                "at_unix": now,
            }
        )

    if not metric_rows:
        raise RuntimeError("No baseline rows collected (model failure?)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_baseline_from_rows(str(model_id), metric_rows, out_path=out_path)

    if rows_out is not None:
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        rows_out.write_text(json.dumps(docs_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        obj = json.loads(out_path.read_text(encoding="utf-8"))
        obj["build_meta"] = {
            "source": "corpus_jsonl",
            "in_path": str(in_path),
            "sources": sorted(src_set),
            "unique_groups_used": int(len(metric_rows)),
            "seed": int(seed),
            "max_input_tokens": int(max_input_tokens),
            "normalize_text": bool(normalize_text),
            "compute_cohesion": bool(compute_cohesion),
        }
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return BuildBaselineCorpusResult(
        baseline_path=out_path,
        rows_path=rows_out,
        n_rows=len(metric_rows),
        model_id=str(model_id),
        doc_type=str(doc_type),
        max_input_tokens=int(max_input_tokens),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a baseline distribution from an existing samples.jsonl/split.jsonl corpus.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL with at least {text, group_id, source}")
    ap.add_argument("--model-id", default="gpt2")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "hf"])
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--source", action="append", default=[], help="Restrict to these source values (repeatable)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-groups", type=int, default=1200, help="Max unique group_id sampled (0 = all)")
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--compute-cohesion", action="store_true", help="Slower; includes cohesion_delta if possible")
    ap.add_argument("--out-path", default=None, help="Explicit baseline JSON output path")
    ap.add_argument("--out-dir", default="data/baselines")
    ap.add_argument("--rows-out", default=None, help="Optional JSON file of docs_meta used for baseline")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    max_groups = int(args.max_groups) if int(args.max_groups) > 0 else None

    in_path = Path(str(args.in_path))
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_path: Path
    if args.out_path:
        out_path = Path(str(args.out_path))
    else:
        out_dir = Path(str(args.out_dir))
        tag = safe_model_id(str(args.model_id))
        suffix = f"corpus_{int(args.max_input_tokens)}"
        out_path = out_dir / f"{tag}_{suffix}_docs.json"

    rows_out = Path(str(args.rows_out)) if args.rows_out else None

    res = build_baseline_from_corpus(
        in_path=in_path,
        model_id=str(args.model_id),
        backend=str(args.backend),
        doc_type=str(args.doc_type),
        max_input_tokens=int(args.max_input_tokens),
        sources=tuple(args.source or []),
        seed=int(args.seed),
        max_groups=max_groups,
        normalize_text=bool(normalize_text),
        compute_cohesion=bool(args.compute_cohesion),
        out_path=out_path,
        rows_out=rows_out,
    )
    payload = {**asdict(res), "baseline_path": str(res.baseline_path), "rows_path": str(res.rows_path) if res.rows_path else None}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

