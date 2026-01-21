from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline_from_rows, safe_model_id


@dataclass(frozen=True)
class PersonalBaselineResult:
    baseline_path: Path
    docs_path: Path
    n_docs: int
    model_id: str
    doc_type: str


def _iter_files(input_path: Path, *, patterns: Sequence[str]) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for pat in patterns:
        yield from sorted(input_path.glob(pat))


def build_personal_baseline(
    input_path: Path,
    *,
    model_id: str,
    doc_type: str,
    backend: str = "auto",
    max_input_tokens: int = 1024,
    compute_cohesion: bool = False,
    limit: Optional[int] = None,
    out_dir: Path = Path("data/baselines"),
    file_patterns: Sequence[str] = ("**/*.txt",),
) -> PersonalBaselineResult:
    rows: List[dict] = []
    docs_meta: List[dict] = []
    n = 0
    for p in _iter_files(input_path, patterns=file_patterns):
        if limit is not None and n >= int(limit):
            break
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            continue
        res = analyze_text(
            text,
            model_id=model_id,
            doc_type=doc_type,
            backend=backend,
            max_input_tokens=max_input_tokens,
            compute_cohesion=compute_cohesion,
        )
        dm = res.get("doc_metrics") or {}
        if not isinstance(dm, dict):
            continue
        dm = {**dm, "doc_type": dm.get("doc_type") or doc_type, "path": str(p)}
        rows.append(dm)
        docs_meta.append(
            {
                "path": str(p),
                "tokens_count": int(dm.get("tokens_count") or 0),
                "truncated": bool(res.get("truncated")),
            }
        )
        n += 1

    tag = safe_model_id(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / f"personal_{tag}_{doc_type}_docs_meta.json"
    baseline_path = out_dir / f"personal_{tag}_{doc_type}_docs.json"
    docs_path.write_text(json.dumps(docs_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    build_baseline_from_rows(model_id, rows, out_path=baseline_path)
    return PersonalBaselineResult(
        baseline_path=baseline_path,
        docs_path=docs_path,
        n_docs=len(rows),
        model_id=model_id,
        doc_type=doc_type,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a personal baseline from your own writing (metrics only).")
    ap.add_argument("--input", required=True, help="A .txt file or a folder of .txt files")
    ap.add_argument("--model-id", default="gpt2")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "hf"])
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    ap.add_argument("--compute-cohesion", action="store_true", help="Slower; includes cohesion_delta if possible")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="data/baselines")
    ap.add_argument("--pattern", action="append", default=["**/*.txt"], help="Glob pattern(s) under --input folder")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    os.makedirs(args.out_dir, exist_ok=True)

    res = build_personal_baseline(
        input_path,
        model_id=str(args.model_id),
        doc_type=str(args.doc_type),
        backend=str(args.backend),
        max_input_tokens=int(args.max_input_tokens),
        compute_cohesion=bool(args.compute_cohesion),
        limit=int(args.limit) if args.limit is not None else None,
        out_dir=Path(args.out_dir),
        file_patterns=tuple(args.pattern or ["**/*.txt"]),
    )
    print(str(res.baseline_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

