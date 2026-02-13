from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.studio.dataset_utils import iter_jsonl, stable_sha1_hex, write_jsonl
from tools.studio.split_eval_set import split_eval_set


def _author_group_id(author_norm: str) -> str:
    return f"author:{stable_sha1_hex([str(author_norm)])[:12]}"


def _load_originals(
    *,
    path: Path,
    author_holdout: bool,
    positive_source: str,
) -> List[dict]:
    out: List[dict] = []
    for r in iter_jsonl(path):
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        group_id = str(r.get("group_id") or "").strip()
        author_norm = str(r.get("author_norm") or "").strip()
        if bool(author_holdout) and author_norm:
            group_id = _author_group_id(author_norm)
        if not group_id:
            group_id = f"sample:{stable_sha1_hex([text])[:12]}"
        in_meta = dict(r.get("meta") or {}) if isinstance(r.get("meta"), dict) else {}
        out.append(
            {
                "sample_id": str(r.get("sample_id") or stable_sha1_hex([group_id, text])[:12]),
                "group_id": group_id,
                "source": str(positive_source),
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "text": text,
                "fetched_at_unix": int(r.get("fetched_at_unix") or 0),
                "meta": {
                    **in_meta,
                    "author": str(r.get("author") or ""),
                    "author_norm": author_norm,
                    "doc_type": str(r.get("doc_type") or "prose"),
                    "eval_kind": "human_original",
                },
            }
        )
    return out


def _load_negatives(
    *,
    pairs_path: Optional[Path],
    negatives_path: Optional[Path],
    author_holdout: bool,
) -> List[dict]:
    out: List[dict] = []
    if negatives_path is not None and negatives_path.exists():
        for r in iter_jsonl(negatives_path):
            text = str(r.get("text") or "")
            if not text.strip():
                continue
            group_id = str(r.get("group_id") or "").strip()
            meta = dict(r.get("meta") or {}) if isinstance(r.get("meta"), dict) else {}
            author_norm = str(meta.get("author_norm") or "")
            if bool(author_holdout) and author_norm:
                group_id = _author_group_id(author_norm)
            if not group_id:
                group_id = f"neg:{stable_sha1_hex([text])[:12]}"
            out.append(
                {
                    "sample_id": str(r.get("sample_id") or stable_sha1_hex([group_id, text])[:12]),
                    "group_id": group_id,
                    "source": str(r.get("source") or "llm_antipattern"),
                    "title": str(r.get("title") or ""),
                    "url": str(r.get("url") or ""),
                    "text": text,
                    "fetched_at_unix": int(r.get("fetched_at_unix") or 0),
                    "meta": {
                        **meta,
                        "eval_kind": "llm_antipattern",
                    },
                }
            )

    if not out and pairs_path is not None and pairs_path.exists():
        for r in iter_jsonl(pairs_path):
            chosen = str(r.get("chosen_text") or "")
            rejected = str(r.get("rejected_text") or "")
            if not chosen.strip() or not rejected.strip():
                continue
            meta = dict(r.get("meta") or {}) if isinstance(r.get("meta"), dict) else {}
            source = str(meta.get("source_label") or "llm_antipattern")
            group_id = str(r.get("group_id") or "").strip()
            author_norm = str(meta.get("author_norm") or "")
            if bool(author_holdout) and author_norm:
                group_id = _author_group_id(author_norm)
            if not group_id:
                group_id = f"pair:{stable_sha1_hex([chosen])[:12]}"
            out.append(
                {
                    "sample_id": stable_sha1_hex([str(r.get("pair_id") or ""), "neg"])[:12],
                    "group_id": group_id,
                    "source": source,
                    "title": str(meta.get("input_title") or ""),
                    "url": str(meta.get("input_url") or ""),
                    "text": rejected,
                    "fetched_at_unix": int(meta.get("created_at_unix") or 0),
                    "meta": {
                        **meta,
                        "eval_kind": "llm_antipattern",
                    },
                }
            )
    return out


def build_ai_overfit_eval(
    *,
    originals_path: Path,
    out_dir: Path,
    pairs_path: Optional[Path],
    negatives_path: Optional[Path],
    seed: int,
    train_frac: float,
    val_frac: float,
    max_humans: int,
    max_negatives: int,
    author_holdout: bool,
    positive_source: str,
) -> Dict[str, Any]:
    humans = _load_originals(path=Path(originals_path), author_holdout=bool(author_holdout), positive_source=str(positive_source))
    negatives = _load_negatives(
        pairs_path=Path(pairs_path) if pairs_path is not None else None,
        negatives_path=Path(negatives_path) if negatives_path is not None else None,
        author_holdout=bool(author_holdout),
    )
    if int(max_humans) > 0:
        humans = humans[: int(max_humans)]
    if int(max_negatives) > 0:
        negatives = negatives[: int(max_negatives)]

    rows = humans + negatives
    if not rows:
        raise RuntimeError("No rows available to build AI-overfit eval set.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    write_jsonl(samples_path, rows)

    manifest = split_eval_set(
        samples_path=samples_path,
        out_dir=out_dir / "splits",
        seed=int(seed),
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        stratify_by_source=True,
    )

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "author_holdout": bool(author_holdout),
        "positive_source": str(positive_source),
        "n_human": int(len(humans)),
        "n_negative": int(len(negatives)),
        "n_total": int(len(rows)),
        "sources": {},
        "manifest_path": str((out_dir / "splits" / "manifest.json")),
    }
    for r in rows:
        src = str(r.get("source") or "unknown")
        stats["sources"][src] = int(stats["sources"].get(src, 0)) + 1

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps({"stats": stats, "split_manifest": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "samples_path": str(samples_path),
        "splits_dir": str(out_dir / "splits"),
        "stats_path": str(stats_path),
        "stats": stats,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a held-out AI-overfit evaluation set from human originals plus LLM anti-pattern negatives."
        )
    )
    ap.add_argument("--originals", required=True, help="Originals JSONL (human passages)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--pairs", default="", help="Optional pairs JSONL (chosen/rejected)")
    ap.add_argument("--negatives", default="", help="Optional negatives JSONL (classification rows)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-humans", type=int, default=0, help="Cap human rows (0 = all)")
    ap.add_argument("--max-negatives", type=int, default=0, help="Cap negative rows (0 = all)")
    ap.add_argument("--author-holdout", action="store_true", help="Split by author group (stronger holdout)")
    ap.add_argument("--positive-source", default="human_original")
    args = ap.parse_args(argv)

    res = build_ai_overfit_eval(
        originals_path=Path(str(args.originals)),
        out_dir=Path(str(args.out_dir)),
        pairs_path=Path(str(args.pairs)) if str(args.pairs).strip() else None,
        negatives_path=Path(str(args.negatives)) if str(args.negatives).strip() else None,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        max_humans=int(args.max_humans),
        max_negatives=int(args.max_negatives),
        author_holdout=bool(args.author_holdout),
        positive_source=str(args.positive_source),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
