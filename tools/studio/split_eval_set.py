from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _sha1_12(s: str) -> str:
    h = hashlib.sha1()
    h.update((s or "").encode("utf-8"))
    return h.hexdigest()[:12]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _group_id_for_row(row: Mapping[str, Any]) -> str:
    gid = str(row.get("group_id") or "").strip()
    if gid:
        return gid
    src = str(row.get("source") or "unknown").strip() or "unknown"
    stable = str(row.get("url") or row.get("title") or row.get("sample_id") or "").strip()
    if not stable:
        stable = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return f"{src}:{_sha1_12(stable)}"


def _stable_int_seed(seed: int, salt: str) -> int:
    h = hashlib.sha1()
    h.update(str(int(seed)).encode("utf-8"))
    h.update(b"\0")
    h.update((salt or "").encode("utf-8"))
    return int(h.hexdigest()[:8], 16)


def _split_group_ids(
    group_ids: List[str],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    gids = sorted({g for g in group_ids if str(g).strip()})
    n = len(gids)
    if n == 0:
        return [], [], []

    tf = float(train_frac)
    vf = float(val_frac)
    if not (0.0 < tf < 1.0) or not (0.0 <= vf < 1.0) or (tf + vf) >= 1.0:
        raise ValueError("Invalid split fractions: require 0<train<1, 0<=val<1, train+val<1")

    import random

    rng = random.Random(_stable_int_seed(int(seed), f"split:{n}"))
    rng.shuffle(gids)

    if n == 1:
        return [], [], gids
    if n == 2:
        return gids[:1], [], gids[1:]

    n_train = int(round(n * tf))
    n_val = int(round(n * vf))

    # Ensure each split has at least 1 group when possible.
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - 1 - n_train, n_val))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1

    train = gids[:n_train]
    val = gids[n_train : n_train + n_val]
    test = gids[n_train + n_val :]
    return train, val, test


def split_eval_set(
    *,
    samples_path: Path,
    out_dir: Path,
    seed: int,
    train_frac: float,
    val_frac: float,
    stratify_by_source: bool = True,
) -> Dict[str, Any]:
    rows = list(_iter_jsonl(samples_path))
    if not rows:
        raise RuntimeError(f"No rows loaded from {samples_path}")

    # group_id -> (source, rows)
    group_to_source: Dict[str, str] = {}
    group_to_rows: Dict[str, List[dict]] = {}
    for r in rows:
        gid = _group_id_for_row(r)
        src = str(r.get("source") or "unknown").strip() or "unknown"
        group_to_source.setdefault(gid, src)
        group_to_rows.setdefault(gid, []).append(dict(r))

    # Partition groups (optionally stratified by source).
    train_groups: List[str] = []
    val_groups: List[str] = []
    test_groups: List[str] = []

    by_source: Dict[str, List[str]] = {}
    for gid, src in group_to_source.items():
        by_source.setdefault(src, []).append(gid)

    for src, gids in sorted(by_source.items(), key=lambda kv: kv[0]):
        if not stratify_by_source:
            continue
        tr, va, te = _split_group_ids(
            gids,
            seed=_stable_int_seed(int(seed), f"source:{src}"),
            train_frac=float(train_frac),
            val_frac=float(val_frac),
        )
        train_groups.extend(tr)
        val_groups.extend(va)
        test_groups.extend(te)

    if not stratify_by_source:
        all_groups = list(group_to_source.keys())
        tr, va, te = _split_group_ids(all_groups, seed=int(seed), train_frac=float(train_frac), val_frac=float(val_frac))
        train_groups, val_groups, test_groups = tr, va, te

    group_split: Dict[str, str] = {}
    for gid in train_groups:
        group_split[gid] = "train"
    for gid in val_groups:
        group_split[gid] = "val"
    for gid in test_groups:
        group_split[gid] = "test"

    # Write split JSONLs
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.jsonl",
        "val": out_dir / "val.jsonl",
        "test": out_dir / "test.jsonl",
    }
    writers = {k: p.open("w", encoding="utf-8") for k, p in paths.items()}
    try:
        for gid, rs in group_to_rows.items():
            split = group_split.get(gid, "test")
            w = writers.get(split)
            if w is None:
                continue
            for r in rs:
                r["group_id"] = gid
                r["split"] = split
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass

    # Basic manifest
    def _count_samples(split: str) -> int:
        return sum(len(group_to_rows[g]) for g, sp in group_split.items() if sp == split and g in group_to_rows)

    def _count_groups(split: str) -> int:
        return sum(1 for g, sp in group_split.items() if sp == split)

    per_source: Dict[str, Dict[str, Any]] = {}
    for src, gids in sorted(by_source.items(), key=lambda kv: kv[0]):
        gset = set(gids)
        per_source[src] = {
            "groups": len(gids),
            "samples": sum(len(group_to_rows[g]) for g in gids),
            "train_groups": sum(1 for g in gids if group_split.get(g) == "train"),
            "val_groups": sum(1 for g in gids if group_split.get(g) == "val"),
            "test_groups": sum(1 for g in gids if group_split.get(g) == "test"),
        }

    # Sample-level breakdown by row["source"] (works even when a group contains multiple sources,
    # e.g. original excerpt + corruptions that share a group_id).
    src_to_groups_all: Dict[str, set[str]] = {}
    src_to_groups_by_split: Dict[str, Dict[str, set[str]]] = {}
    src_to_samples_by_split: Dict[str, Dict[str, int]] = {}
    for r in rows:
        src = str(r.get("source") or "unknown").strip() or "unknown"
        gid = _group_id_for_row(r)
        split = group_split.get(gid, "test")
        src_to_groups_all.setdefault(src, set()).add(gid)
        src_to_groups_by_split.setdefault(src, {}).setdefault(split, set()).add(gid)
        src_to_samples_by_split.setdefault(src, {}).setdefault(split, 0)
        src_to_samples_by_split[src][split] += 1

    per_sample_source: Dict[str, Dict[str, Any]] = {}
    for src in sorted(src_to_groups_all.keys()):
        per_sample_source[src] = {
            "groups": int(len(src_to_groups_all.get(src) or set())),
            "samples": int(sum((src_to_samples_by_split.get(src) or {}).values())),
            "train_groups": int(len((src_to_groups_by_split.get(src) or {}).get("train") or set())),
            "val_groups": int(len((src_to_groups_by_split.get(src) or {}).get("val") or set())),
            "test_groups": int(len((src_to_groups_by_split.get(src) or {}).get("test") or set())),
            "train_samples": int((src_to_samples_by_split.get(src) or {}).get("train") or 0),
            "val_samples": int((src_to_samples_by_split.get(src) or {}).get("val") or 0),
            "test_samples": int((src_to_samples_by_split.get(src) or {}).get("test") or 0),
        }

    manifest: Dict[str, Any] = {
        "samples_path": str(samples_path),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "test_frac": float(max(0.0, 1.0 - float(train_frac) - float(val_frac))),
        "n_groups": int(len(group_to_rows)),
        "n_samples": int(len(rows)),
        "splits": {
            "train": {"groups": _count_groups("train"), "samples": _count_samples("train"), "path": str(paths["train"])},
            "val": {"groups": _count_groups("val"), "samples": _count_samples("val"), "path": str(paths["val"])},
            "test": {"groups": _count_groups("test"), "samples": _count_samples("test"), "path": str(paths["test"])},
        },
        "per_source": per_source,
        "per_sample_source": per_sample_source,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    splits_payload = {
        "seed": int(seed),
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
    }
    (out_dir / "splits.json").write_text(json.dumps(splits_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Split a Studio eval set JSONL into train/val/test without group leakage.")
    ap.add_argument("--samples", default="data/eval_sets/studio_fixed_v1.jsonl")
    ap.add_argument("--out-dir", default="data/eval_sets/studio_fixed_v1_splits")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-stratify-by-source", action="store_true", help="Disable per-source stratification")
    args = ap.parse_args(argv)

    manifest = split_eval_set(
        samples_path=Path(str(args.samples)),
        out_dir=Path(str(args.out_dir)),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        stratify_by_source=not bool(args.no_stratify_by_source),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
