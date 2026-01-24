from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _find_json_array(raw: str) -> str:
    # modal run output often includes logs before the JSON; find the first line that starts a JSON list.
    start = None
    for line in raw.splitlines(True):
        if line.lstrip().startswith("["):
            start = raw.find(line)
            break
    if start is None:
        raise ValueError("Could not find JSON array start '[' in file")
    s = raw[start:]
    end = s.rfind("]")
    if end < 0:
        raise ValueError("Could not find JSON array end ']' in file")
    return s[: end + 1]


def load_rows(path: Path) -> List[dict]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    obj = json.loads(_find_json_array(raw))
    if not isinstance(obj, list):
        raise ValueError("Expected top-level JSON list")
    return [x for x in obj if isinstance(x, dict)]


def _get_path(obj: Any, dotted: str) -> Optional[float]:
    cur: Any = obj
    for part in (dotted or "").split("."):
        key = part.strip()
        if not key:
            continue
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def compare_rows(
    a_rows: Iterable[dict],
    b_rows: Iterable[dict],
    *,
    metric: str,
) -> List[Tuple[float, float, float, str, str]]:
    a = {str(r.get("url") or ""): r for r in a_rows}
    b = {str(r.get("url") or ""): r for r in b_rows}
    out: List[Tuple[float, float, float, str, str]] = []
    for url in sorted(set(a) & set(b)):
        va = _get_path(a[url], metric)
        vb = _get_path(b[url], metric)
        if va is None or vb is None:
            continue
        title = str((b[url].get("title") or a[url].get("title") or "")).strip()
        out.append((float(vb - va), float(va), float(vb), url, title))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two modal-score-urls JSON outputs (e.g. v4 vs v5).")
    ap.add_argument("--a", required=True, help="First JSON file (baseline)")
    ap.add_argument("--b", required=True, help="Second JSON file (candidate)")
    ap.add_argument(
        "--metric",
        default="trained_score.overall_0_100",
        help="Dotted metric path (default: trained_score.overall_0_100)",
    )
    ap.add_argument("--limit", type=int, default=5, help="How many top/bottom deltas to show")
    args = ap.parse_args(argv)

    a_rows = load_rows(Path(str(args.a)))
    b_rows = load_rows(Path(str(args.b)))
    rows = compare_rows(a_rows, b_rows, metric=str(args.metric))
    if not rows:
        raise SystemExit("No comparable rows (check URLs + metric path).")

    limit = max(1, int(args.limit))
    mean_delta = sum(d for d, *_ in rows) / float(len(rows))
    print(f"n={len(rows)} mean_delta={mean_delta:+.4f} metric={args.metric}")

    rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
    print("\nTOP")
    for d, va, vb, _, title in rows_sorted[:limit]:
        name = title or "(untitled)"
        print(f"{d:+7.2f}  {va:7.2f} -> {vb:7.2f}  {name}")

    print("\nBOTTOM")
    for d, va, vb, _, title in rows_sorted[-limit:]:
        name = title or "(untitled)"
        print(f"{d:+7.2f}  {va:7.2f} -> {vb:7.2f}  {name}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

