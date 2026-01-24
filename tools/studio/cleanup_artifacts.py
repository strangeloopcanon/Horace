from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Candidate:
    path: Path
    size_bytes: int


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists():
            return p
    return Path.cwd()


def _is_tracked(path: Path) -> bool:
    root = _repo_root()
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        rel = path
    try:
        cp = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(rel)],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return cp.returncode == 0
    except Exception:
        return False


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            yield p
            continue
        if p.is_dir():
            yield from (x for x in p.rglob("*") if x.is_file())


def _candidates(*, kind: str) -> List[Candidate]:
    root = _repo_root()
    targets: List[Path] = []
    if kind in {"all", "modal"}:
        targets.append(root / "reports" / "modal")
    if kind in {"all", "urls"}:
        targets.append(root / "data" / "benchmarks" / "urls")

    out: List[Candidate] = []
    for p in _iter_files(targets):
        if _is_tracked(p):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(Candidate(path=p, size_bytes=int(st.st_size)))
    return sorted(out, key=lambda c: (c.path.as_posix()))


def _should_keep(path: Path, *, keep_substrings: List[str], keep_snapshots: bool) -> bool:
    s = path.name.lower()
    if keep_snapshots and "snapshot" in s:
        return True
    full = path.as_posix()
    return any(k in full for k in keep_substrings if k)


def _fmt_bytes(n: int) -> str:
    x = float(max(0, int(n)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or unit == "TB":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024.0
    return f"{x:.1f}TB"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="List (and optionally delete) local run artifacts (safe: skips git-tracked files).")
    ap.add_argument("--kind", choices=["all", "modal", "urls"], default="all")
    ap.add_argument("--apply", action="store_true", help="Actually delete files (default: dry-run).")
    ap.add_argument("--keep", action="append", default=[], help="Keep files whose path contains this substring (repeatable).")
    ap.add_argument(
        "--keep-snapshots",
        action="store_true",
        default=True,
        help="Keep files with 'snapshot' in the name (default: on).",
    )
    ap.add_argument(
        "--delete-snapshots",
        action="store_true",
        help="Allow deleting files with 'snapshot' in the name.",
    )
    args = ap.parse_args(argv)

    keep_snapshots = bool(args.keep_snapshots) and not bool(args.delete_snapshots)
    keep_substrings = [str(x) for x in (args.keep or []) if str(x)]
    cands = _candidates(kind=str(args.kind))
    to_delete = [c for c in cands if not _should_keep(c.path, keep_substrings=keep_substrings, keep_snapshots=keep_snapshots)]
    total = sum(c.size_bytes for c in to_delete)

    print(f"kind={args.kind} candidates={len(cands)} delete={len(to_delete)} total={_fmt_bytes(total)} apply={bool(args.apply)}")
    for c in to_delete:
        print(f"- {c.path.as_posix()} ({_fmt_bytes(c.size_bytes)})")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")
        return 0

    deleted = 0
    for c in to_delete:
        try:
            c.path.unlink()
            deleted += 1
        except OSError:
            continue
    print(f"deleted={deleted}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

