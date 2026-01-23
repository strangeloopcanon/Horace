from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_hex(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class UrlSnapshotRow:
    url: str
    fetched_at: str
    fetched_at_unix: int
    title: Optional[str]
    extractor: str
    html_sha1: Optional[str]
    text_sha1: Optional[str]
    chars: int
    text: Optional[str]
    error: Optional[str] = None
    meta: Dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        if obj.get("meta") is None:
            obj["meta"] = {}
        return obj


def iter_urls_from_file(path: Path) -> List[str]:
    urls: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls


def _iter_json_rows(path: Path) -> Iterator[dict]:
    raw = path.read_text(encoding="utf-8")
    s = raw.lstrip()
    if not s:
        return iter(())
    if s.startswith("["):
        try:
            obj = json.loads(raw)
        except Exception:
            return iter(())
        if isinstance(obj, list):
            return (x for x in obj if isinstance(x, dict))
        return iter(())

    def _gen() -> Iterator[dict]:
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

    return _gen()


def iter_snapshot_rows(paths: Sequence[Path]) -> Iterable[dict]:
    for p in paths:
        for row in _iter_json_rows(Path(p)):
            yield row


def load_snapshot_text_by_url(paths: Sequence[Path]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in iter_snapshot_rows(paths):
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        out[url] = row
    return out


def write_snapshot_jsonl(rows: Iterable[Dict[str, Any]], *, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def snapshot_run_meta(*, urls: Sequence[str]) -> Dict[str, Any]:
    return {"fetched_at": now_iso_utc(), "fetched_at_unix": int(time.time()), "n_urls": int(len(urls))}

