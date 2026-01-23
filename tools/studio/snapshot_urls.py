from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.studio.url_extract import extract_url
from tools.studio.url_snapshot import UrlSnapshotRow, now_iso_utc, sha1_hex, snapshot_run_meta, write_snapshot_jsonl


def _git_rev() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return str(out)[:12]
    except Exception:
        return ""


def snapshot_urls(*, urls: List[str], out_path: Path, sleep_s: float, timeout_s: float) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    rev = _git_rev()
    run_meta = snapshot_run_meta(urls=urls)
    run_meta["repo_rev"] = rev
    run_meta["extractor_version"] = "v1"

    for i, url in enumerate(urls, 1):
        url = str(url).strip()
        if not url:
            continue
        fetched_at = now_iso_utc()
        fetched_at_unix = int(time.time())
        stage = "fetch_extract"
        try:
            res = extract_url(url, timeout_s=float(timeout_s))
            txt = res.text
            rows.append(
                UrlSnapshotRow(
                    url=str(url),
                    fetched_at=str(fetched_at),
                    fetched_at_unix=int(fetched_at_unix),
                    title=res.title,
                    extractor=res.extractor,
                    html_sha1=str(res.html_sha1),
                    text_sha1=sha1_hex(txt) if txt else None,
                    chars=int(len(txt or "")),
                    text=txt,
                    error=None if txt else "extract_failed",
                    meta={"repo_rev": rev},
                ).to_dict()
            )
        except Exception as e:
            rows.append(
                UrlSnapshotRow(
                    url=str(url),
                    fetched_at=str(fetched_at),
                    fetched_at_unix=int(fetched_at_unix),
                    title=None,
                    extractor="error",
                    html_sha1=None,
                    text_sha1=None,
                    chars=0,
                    text=None,
                    error=f"{type(e).__name__}: {e} (stage={stage})",
                    meta={"repo_rev": rev},
                ).to_dict()
            )

        if float(sleep_s) > 0 and i < len(urls):
            time.sleep(float(sleep_s))

    # Prepend a small meta row for convenience.
    out_rows: List[Dict[str, Any]] = [{"kind": "meta", **run_meta}] + rows
    write_snapshot_jsonl(out_rows, out_path=Path(out_path))
    return {"out_path": str(out_path), "n_rows": len(out_rows), "repo_rev": rev}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch URLs and snapshot extracted plaintext for reproducible benchmarks.")
    ap.add_argument("--urls", default="", help="Comma-separated list of URLs.")
    ap.add_argument("--urls-file", default="", help="File with one URL per line (comments with #).")
    ap.add_argument("--out", required=True, help="Output JSONL path, e.g. data/benchmarks/urls/snapshot_v1.jsonl")
    ap.add_argument("--sleep-s", type=float, default=0.0, help="Sleep between requests (politeness).")
    ap.add_argument("--timeout-s", type=float, default=45.0, help="Fetch timeout per URL.")
    args = ap.parse_args()

    urls: List[str] = []
    if str(args.urls).strip():
        urls.extend([u.strip() for u in str(args.urls).split(",") if u.strip()])
    if str(args.urls_file).strip():
        p = Path(str(args.urls_file))
        if p.exists():
            urls.extend([u.strip() for u in p.read_text(encoding="utf-8").splitlines() if u.strip() and not u.strip().startswith("#")])
    # Deduplicate, preserve order.
    seen: set[str] = set()
    deduped: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)

    if not deduped:
        raise SystemExit("No URLs provided. Use --urls or --urls-file.")

    info = snapshot_urls(
        urls=deduped,
        out_path=Path(str(args.out)),
        sleep_s=float(args.sleep_s),
        timeout_s=float(args.timeout_s),
    )
    print(info["out_path"])


if __name__ == "__main__":  # pragma: no cover
    main()

