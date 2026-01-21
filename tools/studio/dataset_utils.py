from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_UA = "HoraceStudioDatasets/0.1 (+https://example.invalid)"


@dataclass(frozen=True)
class FixedSample:
    sample_id: str
    group_id: str
    source: str
    title: str
    url: str
    text: str
    fetched_at_unix: int
    meta: Dict[str, Any]


def group_id(source: str, *, stable: str) -> str:
    s = str(stable or "").strip()
    if not s:
        return f"{source}:unknown"
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return f"{source}:{h.hexdigest()[:12]}"


def http_get(url: str, *, timeout_s: float = 20.0, user_agent: str = DEFAULT_UA) -> bytes:
    retries = 0
    try:
        retries = int(os.environ.get("HORACE_HTTP_RETRIES") or 0)
    except Exception:
        retries = 0
    base_sleep_s = 0.8
    try:
        base_sleep_s = float(os.environ.get("HORACE_HTTP_RETRY_BASE_SLEEP_S") or base_sleep_s)
    except Exception:
        base_sleep_s = 0.8
    max_sleep_s = 30.0
    try:
        max_sleep_s = float(os.environ.get("HORACE_HTTP_RETRY_MAX_SLEEP_S") or max_sleep_s)
    except Exception:
        max_sleep_s = 30.0

    def _sleep(attempt: int, *, retry_after_s: float = 0.0) -> None:
        ra = float(retry_after_s) if retry_after_s and retry_after_s > 0 else 0.0
        if ra > 0:
            time.sleep(min(float(max_sleep_s), ra))
            return
        jitter = random.random() * 0.25 * float(base_sleep_s)
        backoff = float(base_sleep_s) * (2 ** max(0, int(attempt)))
        time.sleep(min(float(max_sleep_s), backoff + jitter))

    if str(os.environ.get("HORACE_HTTP_NO_CACHE") or "").strip().lower() not in ("1", "true", "yes"):
        cache_dir = Path(str(os.environ.get("HORACE_HTTP_CACHE_DIR") or "data/cache/http"))
        key = stable_sha1_hex([user_agent, url])
        cache_path = cache_dir / f"{key}.bin"
        try:
            if cache_path.exists():
                return cache_path.read_bytes()
        except Exception:
            pass

        attempt = 0
        while True:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": user_agent})
                with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as e:
                if attempt < int(retries) and int(getattr(e, "code", 0) or 0) in (429, 500, 502, 503, 504):
                    retry_after_s = 0.0
                    try:
                        ra = str(e.headers.get("Retry-After") or "").strip()
                        if ra:
                            retry_after_s = float(ra)
                    except Exception:
                        retry_after_s = 0.0
                    _sleep(attempt, retry_after_s=retry_after_s)
                    attempt += 1
                    continue
                raise
            except Exception:
                if attempt < int(retries):
                    _sleep(attempt)
                    attempt += 1
                    continue
                raise
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw)
        except Exception:
            pass
        return raw

    attempt = 0
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if attempt < int(retries) and int(getattr(e, "code", 0) or 0) in (429, 500, 502, 503, 504):
                retry_after_s = 0.0
                try:
                    ra = str(e.headers.get("Retry-After") or "").strip()
                    if ra:
                        retry_after_s = float(ra)
                except Exception:
                    retry_after_s = 0.0
                _sleep(attempt, retry_after_s=retry_after_s)
                attempt += 1
                continue
            raise
        except Exception:
            if attempt < int(retries):
                _sleep(attempt)
                attempt += 1
                continue
            raise


def http_get_text(url: str, *, timeout_s: float = 20.0, user_agent: str = DEFAULT_UA) -> str:
    raw = http_get(url, timeout_s=timeout_s, user_agent=user_agent)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def http_get_json(url: str, *, timeout_s: float = 20.0, user_agent: str = DEFAULT_UA) -> dict:
    return json.loads(http_get_text(url, timeout_s=timeout_s, user_agent=user_agent))


def clean_text(s: str) -> str:
    t = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


def sample_window(text: str, *, rng: random.Random, max_chars: int, min_chars: int = 0) -> str:
    t = clean_text(text)
    if max_chars <= 0 or len(t) <= max_chars:
        return t

    window_chars = int(max_chars)
    if int(min_chars) > 0 and int(min_chars) < int(max_chars):
        window_chars = rng.randint(int(min_chars), int(max_chars))

    # Prefer a window that starts on a paragraph boundary for readability.
    candidates = [m.start() for m in re.finditer(r"\n\s*\n", t)]
    if not candidates:
        start = rng.randint(0, max(0, len(t) - window_chars))
        return t[start : start + window_chars].strip()

    start = rng.choice(candidates)
    start = min(start, max(0, len(t) - window_chars))
    return t[start : start + window_chars].strip()


def make_sample_id(source: str, title: str, url: str, text: str) -> str:
    h = hashlib.sha1()
    h.update((source or "").encode("utf-8"))
    h.update(b"\0")
    h.update((title or "").encode("utf-8"))
    h.update(b"\0")
    h.update((url or "").encode("utf-8"))
    h.update(b"\0")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()[:12]


def strip_gutenberg_boilerplate(text: str) -> str:
    t = text or ""
    start_idx = None
    end_idx = None

    start_match = re.search(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", t, flags=re.I | re.S)
    if start_match:
        start_idx = start_match.end()
    end_match = re.search(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", t, flags=re.I | re.S)
    if end_match:
        end_idx = end_match.start()

    if start_idx is not None and end_idx is not None and end_idx > start_idx:
        t = t[start_idx:end_idx]
    elif start_idx is not None:
        t = t[start_idx:]
    elif end_idx is not None:
        t = t[:end_idx]
    return clean_text(t)


class _TextHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def html_to_text(raw_html: str) -> str:
    """Best-effort HTML -> readable plain text.

    Note: intentionally simple. For production-grade extraction we'd likely use a
    readability-style algorithm, but we avoid adding deps here.
    """
    t = (raw_html or "").replace("\r\n", "\n").replace("\r", "\n")
    # Drop scripts/styles quickly
    t = re.sub(r"<script\b.*?</script>", " ", t, flags=re.I | re.S)
    t = re.sub(r"<style\b.*?</style>", " ", t, flags=re.I | re.S)
    # Convert common block separators to newlines
    t = re.sub(r"</(?:p|div|section|article|h[1-6]|li|br|tr)>", "\n", t, flags=re.I)
    # Parse remaining tags
    parser = _TextHTMLParser()
    try:
        parser.feed(t)
        out = parser.get_text()
    except Exception:
        out = re.sub(r"<[^>]+>", " ", t)
    out = html.unescape(out)
    out = clean_text(out)
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict]:
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


def stable_sha1_hex(parts: List[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def safe_slug(text: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return t or "item"
