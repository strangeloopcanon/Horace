from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from tools.studio.dataset_utils import http_get, http_get_text
from tools.studio.epub_extract import extract_epub_text


STANDARD_EBOOKS_BASE = "https://standardebooks.org"

# Some Standard Ebooks endpoints return 401 for non-browser UAs.
STANDARD_EBOOKS_UA = (
    "Mozilla/5.0 (compatible; HoraceStudio/0.1; +https://example.invalid) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)


@dataclass(frozen=True)
class StandardEbook:
    path: str
    url: str
    title: str
    author: str
    download_epub_url: str
    gutenberg_id: Optional[int]


def _abs_url(url_or_path: str) -> str:
    u = (url_or_path or "").strip()
    if not u:
        return ""
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if not u.startswith("/"):
        u = "/" + u
    return STANDARD_EBOOKS_BASE + u


def list_ebook_paths(*, max_pages: int = 50, start_page: int = 1, max_consecutive_failures: int = 3) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    start = max(1, int(start_page))
    pages = max(0, int(max_pages))
    if pages <= 0:
        return []

    consecutive_failures = 0
    max_fail = max(1, int(max_consecutive_failures))
    for page in range(start, start + pages):
        url = f"{STANDARD_EBOOKS_BASE}/ebooks" if page == 1 else f"{STANDARD_EBOOKS_BASE}/ebooks?page={page}"
        try:
            html = http_get_text(url, user_agent=STANDARD_EBOOKS_UA)
        except Exception:
            consecutive_failures += 1
            if consecutive_failures >= max_fail:
                break
            continue
        consecutive_failures = 0

        # The index pages are simple enough to parse with a conservative regex.
        links = re.findall(r'href="(/ebooks/[^"#?]+)"', html)
        added_any = False
        for p in links:
            path = str(p or "").strip()
            if not path or path == "/ebooks":
                continue
            if path.startswith("/ebooks?page="):
                continue
            # Ebook pages are /ebooks/<author>/<title> (or similar).
            if path.count("/") < 3:
                continue
            if path.endswith("/"):
                path = path[:-1]
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
            added_any = True

        if not added_any:
            break
    return out


def _extract_og_title(html: str) -> str:
    # Find the meta tag containing property="og:title", then extract content="...".
    m = re.search(r"<meta\b[^>]*\bproperty=\"og:title\"[^>]*>", html, flags=re.I)
    if not m:
        m = re.search(r"<meta\b[^>]*\bcontent=\"[^\"]*\"[^>]*\bproperty=\"og:title\"[^>]*>", html, flags=re.I)
    if not m:
        return ""
    tag = m.group(0)
    mc = re.search(r'\bcontent=\"([^\"]+)\"', tag, flags=re.I)
    return (mc.group(1) if mc else "").strip()


def _parse_title_author(og_title: str) -> tuple[str, str]:
    t = (og_title or "").strip()
    if not t:
        return "", ""

    # "Pride and Prejudice, by Jane Austen - Free ebook download"
    t = re.sub(r"\s+-\s+Free ebook download\s*$", "", t, flags=re.I)
    if ", by " in t:
        title, author = t.split(", by ", 1)
        return title.strip(), author.strip()
    return t.strip(), ""


def _pick_download_epub_url(html: str) -> str:
    cands = re.findall(r'href="(/ebooks/[^"#?]+/downloads/[^"#?]+\.epub)"', html)
    for c in cands:
        low = c.lower()
        if ".kepub." in low:
            continue
        if low.endswith("_advanced.epub"):
            continue
        return _abs_url(c)
    return ""


def _ensure_download_file_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "source=download" in u:
        return u
    return u + ("&source=download" if "?" in u else "?source=download")


def _extract_gutenberg_id(html: str) -> Optional[int]:
    m = re.search(r"https?://www\\.gutenberg\\.org/ebooks/(\\d+)", html)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def fetch_ebook(path: str) -> StandardEbook:
    p = str(path or "").strip()
    if not p:
        raise ValueError("Missing Standard Ebooks path")
    if not p.startswith("/"):
        p = "/" + p
    url = _abs_url(p)
    html = http_get_text(url, user_agent=STANDARD_EBOOKS_UA)

    og_title = _extract_og_title(html)
    title, author = _parse_title_author(og_title)
    dl = _ensure_download_file_url(_pick_download_epub_url(html))
    if not title:
        raise RuntimeError(f"Failed to parse title from {url}")
    if not dl:
        raise RuntimeError(f"Failed to locate download .epub for {url}")

    return StandardEbook(
        path=p,
        url=url,
        title=title,
        author=author,
        download_epub_url=dl,
        gutenberg_id=_extract_gutenberg_id(html),
    )


def download_epub_bytes(ebook: StandardEbook, *, timeout_s: float = 60.0) -> bytes:
    raw = http_get(ebook.download_epub_url, timeout_s=float(timeout_s), user_agent=STANDARD_EBOOKS_UA)
    if raw[:4] == b"PK\x03\x04":
        return raw

    # Some endpoints return an XHTML landing page with a meta refresh to the actual file.
    if raw[:1] == b"<":
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:
            html = ""
        m = re.search(r'http-equiv="refresh"\s+content="[^\"]*url=([^\"]+)"', html, flags=re.I)
        if m:
            target = _abs_url(m.group(1))
            raw2 = http_get(target, timeout_s=float(timeout_s), user_agent=STANDARD_EBOOKS_UA)
            if raw2[:4] == b"PK\x03\x04":
                return raw2

    return raw


def extract_text_from_epub(epub_bytes: bytes) -> str:
    return extract_epub_text(epub_bytes)


def fetch_ebook_text(ebook: StandardEbook) -> str:
    return extract_text_from_epub(download_epub_bytes(ebook))


def fetch_corpus_texts(
    paths: Sequence[str],
    *,
    max_books: Optional[int] = None,
) -> List[tuple[StandardEbook, str]]:
    out: List[tuple[StandardEbook, str]] = []
    for p in paths[: (int(max_books) if max_books is not None else None)]:
        eb = fetch_ebook(str(p))
        out.append((eb, fetch_ebook_text(eb)))
    return out
