from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional

from defusedxml import ElementTree as DefusedET

MAX_FEED_XML_CHARS = 2_000_000


def _strip_ns(tag: str) -> str:
    t = tag or ""
    if "}" in t:
        return t.split("}", 1)[1]
    return t


def _norm_text(s: Optional[str]) -> str:
    return str(s or "").strip()


@dataclass(frozen=True)
class FeedEntry:
    title: str
    url: str
    content_html: str
    published_unix: Optional[int]


_RFC822_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_rfc822_date(s: str) -> Optional[int]:
    # Example: "Sat, 17 Jan 2026 02:57:53 GMT"
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\b", s or "")
    if not m:
        return None
    try:
        day = int(m.group(1))
        mon = _RFC822_MONTHS.get(m.group(2).lower())
        year = int(m.group(3))
        hh = int(m.group(4))
        mm = int(m.group(5))
        ss = int(m.group(6))
        if mon is None:
            return None
        import datetime as _dt

        dt = _dt.datetime(year, mon, day, hh, mm, ss, tzinfo=_dt.timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _parse_iso8601_date(s: str) -> Optional[int]:
    # Example: "2026-01-19T17:39:34Z"
    t = (s or "").strip()
    if not t:
        return None
    try:
        import datetime as _dt

        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _parse_published_unix(s: str) -> Optional[int]:
    t = (s or "").strip()
    if not t:
        return None
    out = _parse_iso8601_date(t)
    if out is not None:
        return out
    out = _parse_rfc822_date(t)
    if out is not None:
        return out
    return None


def parse_feed(xml_text: str) -> List[FeedEntry]:
    """Parse an Atom or RSS feed into lightweight entries (best-effort).

    So what: feed payloads are untrusted network input; we cap payload size and use
    defused XML parsing to avoid parser abuse while keeping ingestion lightweight.
    """
    t = (xml_text or "").strip()
    if not t:
        return []
    if len(t) > int(MAX_FEED_XML_CHARS):
        return []

    try:
        root = DefusedET.fromstring(t)
    except Exception:
        return []

    root_tag = _strip_ns(root.tag).lower()
    entries: List[FeedEntry] = []
    now = int(time.time())

    if root_tag == "feed":  # Atom
        for ent in root.findall(".//{*}entry"):
            title = _norm_text(ent.findtext("{*}title"))
            url = ""
            for ln in ent.findall("{*}link"):
                rel = (ln.attrib.get("rel") or "").strip().lower()
                href = (ln.attrib.get("href") or "").strip()
                if not href:
                    continue
                if rel in ("", "alternate"):
                    url = href
                    break
                if not url:
                    url = href
            content_html = _norm_text(ent.findtext("{*}content")) or _norm_text(ent.findtext("{*}summary"))
            published = (
                _norm_text(ent.findtext("{*}published"))
                or _norm_text(ent.findtext("{*}updated"))
                or _norm_text(ent.findtext("{*}issued"))
            )
            pu = _parse_published_unix(published) or now
            if not (title or url or content_html):
                continue
            entries.append(FeedEntry(title=title, url=url, content_html=content_html, published_unix=pu))
        return entries

    # RSS 2.0 or similar: <rss><channel><item>...</item></channel></rss>
    if root_tag in ("rss", "rdf", "rdf:rdf"):
        for item in root.findall(".//item"):
            title = _norm_text(item.findtext("title"))
            url = _norm_text(item.findtext("link"))
            if not url:
                guid = _norm_text(item.findtext("guid"))
                if guid and guid.startswith("http"):
                    url = guid

            # Prefer content:encoded when present.
            content_html = ""
            for child in list(item):
                tag = _strip_ns(child.tag).lower()
                if tag in ("encoded", "content", "content:encoded"):
                    content_html = _norm_text(child.text)
                    if content_html:
                        break
            if not content_html:
                content_html = _norm_text(item.findtext("description")) or _norm_text(item.findtext("summary"))

            published = _norm_text(item.findtext("pubDate")) or _norm_text(item.findtext("published")) or _norm_text(
                item.findtext("updated")
            )
            pu = _parse_published_unix(published) or now

            if not (title or url or content_html):
                continue
            entries.append(FeedEntry(title=title, url=url, content_html=content_html, published_unix=pu))
        return entries

    return []
