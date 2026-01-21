from __future__ import annotations

import io
import posixpath
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tools.studio.dataset_utils import clean_text, html_to_text


_SKIP_BASENAMES = {
    # Common front/back matter in Standard Ebooks + many EPUBs.
    "titlepage.xhtml",
    "halftitle.xhtml",
    "imprint.xhtml",
    "colophon.xhtml",
    "uncopyright.xhtml",
    "toc.xhtml",
    "nav.xhtml",
    "loi.xhtml",
    "loii.xhtml",
    "endnotes.xhtml",
}


def _read_zip_text(z: zipfile.ZipFile, path: str) -> str:
    raw = z.read(path)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _find_opf_path(z: zipfile.ZipFile) -> str:
    try:
        container = _read_zip_text(z, "META-INF/container.xml")
        root = ET.fromstring(container)
        rf = root.find(".//{*}rootfile")
        if rf is not None:
            full = (rf.attrib.get("full-path") or rf.attrib.get("full_path") or "").strip()
            if full:
                return full
    except Exception:
        pass

    # Fallback: first .opf in the archive.
    for name in z.namelist():
        if name.lower().endswith(".opf"):
            return name
    raise RuntimeError("Could not locate OPF package document in EPUB")


def _opf_spine_paths(z: zipfile.ZipFile, opf_path: str) -> List[str]:
    opf_raw = _read_zip_text(z, opf_path)
    root = ET.fromstring(opf_raw)
    base_dir = posixpath.dirname(opf_path)

    manifest: Dict[str, Tuple[str, str]] = {}
    for item in root.findall(".//{*}manifest/{*}item"):
        iid = (item.attrib.get("id") or "").strip()
        href = (item.attrib.get("href") or "").strip()
        media = (item.attrib.get("media-type") or item.attrib.get("mediaType") or "").strip()
        if iid and href:
            manifest[iid] = (href, media)

    spine_hrefs: List[str] = []
    for itemref in root.findall(".//{*}spine/{*}itemref"):
        idref = (itemref.attrib.get("idref") or "").strip()
        if not idref or idref not in manifest:
            continue
        href, media = manifest[idref]
        # Prefer XHTML/HTML.
        if media and ("html" not in media.lower()) and ("xhtml" not in media.lower()):
            continue
        spine_hrefs.append(href)

    out: List[str] = []
    for href in spine_hrefs:
        joined = posixpath.normpath(posixpath.join(base_dir, href))
        out.append(joined)
    return out


def _fallback_html_paths(z: zipfile.ZipFile) -> List[str]:
    names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
    # Prefer primary text folders when present.
    prefer = [n for n in names if "/text/" in n.lower()]
    (prefer or names).sort()
    return prefer or names


def extract_epub_text(
    epub_bytes: bytes,
    *,
    skip_basenames: Optional[Sequence[str]] = None,
) -> str:
    """Extract readable plain text from an EPUB byte string.

    So what: lets us ingest curated literary sources (e.g. Standard Ebooks) without
    extra parsing dependencies.
    """
    if not epub_bytes:
        return ""

    skip = {str(x).strip().lower() for x in (skip_basenames or []) if str(x).strip()}
    skip = skip or set(_SKIP_BASENAMES)

    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as z:
        try:
            opf_path = _find_opf_path(z)
            paths = _opf_spine_paths(z, opf_path)
        except Exception:
            paths = []

        if not paths:
            paths = _fallback_html_paths(z)

        parts: List[str] = []
        seen: set[str] = set()
        for p in paths:
            if not p or p in seen:
                continue
            seen.add(p)
            base = posixpath.basename(p).lower()
            if base in skip:
                continue
            try:
                raw_html = _read_zip_text(z, p)
            except Exception:
                continue
            text = html_to_text(raw_html)
            if text.strip():
                parts.append(text)

    return clean_text("\n\n".join(parts))

