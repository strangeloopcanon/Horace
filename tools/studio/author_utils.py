from __future__ import annotations

import re
from typing import Iterable, Set


def normalize_author_name(name: str) -> str:
    """Normalize an author string to a stable, matchable key.

    Handles common Gutenberg header formats like "Last, First, 1775-1817".
    """
    t = str(name or "").strip()
    if not t:
        return ""

    # Standard Ebooks often includes contributors in the author field, e.g.
    # "Aeschylus. Translated by Gilbert Murray". Keep the primary author.
    t = re.sub(
        r"\.\s*(?:translated|edited|with|introduction|introduced|preface|illustrated|compiled|selected|arranged|retold)\b.*$",
        "",
        t,
        flags=re.I,
    ).strip()

    # Drop bracketed / parenthetical suffixes and trailing dates.
    t = re.sub(r"\s*\(.*?\)\s*", " ", t)
    t = re.sub(r",?\s*\d{4}\s*[-–]\s*\d{0,4}\s*$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()

    # Reorder "Last, First" -> "First Last"
    if "," in t:
        parts = [p.strip() for p in t.split(",") if p.strip()]
        if len(parts) >= 2:
            last = parts[0]
            first = parts[1]
            rest = " ".join(parts[2:]).strip()
            t = f"{first} {last}".strip()
            if rest:
                t = f"{t} {rest}".strip()

    # Light cleanup
    t = t.replace(".", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def load_author_list(lines: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for ln in lines:
        s = str(ln or "").strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        out.add(normalize_author_name(s))
    return out
