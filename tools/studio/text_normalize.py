from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, Tuple


_HYPHEN_CHARS = "-‐‑"  # hyphen, hyphen (U+2010), non-breaking hyphen (U+2011)
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060"  # ZWSP/ZWNJ/ZWJ/WORD JOINER


@dataclass(frozen=True)
class NormalizationMeta:
    enabled: bool
    applied: bool
    doc_type: str
    original_chars: int
    normalized_chars: int
    original_newlines: int
    normalized_newlines: int
    replaced_single_newlines: int
    joined_hyphen_breaks: int
    preserved_newlines: bool
    removed_zero_width_chars: int
    removed_soft_hyphens: int
    stripped_gutenberg_boilerplate: bool


def _strip_gutenberg_boilerplate_if_present(text: str) -> Tuple[str, bool]:
    t = text or ""
    low = t.lower()
    if "project gutenberg" not in low:
        return t, False
    # Only strip when we see canonical markers or the common eBook header.
    if ("*** start of" not in low) and ("*** end of" not in low) and ("project gutenberg ebook" not in low):
        return t, False

    start_idx = None
    end_idx = None
    start_match = re.search(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", t, flags=re.I | re.S)
    if start_match:
        start_idx = start_match.end()
    end_match = re.search(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", t, flags=re.I | re.S)
    if end_match:
        end_idx = end_match.start()

    if start_idx is not None and end_idx is not None and end_idx > start_idx:
        return t[start_idx:end_idx].strip(), True
    if start_idx is not None:
        return t[start_idx:].strip(), True
    if end_idx is not None:
        return t[:end_idx].strip(), True
    return t, False


def normalize_for_studio(text: str, *, doc_type: str, enabled: bool = True) -> Tuple[str, Dict]:
    """Normalize pasted text so scoring reflects writing, not formatting artifacts.

    Primary target: hard-wrapped plain text (single newline every ~70 chars), common in
    Project Gutenberg and RFC plaintext.

    Rules:
    - Always normalize CRLF -> LF.
    - For non-poetry prose: dewrap single newlines into spaces, preserving paragraph breaks.
      Exception: keep newlines when the text looks like code, lists, or verse, to avoid
      destroying meaningful structure.
    - Join common hyphenated line breaks when dewrapping: "exam-\\nple" -> "example".
    """
    original = text or ""
    t = original.replace("\r\n", "\n").replace("\r", "\n")
    dt = (doc_type or "").strip().lower()

    if not enabled:
        meta = NormalizationMeta(
            enabled=False,
            applied=False,
            doc_type=dt,
            original_chars=len(original),
            normalized_chars=len(t),
            original_newlines=original.count("\n"),
            normalized_newlines=t.count("\n"),
            replaced_single_newlines=0,
            joined_hyphen_breaks=0,
            preserved_newlines=False,
            removed_zero_width_chars=0,
            removed_soft_hyphens=0,
            stripped_gutenberg_boilerplate=False,
        )
        return t, asdict(meta)

    # Unicode cleanup (safe for all types)
    t = t.lstrip("\ufeff")
    removed_zero_width_chars = len(re.findall(rf"[{re.escape(_ZERO_WIDTH)}]", t))
    if removed_zero_width_chars:
        t = re.sub(rf"[{re.escape(_ZERO_WIDTH)}]", "", t)
    removed_soft_hyphens = t.count("\u00ad")
    if removed_soft_hyphens:
        t = t.replace("\u00ad", "")
    try:
        t = unicodedata.normalize("NFKC", t)
    except Exception:
        pass

    # Light whitespace cleanup (safe for all types)
    t = t.replace("\u00a0", " ")
    t = t.replace("\u202f", " ").replace("\u2009", " ")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)

    joined_hyphen_breaks = 0
    replaced_single_newlines = 0
    stripped_gutenberg_boilerplate = False
    preserved_newlines = False

    if dt != "poem":
        t, stripped_gutenberg_boilerplate = _strip_gutenberg_boilerplate_if_present(t)

        def _preserve_newlines(s: str) -> bool:
            if "```" in s or "~~~" in s:
                return True
            lines = (s or "").split("\n")
            nonempty = [ln for ln in lines if ln.strip()]
            if len(nonempty) < 6:
                return False

            bullet_pat = re.compile(r"^\s*(?:[-*+]|•|\d+[.)])\s+\S")
            bullets = sum(1 for ln in nonempty if bullet_pat.match(ln))
            indented = sum(1 for ln in nonempty if re.match(r"^(?:\t+|\s{4,})\S", ln))
            heading = sum(1 for ln in nonempty if re.match(r"^\s*#{1,6}\s+\S", ln))

            alpha = sum(1 for ch in s if ch.isalpha())
            symbols = sum(1 for ch in s if ch in "{}[]();<>/\\=+-*#@")
            symbol_ratio = float(symbols) / float(max(1, alpha + symbols))

            bullet_frac = float(bullets) / float(len(nonempty))
            indented_frac = float(indented) / float(len(nonempty))
            heading_frac = float(heading) / float(len(nonempty))

            lens = sorted(len(ln.strip()) for ln in nonempty)
            med = lens[len(lens) // 2] if lens else 0
            blank_frac = float(sum(1 for ln in lines if not ln.strip())) / float(max(1, len(lines)))

            codeish = (symbol_ratio >= 0.28) or (indented_frac >= 0.12)
            bulletish = bullet_frac >= 0.25
            verseish = (len(nonempty) >= 10) and (med <= 44) and (blank_frac <= 0.25) and (symbol_ratio <= 0.20)

            return bool(codeish or bulletish or verseish or (heading_frac >= 0.35))

        preserved_newlines = _preserve_newlines(t)
        if not preserved_newlines:
            # Join hyphenated line breaks: "exam-\nple" -> "example"
            hy_pat = re.compile(rf"([A-Za-z0-9])[{re.escape(_HYPHEN_CHARS)}]\n([A-Za-z0-9])")
            joined_hyphen_breaks = len(hy_pat.findall(t))
            if joined_hyphen_breaks:
                t = hy_pat.sub(r"\1\2", t)

            # Preserve paragraph boundaries (blank lines), but remove single newlines.
            single_nl_pat = re.compile(r"(?<!\n)\n(?!\n)")
            replaced_single_newlines = len(single_nl_pat.findall(t))
            if replaced_single_newlines:
                t = single_nl_pat.sub(" ", t)

            t = re.sub(r"[ \t]+", " ", t)
            t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
            t = re.sub(r"\n{3,}", "\n\n", t)
        else:
            t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
            t = re.sub(r"\n{4,}", "\n\n\n", t)

    t = t.strip()

    meta = NormalizationMeta(
        enabled=True,
        applied=(t != original),
        doc_type=dt,
        original_chars=len(original),
        normalized_chars=len(t),
        original_newlines=original.count("\n"),
        normalized_newlines=t.count("\n"),
        replaced_single_newlines=int(replaced_single_newlines),
        joined_hyphen_breaks=int(joined_hyphen_breaks),
        preserved_newlines=bool(preserved_newlines),
        removed_zero_width_chars=int(removed_zero_width_chars),
        removed_soft_hyphens=int(removed_soft_hyphens),
        stripped_gutenberg_boilerplate=bool(stripped_gutenberg_boilerplate),
    )
    return t, asdict(meta)
