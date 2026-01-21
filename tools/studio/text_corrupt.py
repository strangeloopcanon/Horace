from __future__ import annotations

import random
import re
from typing import List


_PUNCT_RE = re.compile(r"[\"'“”‘’`]+|[.,;:!?]+")


def _split_paragraphs(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    paras = re.split(r"\n\s*\n+", t)
    return [p.strip() for p in paras if p.strip()]


def _split_sentences(paragraph: str) -> List[str]:
    p = (paragraph or "").strip()
    if not p:
        return []
    # Very lightweight sentence split: break on terminal punctuation followed by whitespace/newline.
    parts = re.split(r"(?<=[.!?])\s+", p)
    out = [s.strip() for s in parts if s.strip()]
    return out


def corrupt_shuffle_sentences(text: str, *, rng: random.Random) -> str:
    paras = _split_paragraphs(text)
    out_paras: List[str] = []
    for p in paras:
        sents = _split_sentences(p)
        if len(sents) < 2:
            out_paras.append(p)
            continue

        if len(sents) == 2:
            out_paras.append(f"{sents[1]} {sents[0]}")
            continue

        orig = list(sents)
        for _ in range(8):
            rng.shuffle(sents)
            if sents != orig:
                break
        out_paras.append(" ".join(sents))
    return "\n\n".join(out_paras).strip()


def corrupt_shuffle_paragraphs(text: str, *, rng: random.Random) -> str:
    paras = _split_paragraphs(text)
    if len(paras) < 3:
        return (text or "").strip()

    orig = list(paras)
    for _ in range(8):
        rng.shuffle(paras)
        if paras != orig:
            break
    return "\n\n".join(paras).strip()


def corrupt_shuffle_sentences_global(text: str, *, rng: random.Random) -> str:
    paras = _split_paragraphs(text)
    if not paras:
        return ""

    per_para: List[int] = []
    all_sents: List[str] = []
    for p in paras:
        sents = _split_sentences(p)
        if not sents:
            continue
        per_para.append(len(sents))
        all_sents.extend(sents)

    if len(all_sents) < 4:
        return corrupt_shuffle_sentences(text, rng=rng)

    orig = list(all_sents)
    for _ in range(8):
        rng.shuffle(all_sents)
        if all_sents != orig:
            break

    out_paras: List[str] = []
    i = 0
    for n in per_para:
        chunk = all_sents[i : i + int(n)]
        i += int(n)
        if chunk:
            out_paras.append(" ".join(chunk))
    if i < len(all_sents):
        out_paras.append(" ".join(all_sents[i:]))
    return "\n\n".join(out_paras).strip()


def corrupt_repeat_sentences(text: str, *, rng: random.Random, repeat_prob: float = 0.35) -> str:
    paras = _split_paragraphs(text)
    out_paras: List[str] = []
    for p in paras:
        sents = _split_sentences(p)
        if len(sents) < 2:
            out_paras.append(p)
            continue
        out: List[str] = []
        for s in sents:
            out.append(s)
            if rng.random() < float(repeat_prob):
                out.append(s)
        out_paras.append(" ".join(out))
    return "\n\n".join(out_paras).strip()


def corrupt_drop_punctuation(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = _PUNCT_RE.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


def corrupt_flatten_paragraphs(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return ""
    # Collapse paragraph breaks into spaces, keep single newlines.
    t = re.sub(r"\n\s*\n+", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def corrupt_text(text: str, *, rng: random.Random, kind: str) -> str:
    k = (kind or "").strip().lower()
    if k in ("shuffle", "shuffle_sentences", "shuffle_sents"):
        return corrupt_shuffle_sentences(text, rng=rng)
    if k in ("shuffle_paragraphs", "shuffle_paras", "shuffle_paragraph"):
        return corrupt_shuffle_paragraphs(text, rng=rng)
    if k in ("shuffle_sentences_global", "shuffle_sents_global", "shuffle_global", "shuffle_all_sentences"):
        return corrupt_shuffle_sentences_global(text, rng=rng)
    if k in ("repeat", "repeat_sentences", "repeat_sents"):
        return corrupt_repeat_sentences(text, rng=rng)
    if k in ("drop_punct", "drop_punctuation", "no_punct"):
        return corrupt_drop_punctuation(text)
    if k in ("flatten", "flatten_paras", "flatten_paragraphs"):
        return corrupt_flatten_paragraphs(text)
    raise ValueError(f"Unknown corruption kind: {kind}")
