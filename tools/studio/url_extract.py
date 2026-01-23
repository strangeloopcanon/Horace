from __future__ import annotations

import hashlib
import html as _html
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ExtractionResult:
    url: str
    title: Optional[str]
    text: Optional[str]
    extractor: str
    html_sha1: str


def _sha1_hex(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="replace")).hexdigest()


def fetch_html(url: str, *, timeout_s: float = 45.0) -> str:
    req = Request(str(url), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=float(timeout_s)) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_title(html: str) -> Optional[str]:
    m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', html, flags=re.I)
    if m:
        return _html.unescape(m.group(1)).strip()
    m = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
    if m:
        t = re.sub(r"\s+", " ", m.group(1))
        return _html.unescape(t).strip()
    return None


def _extract_substack_body(html: str) -> Optional[str]:
    # Preferred: Substack renders post content in a `div.body.markup`.
    from html.parser import HTMLParser

    class _BodyParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.in_body = False
            self.depth = 0
            self.parts: List[str] = []
            self._pending_space = False
            self._block_stack: List[str] = []

        def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
            attr = {k.lower(): (v or "") for k, v in attrs}
            if tag.lower() == "div":
                cls = attr.get("class", "")
                if not self.in_body and "body" in cls and "markup" in cls:
                    self.in_body = True
                    self.depth = 1
                    return
                if self.in_body:
                    self.depth += 1
            if self.in_body:
                if tag.lower() in ("p", "li", "blockquote", "h1", "h2", "h3"):
                    self._block_stack.append(tag.lower())

        def handle_endtag(self, tag: str) -> None:
            if self.in_body:
                if tag.lower() == "div":
                    self.depth -= 1
                    if self.depth <= 0:
                        self.in_body = False
                        return
                if self._block_stack and self._block_stack[-1] == tag.lower():
                    self._block_stack.pop()
                    self.parts.append("\n\n")

        def handle_data(self, data: str) -> None:
            if not self.in_body:
                return
            s = _html.unescape(data)
            if not s:
                return
            if s.isspace():
                self._pending_space = True
                return
            if self._pending_space:
                self.parts.append(" ")
                self._pending_space = False
            self.parts.append(s)

    p = _BodyParser()
    p.feed(html)
    out = "".join(p.parts)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = out.strip()
    return out or None


def _extract_div_by_class(html: str, *, class_tokens: Tuple[str, ...]) -> Optional[str]:
    """Extract plaintext from the first <div> whose class contains a target token.

    This is a lightweight fallback for WordPress-style sites (e.g. Slate Star Codex uses
    `div.pjgm-postcontent`). We avoid adding third-party HTML parsing deps.
    """
    from html.parser import HTMLParser

    targets = tuple(str(t).strip().lower() for t in (class_tokens or ()) if str(t).strip())
    if not targets:
        return None

    class _DivParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.capture = False
            self.depth = 0
            self.ignore_depth = 0
            self.parts: List[str] = []
            self._pending_space = False
            self._block_stack: List[str] = []

        def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
            t = tag.lower()
            attr = {k.lower(): (v or "") for k, v in attrs}
            if self.capture and t in ("script", "style"):
                self.ignore_depth += 1
                return
            if t == "div":
                cls = (attr.get("class", "") or "").lower()
                if not self.capture and any(tok in cls for tok in targets):
                    self.capture = True
                    self.depth = 1
                    return
                if self.capture:
                    self.depth += 1
            if not self.capture or self.ignore_depth > 0:
                return
            if t in ("p", "li", "blockquote", "h1", "h2", "h3"):
                self._block_stack.append(t)
            if t == "br":
                self.parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            t = tag.lower()
            if not self.capture:
                return
            if self.ignore_depth > 0 and t in ("script", "style"):
                self.ignore_depth -= 1
                return
            if t == "div":
                self.depth -= 1
                if self.depth <= 0:
                    self.capture = False
                    return
            if self.ignore_depth > 0:
                return
            if self._block_stack and self._block_stack[-1] == t:
                self._block_stack.pop()
                self.parts.append("\n\n")

        def handle_data(self, data: str) -> None:
            if not self.capture or self.ignore_depth > 0:
                return
            s = _html.unescape(data)
            if not s:
                return
            if s.isspace():
                self._pending_space = True
                return
            if self._pending_space:
                self.parts.append(" ")
                self._pending_space = False
            self.parts.append(s)

    p = _DivParser()
    p.feed(html)
    out = "".join(p.parts)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = out.strip()
    return out or None


def _extract_fallback_article(html: str) -> Optional[str]:
    m = re.search(r"<article[^>]*>(.*?)</article>", html, flags=re.S | re.I)
    if not m:
        return None
    frag = m.group(1)
    frag = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", frag, flags=re.S | re.I)
    frag = re.sub(r"<[^>]+>", " ", frag)
    frag = _html.unescape(frag)
    frag = re.sub(r"\s+", " ", frag).strip()
    return frag or None


def _html_to_text(html: str) -> str:
    from html.parser import HTMLParser

    class _TextParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.parts: List[str] = []
            self._pending_space = False
            self._ignore_depth = 0
            self._block_stack: List[str] = []

        def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
            t = tag.lower()
            if t in ("script", "style"):
                self._ignore_depth += 1
                return
            if self._ignore_depth > 0:
                return
            if t in ("p", "li", "blockquote", "h1", "h2", "h3", "h4"):
                self._block_stack.append(t)
            if t == "br":
                self.parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            t = tag.lower()
            if t in ("script", "style") and self._ignore_depth > 0:
                self._ignore_depth -= 1
                return
            if self._ignore_depth > 0:
                return
            if self._block_stack and self._block_stack[-1] == t:
                self._block_stack.pop()
                self.parts.append("\n\n")

        def handle_data(self, data: str) -> None:
            if self._ignore_depth > 0:
                return
            s = _html.unescape(data)
            if not s:
                return
            if s.isspace():
                self._pending_space = True
                return
            if self._pending_space:
                self.parts.append(" ")
                self._pending_space = False
            self.parts.append(s)

    p = _TextParser()
    p.feed(html)
    out = "".join(p.parts)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _extract_gutenberg_first_section(html: str) -> Optional[str]:
    # Project Gutenberg HTML is often a full "book" page. For quick scoring we
    # extract the first non-preface H2 section (usually the first essay/chapter).
    h2_blocks: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"(?is)<h2[^>]*>.*?</h2>", html):
        start, end = m.span()
        block = m.group(0)
        title = re.sub(r"(?is)<[^>]+>", " ", block)
        title = _html.unescape(re.sub(r"\s+", " ", title)).strip()
        h2_blocks.append((start, end, title))

    def _is_front_matter(title: str) -> bool:
        t = title.lower()
        return any(k in t for k in ("contents", "preface", "footnotes"))

    candidates = [(s, e, t) for (s, e, t) in h2_blocks if not _is_front_matter(t)]
    if not candidates:
        return None

    start_idx, _, _ = candidates[0]
    end_idx = len(html)
    if len(candidates) > 1:
        end_idx = candidates[1][0]

    frag = html[start_idx:end_idx]
    frag = re.sub(r"(?is)<(script|style)[^>]*>.*?</\\1>", " ", frag)
    frag = re.sub(r'(?is)<span[^>]*class="pagenum"[^>]*>.*?</span>', " ", frag)
    frag = re.sub(r'(?is)<a[^>]+name="page[^"]+"[^>]*>\\s*</a>', " ", frag)
    text = _html_to_text(frag)
    # Drop Gutenberg page markers that can survive conversion.
    text = re.sub(r"(?m)^p\\.\\s*\\w+\\s*$", "", text)
    text = re.sub(r"(?m)^[ \t]*\\d+[ \t]*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None


def extract_text(url: str, html: str) -> Tuple[Optional[str], str]:
    txt = _extract_substack_body(html)
    if txt:
        return txt, "substack_body"

    txt = _extract_div_by_class(html, class_tokens=("pjgm-postcontent", "entry-content", "post-content"))
    if txt:
        return txt, "div_by_class"

    if "gutenberg.org" in (url or "").lower():
        txt = _extract_gutenberg_first_section(html)
        if txt:
            return txt, "gutenberg_first_section"

    txt = _extract_fallback_article(html)
    if txt:
        return txt, "article_tag"

    return None, "extract_failed"


def extract_url(url: str, *, timeout_s: float = 45.0) -> ExtractionResult:
    html = fetch_html(url, timeout_s=float(timeout_s))
    title = extract_title(html)
    text, extractor = extract_text(url, html)
    return ExtractionResult(
        url=str(url),
        title=title,
        text=text,
        extractor=str(extractor),
        html_sha1=_sha1_hex(html),
    )

