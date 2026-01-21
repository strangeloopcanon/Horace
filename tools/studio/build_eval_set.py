from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tools.studio.dataset_utils import (
    FixedSample,
    clean_text,
    group_id as make_group_id,
    html_to_text,
    http_get_json,
    http_get_text,
    make_sample_id,
    sample_window,
    strip_gutenberg_boilerplate,
    write_jsonl,
)


# A stable, hand-picked set of encyclopedic topics (deterministic; no random API).
_WIKIPEDIA_TITLES: List[str] = [
    "Black hole",
    "Photosynthesis",
    "Jazz",
    "Volcano",
    "Tea",
    "Machine learning",
    "French Revolution",
    "Moby-Dick",
    "Haiku",
    "Metaphor",
    "Entropy",
    "Neural network",
    "Democracy",
    "Shakespeare",
    "Renaissance",
    "Gravity",
]


# Open-access(ish) technical / prescriptive writing.
_PEP_URLS: List[Tuple[str, str]] = [
    ("PEP 8 — Style Guide for Python Code", "https://peps.python.org/pep-0008/"),
    ("PEP 20 — The Zen of Python", "https://peps.python.org/pep-0020/"),
    ("PEP 257 — Docstring Conventions", "https://peps.python.org/pep-0257/"),
]


_PUBLIC_DOMAIN_DOCS: List[Tuple[str, str]] = [
    ("U.S. Constitution (transcript)", "https://www.archives.gov/founding-docs/constitution-transcript"),
    ("Declaration of Independence (transcript)", "https://www.archives.gov/founding-docs/declaration-transcript"),
]


_GUTENBERG_URLS: List[Tuple[str, str]] = [
    ("Pride and Prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("Frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("Dracula", "https://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("Moby-Dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("The Great Gatsby", "https://www.gutenberg.org/cache/epub/64317/pg64317.txt"),
    ("The Picture of Dorian Gray", "https://www.gutenberg.org/cache/epub/174/pg174.txt"),
]


_RFC_URLS: List[Tuple[str, str]] = [
    ("RFC 9110 (HTTP Semantics)", "https://www.rfc-editor.org/rfc/rfc9110.txt"),
    ("RFC 9112 (HTTP/1.1)", "https://www.rfc-editor.org/rfc/rfc9112.txt"),
    ("RFC 2616 (HTTP/1.1)", "https://www.rfc-editor.org/rfc/rfc2616.txt"),
    ("RFC 8259 (JSON)", "https://www.rfc-editor.org/rfc/rfc8259.txt"),
    ("RFC 3986 (URI Generic Syntax)", "https://www.rfc-editor.org/rfc/rfc3986.txt"),
    ("RFC 3339 (Date and Time on the Internet)", "https://www.rfc-editor.org/rfc/rfc3339.txt"),
    ("RFC 7540 (HTTP/2)", "https://www.rfc-editor.org/rfc/rfc7540.txt"),
    ("RFC 9000 (QUIC)", "https://www.rfc-editor.org/rfc/rfc9000.txt"),
    ("RFC 9001 (QUIC TLS)", "https://www.rfc-editor.org/rfc/rfc9001.txt"),
    ("RFC 8446 (TLS 1.3)", "https://www.rfc-editor.org/rfc/rfc8446.txt"),
    ("RFC 1035 (DNS)", "https://www.rfc-editor.org/rfc/rfc1035.txt"),
    ("RFC 5322 (Internet Message Format)", "https://www.rfc-editor.org/rfc/rfc5322.txt"),
    ("RFC 6455 (WebSocket)", "https://www.rfc-editor.org/rfc/rfc6455.txt"),
    ("RFC 6749 (OAuth 2.0)", "https://www.rfc-editor.org/rfc/rfc6749.txt"),
    ("RFC 7519 (JWT)", "https://www.rfc-editor.org/rfc/rfc7519.txt"),
]


def gutenberg_excerpts(n: int, *, rng: random.Random, max_chars: int) -> List[FixedSample]:
    out: List[FixedSample] = []
    cache: Dict[str, str] = {}
    for _ in range(max(0, int(n))):
        title, url = rng.choice(_GUTENBERG_URLS)
        try:
            body = cache.get(url)
            if body is None:
                body = strip_gutenberg_boilerplate(http_get_text(url))
                cache[url] = body
            excerpt = sample_window(body, rng=rng, max_chars=int(max_chars))
            if not excerpt:
                continue
            sid = make_sample_id("gutenberg_excerpt", title, url, excerpt)
            out.append(
                FixedSample(
                    sample_id=sid,
                    group_id=make_group_id("gutenberg_excerpt", stable=url),
                    source="gutenberg_excerpt",
                    title=title,
                    url=url,
                    text=excerpt,
                    fetched_at_unix=int(time.time()),
                    meta={"license_hint": "public_domain_gutenberg"},
                )
            )
        except Exception:
            continue
    return out


def rfc_excerpts(n: int, *, rng: random.Random, max_chars: int) -> List[FixedSample]:
    out: List[FixedSample] = []
    cache: Dict[str, str] = {}
    for _ in range(max(0, int(n))):
        title, url = rng.choice(_RFC_URLS)
        try:
            raw = cache.get(url)
            if raw is None:
                raw = http_get_text(url)
                cache[url] = raw
            excerpt = sample_window(raw, rng=rng, max_chars=int(max_chars))
            if not excerpt:
                continue
            sid = make_sample_id("rfc_excerpt", title, url, excerpt)
            out.append(
                FixedSample(
                    sample_id=sid,
                    group_id=make_group_id("rfc_excerpt", stable=url),
                    source="rfc_excerpt",
                    title=title,
                    url=url,
                    text=excerpt,
                    fetched_at_unix=int(time.time()),
                    meta={"license_hint": "open_internet_standard"},
                )
            )
        except Exception:
            continue
    return out


def gibberish_controls(n: int, *, rng: random.Random, max_chars: int = 1400) -> List[FixedSample]:
    out: List[FixedSample] = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    punctuation = " ,.;:?!\n"
    for i in range(max(0, int(n))):
        chunks: List[str] = []
        while sum(len(c) for c in chunks) < max_chars:
            wlen = rng.randint(2, 12)
            word = "".join(rng.choice(alphabet) for _ in range(wlen))
            if rng.random() < 0.08:
                word = word.capitalize()
            chunks.append(word)
            chunks.append(rng.choice(punctuation))
        text = "".join(chunks)[:max_chars].strip()
        title = f"gibberish_{i+1}"
        sid = make_sample_id("gibberish_control", title, "", text)
        out.append(
            FixedSample(
                sample_id=sid,
                group_id=make_group_id("gibberish_control", stable=title),
                source="gibberish_control",
                title=title,
                url="",
                text=text,
                fetched_at_unix=int(time.time()),
                meta={"license_hint": "synthetic"},
            )
        )
    return out


def wikipedia_summaries_from_titles(titles: Sequence[str]) -> List[FixedSample]:
    out: List[FixedSample] = []
    for title in titles:
        t = str(title).strip()
        if not t:
            continue
        try:
            api = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(t, safe="")
            data = http_get_json(api)
            page_title = str(data.get("title") or t)
            url = str(((data.get("content_urls") or {}).get("desktop") or {}).get("page") or "")
            text = clean_text(str(data.get("extract") or ""))
            if not text:
                continue
            sid = make_sample_id("wikipedia_summary", page_title, url, text)
            out.append(
                FixedSample(
                    sample_id=sid,
                    group_id=make_group_id("wikipedia_summary", stable=url or page_title),
                    source="wikipedia_summary",
                    title=page_title,
                    url=url,
                    text=text,
                    fetched_at_unix=int(time.time()),
                    meta={"license_hint": "cc_by_sa_wikipedia"},
                )
            )
        except Exception:
            continue
    return out


def wikinews_latest_published(n: int) -> List[FixedSample]:
    # Pull the latest published titles, then fetch plaintext extracts in batches.
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Published",
        "cmsort": "timestamp",
        "cmdir": "desc",
        "cmprop": "title|timestamp",
        "cmlimit": max(1, int(n)),
        "format": "json",
    }
    url = "https://en.wikinews.org/w/api.php?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
    members = ((data.get("query") or {}).get("categorymembers") or [])
    titles: List[Tuple[str, str]] = []
    for m in members:
        if not isinstance(m, dict):
            continue
        ttl = str(m.get("title") or "").strip()
        ts = str(m.get("timestamp") or "").strip()
        if ttl:
            titles.append((ttl, ts))

    out: List[FixedSample] = []
    batch: List[Tuple[str, str]] = []
    for ttl, ts in titles:
        batch.append((ttl, ts))
        if len(batch) >= 10:
            out.extend(_wikinews_extract_batch(batch))
            batch = []
    if batch:
        out.extend(_wikinews_extract_batch(batch))
    return out[: max(0, int(n))]


def _wikinews_extract_batch(titles: Sequence[Tuple[str, str]]) -> List[FixedSample]:
    out: List[FixedSample] = []
    ttl_list = [t for t, _ in titles]
    titles_str = "|".join(ttl_list)
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "redirects": "1",
        "titles": titles_str,
        "format": "json",
    }
    url = "https://en.wikinews.org/w/api.php?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
    pages = ((data.get("query") or {}).get("pages") or {}).values()
    ts_by_title = {t: ts for t, ts in titles}
    for p in pages:
        if not isinstance(p, dict):
            continue
        title = str(p.get("title") or "").strip()
        extract = clean_text(str(p.get("extract") or ""))
        if not title or not extract:
            continue
        page_url = "https://en.wikinews.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="")  # noqa: S310
        sid = make_sample_id("wikinews_published", title, page_url, extract)
        out.append(
            FixedSample(
                sample_id=sid,
                group_id=make_group_id("wikinews_published", stable=page_url),
                source="wikinews_published",
                title=title,
                url=page_url,
                text=extract,
                fetched_at_unix=int(time.time()),
                meta={"license_hint": "cc_by_wikinews", "published_timestamp": ts_by_title.get(title, "")},
            )
        )
    return out


def nasa_breaking_news(n: int, *, max_chars: int) -> List[FixedSample]:
    rss_url = "https://www.nasa.gov/rss/dyn/breaking_news.rss"
    rng = random.Random(0)
    out: List[FixedSample] = []
    target = max(0, int(n))

    # Primary: RSS (usually 10 items only).
    try:
        raw = http_get_text(rss_url)
        root = ET.fromstring(raw)
        items = root.findall(".//item")
    except Exception:
        items = []

    seen_urls: set[str] = set()
    for it in items:
        if len(out) >= target:
            break
        title = clean_text((it.findtext("title") or "").strip())
        link = clean_text((it.findtext("link") or "").strip())
        if link:
            if link in seen_urls:
                continue
            seen_urls.add(link)
        desc = it.findtext("description") or ""
        encoded = it.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or ""
        text_html = encoded or desc
        text = html_to_text(text_html)
        if not text:
            continue
        text = sample_window(text, rng=rng, max_chars=int(max_chars))
        if not text:
            continue
        sid = make_sample_id("nasa_breaking_news", title, link, text)
        out.append(
            FixedSample(
                sample_id=sid,
                group_id=make_group_id("nasa_breaking_news", stable=link or title),
                source="nasa_breaking_news",
                title=title or "NASA breaking news",
                url=link,
                text=text,
                fetched_at_unix=int(time.time()),
                meta={
                    "license_hint": "likely_public_domain_us_government",
                    "source_kind": "rss",
                    "feed_url": rss_url,
                },
            )
        )

    if len(out) >= target:
        return out

    # Fallback: NASA WordPress API posts to fill beyond the RSS page size.
    # We use the rendered HTML content (article body only), so extraction is cleaner than scraping pages.
    api_base = "https://www.nasa.gov/wp-json/wp/v2/posts"
    per_page = 100
    remaining = max(0, target - len(out))
    # Allow some slack for skipped/empty posts; cap to keep the job bounded.
    max_pages = min(20, max(1, (remaining + per_page - 1) // per_page) + 2)
    for page in range(1, max_pages + 1):
        if len(out) >= target:
            break
        api_url = f"{api_base}?per_page={per_page}&page={page}"
        try:
            posts = json.loads(http_get_text(api_url))
        except Exception:
            break
        if not isinstance(posts, list) or not posts:
            break
        for post in posts:
            if len(out) >= target:
                break
            if not isinstance(post, dict):
                continue
            link = str(post.get("link") or "").strip()
            if not link or link in seen_urls:
                continue
            title_html = ""
            title_obj = post.get("title")
            if isinstance(title_obj, dict):
                title_html = str(title_obj.get("rendered") or "")
            content_html = ""
            content_obj = post.get("content")
            if isinstance(content_obj, dict):
                content_html = str(content_obj.get("rendered") or "")
            title = html_to_text(title_html) or "NASA post"
            text = html_to_text(content_html)
            if not text:
                continue
            text = sample_window(text, rng=rng, max_chars=int(max_chars))
            if not text:
                continue
            seen_urls.add(link)
            sid = make_sample_id("nasa_breaking_news", title, link, text)
            out.append(
                FixedSample(
                    sample_id=sid,
                    group_id=make_group_id("nasa_breaking_news", stable=link),
                    source="nasa_breaking_news",
                    title=title,
                    url=link,
                    text=text,
                    fetched_at_unix=int(time.time()),
                    meta={
                        "license_hint": "likely_public_domain_us_government",
                        "source_kind": "wp_v2_posts",
                        "api_url": api_url,
                    },
                )
            )

    return out


def html_page_excerpt(source: str, *, title: str, url: str, max_chars: int, rng: random.Random) -> Optional[FixedSample]:
    try:
        raw = http_get_text(url)
        text = html_to_text(raw)
        text = sample_window(text, rng=rng, max_chars=int(max_chars))
        if not text:
            return None
        sid = make_sample_id(source, title, url, text)
        return FixedSample(
            sample_id=sid,
            group_id=make_group_id(str(source), stable=str(url or title)),
            source=source,
            title=title,
            url=url,
            text=text,
            fetched_at_unix=int(time.time()),
            meta={"license_hint": "unknown_html_source"},
        )
    except Exception:
        return None


def build_fixed_eval_set(
    *,
    seed: int,
    max_chars: int,
    gutenberg_n: int,
    wikipedia_titles: Sequence[str],
    wikinews_n: int,
    nasa_n: int,
    rfc_n: int,
    pep_n: int,
    public_domain_n: int,
    gibberish_n: int,
) -> List[FixedSample]:
    rng = random.Random(int(seed))
    samples: List[FixedSample] = []

    samples.extend(gutenberg_excerpts(int(gutenberg_n), rng=rng, max_chars=int(max_chars)))
    samples.extend(wikipedia_summaries_from_titles(list(wikipedia_titles)))
    samples.extend(wikinews_latest_published(int(wikinews_n)))
    samples.extend(nasa_breaking_news(int(nasa_n), max_chars=int(max_chars)))
    samples.extend(rfc_excerpts(int(rfc_n), rng=rng, max_chars=int(max_chars)))
    samples.extend(gibberish_controls(int(gibberish_n), rng=rng, max_chars=min(1400, int(max_chars))))

    # Add a few deterministic HTML sources (technical/prescriptive)
    pep_urls = _PEP_URLS[: max(0, int(pep_n))]
    for ttl, url in pep_urls:
        s = html_page_excerpt("python_pep", title=ttl, url=url, max_chars=int(max_chars), rng=rng)
        if s is not None:
            samples.append(s)

    pd_urls = _PUBLIC_DOMAIN_DOCS[: max(0, int(public_domain_n))]
    for ttl, url in pd_urls:
        s = html_page_excerpt("public_domain_doc", title=ttl, url=url, max_chars=int(max_chars), rng=rng)
        if s is not None:
            samples.append(s)

    # Deduplicate by sample_id
    uniq: List[FixedSample] = []
    seen: set[str] = set()
    for s in samples:
        if s.sample_id in seen:
            continue
        if not (s.text or "").strip():
            continue
        seen.add(s.sample_id)
        uniq.append(s)
    return uniq


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a fixed Studio eval set (saved JSONL; no random APIs).")
    ap.add_argument("--out", default="data/eval_sets/studio_fixed_v1.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-chars", type=int, default=3800)
    ap.add_argument("--gutenberg", type=int, default=24)
    ap.add_argument("--wikinews", type=int, default=24)
    ap.add_argument("--nasa", type=int, default=16)
    ap.add_argument("--rfc", type=int, default=10)
    ap.add_argument("--pep", type=int, default=3)
    ap.add_argument("--public-domain", type=int, default=2)
    ap.add_argument("--gibberish", type=int, default=12)
    args = ap.parse_args(argv)

    samples = build_fixed_eval_set(
        seed=int(args.seed),
        max_chars=int(args.max_chars),
        gutenberg_n=int(args.gutenberg),
        wikipedia_titles=_WIKIPEDIA_TITLES,
        wikinews_n=int(args.wikinews),
        nasa_n=int(args.nasa),
        rfc_n=int(args.rfc),
        pep_n=int(args.pep),
        public_domain_n=int(args.public_domain),
        gibberish_n=int(args.gibberish),
    )
    out_path = Path(str(args.out))
    write_jsonl(out_path, (asdict(s) for s in samples))
    print(str(out_path))
    print(f"n={len(samples)} sources={sorted({s.source for s in samples})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
