from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline, load_baseline_cached
from tools.studio.score import score_text
from tools.studio.text_normalize import normalize_for_studio


DEFAULT_UA = "HoraceStudioEval/0.1 (+https://example.invalid)"


@dataclass(frozen=True)
class TextSample:
    sample_id: str
    source: str
    title: str
    url: str
    text: str
    fetched_at_unix: int


@dataclass(frozen=True)
class ScoredSample:
    sample_id: str
    source: str
    title: str
    url: str
    tokens_count: int
    truncated: bool
    overall_0_100: float
    categories: Dict[str, float]
    text_stats: Dict[str, Any]
    text_normalization: Dict[str, Any]
    doc_metrics: Dict[str, Any]
    rubric_metrics: Dict[str, Any]
    rubric_worst: List[Dict[str, Any]]


def _http_get(url: str, *, timeout_s: float = 15.0, user_agent: str = DEFAULT_UA) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        return resp.read()


def _http_get_text(url: str, *, timeout_s: float = 15.0, user_agent: str = DEFAULT_UA) -> str:
    raw = _http_get(url, timeout_s=timeout_s, user_agent=user_agent)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _http_get_json(url: str, *, timeout_s: float = 15.0, user_agent: str = DEFAULT_UA) -> dict:
    return json.loads(_http_get_text(url, timeout_s=timeout_s, user_agent=user_agent))


def _clean_text(s: str) -> str:
    t = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


def _strip_gutenberg_boilerplate(text: str) -> str:
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
    return _clean_text(t)


def _sample_window(text: str, *, rng: random.Random, max_chars: int) -> str:
    t = _clean_text(text)
    if max_chars <= 0 or len(t) <= max_chars:
        return t

    # Prefer a window that starts on a paragraph boundary for readability.
    candidates = [m.start() for m in re.finditer(r"\n\s*\n", t)]
    if not candidates:
        start = rng.randint(0, max(0, len(t) - max_chars))
        return t[start : start + max_chars].strip()

    start = rng.choice(candidates)
    start = min(start, max(0, len(t) - max_chars))
    return t[start : start + max_chars].strip()


def _make_sample_id(source: str, title: str, url: str, text: str) -> str:
    h = hashlib.sha1()
    h.update((source or "").encode("utf-8"))
    h.update(b"\0")
    h.update((title or "").encode("utf-8"))
    h.update(b"\0")
    h.update((url or "").encode("utf-8"))
    h.update(b"\0")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()[:12]


def wikipedia_random_summaries(n: int, *, rng: random.Random) -> List[TextSample]:
    out: List[TextSample] = []
    for _ in range(max(0, int(n))):
        try:
            data = _http_get_json("https://en.wikipedia.org/api/rest_v1/page/random/summary")
            title = str(data.get("title") or "Wikipedia")
            url = str(((data.get("content_urls") or {}).get("desktop") or {}).get("page") or "https://en.wikipedia.org/")
            text = str(data.get("extract") or "")
            text = _clean_text(text)
            if not text:
                continue
            sid = _make_sample_id("wikipedia_random_summary", title, url, text)
            out.append(
                TextSample(
                    sample_id=sid,
                    source="wikipedia_random_summary",
                    title=title,
                    url=url,
                    text=text,
                    fetched_at_unix=int(time.time()),
                )
            )
        except Exception:
            continue
    return out


_GUTENBERG_URLS: List[Tuple[str, str]] = [
    ("Pride and Prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("Frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("Dracula", "https://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("Moby-Dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("The Great Gatsby", "https://www.gutenberg.org/cache/epub/64317/pg64317.txt"),
    ("The Picture of Dorian Gray", "https://www.gutenberg.org/cache/epub/174/pg174.txt"),
]


def gutenberg_excerpts(n: int, *, rng: random.Random, max_chars: int = 3800) -> List[TextSample]:
    out: List[TextSample] = []
    if n <= 0:
        return out

    cache: Dict[str, str] = {}

    # Sample works with replacement; extract a random window from each.
    for _ in range(max(0, int(n))):
        title, url = rng.choice(_GUTENBERG_URLS)
        try:
            body = cache.get(url)
            if body is None:
                raw = _http_get_text(url)
                body = _strip_gutenberg_boilerplate(raw)
                cache[url] = body
            excerpt = _sample_window(body, rng=rng, max_chars=int(max_chars))
            if not excerpt:
                continue
            sid = _make_sample_id("gutenberg_excerpt", title, url, excerpt)
            out.append(
                TextSample(
                    sample_id=sid,
                    source="gutenberg_excerpt",
                    title=title,
                    url=url,
                    text=excerpt,
                    fetched_at_unix=int(time.time()),
                )
            )
        except Exception:
            continue
    return out


_RFC_URLS: List[Tuple[str, str]] = [
    ("RFC 9110 (HTTP Semantics)", "https://www.rfc-editor.org/rfc/rfc9110.txt"),
    ("RFC 2616 (HTTP/1.1)", "https://www.rfc-editor.org/rfc/rfc2616.txt"),
    ("RFC 8259 (JSON)", "https://www.rfc-editor.org/rfc/rfc8259.txt"),
]


def rfc_excerpts(n: int, *, rng: random.Random, max_chars: int = 3800) -> List[TextSample]:
    out: List[TextSample] = []
    cache: Dict[str, str] = {}
    for _ in range(max(0, int(n))):
        title, url = rng.choice(_RFC_URLS)
        try:
            raw = cache.get(url)
            if raw is None:
                raw = _http_get_text(url)
                cache[url] = raw
            excerpt = _sample_window(raw, rng=rng, max_chars=int(max_chars))
            if not excerpt:
                continue
            sid = _make_sample_id("rfc_excerpt", title, url, excerpt)
            out.append(
                TextSample(
                    sample_id=sid,
                    source="rfc_excerpt",
                    title=title,
                    url=url,
                    text=excerpt,
                    fetched_at_unix=int(time.time()),
                )
            )
        except Exception:
            continue
    return out


def gibberish_controls(n: int, *, rng: random.Random, max_chars: int = 1400) -> List[TextSample]:
    out: List[TextSample] = []
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
        url = ""
        sid = _make_sample_id("gibberish_control", title, url, text)
        out.append(
            TextSample(
                sample_id=sid,
                source="gibberish_control",
                title=title,
                url=url,
                text=text,
                fetched_at_unix=int(time.time()),
            )
        )
    return out


def _ensure_baseline(baseline_model_or_path: str):
    ident = (baseline_model_or_path or "").strip() or "gpt2"
    p = Path(ident)
    if p.exists():
        return load_baseline_cached(ident, path=p)
    try:
        return load_baseline_cached(ident)
    except Exception:
        build_baseline(ident)
        return load_baseline_cached(ident)


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    pos = (len(sorted_vals) - 1) * q
    i = int(pos)
    j = min(len(sorted_vals) - 1, i + 1)
    frac = pos - i
    return float((1.0 - frac) * sorted_vals[i] + frac * sorted_vals[j])


def _summarize_scores(scores: List[float]) -> Dict[str, Any]:
    vals = [float(x) for x in scores]
    vals.sort()
    if not vals:
        nan = float("nan")
        return {"n": 0, "mean": nan, "p10": nan, "p50": nan, "p90": nan, "min": nan, "max": nan}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p10": _quantile(vals, 0.10),
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "min": vals[0],
        "max": vals[-1],
    }


def _text_stats(text: str) -> Dict[str, Any]:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    newlines = t.count("\n")
    single_newlines = len(re.findall(r"(?<!\n)\n(?!\n)", t))
    lines = t.split("\n")
    line_lens = [len(ln) for ln in lines if ln]
    avg_line_len = (sum(line_lens) / len(line_lens)) if line_lens else 0.0
    return {
        "chars": len(t),
        "newlines": int(newlines),
        "single_newlines": int(single_newlines),
        "lines": int(len(lines)),
        "avg_line_len": float(avg_line_len),
    }


def _rubric_metrics_dict(score) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, ms in (getattr(score, "metrics", None) or {}).items():
        out[k] = {
            "value": float(ms.value),
            "percentile": ms.percentile,
            "score_0_1": ms.score_0_1,
            "mode": ms.mode,
        }
    return out


def _worst_rubric_metrics(score, *, top_k: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, ms in (getattr(score, "metrics", None) or {}).items():
        if ms.score_0_1 is None:
            continue
        rows.append(
            {
                "metric": k,
                "score_0_1": float(ms.score_0_1),
                "percentile": ms.percentile,
                "mode": ms.mode,
                "value": float(ms.value),
            }
        )
    rows.sort(key=lambda d: float(d["score_0_1"]))
    return rows[: max(0, int(top_k))]


def score_samples(
    samples: Sequence[TextSample],
    *,
    model_id: str,
    baseline_model: str,
    doc_type: str,
    backend: str,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
) -> List[ScoredSample]:
    baseline = _ensure_baseline(baseline_model)
    out: List[ScoredSample] = []
    for s in samples:
        raw_stats = _text_stats(s.text)
        _, norm_meta = normalize_for_studio(s.text, doc_type=doc_type, enabled=bool(normalize_text))
        analysis = analyze_text(
            s.text,
            model_id=model_id,
            doc_type=doc_type,
            backend=backend,
            max_input_tokens=int(max_input_tokens),
            normalize_text=bool(normalize_text),
            compute_cohesion=bool(compute_cohesion),
        )
        score = score_text(analysis["doc_metrics"], baseline, doc_type=doc_type)
        dm = analysis.get("doc_metrics") or {}
        out.append(
            ScoredSample(
                sample_id=s.sample_id,
                source=s.source,
                title=s.title,
                url=s.url,
                tokens_count=int(dm.get("tokens_count") or 0),
                truncated=bool(analysis.get("truncated")),
                overall_0_100=float(score.overall_0_100),
                categories={k: float(v) for k, v in (score.categories or {}).items()},
                text_stats=raw_stats,
                text_normalization=(analysis.get("text_normalization") or norm_meta),
                doc_metrics=dm if isinstance(dm, dict) else {},
                rubric_metrics=_rubric_metrics_dict(score),
                rubric_worst=_worst_rubric_metrics(score),
            )
        )
    return out


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_eval(
    *,
    seed: int,
    wikipedia_n: int,
    gutenberg_n: int,
    rfc_n: int,
    gibberish_n: int,
    model_id: str,
    baseline_model: str,
    doc_type: str,
    backend: str,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
    excerpt_chars: int,
    samples_out: Path,
    report_out: Path,
) -> dict:
    rng = random.Random(int(seed))

    samples: List[TextSample] = []
    samples.extend(wikipedia_random_summaries(int(wikipedia_n), rng=rng))
    samples.extend(gutenberg_excerpts(int(gutenberg_n), rng=rng, max_chars=int(excerpt_chars)))
    samples.extend(rfc_excerpts(int(rfc_n), rng=rng, max_chars=int(excerpt_chars)))
    samples.extend(gibberish_controls(int(gibberish_n), rng=rng))

    _write_jsonl(samples_out, (asdict(s) for s in samples))

    scored = score_samples(
        samples,
        model_id=model_id,
        baseline_model=baseline_model,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=max_input_tokens,
        normalize_text=normalize_text,
        compute_cohesion=compute_cohesion,
    )

    by_source: Dict[str, List[ScoredSample]] = {}
    for s in scored:
        by_source.setdefault(s.source, []).append(s)

    report: Dict[str, Any] = {
        "run_meta": {
            "seed": int(seed),
            "model_id": model_id,
            "baseline_model": baseline_model,
            "doc_type": doc_type,
            "backend": backend,
            "max_input_tokens": int(max_input_tokens),
            "normalize_text": bool(normalize_text),
            "compute_cohesion": bool(compute_cohesion),
            "excerpt_chars": int(excerpt_chars),
            "n_samples": len(samples),
        },
        "sources": {},
        "scored": [asdict(s) for s in scored],
    }

    for src, rows in sorted(by_source.items(), key=lambda kv: kv[0]):
        scores = [r.overall_0_100 for r in rows]
        cat_means: Dict[str, float] = {}
        for r in rows:
            for k, v in (r.categories or {}).items():
                cat_means[k] = cat_means.get(k, 0.0) + float(v)
        for k in list(cat_means.keys()):
            cat_means[k] = float((cat_means[k] / max(1, len(rows))) * 100.0)
        report["sources"][src] = {
            "summary": _summarize_scores(scores),
            "category_means_0_100": cat_means,
            "top": [asdict(r) for r in sorted(rows, key=lambda x: x.overall_0_100, reverse=True)[:3]],
            "bottom": [asdict(r) for r in sorted(rows, key=lambda x: x.overall_0_100)[:3]],
        }

    _write_json(report_out, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch web text samples and evaluate Horace Studio scoring.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--model-id", default="gpt2")
    ap.add_argument("--baseline-model", default="gpt2_gutenberg_512", help="Baseline model id or baseline JSON path")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "hf"])
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--no-normalize-text", action="store_true", help="Disable formatting normalization (debug only)")
    ap.add_argument("--compute-cohesion", action="store_true")
    ap.add_argument("--excerpt-chars", type=int, default=3800)

    ap.add_argument("--wikipedia", type=int, default=12, help="Random Wikipedia summaries")
    ap.add_argument("--gutenberg", type=int, default=10, help="Random Project Gutenberg excerpts")
    ap.add_argument("--rfc", type=int, default=6, help="Random RFC excerpts")
    ap.add_argument("--gibberish", type=int, default=10, help="Generated gibberish controls")

    ap.add_argument("--samples-out", default="reports/studio_eval_web_samples.jsonl")
    ap.add_argument("--report-out", default="reports/studio_eval_web_report.json")
    args = ap.parse_args(argv)

    report = run_eval(
        seed=int(args.seed),
        wikipedia_n=int(args.wikipedia),
        gutenberg_n=int(args.gutenberg),
        rfc_n=int(args.rfc),
        gibberish_n=int(args.gibberish),
        model_id=str(args.model_id),
        baseline_model=str(args.baseline_model),
        doc_type=str(args.doc_type),
        backend=str(args.backend),
        max_input_tokens=int(args.max_input_tokens),
        normalize_text=not bool(args.no_normalize_text),
        compute_cohesion=bool(args.compute_cohesion),
        excerpt_chars=int(args.excerpt_chars),
        samples_out=Path(args.samples_out),
        report_out=Path(args.report_out),
    )

    print("== Horace Studio web eval ==")
    print(json.dumps(report.get("run_meta") or {}, indent=2))
    for src, info in (report.get("sources") or {}).items():
        summ = info.get("summary") or {}
        print(
            f"- {src}: n={summ.get('n')}, mean={summ.get('mean'):.1f}, "
            f"p10={summ.get('p10'):.1f}, p50={summ.get('p50'):.1f}, p90={summ.get('p90'):.1f}"
        )
    print(f"\nWrote samples: {args.samples_out}")
    print(f"Wrote report:  {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
