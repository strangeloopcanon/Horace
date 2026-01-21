"""
Score a list of URLs with the trained Qwen3 scorer on Modal.

This is a convenience tool for quick "paste a link → get a score" checks.

Run:
  make setup-modal
  modal run deploy/modal/score_urls_qwen3.py

Or pass URLs:
  modal run deploy/modal/score_urls_qwen3.py --urls https://a.com/x,https://b.com/y

Include rubric diagnostics (token-level metrics + suggestions):
  modal run deploy/modal/score_urls_qwen3.py --include-rubric
"""

from __future__ import annotations

import html as _html
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-score-urls-qwen3"
REPO_REMOTE_PATH = "/root/horace"


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists():
            return p
    return Path.cwd()


_LOCAL_REPO_ROOT = _local_repo_root()

data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("horace-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu121", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy>=1.24.0", "transformers>=4.40.0", "safetensors>=0.4.0")
)
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")

app = modal.App(APP_NAME)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HF_HOME", "/cache/hf")


@app.function(image=image, gpu="any", timeout=60 * 20, volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol})
def score_remote(
    text: str,
    *,
    scorer_model_path: str = "/vol/models/scorer_qwen3_great_other_v1",
    scorer_max_length: int = 512,
    doc_type: str = "prose",
    normalize_text: bool = True,
) -> Dict[str, Any]:
    _bootstrap_repo()
    from tools.studio.scorer_model import score_with_scorer

    res = score_with_scorer(
        text,
        model_path_or_id=str(scorer_model_path),
        doc_type=str(doc_type),
        normalize_text=bool(normalize_text),
        max_length=int(scorer_max_length),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
    )
    hf_cache_vol.commit()
    return asdict(res)


def _fetch_html(url: str) -> str:
    req = Request(str(url), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_og_title(html: str) -> Optional[str]:
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


def _extract_essay(html: str) -> Optional[str]:
    return _extract_substack_body(html) or _extract_fallback_article(html)


def _rubric_for_text(
    text: str,
    *,
    doc_type: str,
    scoring_model_id: str,
    baseline_model: str,
    backend: str,
    max_input_tokens: int,
    normalize_text: bool,
    compute_cohesion: bool,
) -> Dict[str, Any]:
    # Local rubric computation (deterministic); Modal is used only for the trained scorer inference.
    from tools.studio.analyze import analyze_text
    from tools.studio.baselines import load_baseline_cached
    from tools.studio.critique import suggest_edits
    from tools.studio.score import score_text

    ident = (baseline_model or "").strip() or "gpt2"
    p = Path(ident)
    if p.exists():
        baseline = load_baseline_cached(ident, path=p)
    else:
        baseline = load_baseline_cached(ident)

    analysis = analyze_text(
        text,
        model_id=str(scoring_model_id),
        doc_type=str(doc_type),
        backend=str(backend),
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        compute_cohesion=bool(compute_cohesion),
    )
    score = score_text(analysis["doc_metrics"], baseline, doc_type=str(doc_type))
    critique = suggest_edits(
        doc_metrics=analysis["doc_metrics"],
        score=score,
        spikes=analysis.get("spikes") or [],
        segments=analysis.get("segments") or {},
    )
    return {
        "analysis_meta": {
            "model_id": analysis.get("model_id"),
            "truncated": bool(analysis.get("truncated")),
            "tokens_count": int((analysis.get("doc_metrics") or {}).get("tokens_count") or 0),
            "text_normalization": analysis.get("text_normalization") or {},
        },
        "rubric_score": {
            "overall_0_100": float(score.overall_0_100),
            "categories": dict(score.categories),
            "metrics": {
                k: {"value": v.value, "percentile": v.percentile, "score_0_1": v.score_0_1, "mode": v.mode}
                for k, v in score.metrics.items()
            },
        },
        "critique": critique,
    }


@app.local_entrypoint()
def main(
    urls: str = "",
    scorer_model_path: str = "/vol/models/scorer_qwen3_great_other_v1",
    scorer_max_length: int = 512,
    include_rubric: bool = False,
    rubric_model_id: str = "gpt2",
    baseline_model: str = "gpt2_gutenberg_512",
    rubric_backend: str = "auto",
    rubric_max_input_tokens: int = 512,
    rubric_compute_cohesion: bool = False,
) -> None:  # pragma: no cover
    default_urls = [
        "https://www.astralcodexten.com/p/the-dilbert-afterlife",
        "https://www.strangeloopcanon.com/p/life-in-india-is-a-series-of-bilateral",
        "https://hollisrobbinsanecdotal.substack.com/p/llm-poetry-and-the-greatness-question",
    ]
    url_list = [u.strip() for u in str(urls).split(",") if u.strip()] or default_urls

    rows: List[Dict[str, Any]] = []
    for url in url_list:
        html = _fetch_html(url)
        title = _extract_og_title(html)
        text = _extract_essay(html)
        if not text:
            rows.append({"url": url, "title": title, "error": "extract_failed"})
            continue
        trained_score = None
        trained_err = None
        try:
            ts = score_remote.remote(
                text,
                scorer_model_path=str(scorer_model_path),
                scorer_max_length=int(scorer_max_length),
                doc_type="prose",
                normalize_text=True,
            )
            trained_score = {
                "overall_0_100": ts.get("score_0_100"),
                "prob_0_1": ts.get("prob_0_1"),
                "model_path_or_id": ts.get("model_path_or_id") or str(scorer_model_path),
                "device": ts.get("device"),
                "max_length": ts.get("max_length"),
                "n_windows": ts.get("n_windows"),
                "windows_capped": ts.get("windows_capped"),
            }
        except Exception as e:
            trained_err = f"{type(e).__name__}: {e}"

        rubric = None
        rubric_err = None
        if bool(include_rubric):
            try:
                rubric = _rubric_for_text(
                    text,
                    doc_type="prose",
                    scoring_model_id=str(rubric_model_id),
                    baseline_model=str(baseline_model),
                    backend=str(rubric_backend),
                    max_input_tokens=int(rubric_max_input_tokens),
                    normalize_text=True,
                    compute_cohesion=bool(rubric_compute_cohesion),
                )
            except Exception as e:
                rubric_err = f"{type(e).__name__}: {e}"

        row: Dict[str, Any] = {
            "url": url,
            "title": title,
            "chars": len(text),
        }
        if trained_score is not None:
            row["trained_score"] = trained_score
            ts0 = trained_score.get("overall_0_100")
            row["primary_score"] = {
                "overall_0_100": float(ts0) if isinstance(ts0, (int, float)) else 0.0,
                "source": "trained_scorer",
            }
        if trained_err is not None:
            row["trained_score_error"] = trained_err
        if rubric is not None:
            row["rubric"] = rubric
            if row.get("primary_score") is None:
                row["primary_score"] = {
                    "overall_0_100": float((rubric.get("rubric_score") or {}).get("overall_0_100") or 0.0),
                    "source": "rubric",
                }
        if rubric_err is not None:
            row["rubric_error"] = rubric_err
        if row.get("primary_score") is None:
            row["primary_score"] = {"overall_0_100": 0.0, "source": "none"}

        rows.append(row)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
