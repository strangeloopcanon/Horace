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

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    from tools.studio.baselines import load_baseline_cached
    from tools.studio.critique import suggest_edits
    from tools.studio.windowed_rubric import windowed_rubric_for_text

    ident = (baseline_model or "").strip() or "gpt2"
    p = Path(ident)
    if p.exists():
        baseline = load_baseline_cached(ident, path=p)
    else:
        baseline = load_baseline_cached(ident)

    wr = windowed_rubric_for_text(
        text,
        doc_type=str(doc_type),
        scoring_model_id=str(scoring_model_id),
        baseline=baseline,
        backend=str(backend),
        max_input_tokens=int(max_input_tokens),
        normalize_text=bool(normalize_text),
        compute_cohesion=bool(compute_cohesion),
    )
    analysis = wr.worst_analysis
    score = wr.worst
    critique = suggest_edits(
        doc_metrics=analysis["doc_metrics"],
        score=score,
        spikes=analysis.get("spikes") or [],
        segments=analysis.get("segments") or {},
    )
    return {
        "analysis_meta": {
            "model_id": analysis.get("model_id") or str(scoring_model_id),
            "truncated": bool(analysis.get("truncated")),
            "tokens_count": int((analysis.get("doc_metrics") or {}).get("tokens_count") or 0),
            "text_normalization": wr.text_normalization or {},
        },
        "rubric_score": {
            "overall_0_100": float(wr.aggregate.overall_0_100),
            "categories": dict(wr.aggregate.categories),
            "metrics": {
                k: {"value": v.value, "percentile": v.percentile, "score_0_1": v.score_0_1, "mode": v.mode}
                for k, v in score.metrics.items()
            },
        },
        "rubric_windows": {
            "worst_window_index": int(wr.worst_window_index),
            "best_window_index": int(wr.best_window_index),
            "windows": list(wr.windows),
        },
        "critique": critique,
    }


@app.local_entrypoint()
def main(
    urls: str = "",
    snapshot: str = "",
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
    url_list = [u.strip() for u in str(urls).split(",") if u.strip()] or []

    snap_map: Optional[Dict[str, dict]] = None
    if str(snapshot).strip():
        from tools.studio.url_snapshot import load_snapshot_text_by_url

        paths = [Path(p.strip()) for p in str(snapshot).split(",") if p.strip()]
        snap_map = load_snapshot_text_by_url(paths)
        if not url_list:
            url_list = sorted(snap_map.keys())

    if not url_list:
        url_list = default_urls

    rows: List[Dict[str, Any]] = []
    for url in url_list:
        title = None
        text = None
        extractor = None
        if snap_map is not None:
            snap = snap_map.get(str(url))
            if isinstance(snap, dict):
                title = snap.get("title")
                text = snap.get("text")
                extractor = snap.get("extractor") or "snapshot"
            if not text:
                rows.append({"url": url, "title": title, "error": "snapshot_missing_or_empty"})
                continue
        else:
            from tools.studio.url_extract import extract_text, extract_title, fetch_html

            html = fetch_html(url)
            title = extract_title(html)
            text, extractor = extract_text(url, html)
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
                "head_labels": ts.get("head_labels"),
                "head_probs_0_1": ts.get("head_probs_0_1"),
                "head_probs_by_label": ts.get("head_probs_by_label"),
                "primary_from_heads": ts.get("primary_from_heads"),
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
            "extractor": extractor,
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
