"""Horace Studio: analyze arbitrary text and produce score/profile/suggestions/rewrites.

So what: keep package imports lightweight so dataset builders (and Modal wrappers)
don't accidentally pull in heavy ML dependencies unless needed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["analyze_text", "build_baseline", "load_baseline", "score_text", "suggest_edits", "rewrite_and_rerank"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "analyze_text":
        from tools.studio.analyze import analyze_text

        return analyze_text
    if name == "build_baseline":
        from tools.studio.baselines import build_baseline

        return build_baseline
    if name == "load_baseline":
        from tools.studio.baselines import load_baseline

        return load_baseline
    if name == "score_text":
        from tools.studio.score import score_text

        return score_text
    if name == "suggest_edits":
        from tools.studio.critique import suggest_edits

        return suggest_edits
    if name == "rewrite_and_rerank":
        from tools.studio.rewrite import rewrite_and_rerank

        return rewrite_and_rerank
    raise AttributeError(name)
