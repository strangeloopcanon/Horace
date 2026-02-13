from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

_DEFAULT_REMOTE_MODEL_ALLOWLIST: Set[str] = {
    "gpt2",
    "distilbert-base-uncased",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct",
    "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    "Skywork/Skywork-Reward-V2-Qwen3-4B",
    "Skywork/Skywork-Reward-V2-Qwen3-8B",
}


@dataclass(frozen=True)
class ModelSource:
    source_id: str
    revision: Optional[str]
    is_local: bool


def _truthy_env(name: str) -> bool:
    v = str(os.environ.get(name) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def remote_code_enabled() -> bool:
    return _truthy_env("HORACE_ALLOW_REMOTE_CODE")


def _allowlisted_remote_models() -> Set[str]:
    out = set(_DEFAULT_REMOTE_MODEL_ALLOWLIST)
    extra = str(os.environ.get("HORACE_MODEL_ID_ALLOWLIST") or "").strip()
    if extra:
        for part in extra.split(","):
            model_id = str(part or "").strip()
            if model_id:
                out.add(model_id)
    return out


def split_model_revision(model_path_or_id: str) -> Tuple[str, Optional[str]]:
    """Split '<repo>@<revision>' while preserving local filename semantics."""
    ident = str(model_path_or_id or "").strip()
    if "@" not in ident:
        return ident, None
    repo, rev = ident.rsplit("@", 1)
    repo = repo.strip()
    rev = rev.strip()
    if not repo or not rev:
        return ident, None
    if "/" not in repo:
        # Avoid interpreting local filenames like "model@v2" as remote refs.
        return ident, None
    return repo, rev


def _split_remote_revision(model_path_or_id: str) -> Tuple[str, Optional[str]]:
    # Backward-compatible alias kept for internal callers.
    return split_model_revision(model_path_or_id)


def _is_pinned_revision(revision: str) -> bool:
    rev = str(revision or "").strip().lower()
    if not rev:
        return False
    return rev not in {"main", "master", "head", "latest", "trunk"}


def resolve_model_source(model_path_or_id: str, *, purpose: str) -> ModelSource:
    """Resolve and validate a model source for HF loading.

    So what: remote model IDs can execute third-party code or drift over time. We only
    allow local paths, allowlisted remote IDs, or explicitly pinned remote revisions.
    """
    ident = str(model_path_or_id or "").strip()
    if not ident:
        raise ValueError(f"{purpose} is empty")

    p = Path(ident)
    if p.exists():
        try:
            return ModelSource(source_id=str(p.resolve()), revision=None, is_local=True)
        except Exception:
            return ModelSource(source_id=str(p), revision=None, is_local=True)

    repo, revision = split_model_revision(ident)
    if revision and _is_pinned_revision(revision):
        return ModelSource(source_id=repo, revision=revision, is_local=False)

    allowlist = _allowlisted_remote_models()
    if repo in allowlist:
        return ModelSource(source_id=repo, revision=None, is_local=False)

    raise ValueError(
        f"{purpose} must be a local path, an allowlisted remote model id, or a pinned "
        f"reference like '<repo>@<revision>'; got: {ident}"
    )


def resolve_trust_remote_code(source: ModelSource, *, requested: bool, purpose: str) -> bool:
    if not requested:
        return False
    if source.is_local:
        return True
    return remote_code_enabled()
