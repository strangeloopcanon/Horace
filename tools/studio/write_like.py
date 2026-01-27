"""Cadence Match: Generate text that matches a reference cadence profile.

So what: this turns Horace's cadence measurement into a generation control loop:
extract a cadence profile from a reference passage, then generate with the same
spike/cooldown rhythm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.analyze import pick_backend
from tools.sampler import CadenceSampler
from tools.studio.analyze import analyze_text
from tools.studio.cadence_profile import (
    CadenceProfile,
    compare_cadence_profiles,
    extract_cadence_profile,
    profile_to_poetry_config,
)


@dataclass
class WriteLikeResult:
    """Result of a cadence-matched generation."""

    generated_text: str
    reference_profile: CadenceProfile
    generated_profile: CadenceProfile
    cadence_match: Dict[str, Any]  # From compare_cadence_profiles
    generation_meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_text": self.generated_text,
            "reference_profile": self.reference_profile.to_dict(),
            "generated_profile": self.generated_profile.to_dict(),
            "cadence_match": self.cadence_match,
            "generation_meta": self.generation_meta,
        }


def write_like(
    prompt: str,
    *,
    reference_text: str,
    model_id: str = "gpt2",
    backend: str = "auto",
    doc_type: str = "prose",
    max_new_tokens: int = 200,
    seed: Optional[int] = 7,
) -> WriteLikeResult:
    """Generate text matching the cadence of a reference text.

    Args:
        prompt: Starting text / context to continue from
        reference_text: Text whose cadence should be matched
        model_id: Model to use for generation
        backend: 'auto', 'mlx', or 'hf'
        doc_type: 'prose' or 'poem'
        max_new_tokens: Maximum tokens to generate
        seed: Random seed for reproducibility

    Returns:
        WriteLikeResult with generated text and cadence comparison
    """
    # Extract cadence profile from reference
    reference_profile = extract_cadence_profile(
        reference_text,
        model_id=model_id,
        backend=backend,
        max_input_tokens=512,
        doc_type=doc_type,
    )

    # Convert to generation config
    config = profile_to_poetry_config(reference_profile)

    # Initialize backend and sampler
    be = pick_backend(model_id, backend=backend)
    sampler = CadenceSampler(be, config, seed=seed)

    # Generate
    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        prompt_clean = " "  # Need at least something

    generated_full = sampler.generate(prompt_clean, max_new_tokens=max_new_tokens)

    # Extract just the new text
    if generated_full.startswith(prompt_clean):
        generated_text = generated_full[len(prompt_clean):].strip()
    else:
        generated_text = generated_full.strip()

    # Analyze the generated text to get its cadence profile
    generated_profile = extract_cadence_profile(
        generated_text,
        model_id=model_id,
        backend=backend,
        max_input_tokens=256,
        doc_type=doc_type,
    )

    # Compare profiles
    cadence_match = compare_cadence_profiles(generated_profile, reference_profile)

    return WriteLikeResult(
        generated_text=generated_text,
        reference_profile=reference_profile,
        generated_profile=generated_profile,
        cadence_match=cadence_match,
        generation_meta={
            "model_id": model_id,
            "doc_type": doc_type,
            "max_new_tokens": max_new_tokens,
            "prompt_length": len(prompt_clean),
            "generated_length": len(generated_text),
        },
    )


def write_like_with_candidates(
    prompt: str,
    *,
    reference_text: str,
    model_id: str = "gpt2",
    backend: str = "auto",
    doc_type: str = "prose",
    max_new_tokens: int = 200,
    n_candidates: int = 3,
    seed: Optional[int] = 7,
) -> List[WriteLikeResult]:
    """Generate multiple candidates and return them sorted by cadence match.

    Args:
        prompt: Starting text / context to continue from
        reference_text: Text whose cadence should be matched
        model_id: Model for generation
        backend: 'auto', 'mlx', or 'hf'
        doc_type: 'prose' or 'poem'
        max_new_tokens: Maximum tokens per candidate
        n_candidates: Number of candidates to generate
        seed: Base random seed

    Returns:
        List of WriteLikeResult sorted by cadence similarity (best first)
    """
    results: List[WriteLikeResult] = []

    # Extract reference profile once
    reference_profile = extract_cadence_profile(
        reference_text,
        model_id=model_id,
        backend=backend,
        max_input_tokens=512,
        doc_type=doc_type,
    )
    config = profile_to_poetry_config(reference_profile)
    be = pick_backend(model_id, backend=backend)

    prompt_clean = (prompt or "").strip() or " "

    for i in range(max(1, int(n_candidates))):
        sampler = CadenceSampler(
            be, config, seed=(int(seed) + i + 1) if seed else None
        )
        generated_full = sampler.generate(prompt_clean, max_new_tokens=max_new_tokens)

        if generated_full.startswith(prompt_clean):
            generated_text = generated_full[len(prompt_clean):].strip()
        else:
            generated_text = generated_full.strip()

        if not generated_text:
            continue

        generated_profile = extract_cadence_profile(
            generated_text,
            model_id=model_id,
            backend=backend,
            max_input_tokens=256,
            doc_type=doc_type,
        )

        cadence_match = compare_cadence_profiles(generated_profile, reference_profile)

        results.append(
            WriteLikeResult(
                generated_text=generated_text,
                reference_profile=reference_profile,
                generated_profile=generated_profile,
                cadence_match=cadence_match,
                generation_meta={
                    "model_id": model_id,
                    "doc_type": doc_type,
                    "max_new_tokens": max_new_tokens,
                    "candidate_index": i,
                    "seed": (int(seed) + i + 1) if seed else None,
                },
            )
        )

    # Sort by similarity (higher is better)
    results.sort(
        key=lambda r: r.cadence_match.get("similarity_0_1", 0.0),
        reverse=True,
    )

    return results


def extract_cadence_for_display(
    text: str,
    *,
    model_id: str = "gpt2",
    backend: str = "auto",
    doc_type: str = "prose",
) -> Dict[str, Any]:
    """Extract cadence metrics in a format suitable for UI display.

    Returns a dict with both the profile and key display metrics.
    """
    profile = extract_cadence_profile(
        text,
        model_id=model_id,
        backend=backend,
        max_input_tokens=512,
        doc_type=doc_type,
    )

    # Get raw analysis for the plot data
    analysis = analyze_text(
        text,
        model_id=model_id,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=512,
        normalize_text=True,
        compute_cohesion=False,
        include_token_metrics=True,
    )

    doc_metrics = analysis.get("doc_metrics") or {}
    token_data = analysis.get("token_metrics") or {}
    series = analysis.get("series") or {}
    series_surprisal = series.get("surprisal") if isinstance(series, dict) else None
    token_surprisal = None
    if isinstance(token_data, dict):
        token_surprisal = token_data.get("surprisal")
    if token_surprisal is None and isinstance(series_surprisal, list):
        token_surprisal = series_surprisal

    return {
        "profile": profile.to_dict(),
        "display_metrics": {
            "spike_rate": doc_metrics.get("spike_rate"),
            "ipi_mean": doc_metrics.get("ipi_mean") or doc_metrics.get("high_surprise_ipi_mean"),
            "surprisal_cv": doc_metrics.get("surprisal_cv"),
            "content_fraction": doc_metrics.get("content_fraction"),
            "sent_burst_cv": doc_metrics.get("sent_burst_cv"),
        },
        "token_surprisal": token_surprisal,
        "spikes": analysis.get("spikes"),
    }
