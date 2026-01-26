from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.studio.analyze import analyze_text
from tools.studio.baselines import build_baseline, load_baseline_cached
from tools.studio.calibrator import featurize_from_report_row, load_logistic_calibrator
from tools.studio.score import ScoreReport, score_text
from tools.studio.text_normalize import normalize_for_studio


_HF_CACHE: Dict[str, Tuple[Any, Any, str]] = {}
_CALIBRATOR_CACHE: Dict[str, Any] = {}


def _pick_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_hf(model_id: str, device: Optional[str] = None):
    if model_id in _HF_CACHE:
        return _HF_CACHE[model_id]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or _pick_device()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(dev)
    model.eval()
    _HF_CACHE[model_id] = (tok, model, dev)
    return tok, model, dev


def _extract_rewrite(text: str) -> str:
    # Strip common wrappers ("Rewrite:" etc) and fenced blocks.
    t = (text or "").strip()
    if not t:
        return ""
    fenced = re.findall(r"```(?:text)?\n(.*?)```", t, flags=re.S)
    if fenced:
        t = fenced[0].strip()
    # Drop leading labels
    t = re.sub(r"^(rewritten|rewrite|output)\s*:\s*", "", t, flags=re.I)
    return t.strip()


def generate_rewrites(
    text: str,
    *,
    doc_type: str,
    rewrite_model_id: str,
    n: int = 4,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_p: float = 0.92,
    seed: Optional[int] = 7,
) -> List[str]:
    tok, model, dev = _get_hf(rewrite_model_id)

    style = "poem" if doc_type == "poem" else "prose"
    sys_prompt = (
        "You are a literary editor.\n"
        "Return ONLY the rewritten text (no preface, no explanation, no quotes)."
    )
    user_prompt = (
        "Rewrite the following text into stronger literary "
        f"{style} while preserving meaning and voice.\n"
        "- Keep the same viewpoint and core imagery.\n"
        "- Remove generic phrasing; add concrete sensory detail.\n"
        "- Improve rhythm/cadence; avoid purple prose.\n"
        "- Avoid adding new factual claims.\n\n"
        "TEXT:\n"
        + text.strip()
    )

    # Prefer chat template when the tokenizer actually has one configured.
    use_chat = bool(getattr(tok, "chat_template", None))
    if use_chat:
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
            inputs = {"input_ids": input_ids}
        except Exception:
            use_chat = False
    if not use_chat:
        prompt = sys_prompt + "\n\n" + user_prompt + "\n\nREWRITE:\n"
        inputs = tok(prompt, return_tensors="pt").to(dev)

    outputs: List[str] = []
    try:
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        for i in range(max(1, int(n))):
            # Slight seed jitter between candidates
            if seed is not None:
                torch.manual_seed(int(seed) + i + 1)
                np.random.seed(int(seed) + i + 1)
            gen = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            raw = tok.decode(gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            out = _extract_rewrite(raw)
            if out:
                outputs.append(out)
    except Exception:
        pass
    # Deduplicate
    uniq = []
    seen = set()
    for o in outputs:
        key = re.sub(r"\s+", " ", o.strip().lower())
        if key and key not in seen:
            seen.add(key)
            uniq.append(o)
    return uniq


def generate_span_rewrites(
    text: str,
    *,
    start_char: int,
    end_char: int,
    doc_type: str,
    rewrite_model_id: str,
    n: int = 6,
    max_new_tokens: int = 260,
    temperature: float = 0.8,
    top_p: float = 0.92,
    seed: Optional[int] = 7,
    context_before_chars: int = 520,
    context_after_chars: int = 320,
) -> List[str]:
    """Generate rewrites for a specific span only (returns replacement strings).

    So what: this supports "span patching" workflows where we only change the parts we can
    verify (meaning-lock), rather than rewriting the whole document.
    """
    t = text or ""
    s = max(0, min(int(start_char), len(t)))
    e = max(s, min(int(end_char), len(t)))
    span = t[s:e].strip()
    if not span:
        return []

    before = t[max(0, s - int(context_before_chars)) : s].strip()
    after = t[e : min(len(t), e + int(context_after_chars))].strip()

    tok, model, dev = _get_hf(rewrite_model_id)
    style = "poem" if doc_type == "poem" else "prose"
    sys_prompt = (
        "You are a careful editor.\n"
        "Return ONLY the rewritten SPAN text (no preface, no explanation, no quotes, no tags)."
    )
    user_prompt = (
        f"You will rewrite ONLY the SPAN in a {style} document.\n"
        "Rules:\n"
        "- Preserve meaning, viewpoint, and tense.\n"
        "- Keep named entities and numbers unchanged.\n"
        "- Do NOT add new facts.\n"
        "- Keep edits minimal (small local changes; avoid rewriting everything).\n"
        "- Reduce monotony: vary cadence and sentence shape; prefer concrete verbs/nouns.\n\n"
        "CONTEXT BEFORE:\n"
        + (before if before else "(none)")
        + "\n\nSPAN:\n"
        + span
        + "\n\nCONTEXT AFTER:\n"
        + (after if after else "(none)")
        + "\n\nREWRITE SPAN:\n"
    )

    use_chat = bool(getattr(tok, "chat_template", None))
    if use_chat:
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
            inputs = {"input_ids": input_ids}
        except Exception:
            use_chat = False
    if not use_chat:
        prompt = sys_prompt + "\n\n" + user_prompt
        inputs = tok(prompt, return_tensors="pt").to(dev)

    outputs: List[str] = []
    try:
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        for i in range(max(1, int(n))):
            if seed is not None:
                torch.manual_seed(int(seed) + i + 1)
                np.random.seed(int(seed) + i + 1)
            gen = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            raw = tok.decode(gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            out = _extract_rewrite(raw)
            # Strip accidental wrappers
            out = re.sub(r"^<span>\\s*|\\s*</span>$", "", out, flags=re.I)
            out = out.strip()
            if out:
                outputs.append(out)
    except Exception:
        pass

    uniq: List[str] = []
    seen: set[str] = set()
    for o in outputs:
        key = re.sub(r"\\s+", " ", o.strip().lower())
        if key and key not in seen:
            seen.add(key)
            uniq.append(o)
    return uniq


def generate_dulled_rewrites(
    text: str,
    *,
    doc_type: str,
    rewrite_model_id: str,
    strength: str = "mild",
    n: int = 1,
    max_new_tokens: int = 320,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: Optional[int] = 11,
) -> List[str]:
    """Generate intentionally *less literary* rewrites (meaning preserved; style flattened).

    So what: these create within-content preference pairs (original > dulled) that teach the
    scorer to focus on cadence/voice rather than topic/domain.
    """
    tok, model, dev = _get_hf(rewrite_model_id)

    style = "poem" if doc_type == "poem" else "prose"
    strength_norm = str(strength or "mild").strip().lower()
    if strength_norm not in ("mild", "strong"):
        strength_norm = "mild"

    sys_prompt = (
        "You are a careful editor.\n"
        "Return ONLY the rewritten text (no preface, no explanation, no quotes)."
    )
    if strength_norm == "strong":
        user_prompt = (
            f"Rewrite the following {style} into intentionally plain, utilitarian prose while preserving meaning.\n"
            "- Keep the same facts, events, and viewpoint.\n"
            "- Remove imagery, metaphor, and rhetorical flourishes.\n"
            "- Prefer simple words and straightforward sentences.\n"
            "- Make rhythm/cadence flatter and more uniform.\n"
            "- Do NOT add new details.\n\n"
            "TEXT:\n"
            + text.strip()
        )
    else:
        user_prompt = (
            f"Rewrite the following {style} to be slightly more plain while preserving meaning.\n"
            "- Keep the same facts, events, and viewpoint.\n"
            "- Reduce stylistic flourish a bit (less imagery, fewer ornate turns).\n"
            "- Keep it natural and readable; avoid sounding broken or shuffled.\n"
            "- Do NOT add new details.\n\n"
            "TEXT:\n"
            + text.strip()
        )

    # Prefer chat template when available.
    use_chat = bool(getattr(tok, "chat_template", None))
    if use_chat:
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
            inputs = {"input_ids": input_ids}
        except Exception:
            use_chat = False
    if not use_chat:
        prompt = sys_prompt + "\n\n" + user_prompt + "\n\nREWRITE:\n"
        inputs = tok(prompt, return_tensors="pt").to(dev)

    outputs: List[str] = []
    try:
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        for i in range(max(1, int(n))):
            if seed is not None:
                torch.manual_seed(int(seed) + i + 1)
                np.random.seed(int(seed) + i + 1)
            gen = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            raw = tok.decode(gen[0], skip_special_tokens=True)
            # When using chat templates the decoded string can include the prompt.
            out = raw.split("REWRITE:", 1)[-1] if "REWRITE:" in raw else raw
            out = _extract_rewrite(out)
            if out.strip():
                outputs.append(out.strip())
    except Exception:
        pass

    uniq: List[str] = []
    seen: set[str] = set()
    for o in outputs:
        key = re.sub(r"\s+", " ", o.strip().lower())
        if key and key not in seen:
            seen.add(key)
            uniq.append(o)
    return uniq


def _ensure_baseline(baseline_model_id: str):
    ident = (baseline_model_id or "").strip()
    p = Path(ident)
    if p.exists():
        return load_baseline_cached(ident, path=p)
    path = os.path.join("data", "baselines")
    os.makedirs(path, exist_ok=True)
    try:
        return load_baseline_cached(ident)
    except Exception:
        build_baseline(ident)
        return load_baseline_cached(ident)


def _ensure_calibrator(calibrator_path: str):
    ident = (calibrator_path or "").strip()
    if not ident:
        return None
    cached = _CALIBRATOR_CACHE.get(ident)
    if cached is not None:
        return cached
    p = Path(ident)
    if not p.exists():
        raise FileNotFoundError(ident)
    cal = load_logistic_calibrator(p)
    _CALIBRATOR_CACHE[ident] = cal
    return cal


def _calibrated_score_0_100(
    score: ScoreReport,
    calibrator,
    *,
    doc_metrics: Dict[str, Any],
    max_input_tokens: int,
    missing_value: float,
) -> float:
    rubric_metrics = {k: {"score_0_1": v.score_0_1} for k, v in (score.metrics or {}).items()}
    feats = featurize_from_report_row(
        feature_names=calibrator.feature_names,
        categories=score.categories or {},
        rubric_metrics=rubric_metrics,
        doc_metrics=doc_metrics or {},
        max_input_tokens=int(max_input_tokens),
        missing_value=float(missing_value),
    )
    return float(calibrator.score_0_100(feats))


def _score_one(
    text: str,
    *,
    doc_type: str,
    scoring_model_id: str,
    baseline_model_id: str,
    backend: str,
    max_input_tokens: int,
    compute_cohesion: bool,
) -> Tuple[Dict[str, Any], ScoreReport]:
    analysis = analyze_text(
        text,
        model_id=scoring_model_id,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=max_input_tokens,
        compute_cohesion=bool(compute_cohesion),
    )
    baseline = _ensure_baseline(baseline_model_id)
    score = score_text(analysis["doc_metrics"], baseline, doc_type=doc_type)
    return analysis, score


def _delta_report(orig: ScoreReport, cand: ScoreReport, *, top_k: int = 4) -> Dict[str, Any]:
    cat_keys = sorted(set((orig.categories or {}).keys()) | set((cand.categories or {}).keys()))
    cat_delta: Dict[str, float] = {}
    for k in cat_keys:
        a = orig.categories.get(k)
        b = cand.categories.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            cat_delta[k] = float((float(b) - float(a)) * 100.0)

    metric_shifts: List[Dict[str, Any]] = []
    for m, cms in (cand.metrics or {}).items():
        oms = (orig.metrics or {}).get(m)
        if oms is None:
            continue
        if oms.score_0_1 is None or cms.score_0_1 is None:
            continue
        metric_shifts.append(
            {
                "metric": m,
                "mode": cms.mode,
                "delta_score_0_1": float(cms.score_0_1 - oms.score_0_1),
                "from_score_0_1": float(oms.score_0_1),
                "to_score_0_1": float(cms.score_0_1),
                "from_percentile": oms.percentile,
                "to_percentile": cms.percentile,
            }
        )
    metric_shifts.sort(key=lambda d: float(d.get("delta_score_0_1") or 0.0), reverse=True)
    gains = [d for d in metric_shifts if float(d.get("delta_score_0_1") or 0.0) > 0][: int(top_k)]
    losses = [d for d in reversed(metric_shifts) if float(d.get("delta_score_0_1") or 0.0) < 0][: int(top_k)]

    return {
        "overall_delta_0_100": float(cand.overall_0_100 - orig.overall_0_100),
        "categories_delta_0_100": cat_delta,
        "top_metric_gains": gains,
        "top_metric_losses": losses,
    }


def rewrite_and_rerank(
    text: str,
    *,
    doc_type: str = "prose",
    rewrite_model_id: str = "gpt2",
    scoring_model_id: str = "gpt2",
    baseline_model_id: str = "gpt2_gutenberg_512",
    calibrator_path: str = "",
    backend: str = "auto",
    n_candidates: int = 4,
    max_input_tokens: int = 512,
    normalize_text: bool = True,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_p: float = 0.92,
    seed: Optional[int] = 7,
    keep_top: int = 3,
    compute_cohesion: bool = False,
) -> Dict[str, Any]:
    doc_type = (doc_type or "prose").strip().lower()
    backend = (backend or "auto").strip().lower()

    calibrator = _ensure_calibrator(calibrator_path)
    missing_value = float(getattr(calibrator, "meta", {}).get("missing_value", 0.5)) if calibrator is not None else 0.5

    raw_text = text or ""
    norm_text, norm_meta = normalize_for_studio(raw_text, doc_type=doc_type, enabled=bool(normalize_text))

    orig_analysis, orig_score = _score_one(
        norm_text,
        doc_type=doc_type,
        scoring_model_id=scoring_model_id,
        baseline_model_id=baseline_model_id,
        backend=backend,
        max_input_tokens=max_input_tokens,
        compute_cohesion=compute_cohesion,
    )
    orig_cal_score = (
        _calibrated_score_0_100(
            orig_score,
            calibrator,
            doc_metrics=orig_analysis.get("doc_metrics") or {},
            max_input_tokens=int(max_input_tokens),
            missing_value=missing_value,
        )
        if calibrator is not None
        else None
    )

    candidates = generate_rewrites(
        norm_text,
        doc_type=doc_type,
        rewrite_model_id=rewrite_model_id,
        n=n_candidates,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        a, s = _score_one(
            cand,
            doc_type=doc_type,
            scoring_model_id=scoring_model_id,
            baseline_model_id=baseline_model_id,
            backend=backend,
            max_input_tokens=max_input_tokens,
            compute_cohesion=compute_cohesion,
        )
        cal_score = (
            _calibrated_score_0_100(
                s,
                calibrator,
                doc_metrics=a.get("doc_metrics") or {},
                max_input_tokens=int(max_input_tokens),
                missing_value=missing_value,
            )
            if calibrator is not None
            else None
        )
        scored.append(
            {
                "text": cand,
                "score": s.overall_0_100,
                "calibrated_score": cal_score,
                "delta": _delta_report(orig_score, s),
                "categories": s.categories,
                "metrics": {k: asdict(v) for k, v in s.metrics.items()},
                "doc_metrics": a["doc_metrics"],
            }
        )
    sort_key = "calibrated_score" if calibrator is not None else "score"
    scored.sort(key=lambda r: float(r.get(sort_key) or 0.0), reverse=True)

    return {
        "original": {
            "text": raw_text,
            "normalized_text": norm_text,
            "text_normalization": norm_meta,
            "score": orig_score.overall_0_100,
            "calibrated_score": orig_cal_score,
            "categories": orig_score.categories,
            "metrics": {k: asdict(v) for k, v in orig_score.metrics.items()},
            "analysis": orig_analysis,
        },
        "rewrites": scored[: max(1, int(keep_top))],
        "meta": {
            "rewrite_model_id": rewrite_model_id,
            "scoring_model_id": scoring_model_id,
            "baseline_model_id": baseline_model_id,
            "calibrator_path": calibrator_path,
            "rank_by": sort_key,
            "n_candidates": int(n_candidates),
            "normalize_text": bool(normalize_text),
        },
    }
