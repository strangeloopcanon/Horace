from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tools.studio.score import ScoreReport

_HF_CACHE: Dict[str, Tuple[Any, Any, str]] = {}


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


def _extract_json(text: str) -> Optional[dict]:
    t = (text or "").strip()
    if not t:
        return None
    # Strip fences if present
    fenced = re.findall(r"```(?:json)?\n(.*?)```", t, flags=re.S | re.I)
    if fenced:
        t = fenced[0].strip()
    # Best-effort: locate outermost JSON object
    a = t.find("{")
    b = t.rfind("}")
    if a >= 0 and b > a:
        t = t[a : b + 1]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def llm_critique(
    *,
    text: str,
    doc_type: str,
    score: ScoreReport,
    doc_metrics: Dict[str, Any],
    spikes: list[dict],
    segments: Dict[str, Any],
    model_id: str,
    max_new_tokens: int = 450,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Non-deterministic critic (optional).

    This is intentionally grounded: it receives measured metrics and spike excerpts
    and must cite them in suggestions. Deterministic scoring remains the source of truth.
    """
    tok, model, dev = _get_hf(model_id)

    # Keep prompt small; pass a snippet + structured metrics.
    snippet = (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if len(snippet) > 4000:
        snippet = snippet[:4000] + "\n…"

    metric_lines = []
    metric_lines.append(f"overall_0_100={score.overall_0_100:.1f}")
    for k, v in (score.categories or {}).items():
        metric_lines.append(f"category_{k}={v*100:.0f}/100")
    # Rubric metric details
    for k, ms in (score.metrics or {}).items():
        p = "null" if ms.percentile is None else f"{ms.percentile:.0f}"
        metric_lines.append(f"{k}: value={ms.value:.4f}, percentile={p}, mode={ms.mode}")

    seg_lines = []
    try:
        for name in ("sentences", "paragraphs", "lines"):
            s = (segments.get(name) or {}) if isinstance(segments, dict) else {}
            if not isinstance(s, dict):
                continue
            if "burst_cv" in s and isinstance(s["burst_cv"], (int, float)):
                seg_lines.append(f"{name}.burst_cv={float(s['burst_cv']):.3f}")
            if "len_cv" in s and isinstance(s["len_cv"], (int, float)):
                seg_lines.append(f"{name}.len_cv={float(s['len_cv']):.3f}")
    except Exception:
        pass

    spike_lines = []
    for s in (spikes or [])[:6]:
        ctx = str(s.get("context") or "").replace("\n", "\\n")
        spike_lines.append(
            f"- surprisal={float(s.get('surprisal') or 0.0):.2f}, entropy={float(s.get('entropy') or 0.0):.2f}, "
            f"content={bool(s.get('is_content'))}, punct={bool(s.get('is_punct'))}, line_pos={s.get('line_pos')} :: {ctx}"
        )

    prompt = (
        "You are HoraceCritic, a literary editor.\n"
        "You will receive a piece of writing plus measured cadence/signature metrics vs a reference baseline.\n"
        "Your job is to produce a grounded critique. Be specific and practical.\n\n"
        "Hard constraints:\n"
        "- Output MUST be valid JSON (no markdown, no code fences).\n"
        "- Output schema: {\"summary\": str, \"suggestions\": [{\"title\": str, \"why\": str, \"what_to_try\": str, \"evidence\": str|null}]}.\n"
        "- Every suggestion MUST cite at least one metric name (e.g. entropy_mean, ipi_mean, cohesion_delta).\n"
        "- If you cite text, quote only short phrases.\n\n"
        f"doc_type: {doc_type}\n\n"
        "metrics:\n"
        + "\n".join(metric_lines)
        + "\n\n"
        + ("segments:\n" + "\n".join(seg_lines) + "\n\n" if seg_lines else "")
        + ("spikes:\n" + "\n".join(spike_lines) + "\n\n" if spike_lines else "")
        + "text:\n"
        + snippet
        + "\n\nReturn JSON now."
    )

    try:
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
    except Exception:
        pass

    # Prefer chat template when available.
    try:
        messages = [
            {"role": "system", "content": "You are HoraceCritic."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
        gen = model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        raw = tok.decode(gen[0][input_ids.shape[-1] :], skip_special_tokens=True)
    except Exception:
        inputs = tok(prompt, return_tensors="pt").to(dev)
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

    parsed = _extract_json(raw)
    if parsed is None:
        return {
            "summary": (raw or "").strip()[:2000],
            "suggestions": [],
            "raw": (raw or "").strip(),
        }

    # Ensure required keys exist (best-effort)
    if "summary" not in parsed:
        parsed["summary"] = ""
    if "suggestions" not in parsed or not isinstance(parsed.get("suggestions"), list):
        parsed["suggestions"] = []
    return parsed

