from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tools.studio.dataset_utils import iter_jsonl, stable_sha1_hex, write_jsonl

DEFAULT_PROVIDER_MODELS = (
    "openai:gpt-5",
    "google:gemini-3-flash",
    "anthropic:claude-haiku-4.5",
)
DEFAULT_TIERS = ("write_like", "continue_from", "rewrite_from_memory")
BEST_EFFORT_TIER = "best_effort"


@dataclass(frozen=True)
class ProviderModel:
    provider: str
    model: str


def _parse_provider_specs(specs: Sequence[str]) -> List[ProviderModel]:
    out: List[ProviderModel] = []
    seen: set[Tuple[str, str]] = set()
    for raw in specs:
        s = str(raw or "").strip()
        if not s:
            continue
        if ":" not in s:
            raise ValueError(f"Invalid provider/model spec '{s}'. Expected 'provider:model'.")
        provider, model = s.split(":", 1)
        p = provider.strip().lower()
        m = model.strip()
        if p not in ("openai", "google", "anthropic"):
            raise ValueError(f"Unsupported provider '{provider}'. Use openai|google|anthropic.")
        if not m:
            raise ValueError(f"Missing model in provider/model spec '{s}'.")
        key = (p, m)
        if key in seen:
            continue
        seen.add(key)
        out.append(ProviderModel(provider=p, model=m))
    return out


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text or "")))


def _topic_hint(text: str, *, max_words: int = 24) -> str:
    t = " ".join(str(text or "").strip().split())
    if not t:
        return "the themes and events implied by the source"
    first_sent = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)[0].strip()
    words = first_sent.split()
    if not words:
        words = t.split()
    if len(words) > int(max_words):
        words = words[: int(max_words)]
    out = " ".join(words).strip(" -:;,.!?")
    return out or "the themes and events implied by the source"


def _short_prefix(text: str, *, frac: float = 0.40, max_chars: int = 1400) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    n = max(1, min(len(t), int(len(t) * float(frac))))
    n = min(n, int(max_chars))
    return t[:n].strip()


def _tier_prompt(
    *,
    tier: str,
    author: str,
    source_text: str,
    target_words: int,
) -> Tuple[str, str]:
    who = str(author or "").strip() or "the source author"
    topic = _topic_hint(source_text)
    wc = max(40, int(target_words))
    prefix = _short_prefix(source_text)

    if tier == "write_like":
        system = "You are a literary writing assistant."
        user = (
            f"Write an original passage of about {wc} words in the style of {who}.\n"
            f"Topic: {topic}.\n"
            "- Keep the prose natural and coherent.\n"
            "- Do not copy any existing text.\n"
            "- Avoid mentioning that this is an imitation."
        )
        return system, user

    if tier == "continue_from":
        system = "You are a literary writing assistant."
        user = (
            f"Here is a passage associated with {who}:\n"
            f"\"\"\"\n{prefix}\n\"\"\"\n\n"
            f"Continue in a similar voice for about {wc} words on the same subject.\n"
            "- Do not quote or paraphrase the source.\n"
            "- Keep continuity of tone and rhythm."
        )
        return system, user

    if tier == "rewrite_from_memory":
        system = (
            f"You are writing in a voice inspired by {who}. "
            "Write naturally and avoid meta-commentary."
        )
        user = (
            f"Write an original passage of about {wc} words about: {topic}.\n"
            "- Keep stylistic affinity with the source voice.\n"
            "- Do not copy source text.\n"
            "- Keep it self-contained."
        )
        return system, user

    if tier == "best_effort":
        system = "You are an accomplished prose writer."
        user = (
            f"Write a polished passage of about {wc} words about: {topic}.\n"
            "- Write your absolute best prose: clear, engaging, well-crafted.\n"
            "- Do NOT imitate any particular author.\n"
            "- Do NOT use filler phrases or hedge.\n"
            "- Be concrete and vivid.\n"
            "- No meta-commentary about the writing itself."
        )
        return system, user

    raise ValueError(f"Unknown tier: {tier}")


def _sanitize_generated_text(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    # Remove common wrappers.
    t = re.sub(r"^(?:passage|output|response|rewrite)\s*:\s*", "", t, flags=re.I)
    t = t.strip(" \n\r\t\"“”")
    return t.strip()


def _similarity_ratio(a: str, b: str) -> float:
    aa = " ".join(str(a or "").lower().split())
    bb = " ".join(str(b or "").lower().split())
    if not aa or not bb:
        return 0.0
    return float(SequenceMatcher(None, aa, bb).ratio())


def _post_json(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: float = 60.0,
    retries: int = 2,
    base_sleep_s: float = 1.0,
) -> Dict[str, Any]:
    raw_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=raw_data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise RuntimeError("Unexpected non-dict JSON response")
            return obj
        except urllib.error.HTTPError as e:
            code = int(getattr(e, "code", 0) or 0)
            if attempt < int(retries) and code in (429, 500, 502, 503, 504):
                time.sleep(float(base_sleep_s) * (2 ** attempt))
                attempt += 1
                continue
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise RuntimeError(f"HTTP {code}: {body[:400]}") from e
        except Exception:
            if attempt < int(retries):
                time.sleep(float(base_sleep_s) * (2 ** attempt))
                attempt += 1
                continue
            raise


def _extract_openai_text(body: Dict[str, Any]) -> str:
    # Chat Completions format
    try:
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for c in content:
                    if isinstance(c, dict):
                        tx = c.get("text")
                        if isinstance(tx, str):
                            parts.append(tx)
                if parts:
                    return "\n".join(parts).strip()
    except Exception:
        pass
    # Responses-like format
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = body.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if isinstance(c, dict):
                    tx = c.get("text")
                    if isinstance(tx, str):
                        parts.append(tx)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _is_openai_gpt5_model(model: str) -> bool:
    m = str(model or "").strip().lower()
    return m.startswith("gpt-5")


def _call_openai_chat(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload: Dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": str(system)},
            {"role": "user", "content": str(user)},
        ],
    }
    if _is_openai_gpt5_model(model):
        # GPT-5 chat completions only support default temperature and can consume
        # output budget as reasoning unless effort is set to minimal.
        payload["reasoning_effort"] = "minimal"
    else:
        payload["temperature"] = float(temperature)
    if int(max_output_tokens) > 0:
        payload["max_completion_tokens"] = int(max_output_tokens)
    obj = _post_json(
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        payload=payload,
    )
    out = _extract_openai_text(obj)
    return _sanitize_generated_text(out)


def _call_google_generate(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
    model_name = str(model)
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={urllib.parse.quote(key, safe='')}"
    payload: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": str(system)}]},
        "contents": [{"parts": [{"text": str(user)}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    obj = _post_json(
        url=url,
        headers={"Content-Type": "application/json"},
        payload=payload,
    )
    cands = obj.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise RuntimeError(f"Gemini response missing candidates: {obj}")
    content = (cands[0] or {}).get("content") or {}
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError(f"Gemini response missing content parts: {obj}")
    out_parts: List[str] = []
    for p in parts:
        if isinstance(p, dict):
            tx = p.get("text")
            if isinstance(tx, str) and tx.strip():
                out_parts.append(tx.strip())
    return _sanitize_generated_text("\n".join(out_parts))


def _call_anthropic_messages(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    payload: Dict[str, Any] = {
        "model": str(model),
        "system": str(system),
        "messages": [{"role": "user", "content": str(user)}],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
    }
    obj = _post_json(
        url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        payload=payload,
    )
    content = obj.get("content")
    if not isinstance(content, list):
        raise RuntimeError(f"Anthropic response missing content: {obj}")
    parts: List[str] = []
    for c in content:
        if isinstance(c, dict) and c.get("type") == "text":
            tx = c.get("text")
            if isinstance(tx, str) and tx.strip():
                parts.append(tx.strip())
    return _sanitize_generated_text("\n".join(parts))


def _openai_batch_request_row(
    *,
    custom_id: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": str(system)},
            {"role": "user", "content": str(user)},
        ],
    }
    if _is_openai_gpt5_model(model):
        body["reasoning_effort"] = "minimal"
    else:
        body["temperature"] = float(temperature)
    if int(max_output_tokens) > 0:
        body["max_completion_tokens"] = int(max_output_tokens)
    return {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _openai_result_by_custom_id(paths: Sequence[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            custom_id = str(obj.get("custom_id") or "").strip()
            if not custom_id:
                continue
            text = ""
            response = obj.get("response")
            if isinstance(response, dict):
                body = response.get("body")
                if isinstance(body, dict):
                    text = _extract_openai_text(body)
            if not text and isinstance(obj.get("body"), dict):
                text = _extract_openai_text(obj["body"])
            text = _sanitize_generated_text(text)
            if text:
                out[custom_id] = text
    return out


def _tier_source_label(tier: str) -> str:
    return f"llm_antipattern_{str(tier).strip().lower()}"


def _provider_for_job(
    *,
    rng: random.Random,
    specs: Sequence[ProviderModel],
) -> ProviderModel:
    if not specs:
        raise ValueError("No provider specs configured")
    return rng.choice(list(specs))


def _generate_text(
    *,
    spec: ProviderModel,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    if spec.provider == "openai":
        return _call_openai_chat(
            model=spec.model,
            system=system,
            user=user,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )
    if spec.provider == "google":
        return _call_google_generate(
            model=spec.model,
            system=system,
            user=user,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )
    if spec.provider == "anthropic":
        return _call_anthropic_messages(
            model=spec.model,
            system=system,
            user=user,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )
    raise ValueError(f"Unsupported provider: {spec.provider}")


def build_antipattern_pairs(
    *,
    originals_path: Path,
    out_pairs_path: Path,
    out_negatives_path: Optional[Path],
    out_openai_batch_requests: Optional[Path],
    openai_batch_results: Sequence[Path],
    out_unresolved_jobs_path: Optional[Path],
    seed: int,
    provider_specs: Sequence[str],
    tiers: Sequence[str],
    variants_per_tier: int,
    max_originals: int,
    temperature: float,
    max_output_tokens: int,
    min_generated_chars: int,
    max_generated_chars: int,
    similarity_threshold: float,
    run_online: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    specs = _parse_provider_specs(provider_specs)
    if not specs:
        raise ValueError("At least one --provider-model is required")
    use_tiers = [str(t).strip().lower() for t in tiers if str(t).strip()]
    if not use_tiers:
        raise ValueError("At least one --tier is required")
    for t in use_tiers:
        if t not in ("write_like", "continue_from", "rewrite_from_memory", "best_effort"):
            raise ValueError(f"Unsupported tier: {t}")

    originals = list(iter_jsonl(Path(originals_path)))
    rng.shuffle(originals)
    if int(max_originals) > 0:
        originals = originals[: int(max_originals)]

    openai_by_id = _openai_result_by_custom_id(openai_batch_results)

    pair_rows: List[dict] = []
    neg_rows: List[dict] = []
    openai_batch_rows: List[dict] = []
    unresolved_rows: List[dict] = []

    stats: Dict[str, Any] = {
        "seed": int(seed),
        "originals_path": str(originals_path),
        "n_originals_in": int(len(originals)),
        "provider_specs": [f"{s.provider}:{s.model}" for s in specs],
        "tiers": list(use_tiers),
        "variants_per_tier": int(variants_per_tier),
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "run_online": bool(run_online),
        "dry_run": bool(dry_run),
        "pairs_out": 0,
        "negatives_out": 0,
        "openai_batch_requests_out": 0,
        "unresolved_jobs": 0,
        "skipped_empty_original": 0,
        "skipped_empty_generated": 0,
        "skipped_short_generated": 0,
        "skipped_similar_generated": 0,
        "provider_success": {},
        "provider_errors": {},
        "tier_success": {},
    }

    for r in originals:
        original_text = str(r.get("text") or "")
        if not original_text.strip():
            stats["skipped_empty_original"] += 1
            continue
        original_text = original_text.strip()
        original_wc = _word_count(original_text)
        target_words = max(50, min(520, original_wc))

        sample_id = str(r.get("sample_id") or "").strip() or stable_sha1_hex([original_text])[:12]
        group_id = str(r.get("group_id") or "").strip() or f"sample:{sample_id}"
        title = str(r.get("title") or "")
        url = str(r.get("url") or "")
        author = str(r.get("author") or "")
        author_norm = str(r.get("author_norm") or "")
        source = str(r.get("source") or "unknown")
        doc_type = str(r.get("doc_type") or "prose")
        in_meta = dict(r.get("meta") or {}) if isinstance(r.get("meta"), dict) else {}

        for tier in use_tiers:
            for variant_idx in range(max(1, int(variants_per_tier))):
                spec = _provider_for_job(rng=rng, specs=specs)
                job_id = stable_sha1_hex(
                    [
                        str(sample_id),
                        str(group_id),
                        str(tier),
                        str(variant_idx),
                        str(spec.provider),
                        str(spec.model),
                        str(seed),
                    ]
                )[:20]
                system, user = _tier_prompt(
                    tier=tier,
                    author=author or author_norm,
                    source_text=original_text,
                    target_words=target_words,
                )

                generated = ""
                err = ""

                if spec.provider == "openai" and job_id in openai_by_id:
                    generated = openai_by_id[job_id]
                elif bool(dry_run):
                    generated = (
                        f"[DRY-RUN {tier}] "
                        f"{_topic_hint(original_text, max_words=20)} "
                        f"(style hint: {author or author_norm or 'unknown'})"
                    )
                elif bool(run_online):
                    try:
                        generated = _generate_text(
                            spec=spec,
                            system=system,
                            user=user,
                            temperature=float(temperature),
                            max_output_tokens=int(max_output_tokens),
                        )
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                elif spec.provider == "openai" and out_openai_batch_requests is not None:
                    openai_batch_rows.append(
                        _openai_batch_request_row(
                            custom_id=job_id,
                            model=spec.model,
                            system=system,
                            user=user,
                            temperature=float(temperature),
                            max_output_tokens=int(max_output_tokens),
                        )
                    )
                    unresolved_rows.append(
                        {
                            "job_id": job_id,
                            "sample_id": sample_id,
                            "group_id": group_id,
                            "tier": tier,
                            "variant_idx": int(variant_idx),
                            "provider": spec.provider,
                            "model": spec.model,
                            "status": "pending_openai_batch",
                            "author": author,
                            "source": source,
                        }
                    )
                    continue
                else:
                    unresolved_rows.append(
                        {
                            "job_id": job_id,
                            "sample_id": sample_id,
                            "group_id": group_id,
                            "tier": tier,
                            "variant_idx": int(variant_idx),
                            "provider": spec.provider,
                            "model": spec.model,
                            "status": "pending_provider_generation",
                            "author": author,
                            "source": source,
                        }
                    )
                    continue

                generated = _sanitize_generated_text(generated)
                if not generated:
                    if err:
                        stats["provider_errors"][f"{spec.provider}:{spec.model}"] = int(
                            stats["provider_errors"].get(f"{spec.provider}:{spec.model}", 0)
                        ) + 1
                    stats["skipped_empty_generated"] += 1
                    continue

                if len(generated) < int(min_generated_chars):
                    stats["skipped_short_generated"] += 1
                    continue
                if len(generated) > int(max_generated_chars) > 0:
                    generated = generated[: int(max_generated_chars)].strip()

                sim = _similarity_ratio(original_text, generated)
                if sim >= float(similarity_threshold):
                    stats["skipped_similar_generated"] += 1
                    continue

                pair_id = stable_sha1_hex([job_id, "pair"])[:12]
                source_label = _tier_source_label(tier)
                pair_meta = {
                    "created_at_unix": int(time.time()),
                    "provider": spec.provider,
                    "model": spec.model,
                    "tier": tier,
                    "prompt_template": f"{tier}_v1",
                    "variant_idx": int(variant_idx),
                    "source_label": source_label,
                    "author": author,
                    "author_norm": author_norm,
                    "doc_type": doc_type,
                    "input_source": source,
                    "input_title": title,
                    "input_url": url,
                    "input_sample_id": sample_id,
                    "similarity_to_original": float(sim),
                    "job_id": job_id,
                }
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "group_id": group_id,
                        "chosen_text": original_text,
                        "rejected_text": generated,
                        "meta": {**in_meta, **pair_meta},
                    }
                )
                if out_negatives_path is not None:
                    neg_sample_id = stable_sha1_hex([job_id, "neg"])[:12]
                    neg_rows.append(
                        {
                            "sample_id": neg_sample_id,
                            "group_id": group_id,
                            "source": source_label,
                            "title": title,
                            "url": url,
                            "text": generated,
                            "fetched_at_unix": int(time.time()),
                            "meta": {**in_meta, **pair_meta},
                        }
                    )

                stats["pairs_out"] += 1
                stats["provider_success"][f"{spec.provider}:{spec.model}"] = int(
                    stats["provider_success"].get(f"{spec.provider}:{spec.model}", 0)
                ) + 1
                stats["tier_success"][tier] = int(stats["tier_success"].get(tier, 0)) + 1

    if out_pairs_path:
        write_jsonl(Path(out_pairs_path), pair_rows)
    if out_negatives_path is not None:
        write_jsonl(Path(out_negatives_path), neg_rows)
        stats["negatives_out"] = int(len(neg_rows))
    if out_openai_batch_requests is not None:
        write_jsonl(Path(out_openai_batch_requests), openai_batch_rows)
        stats["openai_batch_requests_out"] = int(len(openai_batch_rows))
    if out_unresolved_jobs_path is not None:
        write_jsonl(Path(out_unresolved_jobs_path), unresolved_rows)
    stats["unresolved_jobs"] = int(len(unresolved_rows))

    stats_path = Path(out_pairs_path).with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "out_pairs_path": str(out_pairs_path),
        "out_negatives_path": str(out_negatives_path) if out_negatives_path is not None else "",
        "out_openai_batch_requests": str(out_openai_batch_requests) if out_openai_batch_requests is not None else "",
        "out_unresolved_jobs_path": str(out_unresolved_jobs_path) if out_unresolved_jobs_path is not None else "",
        "stats_path": str(stats_path),
        "stats": stats,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build anti-pattern preference pairs from curated originals using multi-provider LLM generation "
            "(OpenAI/Gemini/Claude) with deterministic random provider assignment."
        )
    )
    ap.add_argument("--originals", required=True, help="Input originals JSONL")
    ap.add_argument("--out-pairs", required=True, help="Output preference pairs JSONL")
    ap.add_argument("--out-negatives", default="", help="Optional output JSONL for classification negatives")
    ap.add_argument(
        "--provider-model",
        action="append",
        default=None,
        help="Provider/model spec provider:model (repeatable)",
    )
    ap.add_argument(
        "--tier",
        action="append",
        default=None,
        help="Generation tier (repeatable)",
    )
    ap.add_argument("--variants-per-tier", type=int, default=1)
    ap.add_argument("--max-originals", type=int, default=0, help="Cap originals (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-output-tokens", type=int, default=700)
    ap.add_argument("--min-generated-chars", type=int, default=220)
    ap.add_argument("--max-generated-chars", type=int, default=4200)
    ap.add_argument("--similarity-threshold", type=float, default=0.92)
    ap.add_argument("--run-online", action="store_true", help="Call provider APIs directly")
    ap.add_argument("--dry-run", action="store_true", help="Generate synthetic local outputs (no API calls)")
    ap.add_argument("--openai-batch-requests-out", default="", help="Optional OpenAI Batch request JSONL output")
    ap.add_argument(
        "--openai-batch-results",
        action="append",
        default=[],
        help="OpenAI Batch results JSONL (repeatable). Matching custom_id results are merged.",
    )
    ap.add_argument("--out-unresolved-jobs", default="", help="Optional JSONL path for unresolved jobs")
    args = ap.parse_args(argv)

    res = build_antipattern_pairs(
        originals_path=Path(str(args.originals)),
        out_pairs_path=Path(str(args.out_pairs)),
        out_negatives_path=Path(str(args.out_negatives)) if str(args.out_negatives).strip() else None,
        out_openai_batch_requests=Path(str(args.openai_batch_requests_out)) if str(args.openai_batch_requests_out).strip() else None,
        openai_batch_results=tuple(Path(str(x)) for x in (args.openai_batch_results or []) if str(x).strip()),
        out_unresolved_jobs_path=Path(str(args.out_unresolved_jobs)) if str(args.out_unresolved_jobs).strip() else None,
        seed=int(args.seed),
        provider_specs=tuple(args.provider_model or DEFAULT_PROVIDER_MODELS),
        tiers=tuple(args.tier or DEFAULT_TIERS),
        variants_per_tier=int(args.variants_per_tier),
        max_originals=int(args.max_originals),
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
        min_generated_chars=int(args.min_generated_chars),
        max_generated_chars=int(args.max_generated_chars),
        similarity_threshold=float(args.similarity_threshold),
        run_online=bool(args.run_online),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
