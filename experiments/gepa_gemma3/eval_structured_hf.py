import os
import re
import json
import time
import pathlib
from typing import Dict, Any, Tuple

from jsonschema import validate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG = pathlib.Path("reflection_eval_hf.jsonl")
MODEL_ID = os.getenv("HF_MODEL", "google/gemma-3-4b-it")


def load_latest_json_prompt() -> str:
    root = pathlib.Path("experiments/gepa_gemma3/outputs_json")
    if root.exists():
        runs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
        for rd in runs:
            fp = rd / "optimized_prompt.txt"
            if fp.exists():
                return fp.read_text(encoding="utf-8").strip()
    # Fallback to creative-writing prompt if needed
    latest = pathlib.Path("experiments/gepa_gemma3/latest/optimized_prompt.txt")
    if latest.exists():
        return latest.read_text(encoding="utf-8").strip()
    return "Return one JSON object only."


JSON_ENFORCER = (
    "Strict JSON Mode: Return ONE JSON object only. First char '{' last '}'. "
    "No prose, no code fences, no explanations. If a skeleton is provided, fill only values."
)

SYSTEM_PROMPT = load_latest_json_prompt() + "\n\n" + JSON_ENFORCER


ASSESS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
        "next_fix": {"type": "string"},
    },
    "required": ["summary", "confidence", "issues", "next_fix"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "error_type": {
            "type": "string",
            "enum": ["missing_requirement", "logic_gap", "format_error", "factual_risk", "none"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "patch": {"type": "string"},
    },
    "required": ["verdict", "error_type", "confidence", "patch"],
}

TASKS = [
    dict(
        name="Sort check",
        schema=ASSESS_SCHEMA,
        context=(
            'USER REQUEST: "Sort this list [5,2,9,1] ascending."\n'
            'MODEL ANSWER: "[5,2,9,1]"\n'
            "Return JSON with fields summary, confidence [0,1], issues (array), next_fix (string)."
        ),
        expect=lambda o: (
            ("sort" in o.get("summary", "").lower())
            and ("ascending" in o.get("summary", "").lower())
            and o.get("next_fix", "").replace(" ", "") == "[1,2,5,9]"
            and any(k in [x.lower() for x in o.get("issues", [])] for k in ["missing_requirement", "format_error"])  # noqa: E501
        ),
        skeleton='{"summary":"","confidence":0.0,"issues":[],"next_fix":""}',
    ),
    dict(
        name="Math type verdict",
        schema=VERDICT_SCHEMA,
        context=(
            "User asked: \"Return the sum of 13 and 29 as JSON {'sum': <int>}.\"\n"
            "Model output: \"{'sum': '42'}\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        expect=lambda o: (
            o.get("verdict") == "incorrect"
            and o.get("error_type") in ["format_error", "missing_requirement"]
            and ("int" in o.get("patch", "").lower() or "integer" in o.get("patch", "").lower())
        ),
        skeleton='{"verdict":"","error_type":"","confidence":0.0,"patch":""}',
    ),
    dict(
        name="Spec compliance verdict",
        schema=VERDICT_SCHEMA,
        context=(
            "User requested: \"Give 3 bullet points only.\"\n"
            "Model output: \"One long paragraph without bullets.\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        expect=lambda o: (
            o.get("verdict") == "incorrect" and o.get("error_type") == "missing_requirement" and ("bullet" in o.get("patch", "").lower() or "•" in o.get("patch", ""))
        ),
        skeleton='{"verdict":"","error_type":"","confidence":0.0,"patch":""}',
    ),
]


def build_messages(user_text: str):
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"},
    ]


def apply_chat(tokenizer, messages) -> Dict[str, torch.Tensor]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def hf_generate(model, tokenizer, user_text: str, max_new_tokens=120) -> str:
    messages = build_messages(user_text)
    inputs = apply_chat(tokenizer, messages)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        gen = model.generate(
            **{k: v for k, v in inputs.items()},
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()


def extract_json(text: str) -> str:
    # Strip code fences if present
    m = re.search(r"```(?:json)?\n(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # Find first JSON object
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        # Optionally trim trailing prose after the end of first object
        s = m.group(0)
        # Attempt to balance braces quickly
        depth = 0
        for i, ch in enumerate(s):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[: i + 1]
        return s
    return text


def try_parse(text: str, schema) -> Tuple[Dict[str, Any] | None, str]:
    raw = extract_json(text)
    try:
        obj = json.loads(raw)
        validate(obj, schema)
        return obj, raw
    except Exception as e:
        return None, raw


def run_task(model, tokenizer, t) -> Dict[str, Any]:
    ctx = t["context"]
    # Stage 1: free form JSON
    out1 = hf_generate(model, tokenizer, f"Return one JSON object ONLY. First char '{{' last '}}'. No prose.\n\nContext:\n{ctx}\n\nOutput:")
    obj, raw = try_parse(out1, t["schema"])
    if obj is None:
        # Stage 2: skeleton fill
        out2 = hf_generate(model, tokenizer, f"Replace blanks in EXACTLY this JSON. Do not add keys. Output JSON ONLY.\n\n{t['skeleton']}")
        obj, raw = try_parse(out2, t["schema"])
        if obj is None:
            # Stage 3: explicit constraints
            out3 = hf_generate(model, tokenizer, f"Return ONE JSON object. Enforce constraints in the context above. First char '{{' last '}}'. No prose.\n\nContext:\n{ctx}\n\nOutput:")
            obj, raw = try_parse(out3, t["schema"])
    schema_ok = obj is not None
    sem_ok = bool(obj) and t["expect"](obj) if schema_ok else False
    return dict(name=t["name"], schema_ok=schema_ok, semantic_ok=sem_ok, ok=schema_ok and sem_ok, json=obj if obj else raw)


def main():
    print(f"Loading HF model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    res_lines = []
    pass_n = 0
    for t in TASKS:
        t0 = time.time()
        res = run_task(model, tok, t)
        dt = time.time() - t0
        print(f"[{('PASS' if res['ok'] else 'FAIL')}] {t['name']}: schema={'OK' if res['schema_ok'] else 'NO'}; semantic={'OK' if res['semantic_ok'] else 'NO'}; time={dt:.1f}s")
        res_lines.append(json.dumps(res, ensure_ascii=False))
        pass_n += int(res["ok"])    
    LOG.write_text("\n".join(res_lines), encoding="utf-8")
    print(f"\nSummary: {pass_n}/{len(TASKS)} passed. Log: {LOG.resolve()}")


if __name__ == "__main__":
    main()

