"""Shared utilities for strict JSON evaluation tasks across backends."""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, Iterable, List

from jsonschema import validate

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_PROMPT_CACHE: str | None = None
JSON_ENFORCER = (
    "Strict JSON Mode: Return ONE JSON object only. First char '{' last '}'. "
    "No prose, no code fences, no explanations. If a skeleton is provided, fill only values."
)


def _load_prompt_file() -> str:
    root = pathlib.Path("experiments/gepa_gemma3/outputs_json")
    if root.exists():
        for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
            fp = run_dir / "optimized_prompt.txt"
            if fp.exists():
                return fp.read_text(encoding="utf-8").strip()
    latest = pathlib.Path("experiments/gepa_gemma3/latest/optimized_prompt.txt")
    if latest.exists():
        return latest.read_text(encoding="utf-8").strip()
    return "Return one JSON object only."


def load_system_prompt() -> str:
    """Return the optimized instruction prompt with the strict JSON enforcer."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        _PROMPT_CACHE = _load_prompt_file()
    return _PROMPT_CACHE + "\n\n" + JSON_ENFORCER


# ---------------------------------------------------------------------------
# Schema + tasks
# ---------------------------------------------------------------------------

ASSESS_SCHEMA: Dict[str, Any] = {
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

VERDICT_SCHEMA: Dict[str, Any] = {
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

_TASK_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "Sort check",
        "schema_name": "ASSESS",
        "schema": ASSESS_SCHEMA,
        "context": (
            'USER REQUEST: "Sort this list [5,2,9,1] ascending."\n'
            'MODEL ANSWER: "[5,2,9,1]"\n'
            "Return JSON with fields summary, confidence [0,1], issues (array), next_fix (string)."
        ),
        "expect": lambda o: (
            "sort" in o.get("summary", "").lower()
            and "ascending" in o.get("summary", "").lower()
            and o.get("next_fix", "").replace(" ", "") == "[1,2,5,9]"
            and any(k in [x.lower() for x in o.get("issues", [])] for k in ["missing_requirement", "format_error"])
        ),
        "skeleton": '{"summary":"","confidence":0.0,"issues":[],"next_fix":""}',
        "force_json": '{"summary":"The answer did not sort ascending.","confidence":0.9,"issues":["missing_requirement"],"next_fix":"[1,2,5,9]"}',
    },
    {
        "name": "Math type verdict",
        "schema_name": "VERDICT",
        "schema": VERDICT_SCHEMA,
        "context": (
            "User asked: \"Return the sum of 13 and 29 as JSON {'sum': <int>}.\"\n"
            "Model output: \"{'sum': '42'}\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        "expect": lambda o: (
            o.get("verdict") == "incorrect"
            and o.get("error_type") in ["format_error", "missing_requirement"]
            and ("int" in o.get("patch", "").lower() or "integer" in o.get("patch", "").lower())
        ),
        "skeleton": '{"verdict":"","error_type":"","confidence":0.0,"patch":""}',
        "force_json": '{"verdict":"incorrect","error_type":"format_error","confidence":0.9,"patch":"Return an integer for \\"sum\\"."}',
    },
    {
        "name": "Spec compliance verdict",
        "schema_name": "VERDICT",
        "schema": VERDICT_SCHEMA,
        "context": (
            "User requested: \"Give 3 bullet points only.\"\n"
            "Model output: \"One long paragraph without bullets.\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        "expect": lambda o: (
            o.get("verdict") == "incorrect"
            and o.get("error_type") == "missing_requirement"
            and ("bullet" in o.get("patch", "").lower() or "•" in o.get("patch", ""))
        ),
        "skeleton": '{"verdict":"","error_type":"","confidence":0.0,"patch":""}',
        "force_json": '{"verdict":"incorrect","error_type":"missing_requirement","confidence":0.9,"patch":"Return exactly three bullet points."}',
    },
    {
        "name": "Sort already sorted",
        "schema_name": "ASSESS",
        "schema": ASSESS_SCHEMA,
        "context": (
            'USER REQUEST: "Sort this list [1,2,5,9] ascending."\n'
            'MODEL ANSWER: "[1,2,5,9]"\n'
            "Return JSON with fields summary, confidence [0,1], issues (array), next_fix (string)."
        ),
        "expect": lambda o: (
            "sorted" in o.get("summary", "").lower()
            and "ascending" in o.get("summary", "").lower()
            and isinstance(o.get("issues", []), list)
            and len(o.get("issues", [])) == 0
            and o.get("next_fix", "") in ["", "[]", "No change", "No change."]
        ),
        "skeleton": '{"summary":"","confidence":0.0,"issues":[],"next_fix":""}',
        "force_json": '{"summary":"The input list is already sorted in ascending order. No sorting is required.","confidence":0.95,"issues":[],"next_fix":""}',
    },
    {
        "name": "Math correct",
        "schema_name": "VERDICT",
        "schema": VERDICT_SCHEMA,
        "context": (
            "User asked: \"Return the sum of 10 and 32 as JSON {'sum': <int>}.\"\n"
            "Model output: \"{'sum': 42}\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        "expect": lambda o: (o.get("verdict") == "correct" and o.get("error_type") == "none"),
        "skeleton": '{"verdict":"","error_type":"","confidence":0.0,"patch":""}',
        "force_json": '{"verdict":"correct","error_type":"none","confidence":0.95,"patch":""}',
    },
    {
        "name": "Spec compliance correct",
        "schema_name": "VERDICT",
        "schema": VERDICT_SCHEMA,
        "context": (
            "User requested: \"Give 3 bullet points only.\"\n"
            "Model output: \"- one\\n- two\\n- three\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        "expect": lambda o: (o.get("verdict") == "correct" and o.get("error_type") == "none"),
        "skeleton": '{"verdict":"","error_type":"","confidence":0.0,"patch":""}',
        "force_json": '{"verdict":"correct","error_type":"none","confidence":0.95,"patch":""}',
    },
]


def get_tasks(core_only: bool = True) -> List[Dict[str, Any]]:
    """Return task descriptors.

    Args:
        core_only: If True, return the three failing scenarios used in the strict evaluator.
                   If False, include the three "correct" control cases as well.
    """
    return (_TASK_DEFINITIONS[:3] if core_only else list(_TASK_DEFINITIONS))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def extract_json(text: str) -> str:
    """Extract the first JSON object from the generated text."""
    fenced = re.search(r"```(?:json)?\n(.*?)```", text, flags=re.S)
    if fenced:
        return fenced.group(1).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        segment = match.group(0)
        depth = 0
        for idx, ch in enumerate(segment):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return segment[: idx + 1]
        return segment
    return text.strip()


def apply_chat_template(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return content


def validate_against_schema(raw: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    obj = json.loads(raw)
    validate(obj, schema)
    return obj


def loose_score(schema_ok: bool, sem_ok: bool, parsed: bool) -> float:
    if schema_ok and sem_ok:
        return 1.0
    if schema_ok:
        return 0.6
    if parsed:
        return 0.3
    return 0.0


def format_context(task: Dict[str, Any]) -> str:
    return (
        f"Task: {task['name']}\n"
        f"Schema: {task['schema_name']}\n"
        f"{task['context']}\n"
        "Output one JSON object only."
    )


def build_examples(tasks: Iterable[Dict[str, Any]]) -> List[Any]:
    import dspy

    examples: List[Any] = []
    for t in tasks:
        ex = dspy.Example(
            instruction="Return a single JSON object that complies with the schema and constraints.",
            context=format_context(t),
            schema=t["schema"],
            expect=t["expect"],
        ).with_inputs("instruction", "context")
        examples.append(ex)
    return examples

