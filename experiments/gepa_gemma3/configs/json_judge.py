from __future__ import annotations

import json
from typing import Tuple

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from jsonschema import validate

from experiments.gepa_gemma3.json_eval_common import (
    JSON_ENFORCER,
    build_examples,
    extract_json,
    get_tasks,
)


class StrictJSONSig(dspy.Signature):
    instruction = dspy.InputField(desc="Task instruction")
    context = dspy.InputField(desc="Context with schema cues")
    json_text = dspy.OutputField(desc="Strict JSON response")


class JSONWriter(dspy.Module):
    def __init__(self, instruction_seed: str):
        super().__init__()
        self.step = dspy.Predict(StrictJSONSig, instructions=instruction_seed)

    def forward(self, instruction: str, context: str) -> dspy.Prediction:
        out = self.step(instruction=instruction, context=context)
        return dspy.Prediction(json=out.json_text)


def _tasks():
    return get_tasks(core_only=False)


def get_seed_prompt() -> str:
    return (
        "You are a strict JSON agent. Return ONE JSON object only.\n"
        "Rules:\n"
        "- First character '{' and last '}'. No prose, no code fences, no extra text.\n"
        "- If a skeleton JSON appears in the prompt, fill only the values; do not add keys.\n"
        "- Obey the schema and label sets described in the context.\n"
        "- Prefer concise strings; confidence in [0,1].\n"
        "- If the task provides acceptance criteria, satisfy them exactly.\n"
        f"\n\n{JSON_ENFORCER}"
    )


def build_datasets(train_count: int, val_count: int, seed: int) -> Tuple[list, list]:
    examples = build_examples(_tasks())
    # Small task set; reuse for both train/val, optionally respecting counts
    train = examples[:max(0, min(len(examples), train_count))] or examples
    val = examples[:max(0, min(len(examples), val_count))] or examples
    return train, val


def build_student(seed_instruction: str) -> dspy.Module:
    return JSONWriter(seed_instruction)


def json_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    text = (getattr(pred, "json", None) or "").strip()
    problems = []

    framing_ok = text.startswith("{") and text.rstrip().endswith("}") and "`" not in text
    if not framing_ok:
        problems.append("Output must be a single JSON object; no prose or code fences")

    schema_ok = False
    parsed = None
    try:
        raw = extract_json(text)
        parsed = json.loads(raw)
        validate(parsed, gold.schema)
        schema_ok = True
    except Exception as exc:
        problems.append(f"Schema/parse error: {exc}")

    sem_ok = False
    if schema_ok:
        try:
            sem_ok = bool(gold.expect(parsed))
            if not sem_ok:
                problems.append("Semantic constraints not satisfied")
        except Exception as exc:
            problems.append(f"Semantic check exception: {exc}")

    if schema_ok and sem_ok:
        score = 1.0
    elif schema_ok:
        score = 0.6
    elif framing_ok:
        score = 0.3
    else:
        score = 0.0

    feedback = (
        f"Score={score:.2f} | framing={'OK' if framing_ok else 'NO'} | schema={'OK' if schema_ok else 'NO'} | semantic={'OK' if sem_ok else 'NO'}"
        + (" | " + " | ".join(problems) if problems else "")
    )
    return ScoreWithFeedback(score=score, feedback=feedback)


def metric():
    return json_metric


def prediction_to_dict(pred: dspy.Prediction) -> dict:
    return {"json": getattr(pred, "json", "")}
