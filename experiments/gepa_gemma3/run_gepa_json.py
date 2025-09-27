#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

from dotenv import load_dotenv
from jsonschema import validate, ValidationError

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from experiments.gepa_gemma3.run_gepa import AIStudioGemmaLM, LocalGemmaLM  # reuse clients


class StrictJSONSig(dspy.Signature):
    """Given instruction + context, return a single JSON object as string."""

    instruction = dspy.InputField(desc="Task instruction")
    context = dspy.InputField(desc="Context with schema cues")
    json = dspy.OutputField(desc="One JSON object only; no prose")


class JSONWriter(dspy.Module):
    def __init__(self, instruction_seed: str):
        super().__init__()
        self.step = dspy.Predict(StrictJSONSig, instructions=instruction_seed)

    def forward(self, instruction: str, context: str) -> dspy.Prediction:
        out = self.step(instruction=instruction, context=context)
        return dspy.Prediction(json=out.json)


def default_seed() -> str:
    return (
        "You are a strict JSON agent. Return ONE JSON object only.\n"
        "Rules:\n"
        "- First character '{' and last '}'. No prose, no code fences, no extra text.\n"
        "- If a skeleton JSON appears in the prompt, fill only the values; do not add keys.\n"
        "- Obey the schema and label sets described in the context.\n"
        "- Prefer concise strings; confidence in [0,1].\n"
        "- If the task provides acceptance criteria, satisfy them exactly.\n"
    )


# Task contexts and semantic checks (mirrors user's harness)
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
        schema_name="ASSESS",
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
    ),
    dict(
        name="Math type verdict",
        schema_name="VERDICT",
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
    ),
    dict(
        name="Spec compliance verdict",
        schema_name="VERDICT",
        schema=VERDICT_SCHEMA,
        context=(
            "User requested: \"Give 3 bullet points only.\"\n"
            "Model output: \"One long paragraph without bullets.\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        expect=lambda o: (
            o.get("verdict") == "incorrect"
            and o.get("error_type") == "missing_requirement"
            and ("bullet" in o.get("patch", "").lower() or "•" in o.get("patch", ""))
        ),
    ),
    # Correctly sorted case
    dict(
        name="Sort already sorted",
        schema_name="ASSESS",
        schema=ASSESS_SCHEMA,
        context=(
            'USER REQUEST: "Sort this list [1,2,5,9] ascending."\n'
            'MODEL ANSWER: "[1,2,5,9]"\n'
            "Return JSON with fields summary, confidence [0,1], issues (array), next_fix (string)."
        ),
        expect=lambda o: (
            ("sorted" in o.get("summary", "").lower())
            and ("ascending" in o.get("summary", "").lower())
            and isinstance(o.get("issues", []), list) and len(o.get("issues", [])) == 0
            and o.get("next_fix", "") in ["", "[]", "No change", "No change."]
        ),
    ),
    # Correct math type
    dict(
        name="Math correct",
        schema_name="VERDICT",
        schema=VERDICT_SCHEMA,
        context=(
            "User asked: \"Return the sum of 10 and 32 as JSON {'sum': <int>}.\"\n"
            "Model output: \"{'sum': 42}\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        expect=lambda o: (o.get("verdict") == "correct" and o.get("error_type") == "none"),
    ),
    # Bullet compliance correct
    dict(
        name="Spec compliance correct",
        schema_name="VERDICT",
        schema=VERDICT_SCHEMA,
        context=(
            "User requested: \"Give 3 bullet points only.\"\n"
            "Model output: \"- one\\n- two\\n- three\"\n"
            "Return JSON with fields verdict in {correct,incorrect}, error_type in {missing_requirement,logic_gap,format_error,factual_risk,none}, confidence [0,1], patch (string)."
        ),
        expect=lambda o: (o.get("verdict") == "correct" and o.get("error_type") == "none"),
    ),
]


def json_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    text = (getattr(pred, "json", None) or "").strip()
    problems = []

    # Basic framing check
    framing_ok = text.startswith("{") and text.rstrip().endswith("}") and "`" not in text
    if not framing_ok:
        problems.append("Output must be a single JSON object; no prose or code fences")

    # Schema validation
    schema_ok = False
    parsed = None
    try:
        parsed = json.loads(text)
        validate(parsed, gold.schema)
        schema_ok = True
    except Exception as e:
        problems.append(f"Schema/parse error: {e}")

    # Semantic checks
    sem_ok = False
    if schema_ok:
        try:
            sem_ok = bool(gold.expect(parsed))
            if not sem_ok:
                problems.append("Semantic constraints not satisfied")
        except Exception as e:
            problems.append(f"Semantic check exception: {e}")

    # Score
    score = 0.0
    if schema_ok and framing_ok and sem_ok:
        score = 1.0
    elif schema_ok and framing_ok:
        score = 0.6
    elif framing_ok:
        score = 0.3

    fb = (
        f"Score={score:.2f} | framing={'OK' if framing_ok else 'NO'} | schema={'OK' if schema_ok else 'NO'} | semantic={'OK' if sem_ok else 'NO'}"
        + (" | " + " | ".join(problems) if problems else "")
    )
    return ScoreWithFeedback(score=score, feedback=fb)


def build_examples() -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    for t in TASKS:
        # we include the schema name and instructions in context to reinforce constraints
        ctx = (
            f"Task: {t['name']}\n"
            f"Schema: {t['schema_name']}\n"
            f"{t['context']}\n"
            "Output one JSON object only."
        )
        ex = dspy.Example(
            instruction="Return a single JSON object that complies with the schema and constraints.",
            context=ctx,
            schema=t["schema"],
            expect=t["expect"],
        ).with_inputs("instruction", "context")
        examples.append(ex)
    return examples


def ensure_lm(task_model: str):
    if task_model.startswith("gemma-3") or task_model.startswith("aistudio/"):
        lm = AIStudioGemmaLM(task_model.split("/", 1)[1] if "/" in task_model else task_model)
    elif task_model.startswith("local:"):
        lm = LocalGemmaLM(task_model.split(":", 1)[1])
    else:
        lm = dspy.LM(task_model)
    dspy.configure(lm=lm)
    return lm


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-metric-calls", type=int, default=45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-model", type=str, default=os.getenv("TASK_MODEL", "gemma-3-4b-it"))
    args = parser.parse_args()

    ensure_lm(args.task_model)
    reflect_lm = dspy.LM(os.getenv("REFLECT_MODEL", "openai/gpt-4.1-mini"))

    trainset = build_examples()
    valset = build_examples()

    seed_instruction = default_seed()
    student = JSONWriter(seed_instruction)
    gepa = dspy.GEPA(
        metric=json_metric,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=2,
        reflection_lm=reflect_lm,
        candidate_selection_strategy="pareto",
        track_stats=True,
        track_best_outputs=True,
    )

    optimized = gepa.compile(student, trainset=trainset, valset=valset)
    prompt = optimized.step.signature.instructions
    out_dir = Path("experiments/gepa_gemma3/outputs_json")
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    run_dir = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "optimized_prompt.txt").write_text(prompt, encoding="utf-8")

    # quick eval on valset
    rows = []
    for ex in valset:
        pred = optimized(instruction=ex.instruction, context=ex.context)
        m = json_metric(ex, pred)
        rows.append({"task": ex.context.split("\n", 1)[0], "score": m["score"], "feedback": m["feedback"], "json": getattr(pred, "json", "")})
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("=== Optimized JSON prompt ===\n", prompt)
    print("Results in", run_dir)


if __name__ == "__main__":
    main()
