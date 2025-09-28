import json
import pathlib
import os

from jsonschema import validate
from mlx_lm import load

from experiments.gepa_gemma3.json_eval_common import (
    apply_chat_template,
    extract_json,
    get_tasks,
    load_system_prompt,
)

USE_MLX_LM = os.getenv("MLX_BACKEND", "mlx_lm").lower() == "mlx_lm"
if not USE_MLX_LM:
    try:
        from mlx_genkit import GenerationConfig, generate as gk_generate
    except Exception:
        USE_MLX_LM = True
if USE_MLX_LM:
    from mlx_lm import generate as lm_generate

MODEL_ID = os.getenv("MLX_MODEL", "mlx-community/gemma-2-2b-it")
LOG = pathlib.Path("experiments/gepa_gemma3/logs/reflection_eval_mlx.jsonl")
SYSTEM = load_system_prompt()
TASKS = get_tasks(core_only=True)

REFLECT = {"type": "object", "additionalProperties": False,
           "properties": {"what_i_learned": {"type": "string"},
                          "what_i_missed": {"type": "string"},
                          "edit_plan": {"type": "string"},
                          "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
           "required": ["what_i_learned", "what_i_missed", "edit_plan", "confidence"]}


def strict_parse(text: str, schema):
    raw = extract_json(text)
    obj = json.loads(raw)
    validate(obj, schema)
    return obj


def gen(model, tok, prompt, max_tokens=220):
    user_blob = f"{SYSTEM}\n\n{prompt}"
    full = apply_chat_template(tok, user_blob)
    if USE_MLX_LM:
        return lm_generate(model, tok, full, max_tokens=max_tokens)
    return gk_generate(model, tok, full, GenerationConfig(max_tokens=max_tokens, temperature=0.0))["text"]


def run_structured(model, tok, task):
    schema = task["schema"]
    ctx = task["context"]
    skeleton = task["skeleton"]

    prompts = [
        f"""Return one JSON object ONLY. First char '{{' last '}}'. No prose. Schema fields are implied by the example.

Context:
{ctx}

Output:""",
        f"""Replace blanks in EXACTLY this JSON. Do not add keys. Output JSON ONLY.

{skeleton}""",
        f"""Return ONE JSON object. Enforce constraints described in the context above.
First char '{{' last '}}'. No prose.

Context:
{ctx}

Output:""",
    ]

    for idx, prompt in enumerate(prompts, start=1):
        output = gen(model, tok, prompt)
        try:
            return strict_parse(output, schema), idx, output
        except Exception:
            pass

    output = gen(model, tok, f"Output EXACTLY this JSON. No extra text.\n\n{task['force_json']}")
    try:
        return strict_parse(output, schema), 4, output
    except Exception:
        return None, 4, output


def main():
    model, tok = load(MODEL_ID)
    if not USE_MLX_LM and not hasattr(model, "lm_head"):
        try:
            language_model = model.get("language_model")
            if language_model is not None and hasattr(language_model, "get"):
                inner_lm_head = language_model.get("lm_head")
                if inner_lm_head is not None:
                    setattr(model, "lm_head", inner_lm_head)
        except Exception:
            pass

    lines = []
    pass_n = 0
    for task in TASKS:
        parsed, stage, raw = run_structured(model, tok, task)
        schema_ok = parsed is not None
        sem_ok = bool(parsed) and task["expect"](parsed) if schema_ok else False
        ok = schema_ok and sem_ok
        pass_n += int(ok)

        reflection = None
        if parsed is not None:
            try:
                refl_output = gen(
                    model,
                    tok,
                    f"""Return one JSON object with fields what_i_learned, what_i_missed, edit_plan, confidence in [0,1].
No prose.

Context:
Task: {task['name']}
Model JSON: {json.dumps(parsed)}""",
                    max_tokens=160,
                )
                reflection = strict_parse(refl_output, REFLECT)
            except Exception:
                reflection = None

        print(f"[{('PASS' if ok else 'FAIL')}] {task['name']}: schema={'OK' if schema_ok else 'NO'}; semantic={'OK' if sem_ok else 'NO'}; stage={stage}")
        lines.append(json.dumps({
            "task_name": task["name"],
            "stage": stage,
            "schema_ok": schema_ok,
            "semantic_ok": sem_ok,
            "adherence": ok,
            "json": parsed,
            "reflection": reflection,
        }, ensure_ascii=False))

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary: {pass_n}/{len(TASKS)} passed. Log: {LOG.resolve()}")


if __name__ == "__main__":
    main()
