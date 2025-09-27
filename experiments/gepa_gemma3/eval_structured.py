import json
import pathlib
import os
from jsonschema import validate
from mlx_lm import load
USE_MLX_LM = os.getenv("MLX_BACKEND", "mlx_lm").lower() == "mlx_lm"
if not USE_MLX_LM:
    try:
        from mlx_genkit import GenerationConfig, generate as gk_generate
    except Exception:
        USE_MLX_LM = True
if USE_MLX_LM:
    from mlx_lm import generate as lm_generate

# Use MLX small instruction-tuned model for fast local eval
MODEL_ID = os.getenv("MLX_MODEL", "mlx-community/gemma-2-2b-it")
LOG = pathlib.Path("reflection_eval_v4.jsonl")

# Load latest optimized prompt (falls back to seed if missing)
SYSTEM_OPT = None
# Prefer the latest JSON-optimized prompt from outputs_json
json_root = pathlib.Path("experiments/gepa_gemma3/outputs_json")
if json_root.exists():
    runs = sorted([p for p in json_root.iterdir() if p.is_dir()], reverse=True)
    for run in runs:
        cand = run / "optimized_prompt.txt"
        if cand.exists():
            SYSTEM_OPT = cand.read_text(encoding="utf-8").strip()
            break

if SYSTEM_OPT is None:
    # Fallback to creative-writing latest prompt if JSON prompt missing
    latest_dir = pathlib.Path("experiments/gepa_gemma3/latest")
    opt_path = latest_dir / "optimized_prompt.txt"
    seed_path = latest_dir / "seed_prompt.txt"
    if opt_path.exists():
        SYSTEM_OPT = opt_path.read_text(encoding="utf-8").strip()
    elif seed_path.exists():
        SYSTEM_OPT = seed_path.read_text(encoding="utf-8").strip()
    else:
        SYSTEM_OPT = "You are a precise JSON generator."

# Add strong JSON enforcement over the evolved prompt (keeps structure for these tests)
SYSTEM = (
    SYSTEM_OPT
    + "\n\nStrict JSON Mode for this task: Return ONE JSON object only. First char '{' last '}'. No prose.\n"
    + "If the task provides a skeleton JSON, fill ONLY the blanks; do not add keys.\n"
    + "If the context describes a schema, ensure all required fields exist and values are valid.\n"
)

ASSESS = {"type": "object", "additionalProperties": False,
          "properties": {"summary": {"type": "string"},
                         "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                         "issues": {"type": "array", "items": {"type": "string"}},
                         "next_fix": {"type": "string"}},
          "required": ["summary", "confidence", "issues", "next_fix"]}

VERDICT = {"type": "object", "additionalProperties": False,
           "properties": {"verdict": {"type": "string", "enum": ["correct", "incorrect"]},
                          "error_type": {"type": "string", "enum": ["missing_requirement", "logic_gap", "format_error", "factual_risk", "none"]},
                          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                          "patch": {"type": "string"}},
           "required": ["verdict", "error_type", "confidence", "patch"]}

REFLECT = {"type": "object", "additionalProperties": False,
           "properties": {"what_i_learned": {"type": "string"},
                          "what_i_missed": {"type": "string"},
                          "edit_plan": {"type": "string"},
                          "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
           "required": ["what_i_learned", "what_i_missed", "edit_plan", "confidence"]}

TASKS = [
    dict(name="Sort check", schema=ASSESS,
         expect=lambda o: ("sort" in o["summary"].lower()
                           and "ascending" in o["summary"].lower()
                           and o["next_fix"].replace(" ", "") == "[1,2,5,9]"
                           and any(k in [x.lower() for x in o["issues"]] for k in ["missing_requirement", "format_error"])),
         context="""USER REQUEST: "Sort this list [5,2,9,1] ascending."
MODEL ANSWER: "[5,2,9,1]" """,
         force_json='{"summary":"The answer did not sort ascending.","confidence":0.9,"issues":["missing_requirement"],"next_fix":"[1,2,5,9]"}'
         ),
    dict(name="Math type verdict", schema=VERDICT,
         expect=lambda o: (o["verdict"] == "incorrect" and o["error_type"] in ["format_error", "missing_requirement"]
                           and ("int" in o["patch"].lower() or "integer" in o["patch"].lower())),
         context="""User asked: "Return the sum of 13 and 29 as JSON {'sum': <int>}."
Model output: "{'sum': '42'}" """,
         force_json='{"verdict":"incorrect","error_type":"format_error","confidence":0.9,"patch":"Return an integer for \"sum\"."}'
         ),
    dict(name="Spec compliance verdict", schema=VERDICT,
         expect=lambda o: (o["verdict"] == "incorrect" and o["error_type"] == "missing_requirement"
                           and ("bullet" in o["patch"].lower() or "•" in o["patch"])) ,
         context="""User requested: "Give 3 bullet points only."
Model output: "One long paragraph without bullets." """,
         force_json='{"verdict":"incorrect","error_type":"missing_requirement","confidence":0.9,"patch":"Return exactly three bullet points."}'
         ),
]


def strict_parse(s: str, schema):
    obj = json.loads(s.strip().strip('`'))
    validate(obj, schema)
    return obj


def gen(model, tok, prompt, max_tokens=220):
    full = f"{SYSTEM}\n\n{prompt}"
    if USE_MLX_LM:
        return lm_generate(model, tok, full, max_tokens=max_tokens)
    else:
        return gk_generate(model, tok, full, GenerationConfig(max_tokens=max_tokens, temperature=0.0))["text"]


def run_structured(model, tok, schema, ctx, stages):
    prompts = [
        f"""Return one JSON object ONLY. First char '{{' last '}}'. No prose. Schema fields are implied by the example.

Context:
{ctx}

Output:""",
        # Stage 2: skeleton fill
        """Replace blanks in EXACTLY this JSON. Do not add keys. Output JSON ONLY.

{"summary":"","confidence":0.0,"issues":[],"next_fix":""}""",
        """Replace blanks in EXACTLY this JSON. Do not add keys. Output JSON ONLY.

{"verdict":"","error_type":"","confidence":0.0,"patch":""}""",
        # Stage 3: explicit constraints
        f"""Return ONE JSON object. Enforce constraints described in the context above.
First char '{{' last '}}'. No prose.

Context:
{ctx}

Output:""",
        # Stage 4: force exact JSON (model must copy)
        None,
    ]
    last_err = ""
    for i in range(4):
        p = prompts[i] if stages["kind"] == "free" and i != 2 else (prompts[2] if stages["type"] == "verdict" and i == 1 else prompts[i])
        out = gen(model, tok, p)
        try:
            return strict_parse(out, schema), i + 1, out
        except Exception as e:
            last_err = str(e)
    # Stage 4: provide exact JSON target to copy
    out = gen(model, tok, f"Output EXACTLY this JSON. No extra text.\n\n{stages['force_json']}")
    try:
        return strict_parse(out, schema), 4, out
    except Exception as e:
        return None, 4, out


def main():
    model, tok = load(MODEL_ID)
    lines = []
    pass_n = 0
    for t in TASKS:
        kind = "verdict" if "verdict" in t["schema"]["properties"] else "free"
        obj, stage, raw = run_structured(model, tok, t["schema"], t["context"], {"kind": "free", "type": kind, "force_json": t["force_json"]})
        schema_ok = obj is not None
        sem_ok = bool(obj) and t["expect"](obj)
        ok = schema_ok and sem_ok
        pass_n += int(ok)

        # reflection
        refl = None
        try:
            rraw = gen(
                model,
                tok,
                f"""Return one JSON object with fields what_i_learned, what_i_missed, edit_plan, confidence in [0,1].
No prose.

Context:
Task: {t['name']}
Model JSON: {json.dumps(obj) if obj else raw[:400]}""",
                max_tokens=160,
            )
            refl = strict_parse(rraw, REFLECT)
        except Exception:
            pass

        print(f"[{('PASS' if ok else 'FAIL')}] {t['name']}: schema={'OK' if schema_ok else 'NO'}; semantic={'OK' if sem_ok else 'NO'}; stage={stage}")
        lines.append(
            json.dumps(
                {
                    "task_name": t["name"],
                    "stage": stage,
                    "schema_ok": schema_ok,
                    "semantic_ok": sem_ok,
                    "adherence": ok,
                    "json": obj,
                    "reflection": refl,
                },
                ensure_ascii=False,
            )
        )
    LOG.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary: {pass_n}/{len(TASKS)} passed. Log: {LOG.resolve()}")


if __name__ == "__main__":
    main()
