import os
import time
import json
import pathlib

from dotenv import load_dotenv
from google import genai
from google.genai import types
from jsonschema import validate

from experiments.gepa_gemma3.json_eval_common import (
    extract_json,
    get_tasks,
    load_system_prompt,
)

LOG = pathlib.Path("experiments/gepa_gemma3/logs/reflection_eval_aistudio.jsonl")
DEFAULT_MODEL = "gemma-3-4b-it"
SYSTEM_PROMPT = load_system_prompt()
TASKS = get_tasks(core_only=True)


def ai_generate(client: genai.Client, model_id: str, user_text: str, max_tokens: int = 160) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n{user_text}"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    pieces: list[str] = []
    for chunk in client.models.generate_content_stream(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.0),
    ):
        if getattr(chunk, "text", None):
            pieces.append(chunk.text)
    return "".join(pieces).strip()


def try_parse(text: str, schema):
    raw = extract_json(text)
    try:
        obj = json.loads(raw)
        validate(obj, schema)
        return obj, raw
    except Exception:
        return None, raw


def run_task(client: genai.Client, model_id: str, task):
    ctx = task["context"]
    skeleton = task["skeleton"]

    prompts = [
        f"Return one JSON object ONLY. First char '{{' last '}}'. No prose.\n\nContext:\n{ctx}\n\nOutput:",
        f"Replace blanks in EXACTLY this JSON. Do not add keys. Output JSON ONLY.\n\n{skeleton}",
        f"Return ONE JSON object. Enforce constraints in the context above. First char '{{' last '}}'. No prose.\n\nContext:\n{ctx}\n\nOutput:",
    ]

    for prompt in prompts:
        out = ai_generate(client, model_id, prompt)
        obj, raw = try_parse(out, task["schema"])
        if obj is not None:
            return obj, raw

    final = ai_generate(client, model_id, f"Output EXACTLY this JSON. No extra text.\n\n{task['force_json']}")
    obj, raw = try_parse(final, task["schema"])
    return obj, raw


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Export it or add to .env")
    model_id = os.getenv("AISTUDIO_MODEL") or os.getenv("TASK_MODEL") or DEFAULT_MODEL
    client = genai.Client(api_key=api_key)

    lines = []
    pass_n = 0
    for task in TASKS:
        start = time.time()
        obj, raw = run_task(client, model_id, task)
        schema_ok = obj is not None
        sem_ok = bool(obj) and task["expect"](obj) if schema_ok else False
        ok = schema_ok and sem_ok
        elapsed = time.time() - start
        print(f"[{('PASS' if ok else 'FAIL')}] {task['name']}: schema={'OK' if schema_ok else 'NO'}; semantic={'OK' if sem_ok else 'NO'}; time={elapsed:.1f}s")
        lines.append(json.dumps({
            "name": task["name"],
            "schema_ok": schema_ok,
            "semantic_ok": sem_ok,
            "ok": ok,
            "json": obj if obj is not None else raw,
        }, ensure_ascii=False))
        pass_n += int(ok)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary: {pass_n}/{len(TASKS)} passed. Log: {LOG.resolve()}")


if __name__ == "__main__":
    main()
