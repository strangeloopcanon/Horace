# GEPA Prompt Optimization — Gemma 3 4B

This workspace holds scripts, configs, and logs for optimizing a writing instruction prompt with GEPA using the curated Horace author tasks. Generated artifacts:

- `run_gepa.py`: launcher that loads tasks, defines the evaluation metric, and runs GEPA.
- `outputs/`: prompt evolution logs, best generations, and summary JSON.
- `reports/`: summary markdown for quick inspection of the optimized prompt.

Run with activated API keys in `.env` (Gemini/OpenRouter as needed):

```bash
pip install -r experiments/gepa_gemma3/requirements.txt
# Using Google AI Studio (Gemini) Gemma
TASK_MODEL=gemma-3-4b-it python experiments/gepa_gemma3/run_gepa.py --max-metric-calls 90 --train 12 --val 6
# Or, with OpenRouter (if OPENROUTER_API_KEY is set)
TASK_MODEL=openrouter/google/gemma-3-4b-it python experiments/gepa_gemma3/run_gepa.py --max-metric-calls 90 --train 12 --val 6
```

The script defaults to `TASK_MODEL=google/gemma-3-4b-it` and `REFLECT_MODEL=openai/gpt-4.1-mini`, but honors environment overrides.

## Artifacts & “Latest” prompts

- Each run writes to `experiments/gepa_gemma3/outputs/<timestamp>/` with:
  - `seed_prompt.txt` — the initial instruction before GEPA.
  - `optimized_prompt.txt` — the evolved instruction from GEPA.
  - `results.json` — scores, metadata, and per-example details.
  - `report.md` — readable summary with samples.
  - `USAGE.md` — how to apply the prompt and expected output shape.
- A convenience mirror of the most recent run is kept in `experiments/gepa_gemma3/latest/` with the same filenames for quick pickup.

## Using the evolved prompt

- Treat `optimized_prompt.txt` as your system/instruction prompt.
- Provide two inputs: `instruction` (task) and `context` (theme, style cues, motifs, cadence stats).
- Return a single JSON object with both the writing and a short self‑review, e.g. `{"answer": "...", "reflection": "word count fit, motif coverage, cadence/style adherence, improvements"}`. Motifs must appear verbatim; keep word count within the window in the context.
