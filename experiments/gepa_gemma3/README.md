# GEPA Prompt Optimization Toolkit

This directory contains reusable tooling to optimise prompts for Gemma models using GEPA.
The workflow is now driven by a single generic runner that loads a configuration module
(Writing, JSON judge, or your own) and handles model wiring, GEPA execution, and reporting.

## Quick start

```bash
pip install -r experiments/gepa_gemma3/requirements.txt

# Writing prompt optimisation (Horace author set)
python experiments/gepa_gemma3/run_gepa_generic.py \
  --config experiments.gepa_gemma3.configs.writing \
  --train 12 --val 6 --max-metric-calls 90 \
  --task-model gemma-3-4b-it --reflect-model openai/gpt-4.1-mini

# Structured JSON judge prompt (verdict tasks)
python experiments/gepa_gemma3/run_gepa_generic.py \
  --config experiments.gepa_gemma3.configs.json_judge \
  --max-metric-calls 45 --task-model gemma-3-4b-it
```

`TASK_MODEL` can point to `gemma-3-4b-it`, `aistudio/<model>`, `local:<hf-id>`, or an OpenRouter
identifier. The reflection model defaults to `openai/gpt-4.1-mini` but honours the
`--reflect-model` flag or the `REFLECT_MODEL` environment variable. Ensure the relevant API keys
(GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, …) are present in `.env`.

## Outputs

Each run writes a timestamped directory under `experiments/gepa_gemma3/outputs/<config_name>/`:

- `seed_prompt.txt` – the initial instruction passed to the runner
- `optimized_prompt.txt` – the best prompt discovered by GEPA
- `results.json` – metadata, score summaries, and per-example evaluation details

The `latest/` symlink is no longer used; inspect the timestamped directories directly.

## Supplied configurations

Two configuration modules live in `experiments/gepa_gemma3/configs/`:

- `writing.py` – loads the Horace author dataset (`data/tasks/gepa_writing_tasks.json`), implements
the literary metric, and prepares the `Writer` module used previously.
- `json_judge.py` – reuses the strict JSON tasks and metric shared with the evaluation harness.

Each config exposes the following hooks that the generic runner consumes:

```python
build_datasets(train_count: int, val_count: int, seed: int) -> tuple[list[dspy.Example], list[dspy.Example]]
build_student(seed_instruction: str) -> dspy.Module
get_seed_prompt() -> str
metric() -> Callable[[dspy.Example, dspy.Prediction], ScoreWithFeedback]
prediction_to_dict(prediction: dspy.Prediction) -> dict
```

To add a new domain, create a module that implements the same interface—load your golden tasks,
define an evaluation metric that emits `ScoreWithFeedback`, and specify how to serialise
predictions for the results file. Point `run_gepa_generic.py` at the new module via `--config`.

```python
# experiments/gepa_gemma3/configs/my_domain.py
import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

class MySignature(dspy.Signature):
    instruction = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()

class MyModule(dspy.Module):
    def __init__(self, seed):
        super().__init__()
        self.step = dspy.Predict(MySignature, instructions=seed)

    def forward(self, instruction, context):
        out = self.step(instruction=instruction, context=context)
        return dspy.Prediction(answer=out.answer)

def get_seed_prompt() -> str: ...
def build_datasets(train_count, val_count, seed): ...  # returns dspy.Example lists
def build_student(seed_instruction: str) -> dspy.Module: return MyModule(seed_instruction)
def metric(): ...  # return callable producing ScoreWithFeedback
def prediction_to_dict(pred: dspy.Prediction) -> dict: return {"answer": pred.answer}
```

## Evaluation scripts

For regression checks across inference backends (HF, MLX, AI Studio), use:

- `eval_structured.py` (MLX / MLX-genkit)
- `eval_structured_hf.py`
- `eval_structured_aistudio.py`

The shared task logic lives in `json_eval_common.py`, ensuring all backends hit the identical
prompt and schema constraints.

## Cleaning up artefacts

Generated GEPA outputs reside entirely under `experiments/gepa_gemma3/outputs/`; evaluation logs
match `reflection_eval_*.jsonl` and are covered by `.gitignore`. Local model downloads belong in
`models/` (also ignored).
