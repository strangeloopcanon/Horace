# Horace Studio

So what: Horace Studio turns *any* pasted writing into a grounded literary analysis — a **score (0–100)**, a **profile vs “top literature” baselines**, concrete **suggestions**, and optional **rewrites reranked by Horace metrics**.

This is an early prototype meant to make iteration fast and UX-driven; the interpretive layer can later be swapped from heuristics to an LLM-based critic.

## Local UI (prototype)

1) Install deps:
```bash
make setup
```

Optional (recommended): activate the venv created by `make setup`:
```bash
source .venv/bin/activate
```

2) Run:
```bash
python tools/studio_ui.py --host 127.0.0.1 --port 7861
```

Or:
```bash
make run-ui
```

What you get:
- **Analyze** tab: score + sub-scores, metric percentiles, suggestions, spike excerpts, cadence plot.
- Optional: **Trained scorer (fast)**: a single HF model that can score text without token-level analysis (enable “Fast mode” + provide a model path).
- **Rewrite + Rerank** tab: generates N candidates and reranks them by Horace score (slow).
- **Patch (dead zones)** tab: finds flat-cadence spans (repetition / droning density) and proposes **meaning-preserving** span patches (MeaningLock + diffs). Use the **Intensity** knob (clearer ↔ punchier) to steer how candidates are ranked.
- **Cadence Match** tab: generate text that matches the cadence profile of a reference passage (cadence-only, not voice copying).
- Optional: **LLM Critique** accordion for a non-deterministic editor voice (grounded in measured metrics/spikes).
- **Formatting normalization (default on for prose)**: fixes hard-wrapped plaintext (single newlines), common in Gutenberg/RFC copies; preserves newlines when the input looks like code, lists, or verse.

To quickly get a local fast scorer to play with:
```bash
make train-scorer-distill-v4-smoke
```
Then set **Scorer model path** to `models/scorer_v4_distill_smoke`.

## Local API (for a real web frontend)

Run:
```bash
python -m tools.studio_api --host 127.0.0.1 --port 8000
```

Or:
```bash
make run-api
```

Then open:
`http://127.0.0.1:8000/`

API docs:
- Human: `http://127.0.0.1:8000/api`
- OpenAPI: `http://127.0.0.1:8000/docs`

Optional auth: set `HORACE_API_KEY` to require `Authorization: Bearer <key>` or `X-API-Key: <key>` for non-doc routes.

Health:
```bash
curl http://127.0.0.1:8000/healthz
```

Analyze:
```bash
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","doc_type":"prose","scoring_model_id":"gpt2","baseline_model":"gpt2_gutenberg_512","backend":"auto"}'
```

Fast scoring only (single trained model; skips token analysis):
```bash
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","scorer_model_path":"models/scorer_v4_distill_smoke","fast_only":true}'
```

Optional anti-pattern guardrail (same API shape, adjusted primary score):
```bash
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","primary_score_mode":"rubric","antipattern_model_path":"models/scorer_v5_authenticity_v1","antipattern_penalty_weight":0.85,"antipattern_penalty_threshold":0.85,"antipattern_combiner_mode":"adaptive","apply_antipattern_penalty":false}'
```

`apply_antipattern_penalty=false` (default) returns dual scores: quality + authenticity risk without blending them into one number.
`scorer_model_path` is reserved for quality scorers; anti-pattern/authenticity checkpoints are rejected there and must go in `antipattern_model_path`.

Span patching (dead zones → MeaningLock → diffs):

Suggest dead zones:
```bash
curl -s http://127.0.0.1:8000/patch/suggest \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","doc_type":"prose","scoring_model_id":"gpt2","normalize_text":true}'
```

Patch one span (use a zone’s `start_char`/`end_char`):
```bash
curl -s http://127.0.0.1:8000/patch/span \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","doc_type":"prose","start_char":0,"end_char":28,"intensity":0.55,"rewrite_model_id":"Qwen/Qwen2.5-0.5B-Instruct","meaning_lock_min_cosine_sim":0.86}'
```

Optional: apply a learned calibrator (trained from eval reports):
```bash
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","calibrator_path":"reports/calibrators/calibrator.json"}'
```

Non-deterministic critique (optional; downloads a model):
```bash
curl -s http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","use_llm_critic":true,"critic_model_id":"Qwen/Qwen2.5-0.5B-Instruct"}'
```

Cadence match (reference-guided generation):
```bash
curl -s http://127.0.0.1:8000/cadence-match \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The morning light crept through the window","reference_text":"Paste a reference paragraph here.","doc_type":"prose","model_id":"gpt2","max_new_tokens":200}'
```

## Baselines (reference “top literature” bands)

Baselines are built from `data/analysis/<model>/docs_clean.jsonl` and stored as JSON under `data/baselines/`.

Build a baseline:
```bash
python -c "from tools.studio.baselines import build_baseline; build_baseline('gpt2')"
```

The UI will also auto-build a baseline if it’s missing and `data/analysis/<model>/docs_clean.jsonl` is present.

Important caveat: the local corpus currently mixes **full texts** and **short summaries** for prose, which can skew the “top literature” band toward encyclopedic/summary-style writing. For a baseline closer to “real literary prose windows”, build one from Project Gutenberg excerpts:

```bash
python -m tools.studio.build_baseline_web --model-id gpt2 --max-input-tokens 512 --n 200 --gutenberg-books 200 --gutenberg-excerpts-per-book 1
```

Or:
```bash
make build-baseline-web MODEL=gpt2
```

Then point Studio’s **Baseline model id (or baseline JSON path)** at the emitted JSON file (e.g. `data/baselines/gpt2_gutenberg_512_docs.json`).

## Personal baselines (your own writing)

Personal baselines are a low-friction step toward personalization: they don’t train a model, they build a distribution over the same metrics for *your* corpus.

Build from a folder of `.txt` files:
```bash
python -m tools.studio.personal_baseline --input path/to/my_texts --model-id gpt2 --doc-type prose
```

Then, in the Studio UI, set **Baseline model id (or baseline JSON path)** to the emitted JSON path (e.g. `data/baselines/personal_gpt2_prose_docs.json`).

## Modal API (serve to the public)

`deploy/modal/horace_studio.py` exposes:
- `POST /analyze`
- `POST /rewrite`
- `POST /cadence-match` (alias: `POST /write-like`)

It mounts `tools/` and `data/baselines/` into the container and expects baseline JSONs to be present.

Modal setup (first time):
```bash
make setup-modal
make modal-token
```

Before deploy, make sure you have a baseline file locally (it gets mounted into the container):
```bash
make build-baseline MODEL=gpt2
```

Smoke-test a remote call (runs on Modal, prints score JSON):
```bash
.venv/bin/modal run deploy/modal/horace_studio.py --text "At dawn, the city leans into light."
```

Score URLs (trained scorer on Modal; optional rubric breakdown computed locally):
```bash
.venv/bin/modal run deploy/modal/score_urls_qwen3.py --urls https://example.com/a,https://example.com/b
.venv/bin/modal run deploy/modal/score_urls_qwen3.py --urls https://example.com/a --include-rubric
```

Note: rubric breakdown is **windowed** for long inputs (multiple slices across the document), which reduces “intro bias” from only scoring the first N tokens. The JSON includes a `rubric_windows` section with per-window summaries.

Tip: if you want machine-readable JSON on stdout, use Modal’s quiet flag to suppress progress output:
```bash
.venv/bin/python -m modal run -q deploy/modal/score_urls_qwen3.py --urls https://example.com/a --include-rubric > out.json
```

Tip: if `modal ...` fails with `ModuleNotFoundError: No module named 'modal'`, you’re almost certainly invoking a global `modal` binary. Use `make modal-token`/`make modal-…` targets, or run Modal via your venv:
```bash
.venv/bin/python -m modal run deploy/modal/horace_studio.py --text "At dawn, the city leans into light."
```

Fast scoring on Modal (skip token-level analysis; uses a single trained scorer model):
```bash
.venv/bin/modal run deploy/modal/horace_studio.py \
  --text "At dawn, the city leans into light." \
  --scorer-model-path /vol/models/scorer_hybrid_v1 \
  --fast-only true
```

Deploy the web app (serves `GET /` plus the API):
```bash
.venv/bin/modal deploy deploy/modal/horace_studio.py
```

Optional: Hugging Face token (for gated models). Create a Modal secret and add it to the app later when needed:
```bash
.venv/bin/modal secret create huggingface HUGGINGFACE_HUB_TOKEN=YOUR_TOKEN
```

## Modal training (later: your own models)

Horace already includes HF training loops (e.g. GRPO). The wrapper:
- `deploy/modal/grpo_full_train.py`

runs GRPO full training on a GPU and persists:
- outputs under a Modal volume (`/vol`)
- HF cache under a Modal volume (`/cache/hf`)

Example:
```bash
.venv/bin/modal run deploy/modal/grpo_full_train.py --config configs/grpo_full_hf_smoke.json --out-dir /vol/grpo_runs/demo --limit-steps 10
```

Distill the (slow) rubric into a single fast scorer (Qwen3 + LoRA example):
```bash
.venv/bin/python -m modal run -q deploy/modal/studio_distill_scorer_mixed.py \
  --out-dir /vol/models/scorer_mixed_distilled_qwen3_rubricv2_v1 \
  --teacher-model Qwen/Qwen2.5-1.5B --teacher-max-input-tokens 512 \
  --base-model Qwen/Qwen3-1.7B --scorer-max-length 512 --scorer-batch-size 1 --scorer-lr 1e-4 --scorer-epochs 1 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --grad-accum-steps 16 --gradient-checkpointing --bf16 --merge-lora
```

### Modal “train” for the scorer (now: calibrator)

Before training a full reward model, you can train a tiny **calibrator** that learns a better overall score from the rubric outputs (still deterministic at inference).

Run a combined eval + calibrator train job on Modal:
```bash
.venv/bin/modal run deploy/modal/studio_train_calibrator_web.py --out-dir reports/calibrators --max-input-tokens 512
```

Or:
```bash
make modal-train-calibrator-web
```

### Modal training for the single scorer model (v4)

Build the within-domain benchmark + train a single text→score model on Modal (GPU):
```bash
make modal-train-scorer-v4
```

### Modal training for the anti-pattern scorer (v5)

The v5 anti-pattern scorer is now optimized around a reward-model style base.
To retrain the model on the same dataset used by `make train-scorer-v5-antipattern`,
use Skywork Reward V2 on Modal:

```bash
make modal-train-scorer-v5-antipattern-skywork
```

Useful overrides:
- `V5_SCORER_BASE_MODEL` (default `Skywork/Skywork-Reward-V2-Qwen3-1.7B`)
- `V5_SCORER_BATCH_SIZE` (default `1`)
- `V5_SCORER_LORA_R` (default `16`)
- `V5_SCORER_LORA_ALPHA` (default `32`)
- `V5_SCORER_EPOCHS` (default `1`)

Example (legacy Qwen alternative):
```bash
make modal-train-scorer-v5-antipattern-qwen3 \
  V5_SCORER_BASE_MODEL=Qwen/Qwen3-4B \
  V5_SCORER_MAX_LENGTH=512 \
  V5_SCORER_BATCH_SIZE=1
```
Example (larger reward model):
```bash
make modal-train-scorer-v5-antipattern-skywork \
  V5_SCORER_BASE_MODEL=Skywork/Skywork-Reward-V2-Qwen3-4B \
  V5_SCORER_OUT_DIR_SKYWORK=/vol/models/scorer_v5_antipattern_skywork_v2 \
  V5_SCORER_MAX_LENGTH=512 \
  V5_SCORER_BATCH_SIZE=1
```

### Modal distillation for the scorer (recommended)

So what: distill the **slow** rubric score into a **fast** scorer (teacher = rubric; student = encoder).

Run on Modal (GPU; writes to `/vol/models/scorer_v4_distilled`):
```bash
make modal-distill-scorer-v4
```

### Modal hybrid scorer training (distill + cadence preferences)

So what: the rubric teacher is useful but incomplete; we add *within-content* preference supervision so the scorer learns cadence/voice directly.

This trains one scorer model in two phases:
- **Distill**: rubric teacher → encoder scorer (text → scalar).
- **Preference fine-tune**: original excerpt > “dulled rewrite” (meaning preserved; style flattened) and > deterministic cadence corruptions.

Run on Modal (GPU; writes to `/vol/models/scorer_hybrid_v1`):
```bash
make modal-train-scorer-hybrid
```

Tip: Modal pins `torch==2.5.1+cu121`; if your `base_model` only ships `pytorch_model.bin` (no `model.safetensors`), Transformers will refuse to load it. `roberta-base` and `distilbert-base-uncased` work.

### Anti-pattern training (human originals vs LLM imitations)

So what: this adds hard negatives from high-quality AI imitation, reducing false highs on AI-like prose while keeping the existing Studio UX and API shape.

Build originals:
```bash
make build-antipattern-originals
```

Generate pairs (randomized provider/model across OpenAI, Gemini, Anthropic):
```bash
make build-antipattern-pairs
```

Batch-first OpenAI path (cost control):
```bash
make build-antipattern-pairs-openai-batch
```

After downloading OpenAI Batch results:
```bash
make merge-antipattern-openai-batch ANTIPATTERN_OPENAI_BATCH_RES=data/antipattern/openai_batch_results_v1.jsonl
```

Train with anti-pattern negatives:
```bash
make train-scorer-v5-antipattern
make train-scorer-v5-antipattern-skywork   # explicit alias (same as default now)
make train-scorer-v5-antipattern-distilbert  # legacy baseline
```

Held-out AI-overfit eval:
```bash
make eval-ai-overfit
```

## Web evals (random + curated)

So what: if the product promise is “does this match top-literature patterns?”, we need a quick way to sanity-check whether the score separates:
- literature-ish text (Project Gutenberg excerpts)
- encyclopedic prose (Wikipedia summaries)
- technical prose (RFC excerpts)
- nonsense controls (gibberish)

Run locally:
```bash
make eval-web
```

Run on Modal (recommended if local is slow):
```bash
.venv/bin/modal run deploy/modal/studio_eval_web.py --out-dir reports/studio_eval_modal_big --wikipedia 12 --gutenberg 12 --rfc 6 --gibberish 12 --max-input-tokens 512
```

### Fixed eval set (regression-friendly)

So what: random web samples drift; a fixed snapshot lets us regression-test changes to preprocessing/scoring/training.

Build the fixed set (writes `data/eval_sets/studio_fixed_v1.jsonl`):
```bash
make build-eval-set
```

Evaluate it:
```bash
make eval-set
```

Split it into train/val/test with **no group leakage** (e.g. no NASA post or Gutenberg book appears in multiple splits):
```bash
make split-eval-set
```

Train a tiny learned calibrator (optional; **held-out**: trains on train split, reports on test split):
```bash
make train-calibrator-eval-set
```

Debug-only (tainted) calibrator run:
```bash
make train-calibrator-eval-set-tainted
```

### Larger benchmark set (enough data for real held-out evals)

So what: the fixed set above is intentionally small; for stable held-out numbers, build a larger benchmark snapshot.

Build (downloads public sources; writes `data/benchmarks/studio_benchmark_v3/`):
```bash
make build-benchmark-set
```

It produces:
- All samples: `data/benchmarks/studio_benchmark_v3/samples.jsonl`
- Splits: `data/benchmarks/studio_benchmark_v3/splits/train.jsonl`, `val.jsonl`, `test.jsonl` (+ `manifest.json`)

Modal equivalents:
```bash
make modal-eval-set
make modal-train-calibrator-eval-set
```

Outputs:
- Samples JSONL: `reports/**/studio_eval_web_samples.jsonl`
- Scored report JSON: `reports/**/studio_eval_web_report.json`

Current status: with formatting normalization + a Gutenberg-derived baseline, the deterministic score separates literature-ish text from encyclopedia/news/technical prose on average (AUC ~0.83 on the fixed set). The calibrator can improve separation, but you should only trust **held-out** numbers (and the current fixed set is still small, so variance is high).

## Roadmap (UX-first)

<details>
<summary>Click to expand</summary>

1) **Ship MVP UX**: paste → score/profile/suggestions; fast; predictable caps; clear explanations.
2) **Public deployment**: Modal FastAPI + a thin web frontend; rate limiting; basic abuse controls.
3) **Non-deterministic critic**: LLM-generated critique that is *grounded in measured metrics* (structured output, cites spikes, temperature > 0).
4) **Rewrite quality**: train an editor to improve Horace score *while preserving meaning* (semantic constraints + rerank + evals).
5) **Personalization**: per-user baselines + (later) finetunes; job queue; checkpoints stored in volumes; rollback-ready.
6) **Evals & monitoring**: golden sets, latency/cost budgets, regression tests, safety checks.

</details>
