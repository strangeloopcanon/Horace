# Horace — Writing Signatures & Cadence Sampler

Horace is a compact toolkit to:

- Measure token‑level distributions and cadence signatures from causal LMs (GPT‑2, Qwen, etc.).
- Generate better prose/poetry with a cadence‑aware sampler that inserts purposeful spikes and cool‑downs.

It prefers Apple Silicon (MLX) and falls back to Hugging Face + PyTorch.

## Table of Contents

- Overview
- Quick Links
- Install
- Data Layout
- Analysis Pipeline
- Cadence Sampler
- Reports & Final Book
- Usage Examples
- Troubleshooting

## Overview

- Analysis: Per‑token probabilities, entropy, true‑rank, nucleus width, and top‑k; per‑doc/author cadence metrics (IPI, MASD/ACF/FFT, cooldown entropy drop), distributional stats, cohesion shuffle delta, token‑class context.
- Cadence Sampler: Token‑aware controller enforcing Base → Spike → Cooldown with per‑phase `top_p` + temperature; defers spikes on punctuation, aims rhyme at line ends, adds repetition controls and a diversity bonus on spikes.
- Reports: Per‑model and cross‑model dashboards; curated “Final” book with narrative, illustrations, and before/after snippets.
- Why: Great writing isn’t random or safe — it rides a rhythm of focused choices punctuated by turns; we measure those patterns and then make the model follow them.

## Quick Links

- [Final Report (HTML, self‑contained)](reports/final/report.html)
- [Final Report (DOCX)](reports/final/report.docx)
- [Final Report (Markdown)](reports/final/README.md)
- [Model Report — Qwen/Qwen2.5‑1.5B](reports/Qwen/Qwen2.5-1.5B/README.md)
- [Model Report — GPT‑2](reports/gpt2/README.md)
- [Cross‑Model Compare (GPT‑2 vs Qwen)](reports/compare_gpt2_vs_Qwen_Qwen2.5-1.5B/README.md)
- [All Generated Before/After Examples](reports/generated/README.md)
- [GRPO Reward Plan (v0)](docs/reward.md)

## GRPO Rollouts (MLX)
- Configure: `configs/grpo_default.json` (defaults to `Qwen/Qwen3-1.7B`, MLX backend, simple diverse sampler).
- Run rollouts + rewards: `python tools/grpo_runner.py --config configs/grpo_default.json`
- Output: `data/grpo_runs/<run>/rollouts.jsonl` with group rewards, advantages, and component breakdowns.

## GRPO Training (Adapter)
- The repo includes a lightweight GRPO trainer that optimizes a vocab-sized logit-bias adapter (keeps base model fixed; MLX rollouts).
- Configure training under `train` in `configs/grpo_default.json`.
- Start training: `python tools/grpo_train.py --config configs/grpo_default.json`
- Checkpoints/logs in the configured `out_dir`.

## Full GRPO Training (HF)
- Train the base model weights (no LoRA/adapters):
- Set `backend: hf` and adjust `train` hyperparams in `configs/grpo_default.json` (e.g., `lr: 5e-6`, `clip: 0.2`).
- Start: `python tools/grpo_full_train.py --config configs/grpo_default.json`
- Checkpoints are saved under the configured `out_dir`.
- Options: enable `train.amp` (`bf16`/`fp16` on CUDA), set `train.grad_accum` for accumulation, and `train.kl_coef` to add a reference-KL penalty.

## Data Layout

- Input index: `data/index.json` with `{type, author, title, path}` pointing to `.txt` files.
- Per‑token capture: chosen token probability, logprob, rank, entropy, effective support `exp(H)`, nucleus width `w_p`, `topk_ids`/`topk_probs`, cumulative mass, tail mass, offsets, and metadata.
- Aggregations: per‑doc metrics (means/medians/percentiles, cadence & cohesion shuffle delta) and per‑author rollups.
- Storage: `data/analysis/<model_id>/...` for tokens, docs, authors.
- Cadence suite: surprisal CV/MASD/ACF, FFT peak period, high‑surprise IPI, run lengths, permutation entropy, Hurst (R/S).

Requirements
- Python 3.9+ recommended
- MLX (Apple Silicon): `pip install mlx mlx-lm`
- Or HF/PyTorch fallback: `pip install -r requirements.txt`

Note: The CLI auto‑prefers MLX; if MLX imports fail, it falls back to HF/PyTorch.

## Install

- Python 3.9+
- MLX path (Apple Silicon): `pip install mlx mlx-lm`
- HF/PyTorch fallback: `pip install -r requirements.txt`

## Quickstart (GPT‑2 on MLX)
- Install MLX:
  - `pip install mlx mlx-lm`
- Run end‑to‑end with config:
  - `python tools/analyze.py run --config configs/gpt2_mlx.json`

This executes: `init` → `tokens` → `docs` → `authors` for model `gpt2`.

## Running with Hugging Face (fallback)
- `pip install -r requirements.txt`
- `python tools/analyze.py run --config configs/gpt2_mlx.json`

The command is the same; it will print a message if MLX is unavailable and use HF/PyTorch instead.

Configs
- `configs/gpt2_mlx.json`
  - `{ "model": "gpt2", "k": 10, "p": 0.9, "context": 1024, "stride": 896, "limit": null }`
- Subset configs by type:
  - `configs/gpt2_poems.json` → only poems
  - `configs/gpt2_shortstories.json` → only short stories
  - `configs/gpt2_novels.json` → only novels
- `configs/qwen_base.json` (example; adjust model id as needed)
  - `{ "model": "Qwen/Qwen2.5-1.5B", "k": 10, "p": 0.9, "context": 1024, "stride": 896, "limit": 5 }`

Change `limit` to a small number for smoke tests before full runs.

## CLI Reference
- `python tools/analyze.py init --model gpt2 --k 10 --p 0.9 --context 1024 --stride 896 [--backend auto|mlx|hf]`
  - Writes `data/analysis/<model_id>/run_meta.json`.
- `python tools/analyze.py tokens --model gpt2 [--limit N] [--types poem,shortstory,novel] [--resume] [--force] [--backend auto|mlx|hf]`
  - Reads `data/index.json`, emits per‑token JSONL.GZ and per‑doc aggregates.
- `python tools/analyze.py docs --model gpt2 [--types poem,shortstory,novel]`
  - Dedupes to `docs_clean.jsonl`.
- `python tools/analyze.py authors --model gpt2 [--types poem,shortstory,novel]`
  - Aggregates to `authors.jsonl`.
- `python tools/analyze.py run --config <config.json>`
  - Executes the full sequence above.

## Reports
- Generate a Markdown report with visualizations:
  - `python tools/report.py --model gpt2`
  - Outputs to `reports/gpt2/README.md` with PNGs alongside.
  - Requires `matplotlib` (`pip install matplotlib`).
 - Export a PDF with the same figures (one per page):
   - `python tools/make_pdf.py --model gpt2`
   - Writes `reports/gpt2/report.pdf`.

## Outputs
- `data/analysis/<model_id>/run_meta.json`: model/tokenizer info and params.
- `data/analysis/<model_id>/tokens/<doc_id>.jsonl.gz`: per‑token records.
- `data/analysis/<model_id>/docs.jsonl` → `docs_clean.jsonl`: per‑doc signatures.
- `data/analysis/<model_id>/authors.jsonl`: per‑author signatures.

## Per‑Token Schema (high‑level)
- token: `token_id`, `token_str`, `char_start`, `char_end`
- probabilities: `p_true`, `logp_true`, `rank`, `entropy`, `effective_support`, `nucleus_width`
- alternatives: `topk_ids`, `topk_probs`, `cum_mass_topk`, `tail_mass`
- metadata: `doc_id`, `doc_type`, `author`, `title`, `model_id`, `token_index`

## Tips & Notes
- Keep `k` and `p` fixed across models for comparability (defaults: k=10, p=0.9).
- Context/stride: larger context with 25–50% overlap reduces boundary effects; defaults are tuned for GPT‑2.
- Cohesion metric: includes `cohesion_delta = shuffled − original` (negative ⇒ stronger cohesion).
- Determinism: models run in eval mode with temperature=1.0; we store `model_id` and tokenizer identifiers for reproducibility.
 - Resume runs: use `--resume` to skip already‑processed docs; `--force` to recompute.
 - Backend: `--backend mlx` to force MLX; `--backend hf` for HF/PyTorch; `auto` prefers MLX.

## Running Multiple Models
- Run GPT‑2:
  - `python tools/analyze.py run --config configs/gpt2_mlx.json`
- Run Qwen base (HF unless supported by MLX):
  - `python tools/analyze.py run --config configs/qwen_base.json`

After both complete, compare per‑doc or per‑author signatures across `data/analysis/gpt2/` and `data/analysis/<qwen_id>/`.

## Troubleshooting
- MLX not found: install with `pip install mlx mlx-lm`. On first run, models download and cache.
- Slow runs: use `limit` in config for small smoke tests; then remove `limit` for full runs.
- Memory: for large models on HF/PyTorch, consider smaller context, larger stride, or 8‑bit loading (not yet wired here).

## Next Steps
- Add optional Parquet outputs for faster analytics.
- Add cadence metrics alongside generated snippets in the final book.
- Extend MLX backend coverage (e.g., for additional base models) as mlx‑lm support grows.

## Cadence Sampler (generation)
- Compare normal vs fixed‑up using the token‑aware controller (HF backend):
  - `python -m tools.gen_compare --model Qwen/Qwen2.5-1.5B --preset imagist --manual-fixed --task "Write an imagist poem with clear, concrete images." --prompt "At dawn, the city leans into light:\n" --max-new-tokens 90 --seed 303 --save`
- One‑off sampler (HF/MLX) with presets: `python -m tools.sampler <model> --backend <hf|mlx> --preset <poetry_default|sonnet|dickinson|freeverse|couplets> --prompt "..." --max-new-tokens 120`
- Saved outputs go to `data/generated/...` and aggregate at `reports/generated/README.md`.

## Make the Final Book
- Build curated README, DOCX, PDF, and one‑page HTML, pulling in figures and latest samples:
  - `python -m tools.finalize --model Qwen/Qwen2.5-1.5B --out-readme reports/final/README.md --out-docx reports/final/report.docx --out-pdf reports/final/report.pdf`
  - One‑page HTML at `reports/final/report.html` embeds images for easy sharing.
