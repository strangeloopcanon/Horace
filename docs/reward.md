# Horace GRPO Reward Design (v0)

Scope: turn Horace’s cadence analysis into a simple, toggleable reward for GRPO to teach models to write with rhythm, surprise, and purposeful exploration — without over‑specifying grammar or aesthetics.

## Philosophy
- Style-first, author-light: optimize for form/genre presets (sonnet, Dickinson-like, free‑verse, prose). Optionally layer light author cadence signatures later.
- Keep v0 lean: start with five core signals and regularizers; add rhyme/meter and other plug-ins only when needed.
- Exploration without drift: avoid cosine-to-prompt maximization; use a corridor + wander/return arcs to preserve topical intent while allowing excursions.

## Core Signals (v0)
1) Cadence (base→spike→cooldown)
- Per-token entropy/surprisal envelope match per line/clause.
- Small penalty for too-early punctuation; encourage resets.

2) Surprise (controlled bursts)
- Reward 1–2 high-surprisal tokens per line; penalize fully flat or continuously spiky lines.

3) Distinctive Choice (novel but plausible)
- Reward high-IDF content words and rare-but-coherent collocations (PMI-based), softly gated by topic.

4) Coherence Corridor (not cosine maximization)
- Maintain per-line embedding distance to a theme centroid within [r_lo, r_hi] with soft margins.

5) Wander/Return Arc
- Detect excursions (distance peaks) that later reconnect to the theme; score 1–2 clean arcs per stanza/document.

Regularizers (always on)
- KL to reference policy, safety/content gates, light no-repeat n‑gram/length/format constraints.

## Optional Plug-ins (toggleable)
- Rhyme/Meter: phonetic rhyme score and meter fit (slant allowed). Useful for sonnets/couplets.
- Repetition & Safety: stricter repetition control, toxicity/NSFW blocks.
- Format: line/stanza/length requirements; e.g., 14 lines for sonnet.

## Composite Reward
R = w_cad·S_cad + w_sup·S_sup + w_dis·S_dis + w_cor·S_corridor + w_arc·S_arc − w_kl·KL − w_saf·Penalty − w_fmt·Penalty (+ optional plugin terms)

Guidelines
- Toggle-friendly: set unused weights to 0 per preset or CLI.
- Normalize each component to [0,1] via baseline min–max/z-score calibrated on current model.
- Keep weight sum ≈ 1 for interpretability.

## GRPO Loop (practical)
1) For each prompt, sample K responses from current policy (simple decoding; do not use Horace sampler at train time).
2) Compute token-level stats from logits (entropy, surprisal), per-line embeddings, and text stats for distinctiveness.
3) Score using the composite reward; advantage = R_i − mean(R_group).
4) PPO/GRPO update with clipping + adaptive KL to reference policy.

Curriculum
- Phase A: Cadence + Surprise + Corridor.
- Phase B: add Distinctive; widen/warm corridor.
- Phase C (style-bound): enable Rhyme/Meter plug-ins (e.g., sonnet), then tighten KL to preserve fluency.

Anti-gaming
- Pair signals: give Distinctive credit only within the topical mask; deny credit if corridor violated.
- Entropy bands: penalize sustained max-entropy (“word salad”) or min-entropy (“template flatness”).
- Rhyme diversity (when enabled): deny rhyme credit if last-word distribution collapses.

## Minimal API (stubs now added)
- `tools/reward.py`
  - `PresetConfig`, `RewardWeights`, `compute_reward(sample, preset, refs, toggles)`
  - Component scorers: `cadence_score`, `surprise_score`, `distinctive_score`, `corridor_score`, `wander_return_score`
  - Normalization and composite assembly.
- `configs/reward_presets.json`
  - Default weights and corridor bands for `sonnet`, `dickinson`, `freeverse`, `prose`.
- `tools/grpo_runner.py` (MLX-only rollouts)
  - Samples K diverse responses via a simple MLX sampler (temperature/top_p/top_k + jitter), computes rewards and group-relative advantages, and saves to a JSONL for PPO/GRPO trainers.
  - Configured via `configs/grpo_default.json` (default model: Qwen3‑1.7B).

## Data/Models
- Embeddings: small sentence embedding model; theme = centroid(prompt + first 1–2 lines).
- IDF/PMI: static frequency table (e.g., wordfreq/Zipf) to avoid external calls.
- Cadence targets: reuse Horace’s preset envelopes from analysis or hand-tuned JSON.

## Next Steps (pick up here)
1) Wire `analyze_stream` from `tools/analyze.py` to compute per-token entropy/top‑p and punctuation positions.
2) Implement per-line embedding pipeline and distance traces.
3) Implement normalization using baseline caches (save stats per preset/model under `data/analysis/<model>/reward_norm.json`).
4) Add rhyme/meter plug-ins (optional) using CMUdict/g2p; keep weights at 0 by default.
5) Integrate with a PPO/GRPO trainer (TRL/OpenRLHF) — start offline scoring, then move on-policy.
6) Add CLI toggles to enable/disable components per run; log full component breakdowns.
7) Validate on a small prompt set; iterate corridor bands and cadence envelopes.

## Using the GRPO runner
- Edit `configs/grpo_default.json` (model, sampler, reward preset/weights, prompts, out_dir).
- Run rollouts (MLX backend by default):
  - `python tools/grpo_runner.py --config configs/grpo_default.json`
- Outputs: `data/grpo_runs/<run>/rollouts.jsonl` with per-prompt groups, rewards, advantages, and component breakdowns.
 - To evaluate a trained checkpoint with MLX, set `model` in the config to the local checkpoint directory saved by full training (HF checkpoints are loadable by id or local path), and keep `backend: mlx`.

## GRPO Training (logit-bias adapter)
- Add training hyperparams in `configs/grpo_default.json` under `train`.
- Train: `python tools/grpo_train.py --config configs/grpo_default.json`
- The run updates a vocab-sized bias vector applied to logits during sampling, acting as a compact policy adapter trained with GRPO.
- Checkpoints: `data/grpo_runs/<run>/logit_bias.npy` and `train_log.jsonl` (avg reward, grad norm, bias norm).

## Full GRPO Training (HF/PyTorch)
- For full model updates (no LoRA/adapters):
  - Ensure `backend` is `hf` in your config and set a reasonable learning rate (e.g., 5e-6) and clip=0.2.
  - Run: `python tools/grpo_full_train.py --config configs/grpo_default.json`
- This performs on-policy rollouts and PPO-style updates to base weights, saving checkpoints under `out_dir`.

Advanced (stability/speed)
- Mixed precision: set `train.amp=true` and `train.amp_dtype` to `bf16` or `fp16` (CUDA only for scaling). On MPS/CPU, amp falls back safely.
- Gradient accumulation: set `train.grad_accum` to accumulate steps before each optimizer step.
- Reference KL: set `train.kl_coef>0` to penalize KL(new||ref) w.r.t. a frozen copy of the base model.

Open Questions
- How wide should corridor bands be per preset/topic? (Collect stats.)
- Best way to detect and score “reframes” vs literal returns in arcs?
- When enabling meter, how strict should stress alignment be for modern free-verse variants?
