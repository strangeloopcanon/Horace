# Horace GRPO Reward Design (v0 → v0.1)

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

---

## v0.1 Upgrades (targeted, high leverage)

These address the biggest risks we saw in v0: proxy collapse to length/punctuation, brittle arc detection, and drift in embedding/cadence alignment. Each item is drop‑in and backwards compatible (fallbacks preserved).

1) Surprise relative to a reference (avoid gibberish)
- Replace absolute surprisal with surprisal delta against a reference policy: Δs_i = s_i − s_i^ref.
- Reward tokens where Δs_i is in a soft band [a,b]; softly penalize persistent overshoots.
- Gate distinctiveness and punctuation penalties using the reference (see 3 and 8).

2) Cadence via DTW on per‑line envelopes
- Z‑score surprisal within each line and compare to a target envelope with length‑robust dynamic time warping (DTW).
- Score lines by exp(−cost/len), then average. Keep our current Pearson fallback if DTW is unavailable.

3) Distinctiveness gated by acceptability
- Credit IDF/PMI only when the token is locally plausible under the reference (s_ref < τ_ref).
- This prevents rewarding rare nonsense while still encouraging specific, content‑bearing choices.

4) Corridor with Huber margins
- Compute per‑line distance d_ℓ to the theme. Use a Huber penalty outside [r_lo, r_hi], averaged over lines.
- Choose δ = (r_hi − r_lo)/8 so the margin is forgiving but discourages long excursions.

5) Deterministic wander/return detector
- Find peaks in the distance trace with a minimum prominence; define arcs as local‑min → peak → return within ε of the min.
- Score amplitude × duration prior; keep the top 1–2 arcs; zero out if the arc violates the corridor.

6) Normalization that’s length‑robust
- Normalize each component to [0,1] using group‑wise z‑scoring and an EMA baseline per (model, preset). Cache under data/analysis/<model>/reward_norm.json.
- Clip with a temperatured sigmoid to resist gaming via length or punctuation density.

7) Entropy band regularizer (soft)
- Track line‑mean entropy and penalize only persistent drift outside [h_lo, h_hi] with an exponential penalty (half‑life k lines), not a hard cap.

8) Early punctuation penalty that scales with ref
- Penalize premature stops only if the reference model assigns low probability to punctuation at that position. This preserves dash‑heavy styles (e.g., Dickinson) when chosen as a preset.

### Concrete score definitions (reference stubs)

```python
def cadence_score(surprisal_by_line, target_env_by_line):
    # z-score per line
    z_lines = [ (x - x.mean()) / (x.std()+1e-6) for x in surprisal_by_line ]
    # DTW cost per line (fallback to correlation if DTW is absent)
    costs = [dtw_cost(z, target_env_by_line[i]) for i, z in enumerate(z_lines)]
    # convert cost to [0,1]
    line_scores = [np.exp(-c/len(z)) for c, z in zip(costs, z_lines)]
    return float(np.mean(line_scores))
```

```python
def surprise_score(s_curr, s_ref, line_spans, a=0.5, b=2.0, max_spikes=2):
    ds = s_curr - s_ref
    scores = []
    for L in line_spans:
        line_ds = ds[L.start:L.end]
        in_band = (line_ds >= a) & (line_ds <= b)
        picked = np.sort(line_ds[in_band])[-max_spikes:]
        reward = np.mean(np.clip((picked - a)/(b - a + 1e-6), 0, 1)) if picked.size else 0.0
        penalty = float(np.mean(line_ds > b)) if np.sum(line_ds > b) > max_spikes else 0.0
        scores.append(np.clip(reward - 0.5*penalty, 0, 1))
    return float(np.mean(scores))
```

```python
def distinctive_score(tokens, idf, pmi, s_ref, content_mask, corridor_mask):
    vals = []
    for i,tok in enumerate(tokens):
        if not(content_mask[i] and corridor_mask[i]):
            continue
        if s_ref[i] >= 3.0:  # implausible under ref
            continue
        vals.append(sigmoid(0.5*idf[tok]) * sigmoid(0.2*pmi.get((tokens[i-1], tok), 0.0)))
    return np.mean(vals) if vals else 0.0
```

```python
def corridor_score(line_embs, theme_emb, r_lo, r_hi, delta):
    d = np.linalg.norm(line_embs - theme_emb, axis=1)
    def huber_margin(x):
        if x <= 0: return 0.0
        return (x**2)/(2*delta) if x <= delta else x - delta/2
    penalties = [huber_margin(di - r_hi) + huber_margin(r_lo - di) for di in d]
    raw = 1.0 - np.mean(penalties) / (r_hi - r_lo + 1e-6)
    return float(np.clip(raw, 0, 1)), d
```

```python
def wander_return_score(d, prominence, eps, A=0.6, lam=0.03, topk=2):
    peaks, props = find_peaks(d, prominence=prominence)
    arcs = []
    for p in peaks:
        m = local_min_left(d, p)
        r = return_index(d, start=p, target=d[m]+eps)
        if r is None:
            continue
        amp = min(1.0, (d[p]-d[m])/(A+1e-6))
        dur = np.exp(-lam*(r-m))
        arcs.append(amp*dur)
    arcs = sorted(arcs, reverse=True)[:topk]
    return float(np.sum(arcs)) if arcs else 0.0
```

Composite (unchanged): R = Σ w·S − regularizers. Normalize each S via EMA mean/std:

```python
def norm01(x, stats):
    z = (x - stats.mean)/(stats.std+1e-6)
    return float(1/(1+np.exp(-z)))
```

### Defaults that work (presets)

We ship conservative defaults and encourage tightening after Phase B:
- freeverse: {cad:0.28, sup:0.18, dis:0.18, cor:0.26, arc:0.10}; kl:0.08
- prose:    {cad:0.22, sup:0.12, dis:0.22, cor:0.34, arc:0.10}; kl:0.06
- sonnet:   {cad:0.22, sup:0.12, dis:0.14, cor:0.22, arc:0.06, rhyme:0.12, meter:0.12}; kl:0.08
- dickinson: {cad:0.30, sup:0.16, dis:0.18, cor:0.26, arc:0.10}; dash‑tolerant punctuation model.

Suggested corridor bands (cosine distance on unit‑norm sentence embeddings):
- prose: [0.10, 0.28]
- freeverse: [0.14, 0.32]
- sonnet: [0.12, 0.26]

Start wide; tighten 10–15% after Phase B.

### Minimal JSONs (examples)

configs/reward_presets_v01.json

```json
{
  "freeverse": {
    "weights": {"cad":0.28,"sup":0.18,"dis":0.18,"cor":0.26,"arc":0.10,"kl":0.08},
    "corridor": {"r_lo":0.14,"r_hi":0.32,"soft_margin":0.02},
    "bands": {"surprise": [0.5,2.0], "entropy": [2.5,4.0]}
  },
  "sonnet": {
    "weights": {"cad":0.22,"sup":0.12,"dis":0.14,"cor":0.22,"arc":0.06,"rhyme":0.12,"meter":0.12,"kl":0.08},
    "corridor": {"r_lo":0.12,"r_hi":0.26,"soft_margin":0.015},
    "rhyme_scheme": "ABAB CDCD EFEF GG",
    "meter": "iambic_pentameter"
  }
}
```

configs/grpo_default_v01.json

```json
{
  "model": "Qwen/Qwen3-1.7B",
  "backend": "mlx",
  "sampler": {"temperature": 0.8, "top_p": 0.95, "top_k": 60, "jitter": 0.03},
  "reward": {"preset": "freeverse", "weights_override": null},
  "prompts": "data/prompts/freeverse_dev.txt",
  "out_dir": "data/grpo_runs/freeverse_v01",
  "train": {"clip": 0.2, "lr": 5e-6, "kl_coef": 0.08, "target_kl": 0.08, "amp": true, "amp_dtype": "bf16"},
  "K": 8
}
```

### Anti‑gaming extensions (keep in backlog)

- Punctuation doping: penalize if terminal punctuation appears before token 6 more often than the reference (by >2σ).
- Rhyme collapse: enforce last‑word entropy per stanza above a floor before awarding rhyme credit.
- Length skew: tiny penalty if line length variance is too small or too large.
- Boilerplate mask: maintain a banlist of cliché n‑grams (PMI < 0 with theme) and zero their distinctiveness credit.
- Copy‑back shield: zero reward if any line’s cosine similarity with the prompt exceeds 0.9.

### GRPO loop stability tweaks

- Group‑wise whitening: z‑score each component across K samples within a prompt before composing R.
- Adaptive KL: update w_kl ← w_kl · exp(η (KL − k*)) with k* as the target KL.
- Sampling: temperature 0.8, top‑p 0.95, small top‑k jitter; do not use the cadence sampler during training.
- Advantages: A_i = R_i − mean(R_group) (optionally subtract a learned value head later).

Implementation hooks
- analyze_stream: emit tokens, lines, s_curr, s_ref, top_p, and punctuation positions.
- Embeddings: use a small sentence encoder (e5‑small/MiniLM) and unit‑normalize lines; cache IDF/PMI in memory‑mapped files.
- Logging: write per‑component scores to JSONL; plot distance traces to validate arc/corridor behavior.
