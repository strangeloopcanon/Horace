# Learning Signatures of Good Writing — Final Report

## Executive Summary

Great prose and poetry ride a cadence: mostly focused choices, punctuated by purposeful spikes of surprise that turn the scene or idea, followed by a short cooldown that grounds what just happened. Spikes align with content words and rhetorical pivots (not punctuation), with larger sustained shifts every few sentences or lines.

Our analysis measured token-level distributions (p_true, entropy, rank, nucleus width), cadence statistics (spike rate, inter-peak intervals, cooldown entropy drop), cohesion (order vs shuffled), and token-class contexts. We then built a cadence-aware sampler (HF + MLX) that enforces per-phase top_p/temperature, content-aware spikes, cooldowns, and optional rhyme/line nudges.

## Principles
- Cadence, not chaos: base focus → spike → cooldown → repeat
- Spike on content pivots; defer punctuation/newline
- Sustained shifts every 1–3 lines/sentences
- Order matters: negative cohesion delta (original > shuffled)
- Genre dials: denser spikes for poetry; gentler cadence for prose

## How to Read These Signatures

How to read the signatures: Surprisal/entropy show focus vs openness; spike rate and IPI (inter-peak interval) capture rhythm; cooldown entropy drop shows consolidation after turns; cohesion delta quantifies how much word order matters in an author; content vs punctuation alignment around spikes shows whether turns land on meaningful tokens.

## Findings by Genre and Author

Findings by genre and author:
- Sonnets: medium spike density, clear quatrain cadence, a volta near line 9; strong end-words and rhyme.
- Dickinson: higher micro-turn density with short cooldowns; punctuation tolerance (dashes) around content spikes.
- Imagist free verse: moderate spike density but stronger spike intensity; concrete nouns/verbs; sparse function words.
- Whitman/long-line: lower spike density with stanza-level sustained shifts; broader swell windows rather than frequent micro-spikes.
- Wodehouse: playful but coherent cadence; frequent callbacks and entity threading; negative cohesion delta is notable.
- Hemingway: simpler clauses with steadier cadence; fewer spikes, stronger cooldown consolidation; high cohesion.

## Best Practices — Target Bands
- Base top_p ≈ 0.88–0.92, temperature ≈ 0.7–0.85
- Spike top_p ≈ 0.95–0.98, temperature ≈ 0.95–1.08; require content token
- Cooldown top_p ≈ 0.80–0.86, temperature ≈ 0.6–0.75 for 3–8 tokens
- Target spike interval: poetry 8–16; prose 14–24 (tune by author)
- After spike: look for cooldown entropy drop ≥ 1.0 bits (3-token window)
- Maintain content alignment: spike-next-content-rate high; avoid back-to-back punctuation surprises

## Illustrations
## What We Did and Why

What we did and why: We measured token distributions and cadence signatures across authors and genres to learn what ‘good writing’ looks like in terms of focus, rhythm, and cohesion. We then built a cadence‑aware sampler that explicitly follows those patterns — keeping baseline choices focused, inserting purposeful spikes on content pivots, cooling down to consolidate, and periodically opening sustained shifts. We added rhyme nudging for poetry, repetition controls and a diversity bonus on spikes to keep turns fresh, and we save before/after snippets so you can see and measure the effect.

## Generated Snippets (Normal vs Fixed-Up)
