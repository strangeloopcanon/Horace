# Combined Writing Signatures — gpt2 vs Qwen/Qwen2.5-1.5B


### Executive Summary

- surprisal mean (nats): mean Δ (Qwen/Qwen2.5-1.5B − gpt2) = -2.097 (lower is better).
- entropy mean (nats): mean Δ (Qwen/Qwen2.5-1.5B − gpt2) = -2.000 (lower is sharper).
- nucleus width (p=0.9): mean Δ (Qwen/Qwen2.5-1.5B − gpt2) = -643.220 (lower is more focused).
- cohesion delta (shuffled − original): mean Δ (Qwen/Qwen2.5-1.5B − gpt2) = -1.409 (more negative means stronger cohesion).
- Largest surprisal drops (Qwen lower than GPT‑2):
  - Percy Bysshe Shelley: Δ=-4.23  gpt2=4.51 → Qwen/Qwen2.5-1.5B=0.28
  - Wilfred Owen: Δ=-4.21  gpt2=4.89 → Qwen/Qwen2.5-1.5B=0.68
  - John Keats: Δ=-4.21  gpt2=4.75 → Qwen/Qwen2.5-1.5B=0.54
  - William Ernest Henley: Δ=-4.06  gpt2=4.48 → Qwen/Qwen2.5-1.5B=0.42
  - Robert Frost: Δ=-4.03  gpt2=4.36 → Qwen/Qwen2.5-1.5B=0.32
  - William Shakespeare: Δ=-3.91  gpt2=4.42 → Qwen/Qwen2.5-1.5B=0.51
  - William Butler Yeats: Δ=-3.77  gpt2=4.43 → Qwen/Qwen2.5-1.5B=0.65
  - Samuel Taylor Coleridge: Δ=-3.77  gpt2=4.28 → Qwen/Qwen2.5-1.5B=0.51

### Per‑Type Deltas (means of Δ = Qwen/Qwen2.5-1.5B − gpt2)

- poem: surprisal -3.10, entropy -2.62, nucleus_w -785, cohesion_delta -1.89
- shortstory: surprisal -1.34, entropy -1.48, nucleus_w -668, cohesion_delta -0.10
- novel: surprisal -1.24, entropy -1.21, nucleus_w -529, cohesion_delta -0.20

### How to Read These Signatures
- Surprisal mean (nats): lower suggests the model finds tokens predictable; poems naturally run higher than prose.
- Entropy mean (nats): lower means sharper distributions (few strong candidates).
- Nucleus width (p=0.9): lower indicates concentrated probability mass; large values signal open‑ended choices.
- Cohesion delta (shuffled − original): more negative ⇒ stronger reliance on order/structure.
- Cadence (IPI, MASD, ACF): captures rhythm — how often spikes occur and how quickly uncertainty settles.
- Token mix (content_fraction, spike context): shows whether surprises align with content words and line boundaries.

For good‑writing signatures: look for structured cadence (stable IPI with occasional spikes),
negative cohesion delta (order matters), focused distributions (smaller nucleus),
and spikes that coincide with contentful turns (high spike_prev/next content rates).

### Findings by Genre and Author

- Poem: Δ surprisal -3.10, Δ entropy -2.62, Δ nucleus_w -785, Δ cohesion_delta -1.89.
- Shortstory: Δ surprisal -1.34, Δ entropy -1.48, Δ nucleus_w -668, Δ cohesion_delta -0.10.
- Novel: Δ surprisal -1.24, Δ entropy -1.21, Δ nucleus_w -529, Δ cohesion_delta -0.20.

- William Shakespeare: Δ surprisal -3.91, Δ entropy -3.67, Δ nucleus_w -804, Δ cohesion_delta -1.38
- Emily Dickinson: Δ surprisal -2.65, Δ entropy -1.92, Δ nucleus_w -925, Δ cohesion_delta -1.80
- Robert Frost: Δ surprisal -4.03, Δ entropy -3.65, Δ nucleus_w -625, Δ cohesion_delta -2.68
- P G Wodehouse: Δ surprisal -1.16, Δ entropy -1.00, Δ nucleus_w -600, Δ cohesion_delta -0.36
- Ernest Hemingway: Δ surprisal -0.82, Δ entropy -0.86, Δ nucleus_w -283, Δ cohesion_delta -0.09
- Edgar Allan Poe: Δ surprisal -1.69, Δ entropy -1.72, Δ nucleus_w -453, Δ cohesion_delta -0.71

#### Synthesis — What ‘good writing’ patterns emerge
- Structure: more negative cohesion delta (shuffling hurts), especially in poetry; Qwen amplifies this.
- Focus: lower entropy and smaller nucleus widths — confident predictions with occasional, meaningful spikes.
- Cadence: inter‑peak intervals are moderate and fairly regular (CV ≈ 0.9–1.1); spikes are followed by entropy cooldown.
- Semantics: spikes align with content tokens; preceding tokens are often function/punctuation, signaling a turn.
- Genre: poems benefit most (largest Δs), short stories next, novels least — consistent with stylistic density.

### Best Practices — Target Bands (from Qwen/Qwen2.5-1.5B)

- Poem:
  - surprisal mean (nats): typical 1.32 (IQR 0.55..1.71)
  - entropy mean (nats): typical 1.47 (IQR 0.77..1.80)
  - nucleus width (p=0.9): typical 91 (IQR 18..88) lower is more focused
  - cohesion delta (shuffled − original): typical -1.74 (IQR -2.21..-1.39) more negative is better
  - spike rate per 100 tokens: typical 7.86 (IQR 4.91..10.76)
  - inter‑peak interval (tokens): typical 10.68 (IQR 6.49..11.77)
  - post‑spike entropy cooldown (3 tokens): typical 1.27 (IQR 0.98..1.50)
  - content token fraction: typical 0.34 (IQR 0.29..0.39)
  - spike prev content rate: typical 0.27 (IQR 0.21..0.33)
  - spike next content rate: typical 0.48 (IQR 0.39..0.58)

- Shortstory:
  - surprisal mean (nats): typical 2.54 (IQR 2.43..2.74)
  - entropy mean (nats): typical 2.52 (IQR 2.40..2.76)
  - nucleus width (p=0.9): typical 170 (IQR 95..234) lower is more focused
  - cohesion delta (shuffled − original): typical -0.24 (IQR -0.30..-0.15) more negative is better
  - spike rate per 100 tokens: typical 13.30 (IQR 12.67..14.32)
  - inter‑peak interval (tokens): typical 6.42 (IQR 5.71..6.72)
  - post‑spike entropy cooldown (3 tokens): typical 1.57 (IQR 1.26..1.80)
  - content token fraction: typical 0.44 (IQR 0.40..0.48)
  - spike prev content rate: typical 0.46 (IQR 0.35..0.54)
  - spike next content rate: typical 0.74 (IQR 0.68..0.83)

- Novel:
  - surprisal mean (nats): typical 2.44 (IQR 2.16..2.66)
  - entropy mean (nats): typical 2.48 (IQR 2.14..2.84)
  - nucleus width (p=0.9): typical 121 (IQR 68..166) lower is more focused
  - cohesion delta (shuffled − original): typical -0.55 (IQR -0.60..-0.46) more negative is better
  - spike rate per 100 tokens: typical 13.12 (IQR 12.49..13.49)
  - inter‑peak interval (tokens): typical 6.42 (IQR 6.32..6.60)
  - post‑spike entropy cooldown (3 tokens): typical 1.55 (IQR 1.40..1.74)
  - content token fraction: typical 0.45 (IQR 0.39..0.53)
  - spike prev content rate: typical 0.40 (IQR 0.32..0.46)
  - spike next content rate: typical 0.75 (IQR 0.68..0.80)

Use these as sanity bands when evaluating generations: aim for focused yet expressive distributions, 
regular but not rigid cadence (IPI around each genre’s typical mean with CV ≈ 1), and negative cohesion deltas.

## Cross‑Model Panels

![authors_delta_surprisal_mean_mean](authors_delta_surprisal_mean_mean.png)


![authors_delta_cohesion_delta_mean](authors_delta_cohesion_delta_mean.png)


![authors_delta_nucleus_w_mean_mean](authors_delta_nucleus_w_mean_mean.png)


![authors_delta_content_fraction_mean](authors_delta_content_fraction_mean.png)


![authors_delta_spike_prev_content_rate_mean](authors_delta_spike_prev_content_rate_mean.png)


![authors_delta_spike_next_content_rate_mean](authors_delta_spike_next_content_rate_mean.png)


![authors_delta_ipi_mean_mean](authors_delta_ipi_mean_mean.png)


![authors_delta_cooldown_entropy_drop_3_mean](authors_delta_cooldown_entropy_drop_3_mean.png)


![authors_scatter_surprisal](authors_scatter_surprisal.png)


![docs_scatter_surprisal](docs_scatter_surprisal.png)


![docs_delta_surprisal_hist](docs_delta_surprisal_hist.png)


## Model‑Specific Overviews

### gpt2

![authors_top_low_surprisal.png](../gpt2/authors_top_low_surprisal.png)


![authors_entropy_vs_surprisal.png](../gpt2/authors_entropy_vs_surprisal.png)


![authors_top_cohesion_delta.png](../gpt2/authors_top_cohesion_delta.png)


![docs_entropy_vs_surprisal.png](../gpt2/docs_entropy_vs_surprisal.png)


![docs_nucleus_width_by_type.png](../gpt2/docs_nucleus_width_by_type.png)


![docs_top_cohesion_delta.png](../gpt2/docs_top_cohesion_delta.png)


![doc_ts_novel_charlotte_bronte_jane_eyre.png](../gpt2/doc_ts_novel_charlotte_bronte_jane_eyre.png)


![doc_ts_novel_ernest_hemingway_the_sun_also_rises.png](../gpt2/doc_ts_novel_ernest_hemingway_the_sun_also_rises.png)


![doc_ts_poem_laurence_binyon_for_the_fallen.png](../gpt2/doc_ts_poem_laurence_binyon_for_the_fallen.png)


### Qwen/Qwen2.5-1.5B

![authors_top_low_surprisal.png](../Qwen/Qwen2.5-1.5B/authors_top_low_surprisal.png)


![authors_entropy_vs_surprisal.png](../Qwen/Qwen2.5-1.5B/authors_entropy_vs_surprisal.png)


![authors_top_cohesion_delta.png](../Qwen/Qwen2.5-1.5B/authors_top_cohesion_delta.png)


![docs_entropy_vs_surprisal.png](../Qwen/Qwen2.5-1.5B/docs_entropy_vs_surprisal.png)


![docs_nucleus_width_by_type.png](../Qwen/Qwen2.5-1.5B/docs_nucleus_width_by_type.png)


![docs_top_cohesion_delta.png](../Qwen/Qwen2.5-1.5B/docs_top_cohesion_delta.png)


![doc_ts_novel_ernest_hemingway_a_farewell_to_arms.png](../Qwen/Qwen2.5-1.5B/doc_ts_novel_ernest_hemingway_a_farewell_to_arms.png)


![doc_ts_novel_ernest_hemingway_the_sun_also_rises.png](../Qwen/Qwen2.5-1.5B/doc_ts_novel_ernest_hemingway_the_sun_also_rises.png)


![doc_ts_poem_wilfred_owen_anthem_for_doomed_youth.png](../Qwen/Qwen2.5-1.5B/doc_ts_poem_wilfred_owen_anthem_for_doomed_youth.png)


### gpt2: William Shakespeare cadence

![author_william_shakespeare_surprisal_hist.png](../gpt2/author_william_shakespeare_surprisal_hist.png)


![author_william_shakespeare_ipi_hist.png](../gpt2/author_william_shakespeare_ipi_hist.png)


![author_william_shakespeare_series.png](../gpt2/author_william_shakespeare_series.png)


### Qwen/Qwen2.5-1.5B: William Shakespeare cadence

![author_william_shakespeare_surprisal_hist.png](../Qwen/Qwen2.5-1.5B/author_william_shakespeare_surprisal_hist.png)


![author_william_shakespeare_ipi_hist.png](../Qwen/Qwen2.5-1.5B/author_william_shakespeare_ipi_hist.png)


![author_william_shakespeare_series.png](../Qwen/Qwen2.5-1.5B/author_william_shakespeare_series.png)


### gpt2: P G Wodehouse cadence

![author_p_g_wodehouse_surprisal_hist.png](../gpt2/author_p_g_wodehouse_surprisal_hist.png)


![author_p_g_wodehouse_ipi_hist.png](../gpt2/author_p_g_wodehouse_ipi_hist.png)


![author_p_g_wodehouse_series.png](../gpt2/author_p_g_wodehouse_series.png)


### Qwen/Qwen2.5-1.5B: P G Wodehouse cadence

![author_p_g_wodehouse_surprisal_hist.png](../Qwen/Qwen2.5-1.5B/author_p_g_wodehouse_surprisal_hist.png)


![author_p_g_wodehouse_ipi_hist.png](../Qwen/Qwen2.5-1.5B/author_p_g_wodehouse_ipi_hist.png)


![author_p_g_wodehouse_series.png](../Qwen/Qwen2.5-1.5B/author_p_g_wodehouse_series.png)


## Full Reports

- [gpt2 full report](../gpt2/README.md)

- [Qwen/Qwen2.5-1.5B full report](../Qwen/Qwen2.5-1.5B/README.md)
