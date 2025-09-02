# Writer Signature: Core Idea & Pipeline

## 1. Intuition in One Line

A writer’s signature is the shape, rhythm, and placement of surprisal—how often, how strongly, and where they pick unexpected tokens given context.

---

## 2. Pipeline Overview

### Data Preparation

- Gather comparable English translations (multiple translators per author if possible)
- Segment into paragraphs → sentences → tokens
- Label spans (narration vs dialogue; paragraph/sentence positions)
- Normalize: sample fixed-length chunks (e.g., 1–2k tokens) to equalize length

### Model Choice & Settings

- Use a fixed, strong English causal LM (temperature = 1, eval mode)
- Byte-level/BPE tokenizer (robust across vocabularies)
- (Optional) also run a second LM for robustness checks

### Teacher Forcing & Token Stats

For each token x_t:

- Compute surprisal: s_t = -log p(x_t | x_<t)
- Also compute local entropy H_t and normalized surprisal: r_t = s_t / H_t
- Keep per-token metadata: POS tag, is-punct, sentence position, dialogue flag, syntactic depth (if you parse)

### Signature Features (per chunk → per work/author)

- Distributional shape of surprisal (quantiles, tail index)
- Rhythm/temporal structure (autocorrelation / spectrum)
- Placement sensitivity (by position, discourse, syntax, POS)
- Burstiness (within/between sentences)
- Coherence gap (LL of original vs sentence-shuffled)

### Embed, Compare, Classify

- Turn each chunk into a feature vector (the signature)
- Compare authors by distance (cosine / Wasserstein) or train a simple classifier
- For intra-author evolution (e.g., Joyce), track signatures over time and test for shift

### Deconfound Translation

- Multiple translations; leave-one-translator-out validation
- Mixed-effects modeling or domain-adversarial removal of translator signal

---

## 3. The Surprisal Signature (Compact, Robust Feature Set)

### A. Distribution Shape (Lexical Surprise)

- Quantiles of surprisal: Q_{5,10,25,50,75,90,95} of {s_t}
- Tail heaviness: Hill/Pareto tail index on top 10% of {s_t}
- Median-to-upper-tail gap: Q_{90} - Q_{50} (sharpness of peaks)
- Normalized surprisal quantiles: same for {r_t = s_t / H_t} to control for local uncertainty

### B. Rhythm in Time (How Surprise is Spaced)

- ACF of {s_t} at lags 1–10 (vector)
- Power-spectrum slope of {s_t} (log–log): low-vs-high frequency energy (smooth cadence vs jittery bursts)
- Burstiness within sentences: coefficient of variation of mean sentence surprisal
- High-surprisal run lengths: distribution of consecutive tokens where s_t exceeds the chunk’s 90th percentile

### C. Placement Sensitivity (Where Surprise Happens)

- Sentence-position profile: average s_t binned into deciles of sentence position (10-D vector)
- Paragraph-position profile: same per paragraph decile (10-D)
- Dialogue vs narration: mean s_t and r_t inside quotes vs outside
- Punctuation-adjacent: mean s_t for tokens immediately before/after commas, em-dashes, semicolons

### D. Linguistic Conditioning (What Kind of Words are Surprising)

- POS-conditional surprisal: mean s_t and r_t for nouns, verbs, adj/adv, function words
- Depth-conditional surprisal (optional): mean s_t by syntactic depth bins (e.g., 0–2, 3–5, >5)

### E. Coherence

- Coherence gap:

  Delta_coh = log p(x_{1:T}) - E_pi[log p(pi(x_{1:T}))]

  where pi shuffles sentences within a paragraph/chunk. (Estimate with a few shuffles; report the mean gap.)

The signature is simply the concatenation of these features (z-scored).

---

## 4. Using the Signature

### A. Distinguishing Authors (Dostoevsky vs Tolstoy)

- Distance-based: compute author centroids from training chunks; classify a new chunk by nearest centroid (cosine)
- Linear probe: logistic regression on z-scored features with nested CV
- Interpretability: inspect which features carry weight (e.g., Dostoevsky might show higher dialogue-zone surprisal and punctuation-adjacent spikes; Tolstoy more even cadence with narration-zone surprises)

### B. Intra-author Evolution (Joyce Early → Late)

- Slide a window across works in chronological order, compute signatures, and fit a trend
- Test shift with energy distance / MMD between early-period and late-period signature distributions, with bootstrap CIs
- Visualize with PCA/UMAP of signatures to see trajectory in style space

---

## 5. Translation Confounds & How to Handle Them

- Data design: include multiple translators per author and (ideally) the same translator across multiple authors
- Validation: leave-one-translator-out: train on all but one, test on that one. Report accuracy drop
- Mixed-effects: fit a model with random intercepts for translator:
 
  `feature ~ 1 + author + (1|translator)`
 
- Adversarial removal (optional): small MLP on features; add a gradient-reversal translator classifier to encourage translator-invariant author signal

## 6. Evaluation You Can Trust

### Between-Author Discrimination

- Split chunks by work and translator
- Train on some works/translators, test on unseen ones
- Report accuracy, macro-F1, and a confusion matrix
- Control: leave-one-translator-out; if accuracy holds, signal isn’t translator-driven

### Within-Author Evolution (Joyce)

- Divide into early/middle/late periods
- Compare signature distributions with energy distance or MMD; bootstrap confidence intervals
- Plot signatures via PCA/UMAP to visualize stylistic trajectory

### Ablations

- Remove rhythm features → watch accuracy drop?
- Remove placement features → does Dostoevsky/Tolstoy separation shrink (dialogue vs narration)?
- Swap surprisal for plain word frequencies → large drop would indicate the model-conditioned nature matters

---

## 7. What Differences Might You Expect?

- Dostoevsky: more dialogue-heavy with abrupt turns → higher dialogue-zone surprisal, stronger punctuation-adjacent spikes, shorter but frequent high-surprisal runs, livelier ACF at short lags
- Tolstoy: steadier narrative cadence → lower burst CV, smoother spectrum slope (more low-frequency energy), higher coherence gap from scene scaffolding
- Joyce (early → late): rising tail heaviness and run lengths, steeper spectrum slope (more high-frequency energy), growing divergence between function-word and content-word surprisal profiles, bigger punctuation-adjacent contrasts

These are hypotheses; the point of the framework is to test them quantitatively.

---

## 8. Practical Tips

- Chunking: 1–2k tokens per chunk balances stability with sample size; use ≥20 chunks per class when possible
- Z-scoring: standardize features using training means/SD; apply the same transform at test time
- Multiple LMs: repeat with two English LMs; keep features that agree across models
- Reporting: alongside accuracy, show feature importances or coefficients to keep it interpretable
