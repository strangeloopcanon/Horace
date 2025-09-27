Yes. For any causal LLM you can get, for each position t in a given text, the full next-token distribution $p(x_t\mid x_{<t})$, the token’s own probability, and the top-k or top-p alternatives.

How it works

* Teacher-force the text. At step t the model outputs logits over the vocab. Softmax → probabilities.
* Token probability: gather $p(x_t)$ from that vector.
* Top-k: take k highest probabilities.
* Top-p (nucleus): sort by prob, take the smallest prefix whose cumulative prob ≥ p.
* Sequence score: NLL $= -\sum_t \log p(x_t\mid x_{<t})$. Perplexity $= \exp(\text{NLL}/T)$.

Minimal PyTorch/Transformers sketch

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "gpt2"  # or any causal LM
tok = AutoTokenizer.from_pretrained(model_id)
m = AutoModelForCausalLM.from_pretrained(model_id).eval()

text = "Your text goes here."
enc = tok(text, return_tensors="pt")
with torch.no_grad():
    logits = m(**enc).logits  # [B, T, V]

# align: predict token t from prefix up to t-1
logits = logits[:, :-1, :]                 # [B, T-1, V]
labels = enc.input_ids[:, 1:]              # [B, T-1]
probs  = logits.softmax(dim=-1)            # [B, T-1, V]
logprobs = probs.log()                     # if you want log space

# token probabilities for the actual text
tok_p = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
nll = (-tok_p.log()).sum(dim=1)                              # scalar per example
ppl = torch.exp(nll / labels.size(1))

# top-k alternatives at each position
k = 10
topk_p, topk_id = probs.topk(k, dim=-1)   # [B, T-1, k] each step’s k best tokens

# top-p (nucleus) alternatives at each position
p = 0.9
sorted_p, sorted_id = probs.sort(dim=-1, descending=True)         # [B,T-1,V]
cum = sorted_p.cumsum(dim=-1)
keep = cum <= p
# ensure we include the first token that crosses the threshold
keep[..., 0] = True
keep[..., 1:] = keep[..., 1:] | (~keep[..., :-1] & (cum[..., 1:] >= p))
# mask out the rest
topp_p = sorted_p * keep
topp_id = sorted_id * keep + (~keep) * 0  # ids for kept tokens; ignore zeros when reading
```

Notes

* Use `model.eval()` to disable dropout.
* Results depend on tokenizer and the exact checkpoint.
* Temperature rescales logits; for “native” probabilities keep $T=1$.
* For very long texts, process in chunks within the model’s context window.
* Masked LMs (e.g., BERT) can score specific positions but not left-to-right sequence likelihood cleanly.


Now, if you were to take a bunch of the top poems in the world, and bunch of top prose, could you create like a measure from the tokens that were actually used vs the tokens that were around in probability and create some measure from it to understand the "measure of good writing"?

Yes. Build a token-choice signal from the model’s counterfactuals and aggregate it. Treat “good writing” as “better token choices than strong baselines, with coherent long-range dependencies under genre constraints.”

# Core per-token features

For position $t$ with model distribution $p(\cdot\mid x_{<t})$, chosen token $x_t$:

* Surprisal: $s_t=-\log p(x_t)$.
* Local entropy: $H_t=-\sum_i p_i\log p_i$.
* Choice advantage (vs top-k): $a_t^{(k)}=\log p(x_t)-\frac{1}{k}\sum_{i\in \text{Top-}k}\log p_i$.
* Rank percentile: $q_t=\text{percentile\_rank}(x_t)$.
* Support width (nucleus): $w_t^{(p)}=\lvert\{i:\text{cumprob}\le p\}\rvert$.
* Normalized surprise: $r_t=s_t/H_t$ (surprise given local uncertainty).

# Document-level stats (robust, genre-aware)

Compute per doc: means, medians, IQRs of $\{a_t^{(k)}, r_t, q_t, w_t^{(p)}\}$, plus burstiness $=\text{CV}(s_t)$ by sentence. Add a coherence term:

* Long-range cohesion gain: $\Delta_{\text{coh}}=\log p(x_{1:T})-\mathbb{E}_{\pi}[\log p(\pi(x_{1:T}))]$ where $\pi$ randomly shuffles sentences or paragraphs. Higher is better narrative structure.

# Good Writing Score (learned, not hand-waved)

Train a simple model to combine features:

* Positive weights: $\overline{a^{(k)}}, \text{IQR}(r), \Delta_{\text{coh}}$.
* Negative weights: $\overline{w^{(p)}}, \overline{q}$ (too many easy, high-prob tokens or very flat supports).
  Fit logistic regression or a small GBM on labeled sets (canon poems/prose vs matched contemporary baseline). Cross-validate by author and era to avoid memorization.

# Controls

* Topic/style control: condition the scorer on metadata (poem vs prose, meter, dialogue ratio). For poetry add meter/rhyme constraints so you don’t punish form.
* Length control: report per-100-token stats.
* Model control: use a strong general LM and freeze temperature $=1$. Repeat with two architectures to sanity-check.

# Minimal PyTorch sketch

```python
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def token_features(model_id, text, k=20, p=0.9):
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id).eval()
    x = tok(text, return_tensors="pt")
    with torch.no_grad():
        logits = m(**x).logits[:, :-1, :]            # predict t from <t
    probs = logits.softmax(-1)
    labels = x.input_ids[:, 1:]
    # chosen token probs and surprisal
    p_true = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # [1,T-1]
    s = -p_true.log()
    # entropy
    H = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    r = s / H.clamp_min(1e-12)
    # top-k advantage
    topk_p, topk_id = probs.topk(k, -1)
    a_k = p_true.log() - topk_p.clamp_min(1e-12).log().mean(-1)
    # rank percentile
    ranks = (probs >= p_true.unsqueeze(-1)).sum(-1)  # 1=best
    q = (ranks.float() / probs.size(-1))             # in [0,1]
    # nucleus width
    sp, si = probs.sort(-1, descending=True)
    cum = sp.cumsum(-1)
    w = (cum <= p).sum(-1)
    return { "surprisal": s.squeeze().numpy(),
             "entropy": H.squeeze().numpy(),
             "r_norm": r.squeeze().numpy(),
             "adv_k": a_k.squeeze().numpy(),
             "rank_pct": q.squeeze().numpy(),
             "nucleus_w": w.squeeze().numpy() }

# Aggregate to a score later with regression on labeled corpora
```

# Validation

* Discrimination: AUROC on held-out authors between canon vs baseline.
* Sanity: random word swaps should drop $\Delta_{\text{coh}}$ sharply, synonyms should keep it near-flat.
* Adversarial: rare-word stuffing will raise surprisal but also raise nucleus width and hurt cohesion; the combined score should not improve.

Limitations

* “Good” is corpus-relative and model-relative. You are measuring “expert token choice under a strong LM prior,” not timeless beauty.
* Genre and meter matter. Always stratify.
