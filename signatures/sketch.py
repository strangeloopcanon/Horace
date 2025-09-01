# In practice, you’ll also want sentence/paragraph and dialogue spans
# (e.g., from spaCy for sentence boundaries and a simple quote-range detector for dialogue).
# Add POS/syntactic features the same way: filter token-level surprisal by POS/depth masks and take means/quantiles.

from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 1) surprisal stream
def surprisal_stream(model_id, text):
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id).eval()
    x = tok(text, return_tensors="pt")
    with torch.no_grad():
        logits = m(**x).logits[:, :-1, :]  # predict t from <t
    probs = logits.softmax(-1)
    labels = x.input_ids[:, 1:]
    p_true = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1,T-1]
    s = (-p_true.log()).squeeze(0).cpu().numpy()  # surprisal
    H = (-(probs * (probs.clamp_min(1e-12).log())).sum(-1)).squeeze(0).cpu().numpy()
    r = s / np.clip(H, 1e-12, None)  # normalized
    return s, r


# 2) feature helpers
def quantiles(arr, qs=(5, 10, 25, 50, 75, 90, 95)):
    return np.percentile(arr, qs)


def hill_tail_index(arr, top_q=0.9):
    x = arr[arr >= np.quantile(arr, top_q)]
    if len(x) < 5:
        return np.nan
    x = np.sort(x)
    x0 = x[0]
    return np.mean(np.log(x / x0 + 1e-12))


def acf(arr, max_lag=10):
    arr = arr - arr.mean()
    denom = (arr**2).sum() + 1e-12
    out = []
    for k in range(1, max_lag + 1):
        out.append((arr[:-k] * arr[k:]).sum() / denom)
    return np.array(out)


def spectrum_slope(arr):
    x = arr - arr.mean()
    X = np.fft.rfft(x)
    psd = (X * np.conj(X)).real
    freqs = np.fft.rfftfreq(len(x))
    mask = freqs > 0
    if mask.sum() < 5:
        return np.nan
    fx = np.log(freqs[mask])
    py = np.log(psd[mask] + 1e-12)
    A = np.vstack([fx, np.ones_like(fx)]).T
    slope = np.linalg.lstsq(A, py, rcond=None)[0][0]
    return slope


# 3) build signature for one chunk
def surprisal_signature(
    model_id, text, sent_spans=None, para_spans=None, dialogue_spans=None
):
    s, r = surprisal_stream(model_id, text)
    feat = {}
    # A) distribution shape
    feat.update(
        {f"s_q{q}": v for q, v in zip((5, 10, 25, 50, 75, 90, 95), quantiles(s))}
    )
    feat["s_tail"] = hill_tail_index(s)
    feat["s_q90_minus_q50"] = np.percentile(s, 90) - np.percentile(s, 50)
    feat.update(
        {f"r_q{q}": v for q, v in zip((5, 10, 25, 50, 75, 90, 95), quantiles(r))}
    )

    # B) rhythm
    a = acf(s, 10)
    for i, v in enumerate(a, 1):
        feat[f"acf_{i}"] = v
    feat["spec_slope"] = spectrum_slope(s)

    # C) placement (if spans are provided; otherwise skip)
    def span_mean(arr, spans):
        return np.mean([arr[a:b].mean() for a, b in spans]) if spans else np.nan

    feat["dialogue_s_mean"] = span_mean(s, dialogue_spans)
    feat["narration_s_mean"] = span_mean(
        s, para_spans
    )  # crude proxy if no dialogue spans
    # burstiness by sentence
    if sent_spans:
        sent_means = np.array([s[a:b].mean() for a, b in sent_spans])
        feat["burst_cv_sent"] = (sent_means.std() + 1e-12) / (
            abs(sent_means.mean()) + 1e-12
        )
    else:
        feat["burst_cv_sent"] = np.nan

    # high-surprisal runs
    thr = np.quantile(s, 0.9)
    runs, cur = [], 0
    for v in s:
        if v >= thr:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    feat["run_mean"] = np.mean(runs) if runs else 0.0
    feat["run_q90"] = (
        np.quantile(runs, 0.9) if len(runs) >= 5 else (max(runs) if runs else 0.0)
    )
    return feat
