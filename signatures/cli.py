#!/usr/bin/env python3
"""
Signature CLI: compute writer surprisal signatures, classify authors, and plot trajectories.

Inputs:
  - Either a manifest (JSON/JSONL/CSV) with columns/keys: path, author[, title, year]
  - Or repeated --file/--author pairs (optionally --title, --year)

Outputs:
  - JSONL of per-chunk signatures
  - Classification report (JSON) and confusion matrix plot (PNG)
  - Early→late trajectory plot for a chosen author (PNG)

Backends:
  - 'hf' (PyTorch + transformers)
  - 'mlx' (Apple MLX via mlx-lm)

Notes:
  - Computes token surprisal s_t and local entropy H_t (normalized r_t = s_t/H_t)
  - Features cover distribution shape + rhythm + high-surprisal run stats (core of idea.md)
  - Sentence/paragraph/dialogue-conditioned features are deferred unless spans are provided in future
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # We'll error nicely later if plotting is requested

# Optional heavy imports are loaded lazily per backend to allow MLX-only usage
try:
    import torch  # type: ignore
except Exception:  # torch may be absent on MLX-only setups
    torch = None  # type: ignore
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

# MLX imports are also optional
try:
    from mlx_lm import load as mlx_load  # type: ignore
    import mlx.core as mx  # type: ignore
except Exception:
    mlx_load = None  # type: ignore
    mx = None  # type: ignore


# ---------- Data structures ----------

@dataclass
class DocSpec:
    path: Path
    author: str
    title: str
    year: Optional[int] = None


@dataclass
class ChunkSig:
    doc_idx: int
    doc_path: str
    author: str
    title: str
    year: Optional[int]
    chunk_idx: int
    token_start: int
    token_end: int
    features: Dict[str, float]


# ---------- IO helpers ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> List[DocSpec]:
    ext = path.suffix.lower()
    rows: List[DocSpec] = []
    if ext in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    rows.append(_row_to_docspec(obj))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        rows.append(_row_to_docspec(obj))
                else:
                    raise ValueError("JSON manifest must be a list of records")
    elif ext in {".csv", ".tsv"}:
        delim = "," if ext == ".csv" else "\t"
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter=delim)
            for row in r:
                rows.append(_row_to_docspec(row))
    else:
        raise ValueError(f"Unsupported manifest format: {ext}")
    return rows


def _row_to_docspec(obj: Dict) -> DocSpec:
    p = obj.get("path") or obj.get("file") or obj.get("filepath")
    if not p:
        raise ValueError("Manifest row missing 'path' field")
    author = obj.get("author") or obj.get("label") or "Unknown"
    title = obj.get("title") or Path(p).stem
    year = obj.get("year")
    try:
        year = int(year) if year is not None and str(year).strip() != "" else None
    except Exception:
        year = None
    return DocSpec(path=Path(p), author=str(author), title=str(title), year=year)


# ---------- LM + surprisal ----------

def pick_device():
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def surprisal_for_text_hf(
    text: str,
    tok,
    model,
    max_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute surprisal s and entropy H per token for text.

    Returns:
        s: np.ndarray of shape [T-1], where position i corresponds to token i+1
        H: np.ndarray of shape [T-1] (local entropy)
        T: number of tokens
    """
    if torch is None:
        raise RuntimeError("PyTorch is not available — cannot use 'hf' backend")
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    T = int(input_ids.shape[1])
    if T <= 1:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), T

    # windowed processing
    if max_window is None:
        # try model.config.n_positions or max_position_embeddings
        max_window = int(getattr(getattr(model, "config", object()), "n_positions", 1024))
        max_window = int(getattr(getattr(model, "config", object()), "max_position_embeddings", max_window))
        max_window = max(64, min(max_window, 4096))

    step = max_window - 1  # we predict tokens 1..L-1 in each window of length L
    s_list: List[np.ndarray] = []
    H_list: List[np.ndarray] = []

    for start in range(0, T, step):
        end = min(T, start + max_window)
        ids = input_ids[:, start:end]
        am = attn[:, start:end] if attn is not None else None
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=am)
        logits = out.logits[:, :-1, :]  # predict next token
        probs = logits.softmax(-1)
        labels = ids[:, 1:]
        p_true = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        s_win = (-p_true.log()).squeeze(0).detach().cpu().numpy()
        H_win = (-(probs * (probs.clamp_min(1e-12).log())).sum(-1)).squeeze(0).detach().cpu().numpy()
        s_list.append(s_win)
        H_list.append(H_win)
        if end >= T:
            break

    s = np.concatenate(s_list, axis=0)
    H = np.concatenate(H_list, axis=0)
    return s, H, T


def surprisal_for_text_mlx(
    text: str,
    tok,
    model,
    max_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if mlx_load is None or mx is None:
        raise RuntimeError("MLX is not available — cannot use 'mlx' backend")
    # Tokenize to integer ids
    ids: List[int] = tok.encode(text)
    T = len(ids)
    if T <= 1:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), T
    if max_window is None:
        max_window = 1024  # default for GPT‑2; can be overridden by --max-window
    max_window = max(64, min(int(max_window), 8192))
    step = max_window - 1
    s_list: List[np.ndarray] = []
    H_list: List[np.ndarray] = []
    for start in range(0, T, step):
        end = min(T, start + max_window)
        ids_win = ids[start:end]
        x = mx.array([ids_win])
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        # logits: [1, L, V]; we need positions 0..L-2 predicting labels 1..L-1
        logits = out[:, :-1, :]
        # Ensure float32 for safe numpy conversion (avoid bf16/float16 buffer issues)
        try:
            logits = logits.astype(getattr(mx, 'float32'))  # type: ignore[attr-defined]
        except Exception:
            pass
        # Convert to numpy for stability
        logits_np = np.array(logits)
        # softmax along vocab
        e = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
        probs_np = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
        labels = np.array(ids_win[1:], dtype=np.int64)[None, :]  # [1, L-1]
        # gather true probs
        p_true = probs_np[0, np.arange(probs_np.shape[1]), labels[0]]  # [L-1]
        s_win = -np.log(np.clip(p_true, 1e-12, None))
        # local entropy
        H_win = -(probs_np * (np.log(np.clip(probs_np, 1e-12, None)))).sum(axis=-1)[0]
        s_list.append(s_win.astype(np.float32))
        H_list.append(H_win.astype(np.float32))
        if end >= T:
            break
    s = np.concatenate(s_list, axis=0)
    H = np.concatenate(H_list, axis=0)
    return s, H, T


# ---------- Signature features (subset A/B + runs) ----------

def _quantiles(arr: np.ndarray, qs=(5, 10, 25, 50, 75, 90, 95)) -> np.ndarray:
    if arr.size == 0:
        return np.full((len(qs),), np.nan, dtype=np.float32)
    return np.percentile(arr, qs)


def _hill_tail_index(arr: np.ndarray, top_q=0.9) -> float:
    if arr.size == 0:
        return float("nan")
    x = arr[arr >= np.quantile(arr, top_q)]
    if len(x) < 5:
        return float("nan")
    x = np.sort(x)
    x0 = x[0]
    return float(np.mean(np.log(x / (x0 + 1e-12) + 1e-12)))


def _acf(arr: np.ndarray, max_lag=10) -> np.ndarray:
    if arr.size == 0:
        return np.full((max_lag,), np.nan, dtype=np.float32)
    arr = arr - arr.mean()
    denom = float((arr**2).sum() + 1e-12)
    out = []
    for k in range(1, max_lag + 1):
        if k >= len(arr):
            out.append(np.nan)
        else:
            out.append(float((arr[:-k] * arr[k:]).sum() / denom))
    return np.array(out, dtype=np.float32)


def _spectrum_slope(arr: np.ndarray) -> float:
    if arr.size < 8:
        return float("nan")
    x = arr - arr.mean()
    X = np.fft.rfft(x)
    psd = (X * np.conj(X)).real
    freqs = np.fft.rfftfreq(len(x))
    mask = freqs > 0
    if mask.sum() < 5:
        return float("nan")
    fx = np.log(freqs[mask])
    py = np.log(psd[mask] + 1e-12)
    A = np.vstack([fx, np.ones_like(fx)]).T
    slope = np.linalg.lstsq(A, py, rcond=None)[0][0]
    return float(slope)


def signature_from_sr(s: np.ndarray, H: np.ndarray) -> Dict[str, float]:
    r = s / np.clip(H, 1e-12, None)
    feat: Dict[str, float] = {}
    # A) distribution shape
    for q, v in zip((5, 10, 25, 50, 75, 90, 95), _quantiles(s)):
        feat[f"s_q{q}"] = float(v)
    feat["s_tail"] = _hill_tail_index(s)
    feat["s_q90_minus_q50"] = float(np.percentile(s, 90) - np.percentile(s, 50)) if s.size > 0 else float("nan")
    for q, v in zip((5, 10, 25, 50, 75, 90, 95), _quantiles(r)):
        feat[f"r_q{q}"] = float(v)
    # B) rhythm
    a = _acf(s, 10)
    for i, v in enumerate(a, 1):
        feat[f"acf_{i}"] = float(v)
    feat["spec_slope"] = _spectrum_slope(s)
    # High-surprisal runs
    if s.size > 0:
        thr = float(np.quantile(s, 0.9))
        runs, cur = [], 0
        for v in s:
            if v >= thr:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        feat["run_mean"] = float(np.mean(runs)) if runs else 0.0
        feat["run_q90"] = float(np.quantile(runs, 0.9)) if len(runs) >= 5 else (float(max(runs)) if runs else 0.0)
    else:
        feat["run_mean"] = 0.0
        feat["run_q90"] = 0.0
    return feat


# ---------- Feature matrix utils ----------

def dicts_to_matrix(dicts: List[Dict[str, float]], keys: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    if keys is None:
        # union of keys preserving a stable order: use the keys of the first dict, then add new
        seen: Dict[str, None] = {}
        for d in dicts:
            for k in d.keys():
                if k not in seen:
                    seen[k] = None
        keys = list(seen.keys())
    X = np.zeros((len(dicts), len(keys)), dtype=np.float32)
    for i, d in enumerate(dicts):
        for j, k in enumerate(keys):
            v = d.get(k)
            X[i, j] = np.nan if v is None else float(v)
    # replace nans with column means (robust for distance computations)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X, keys


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(1.0 - (a @ b) / (na * nb))


def nearest_centroid_predict(
    X: np.ndarray,
    y: List[str],
    groups: List[int],
    fold_test_group: int,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Train on all chunks not in fold_test_group, predict labels for chunks in that group.
    Returns chunk-level predictions and centroids map.
    """
    train_idx = [i for i, g in enumerate(groups) if g != fold_test_group]
    test_idx = [i for i, g in enumerate(groups) if g == fold_test_group]
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr = [y[i] for i in train_idx]

    # centroids by author
    authors = sorted(set(ytr))
    centroids: Dict[str, np.ndarray] = {}
    for a in authors:
        xa = Xtr[[i for i, lab in enumerate(ytr) if lab == a]]
        centroids[a] = xa.mean(axis=0)

    preds: List[str] = []
    for v in Xte:
        dists = [(cosine_distance(v, c), a) for a, c in centroids.items()]
        dists.sort(key=lambda t: t[0])
        preds.append(dists[0][1])
    return preds, centroids


def confusion_matrix(y_true: List[str], y_pred: List[str]) -> Tuple[np.ndarray, List[str]]:
    labels = sorted(set(y_true) | set(y_pred))
    idx = {lab: i for i, lab in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for yt, yp in zip(y_true, y_pred):
        M[idx[yt], idx[yp]] += 1
    return M, labels


# ---------- Pipeline ----------

def compute_signatures_for_docs(
    docs: List[DocSpec],
    model_id: str,
    chunk_tokens: int = 1500,
    max_window: Optional[int] = None,
    limit_chunks_per_doc: Optional[int] = None,
    backend: str = "auto",
) -> List[ChunkSig]:
    # Select backend
    be = backend.lower()
    if be == "auto":
        if (torch is not None) and (AutoModelForCausalLM is not None) and (AutoTokenizer is not None):
            be = "hf"
        elif (mlx_load is not None) and (mx is not None):
            be = "mlx"
        else:
            raise RuntimeError("No compatible backend found. Install torch+transformers or mlx+mlx-lm.")

    if be == "hf":
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise RuntimeError("HF backend requested but torch/transformers not available")
        device = pick_device()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        compute_fn = lambda txt: surprisal_for_text_hf(txt, tok, model, max_window=max_window)
    elif be == "mlx":
        if mlx_load is None or mx is None:
            raise RuntimeError("MLX backend requested but mlx/mlx-lm not available")
        model, tok = mlx_load(model_id)
        compute_fn = lambda txt: surprisal_for_text_mlx(txt, tok, model, max_window=max_window)
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    all_chunks: List[ChunkSig] = []
    for di, d in enumerate(docs):
        text = Path(d.path).read_text(encoding="utf-8", errors="ignore")
        s, H, T = compute_fn(text)
        if T <= 1:
            continue
        # chunk by token indices [a,b)
        chunks: List[Tuple[int, int]] = []
        for a in range(0, T, chunk_tokens):
            b = min(T, a + chunk_tokens)
            if b - a < max(100, int(0.25 * chunk_tokens)):
                break
            chunks.append((a, b))
        if limit_chunks_per_doc is not None:
            chunks = chunks[: limit_chunks_per_doc]

        for ci, (a, b) in enumerate(chunks):
            # Indices in s/H correspond to labels for tokens [1..T-1].
            # For chunk tokens [a, b), we want s/H covering labels for k in [a, b-1] → s[a : b-1].
            i0 = max(0, a)
            i1 = max(i0, b - 1)
            s_slice = s[i0:i1]
            H_slice = H[i0:i1]
            feats = signature_from_sr(s_slice, H_slice)
            all_chunks.append(
                ChunkSig(
                    doc_idx=di,
                    doc_path=str(d.path),
                    author=d.author,
                    title=d.title,
                    year=d.year,
                    chunk_idx=ci,
                    token_start=a,
                    token_end=b,
                    features=feats,
                )
            )
    return all_chunks


def chunklevel_cv_classification(
    chunks: List[ChunkSig], outdir: Path, figure: bool = True
) -> Dict:
    # Build matrices
    dicts = [c.features for c in chunks]
    X, keys = dicts_to_matrix(dicts)
    y = [c.author for c in chunks]
    groups = [c.doc_idx for c in chunks]
    doc_ids = sorted(set(groups))

    # If there is fewer than 2 documents, we cannot do leave-one-doc-out CV.
    # Return a report with metadata and no confusion matrix.
    if len(doc_ids) < 2:
        return {
            "n_chunks": len(chunks),
            "n_docs": len(doc_ids),
            "authors": sorted(set(y)),
            "feature_keys": keys,
            "acc_chunk_level": None,
            "folds": [],
            "confusion_labels": [],
            "confusion_matrix": [],
            "confusion_figure": None,
            "note": "Insufficient documents for CV classification (need >=2).",
        }

    # CV: leave-one-doc-out
    all_true: List[str] = []
    all_pred: List[str] = []
    fold_reports: List[Dict] = []

    for g in doc_ids:
        # z-score on train only
        train_idx = [i for i, gg in enumerate(groups) if gg != g]
        test_idx = [i for i, gg in enumerate(groups) if gg == g]
        if not train_idx or not test_idx:
            # Skip folds where no training or testing data exists (degenerate split)
            continue
        mu, sd = zscore_fit(X[train_idx])
        Xn = zscore_apply(X, mu, sd)
        preds, cents = nearest_centroid_predict(Xn, y, groups, fold_test_group=g)
        yte = [y[i] for i in test_idx]
        all_true.extend(yte)
        all_pred.extend(preds)
        acc = float(np.mean([yt == yp for yt, yp in zip(yte, preds)])) if yte else float("nan")
        fold_reports.append({"doc_idx": int(g), "n_test_chunks": len(test_idx), "acc": acc})

    # Overall metrics
    acc_all = float(np.mean([yt == yp for yt, yp in zip(all_true, all_pred)])) if all_true else float("nan")
    M, labels = confusion_matrix(all_true, all_pred)

    fig_path = None
    if figure:
        if plt is None:
            raise RuntimeError("matplotlib is required for plotting. Try: pip install matplotlib")
        plt.figure(figsize=(8, 7))
        plt.imshow(M, cmap="Blues")
        plt.title("Author classification (chunk-level) — confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(j, i, str(M[i, j]), ha="center", va="center", color="#222")
        plt.tight_layout()
        fig_path = outdir / "classification_confusion_matrix.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    report = {
        "n_chunks": len(chunks),
        "n_docs": len(doc_ids),
        "authors": sorted(set(y)),
        "feature_keys": keys,
        "acc_chunk_level": acc_all,
        "folds": fold_reports,
        "confusion_labels": labels,
        "confusion_matrix": M.tolist(),
        "confusion_figure": str(fig_path) if fig_path else None,
    }
    return report


def plot_trajectory(
    chunks: List[ChunkSig], author: str, outdir: Path, title_key: str = "title"
) -> Optional[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Try: pip install matplotlib")
    # Aggregate per-doc averages for the chosen author
    sel = [c for c in chunks if c.author.lower() == author.lower()]
    if not sel:
        print(f"No chunks for author '{author}'")
        return None
    # group by doc_idx
    by_doc: Dict[int, List[ChunkSig]] = {}
    for c in sel:
        by_doc.setdefault(c.doc_idx, []).append(c)
    docs = []
    for doc_idx, items in by_doc.items():
        X, keys = dicts_to_matrix([it.features for it in items])
        v = X.mean(axis=0)
        doc = {
            "doc_idx": doc_idx,
            "title": items[0].title,
            "year": items[0].year,
            "path": items[0].doc_path,
            "vec": v,
        }
        docs.append(doc)

    # order by year if available else title
    docs.sort(key=lambda d: (d["year"] if d.get("year") is not None else 10**9, str(d["title"]).lower()))

    # PCA (2D) via SVD on standardized doc vectors
    M = np.stack([d["vec"] for d in docs], axis=0)
    mu, sd = zscore_fit(M)
    Z = zscore_apply(M, mu, sd)
    # center only for SVD
    Zc = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = U[:, :2] * S[:2]  # projected coords

    plt.figure(figsize=(7, 5))
    xs, ys = P[:, 0], P[:, 1] if P.shape[1] > 1 else np.zeros_like(P[:, 0])
    plt.plot(xs, ys, marker="o", linestyle="-", color="#1f77b4", alpha=0.9)
    for i, d in enumerate(docs):
        label = f"{d['title']}" + (f" ({d['year']})" if d.get("year") else "")
        plt.annotate(label, (xs[i], ys[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.title(f"{author}: early→late trajectory (PCA of signatures)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    out = outdir / f"trajectory_{author.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_signatures_jsonl(chunks: List[ChunkSig], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            row = {
                "doc_idx": c.doc_idx,
                "doc_path": c.doc_path,
                "author": c.author,
                "title": c.title,
                "year": c.year,
                "chunk_idx": c.chunk_idx,
                "token_start": c.token_start,
                "token_end": c.token_end,
                "features": c.features,
            }
            f.write(json.dumps(row) + "\n")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute writer signatures, classify authors, and plot trajectories")
    ap.add_argument("--model", default="gpt2", help="HF causal LM model id (default: gpt2)")
    ap.add_argument("--backend", default="auto", choices=["auto", "hf", "mlx"], help="Backend: hf (torch) or mlx (Apple MLX); default auto")
    ap.add_argument("--manifest", type=str, help="Manifest file (json/jsonl/csv with path,author[,title,year])")
    ap.add_argument("--file", action="append", nargs="+", help="One or more file paths for a single author when used with --author")
    ap.add_argument("--author", action="append", help="Author for the preceding --file group; repeatable")
    ap.add_argument("--title", action="append", help="Optional title for the preceding --file group; repeatable, aligns to --file entries")
    ap.add_argument("--year", action="append", help="Optional year for the preceding --file group; repeatable, aligns to --file entries")
    ap.add_argument("--chunk-tokens", type=int, default=1500, help="Tokens per chunk (default 1500)")
    ap.add_argument("--limit-chunks", type=int, default=None, help="Optional cap on chunks per doc for speed")
    ap.add_argument("--max-window", type=int, default=None, help="Model window to use per forward pass (default: model max)")
    ap.add_argument("--outdir", type=str, default="reports/generated", help="Output directory")
    ap.add_argument("--trajectory-author", type=str, default=None, help="Author to plot trajectory for (match case-insensitive)")
    ap.add_argument("--no-plots", action="store_true", help="Skip plots (useful on headless machines)")
    return ap.parse_args()


def collect_docs_from_args(ns: argparse.Namespace) -> List[DocSpec]:
    docs: List[DocSpec] = []
    if ns.manifest:
        docs = load_manifest(Path(ns.manifest))
        # Normalize paths relative to repo root if needed
        docs = [DocSpec(path=Path(d.path), author=d.author, title=d.title, year=d.year) for d in docs]
    else:
        # Build from repeated --file + --author (and optional --title --year)
        files_groups = ns.file or []
        authors = ns.author or []
        titles = ns.title or []
        years = ns.year or []

        flat_files: List[str] = [p for group in files_groups for p in group]
        if authors and len(authors) == 1 and len(flat_files) > 1:
            # one author for many files
            authors = authors * len(flat_files)
        if titles and len(titles) == 1 and len(flat_files) > 1:
            titles = titles * len(flat_files)
        if years and len(years) == 1 and len(flat_files) > 1:
            years = years * len(flat_files)

        if not flat_files or not authors or len(flat_files) != len(authors):
            raise SystemExit("Provide either --manifest or matching --file ... and --author ... entries")

        for i, p in enumerate(flat_files):
            t = titles[i] if i < len(titles) else Path(p).stem
            y = years[i] if i < len(years) else None
            try:
                yv = int(y) if y is not None else None
            except Exception:
                yv = None
            docs.append(DocSpec(path=Path(p), author=authors[i], title=t, year=yv))
    # Validate existence
    missing = [d.path for d in docs if not Path(d.path).exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing[:5]}{' ...' if len(missing)>5 else ''}")
    return docs


def main():
    ns = parse_args()
    outdir = Path(ns.outdir)
    ensure_dir(outdir)

    # Collect docs
    docs = collect_docs_from_args(ns)
    print(f"[info] {len(docs)} documents")

    # Compute signatures (per chunk)
    print("[info] Computing surprisal/signatures …")
    chunks = compute_signatures_for_docs(
        docs,
        model_id=ns.model,
        chunk_tokens=ns.chunk_tokens,
        max_window=ns.max_window,
        limit_chunks_per_doc=ns.limit_chunks,
        backend=ns.backend,
    )
    print(f"[info] Computed {len(chunks)} chunk signatures")

    # Save signatures
    sig_path = outdir / "signatures.jsonl"
    save_signatures_jsonl(chunks, sig_path)
    print(f"[ok] Signatures written: {sig_path}")

    # Classification (author)
    print("[info] Running author classification (leave-one-doc-out, chunk-level)…")
    cls_report = chunklevel_cv_classification(chunks, outdir, figure=not ns.no_plots)
    rep_path = outdir / "classification_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(cls_report, f, indent=2)
    print(f"[ok] Classification report: {rep_path}")
    if cls_report.get("confusion_figure"):
        print(f"[ok] Confusion matrix: {cls_report['confusion_figure']}")

    # Trajectory
    if ns.trajectory_author and not ns.no_plots:
        print(f"[info] Plotting early→late trajectory for {ns.trajectory_author} …")
        traj_fig = plot_trajectory(chunks, ns.trajectory_author, outdir)
        if traj_fig:
            print(f"[ok] Trajectory figure: {traj_fig}")

    print("[done]")


if __name__ == "__main__":
    main()
