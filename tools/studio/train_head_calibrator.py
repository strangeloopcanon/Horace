from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.head_calibrator import HeadCalibrator, save_head_calibrator
from tools.studio.score import rubric_category_weights
from tools.studio.text_normalize import normalize_for_studio


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64, copy=False)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y = y_true.astype(np.int64, copy=False)
    s = y_score.astype(np.float64, copy=False)
    if y.size == 0:
        return None
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, y.size + 1)
    pos_ranks = ranks[y == 1]
    auc = (float(np.sum(pos_ranks)) - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _needs_trust_remote_code(model_path_or_id: str) -> bool:
    s = str(model_path_or_id or "").strip()
    if not s:
        return False
    p = Path(s)
    if p.exists():
        cfg = p / "config.json"
        if cfg.exists():
            try:
                obj = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and "auto_map" in obj:
                    return True
            except Exception:
                return False
        return False
    return "qwen" in s.lower()


def _ensure_pad_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        tokenizer.pad_token = eos
        return
    try:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    except Exception:
        return


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _label_from_source(src: str, *, pos_set: set[str], neg_set: Optional[set[str]]) -> Optional[int]:
    s = str(src or "")
    if s in pos_set:
        return 1
    if neg_set:
        return 0 if s in neg_set else None
    return 0


def _best_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def _head_labels(model, n_heads: int) -> List[str]:
    labels: List[str] = []
    try:
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict):
            labels = [str(id2label.get(i) or f"head_{i}") for i in range(n_heads)]
    except Exception:
        labels = []
    if not labels:
        labels = [f"head_{i}" for i in range(n_heads)]
    return labels


def _derive_rubric_overall(
    *,
    head_labels: List[str],
    probs: np.ndarray,
) -> Optional[np.ndarray]:
    if probs.ndim != 2 or probs.shape[1] != len(head_labels):
        return None
    cat_w = rubric_category_weights()
    idxs: List[int] = []
    ws: List[float] = []
    for j, lab in enumerate(head_labels):
        if not lab.startswith("rubric_"):
            continue
        cat = lab[len("rubric_") :]
        wf = float(cat_w.get(cat, 1.0))
        if not math.isfinite(wf) or wf <= 0:
            continue
        idxs.append(int(j))
        ws.append(float(wf))
    if not idxs or float(sum(ws)) <= 0:
        return None
    wv = (np.asarray(ws, dtype=np.float64) / float(sum(ws))).reshape(1, -1)
    derived = (probs[:, idxs].astype(np.float64) * wv).sum(axis=1)
    return np.clip(derived, 0.0, 1.0)


def _score_texts(
    *,
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    doc_type: str,
    normalize_text: bool,
    device: str,
) -> Tuple[np.ndarray, List[str]]:
    batch = []
    for t in texts:
        t_norm, _ = normalize_for_studio(t, doc_type=str(doc_type), enabled=bool(normalize_text))
        batch.append(t_norm)
    enc = tokenizer(batch, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    head_labels = _head_labels(model, int(probs.shape[1]))
    return probs, head_labels


def _build_feature_matrix(
    *,
    model,
    tokenizer,
    rows: List[dict],
    pos_set: set[str],
    neg_set: Optional[set[str]],
    max_length: int,
    batch_size: int,
    doc_type: str,
    normalize_text: bool,
    device: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    items: List[Tuple[str, int]] = []
    for r in rows:
        text = str(r.get("text") or "")
        if not text.strip():
            continue
        y = _label_from_source(str(r.get("source") or ""), pos_set=pos_set, neg_set=neg_set)
        if y is None:
            continue
        items.append((text, int(y)))

    if not items:
        raise ValueError("No labeled rows after filtering sources")

    rng = random.Random(int(seed))
    rng.shuffle(items)

    all_feats: List[List[float]] = []
    all_labels: List[int] = []
    head_labels: List[str] = []

    for i in range(0, len(items), int(batch_size)):
        chunk = items[i : i + int(batch_size)]
        texts = [t for t, _ in chunk]
        y = [int(lbl) for _, lbl in chunk]
        probs, head_labels = _score_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=int(max_length),
            doc_type=str(doc_type),
            normalize_text=bool(normalize_text),
            device=str(device),
        )

        derived = _derive_rubric_overall(head_labels=head_labels, probs=probs)
        if derived is not None:
            feat_names = list(head_labels) + ["rubric_overall_from_categories"]
            feats = np.concatenate([probs, derived.reshape(-1, 1)], axis=1)
        else:
            feat_names = list(head_labels)
            feats = probs

        all_feats.extend(feats.astype(np.float64).tolist())
        all_labels.extend(y)

    X = np.asarray(all_feats, dtype=np.float64)
    y_arr = np.asarray(all_labels, dtype=np.float64)
    return X, y_arr, feat_names


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-2,
    lr: float = 0.5,
    steps: int = 600,
    seed: int = 1337,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(int(seed))
    n, d = X.shape
    w = rng.normal(0.0, 0.01, size=(d,)).astype(np.float64)
    b = 0.0
    for _ in range(max(1, int(steps))):
        logits = X @ w + b
        p = _sigmoid_np(logits)
        err = p - y
        grad_w = (X.T @ err) / max(1, n) + float(l2) * w
        grad_b = float(np.mean(err)) if n > 0 else 0.0
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w, float(b)


@dataclass(frozen=True)
class TrainReport:
    out_path: str
    n_train: int
    n_test: int
    train_auc: Optional[float]
    test_auc: Optional[float]
    test_acc: Optional[float]
    test_mean_pos: Optional[float]
    test_mean_neg: Optional[float]
    baseline_auc: Optional[float]
    feature_names: Tuple[str, ...]


def train_head_calibrator(
    *,
    model_path_or_id: str,
    train_path: Path,
    test_path: Path,
    out_path: Path,
    positive_sources: Sequence[str],
    negative_sources: Optional[Sequence[str]],
    max_length: int,
    batch_size: int,
    doc_type: str,
    normalize_text: bool,
    l2: float,
    lr: float,
    steps: int,
    seed: int,
    device: Optional[str] = None,
) -> TrainReport:
    m = str(model_path_or_id)
    trc = _needs_trust_remote_code(m)
    tok_kwargs = {"trust_remote_code": trc}
    try:
        tok = AutoTokenizer.from_pretrained(m, fix_mistral_regex=True, **tok_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(m, **tok_kwargs)
    _ensure_pad_token(tok)
    model = AutoModelForSequenceClassification.from_pretrained(m, trust_remote_code=trc)
    if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
        model.config.pad_token_id = tok.pad_token_id
    dev = _best_device(device)
    model.to(dev)
    model.eval()

    pos_set = {str(x) for x in positive_sources if str(x).strip()}
    neg_set = {str(x) for x in (negative_sources or []) if str(x).strip()} if negative_sources else None
    if not pos_set:
        raise ValueError("positive_sources is empty")

    train_rows = list(_iter_jsonl(train_path))
    test_rows = list(_iter_jsonl(test_path))

    X_train, y_train, feat_names = _build_feature_matrix(
        model=model,
        tokenizer=tok,
        rows=train_rows,
        pos_set=pos_set,
        neg_set=neg_set,
        max_length=int(max_length),
        batch_size=int(batch_size),
        doc_type=str(doc_type),
        normalize_text=bool(normalize_text),
        device=str(dev),
        seed=int(seed),
    )
    X_test, y_test, _ = _build_feature_matrix(
        model=model,
        tokenizer=tok,
        rows=test_rows,
        pos_set=pos_set,
        neg_set=neg_set,
        max_length=int(max_length),
        batch_size=int(batch_size),
        doc_type=str(doc_type),
        normalize_text=bool(normalize_text),
        device=str(dev),
        seed=int(seed + 1),
    )

    w, b = train_logistic(X_train, y_train, l2=float(l2), lr=float(lr), steps=int(steps), seed=int(seed))
    train_probs = _sigmoid_np(X_train @ w + b)
    test_probs = _sigmoid_np(X_test @ w + b)

    train_auc = _auc_roc(y_train.astype(np.int64), train_probs)
    test_auc = _auc_roc(y_test.astype(np.int64), test_probs)
    test_preds = (test_probs >= 0.5).astype(np.int64)
    test_acc = float(np.mean(test_preds == y_test.astype(np.int64))) if y_test.size else None
    test_mean_pos = float(np.mean(test_probs[y_test == 1.0])) if int(np.sum(y_test == 1.0)) > 0 else None
    test_mean_neg = float(np.mean(test_probs[y_test == 0.0])) if int(np.sum(y_test == 0.0)) > 0 else None

    baseline_auc = None
    if feat_names and feat_names[0] == "greatness":
        baseline_auc = _auc_roc(y_test.astype(np.int64), X_test[:, 0].astype(np.float64))

    cal = HeadCalibrator(
        feature_names=tuple(feat_names),
        weights=tuple(float(x) for x in w.tolist()),
        bias=float(b),
        meta={
            "type": "head_logistic_v1",
            "model_path_or_id": str(model_path_or_id),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "doc_type": str(doc_type),
            "normalize_text": bool(normalize_text),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "positive_sources": sorted(pos_set),
            "negative_sources": sorted(neg_set) if neg_set else "(all non-positive sources)",
            "l2": float(l2),
            "lr": float(lr),
            "steps": int(steps),
            "seed": int(seed),
            "train_auc": float(train_auc) if train_auc is not None else None,
            "test_auc": float(test_auc) if test_auc is not None else None,
            "baseline_auc": float(baseline_auc) if baseline_auc is not None else None,
        },
    )
    save_head_calibrator(cal, out_path)

    return TrainReport(
        out_path=str(out_path),
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        train_auc=float(train_auc) if train_auc is not None else None,
        test_auc=float(test_auc) if test_auc is not None else None,
        test_acc=float(test_acc) if test_acc is not None else None,
        test_mean_pos=float(test_mean_pos) if test_mean_pos is not None else None,
        test_mean_neg=float(test_mean_neg) if test_mean_neg is not None else None,
        baseline_auc=float(baseline_auc) if baseline_auc is not None else None,
        feature_names=tuple(feat_names),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a tiny calibrator over scorer head probabilities.")
    ap.add_argument("--model", required=True, help="HF model dir or id")
    ap.add_argument("--train", required=True, help="Train split JSONL")
    ap.add_argument("--test", required=True, help="Test split JSONL")
    ap.add_argument("--out", required=True, help="Output calibrator JSON path")

    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", default="")

    ap.add_argument("--pos", action="append", default=[], help="Positive source(s)")
    ap.add_argument("--neg", action="append", default=[], help="Negative source(s); empty => all non-pos")

    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    pos = tuple(args.pos or [])
    neg = tuple(args.neg or []) if (args.neg or []) else None
    if not pos:
        raise SystemExit("--pos is required")

    report = train_head_calibrator(
        model_path_or_id=str(args.model),
        train_path=Path(str(args.train)),
        test_path=Path(str(args.test)),
        out_path=Path(str(args.out)),
        positive_sources=pos,
        negative_sources=neg,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        doc_type=str(args.doc_type),
        normalize_text=bool(normalize_text),
        l2=float(args.l2),
        lr=float(args.lr),
        steps=int(args.steps),
        seed=int(args.seed),
        device=str(args.device).strip() or None,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

