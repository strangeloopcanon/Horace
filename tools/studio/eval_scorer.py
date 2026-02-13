from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.text_normalize import normalize_for_studio


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


def _label_from_source(
    src: str,
    *,
    pos_set: set[str],
    neg_set: Optional[set[str]],
) -> Optional[int]:
    s = str(src or "")
    if s in pos_set:
        return 1
    if neg_set:
        return 0 if s in neg_set else None
    return 0


class _EvalDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        *,
        doc_type: str,
        normalize_text: bool,
        pos_set: set[str],
        neg_set: Optional[set[str]],
        max_length: int,
        tokenizer,
    ):
        self._items: List[Tuple[Dict[str, torch.Tensor], int, str]] = []
        dt = str(doc_type)
        for r in rows:
            text = str(r.get("text") or "")
            if not text.strip():
                continue
            y = _label_from_source(str(r.get("source") or ""), pos_set=pos_set, neg_set=neg_set)
            if y is None:
                continue
            t, _ = normalize_for_studio(text, doc_type=dt, enabled=bool(normalize_text))
            enc = tokenizer(t, truncation=True, max_length=int(max_length), padding=False, return_tensors="pt")
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            self._items.append((enc, int(y), str(r.get("source") or "")))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc, y, src = self._items[idx]
        return {"enc": enc, "y": int(y), "source": src}


def _collate(tokenizer, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    encs = [x["enc"] for x in batch]
    padded = tokenizer.pad(encs, padding=True, return_tensors="pt")
    padded["y"] = torch.tensor([int(x["y"]) for x in batch], dtype=torch.int64)
    padded["source"] = [str(x["source"]) for x in batch]
    return padded


@dataclass(frozen=True)
class EvalResult:
    n: int
    auc: Optional[float]
    mean_pos: Optional[float]
    mean_neg: Optional[float]


def eval_scorer(
    *,
    model_path_or_id: str,
    samples_path: Path,
    positive_sources: Sequence[str],
    negative_sources: Optional[Sequence[str]],
    doc_type: str,
    normalize_text: bool,
    max_length: int,
    batch_size: int,
    device: Optional[str] = None,
) -> Tuple[EvalResult, Dict[str, Any]]:
    pos_set = {str(x) for x in positive_sources if str(x).strip()}
    neg_set = {str(x) for x in (negative_sources or []) if str(x).strip()} if negative_sources else None
    if not pos_set:
        raise ValueError("positive_sources is empty")

    rows = list(_iter_jsonl(samples_path))
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

    ds = _EvalDataset(
        rows,
        doc_type=str(doc_type),
        normalize_text=bool(normalize_text),
        pos_set=pos_set,
        neg_set=neg_set,
        max_length=int(max_length),
        tokenizer=tok,
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, collate_fn=lambda b: _collate(tok, b))

    all_y: List[int] = []
    all_p: List[float] = []
    all_src: List[str] = []
    with torch.no_grad():
        for batch in loader:
            y = batch.pop("y")
            src = batch.pop("source")
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.squeeze(-1)
            p = torch.sigmoid(logits.float()).detach().cpu().numpy()
            all_p.extend([float(x) for x in p.tolist()])
            all_y.extend([int(x) for x in y.detach().cpu().numpy().tolist()])
            all_src.extend([str(x) for x in src])

    y_arr = np.asarray(all_y, dtype=np.int64)
    p_arr = np.asarray(all_p, dtype=np.float64)
    auc = _auc_roc(y_arr, p_arr)
    mean_pos = float(np.mean(p_arr[y_arr == 1])) if y_arr.size and int(np.sum(y_arr == 1)) > 0 else None
    mean_neg = float(np.mean(p_arr[y_arr == 0])) if y_arr.size and int(np.sum(y_arr == 0)) > 0 else None

    by_source: Dict[str, Dict[str, Any]] = {}
    for src in sorted(set(all_src)):
        ps = [p for p, s in zip(all_p, all_src) if s == src]
        by_source[src] = {
            "n": int(len(ps)),
            "mean_prob": float(sum(ps) / len(ps)) if ps else None,
            "p10": float(np.quantile(np.array(ps), 0.10)) if len(ps) >= 2 else None,
            "p50": float(np.quantile(np.array(ps), 0.50)) if len(ps) >= 2 else None,
            "p90": float(np.quantile(np.array(ps), 0.90)) if len(ps) >= 2 else None,
        }

    meta = {
        "model_path_or_id": str(model_path_or_id),
        "samples_path": str(samples_path),
        "doc_type": str(doc_type),
        "normalize_text": bool(normalize_text),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "device": str(dev),
        "positive_sources": sorted(pos_set),
        "negative_sources": sorted(neg_set) if neg_set else "(all non-positive sources)",
        "by_source": by_source,
    }

    return EvalResult(n=int(y_arr.size), auc=auc, mean_pos=mean_pos, mean_neg=mean_neg), meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained scorer model on a benchmark split JSONL.")
    ap.add_argument("--model", required=True, help="HF model dir or id")
    ap.add_argument("--samples", required=True, help="Split JSONL path")
    ap.add_argument("--out", default="", help="Optional report JSON output path")

    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", default="")

    ap.add_argument("--pos", action="append", default=[], help="Positive source(s) (repeatable)")
    ap.add_argument("--neg", action="append", default=[], help="Negative source(s) (repeatable); empty => all non-pos")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    pos = tuple(args.pos or [])
    neg = tuple(args.neg or []) if (args.neg or []) else None
    if not pos:
        raise SystemExit("--pos is required")

    res, meta = eval_scorer(
        model_path_or_id=str(args.model),
        samples_path=Path(str(args.samples)),
        positive_sources=pos,
        negative_sources=neg,
        doc_type=str(args.doc_type),
        normalize_text=bool(normalize_text),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        device=str(args.device).strip() or None,
    )

    payload = {"result": asdict(res), "meta": meta}
    if str(args.out).strip():
        p = Path(str(args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
