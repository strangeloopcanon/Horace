from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.text_normalize import normalize_for_studio


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


class _PairsDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        *,
        doc_type: str,
        normalize_text: bool,
    ):
        self.doc_type = str(doc_type)
        self.normalize_text = bool(normalize_text)
        self._items: List[Tuple[str, str, dict]] = []

        for r in rows:
            chosen = str(r.get("chosen_text") or "")
            rejected = str(r.get("rejected_text") or "")
            if not chosen.strip() or not rejected.strip():
                continue
            c, _ = normalize_for_studio(chosen, doc_type=self.doc_type, enabled=self.normalize_text)
            j, _ = normalize_for_studio(rejected, doc_type=self.doc_type, enabled=self.normalize_text)
            if not c.strip() or not j.strip():
                continue
            meta = {
                "pair_id": r.get("pair_id"),
                "group_id": r.get("group_id"),
                "rewrite_kind": (r.get("meta") or {}).get("rewrite_kind") if isinstance(r.get("meta"), dict) else None,
            }
            self._items.append((c, j, meta))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chosen, rejected, meta = self._items[idx]
        return {"chosen_text": chosen, "rejected_text": rejected, "meta": meta}


def _collate(tokenizer, batch: List[Dict[str, Any]], *, max_length: int) -> Dict[str, Any]:
    chosen = [x["chosen_text"] for x in batch]
    rejected = [x["rejected_text"] for x in batch]

    enc_c = tokenizer(
        chosen,
        truncation=True,
        max_length=int(max_length),
        padding=True,
        return_tensors="pt",
    )
    enc_r = tokenizer(
        rejected,
        truncation=True,
        max_length=int(max_length),
        padding=True,
        return_tensors="pt",
    )
    return {"chosen": enc_c, "rejected": enc_r, "meta": [x["meta"] for x in batch]}


@dataclass(frozen=True)
class PrefTrainSummary:
    out_dir: str
    init_model: str
    base_model: str
    doc_type: str
    train_pairs: int
    val_pairs: int
    test_pairs: int
    max_length: int
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    seed: int
    device: str
    best_val_acc: Optional[float]


def _pair_metrics(*, chosen_logits: np.ndarray, rejected_logits: np.ndarray) -> Dict[str, Any]:
    if chosen_logits.size == 0 or rejected_logits.size == 0 or chosen_logits.size != rejected_logits.size:
        return {"n": 0, "acc": None, "margin_mean": None, "margin_p10": None, "margin_p50": None, "margin_p90": None}
    margins = chosen_logits - rejected_logits
    acc = float(np.mean((margins > 0).astype(np.float64)))
    return {
        "n": int(margins.size),
        "acc": float(acc),
        "margin_mean": float(np.mean(margins)),
        "margin_p10": float(np.quantile(margins, 0.10)),
        "margin_p50": float(np.quantile(margins, 0.50)),
        "margin_p90": float(np.quantile(margins, 0.90)),
    }


def _eval_pairs(model, loader: DataLoader, *, device: str) -> Dict[str, Any]:
    model.eval()
    all_c: List[float] = []
    all_r: List[float] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            enc_c = {k: v.to(device) for k, v in batch["chosen"].items()}
            enc_r = {k: v.to(device) for k, v in batch["rejected"].items()}
            out_c = model(**enc_c)
            out_r = model(**enc_r)
            lc = out_c.logits.squeeze(-1)
            lr = out_r.logits.squeeze(-1)
            loss = F.softplus(-(lc - lr)).mean()
            losses.append(float(loss.detach().cpu().item()))
            all_c.extend([float(x) for x in lc.detach().cpu().numpy().tolist()])
            all_r.extend([float(x) for x in lr.detach().cpu().numpy().tolist()])

    c = np.asarray(all_c, dtype=np.float64)
    r = np.asarray(all_r, dtype=np.float64)
    return {"loss_mean": float(np.mean(losses)) if losses else None, "pairs": _pair_metrics(chosen_logits=c, rejected_logits=r)}


def train_preference_scorer(
    *,
    train_pairs_path: Path,
    val_pairs_path: Optional[Path],
    test_pairs_path: Optional[Path],
    out_dir: Path,
    base_model: str,
    init_model: Optional[str],
    doc_type: str,
    normalize_text: bool,
    max_length: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: Optional[str] = None,
) -> PrefTrainSummary:
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(_iter_jsonl(Path(train_pairs_path)))
    val_rows = list(_iter_jsonl(Path(val_pairs_path))) if val_pairs_path is not None else []
    test_rows = list(_iter_jsonl(Path(test_pairs_path))) if test_pairs_path is not None else []

    train_ds = _PairsDataset(train_rows, doc_type=str(doc_type), normalize_text=bool(normalize_text))
    val_ds = _PairsDataset(val_rows, doc_type=str(doc_type), normalize_text=bool(normalize_text)) if val_rows else None
    test_ds = _PairsDataset(test_rows, doc_type=str(doc_type), normalize_text=bool(normalize_text)) if test_rows else None

    init = str(init_model).strip() if init_model else ""
    model_id = init or str(base_model)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    dev = _best_device(device)
    model.to(dev)

    collate = lambda b: _collate(tok, b, max_length=int(max_length))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate) if test_ds else None

    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val_acc: Optional[float] = None
    best_epoch: Optional[int] = None

    for epoch in range(max(1, int(epochs))):
        model.train()
        pbar = tqdm(train_loader, desc=f"pref epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            enc_c = {k: v.to(dev) for k, v in batch["chosen"].items()}
            enc_r = {k: v.to(dev) for k, v in batch["rejected"].items()}
            out_c = model(**enc_c)
            out_r = model(**enc_r)
            lc = out_c.logits.squeeze(-1)
            lrj = out_r.logits.squeeze(-1)
            loss = F.softplus(-(lc - lrj)).mean()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

        if val_loader is not None:
            metrics = _eval_pairs(model, val_loader, device=dev)
            acc = (metrics.get("pairs") or {}).get("acc")
            if isinstance(acc, (int, float)) and np.isfinite(float(acc)):
                if best_val_acc is None or float(acc) > float(best_val_acc):
                    best_val_acc = float(acc)
                    best_epoch = int(epoch + 1)
                    model.save_pretrained(out_dir)
                    tok.save_pretrained(out_dir)

    if val_loader is None:
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

    report: Dict[str, Any] = {
        "run_meta": {
            "train_pairs_path": str(train_pairs_path),
            "val_pairs_path": str(val_pairs_path) if val_pairs_path is not None else None,
            "test_pairs_path": str(test_pairs_path) if test_pairs_path is not None else None,
            "init_model": init or None,
            "base_model": str(base_model),
            "doc_type": str(doc_type),
            "normalize_text": bool(normalize_text),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "seed": int(seed),
            "device": str(dev),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_acc": float(best_val_acc) if best_val_acc is not None else None,
            "torch_num_threads": int(torch.get_num_threads()),
            "hf_home": str(os.environ.get("HF_HOME") or ""),
        },
        "train": {"pairs": int(len(train_ds))},
        "val": None,
        "test": None,
    }
    if val_loader is not None:
        report["val"] = _eval_pairs(model, val_loader, device=dev)
    if test_loader is not None:
        report["test"] = _eval_pairs(model, test_loader, device=dev)

    (out_dir / "preference_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return PrefTrainSummary(
        out_dir=str(out_dir),
        init_model=str(init or ""),
        base_model=str(base_model),
        doc_type=str(doc_type),
        train_pairs=int(len(train_ds)),
        val_pairs=int(len(val_ds)) if val_ds is not None else 0,
        test_pairs=int(len(test_ds)) if test_ds is not None else 0,
        max_length=int(max_length),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=int(epochs),
        seed=int(seed),
        device=str(dev),
        best_val_acc=float(best_val_acc) if best_val_acc is not None else None,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a single text→score model from preference pairs (chosen > rejected).")
    ap.add_argument("--train", required=True, help="Train pairs.jsonl")
    ap.add_argument("--val", default="", help="Val pairs.jsonl (optional)")
    ap.add_argument("--test", default="", help="Test pairs.jsonl (optional)")
    ap.add_argument("--out-dir", required=True, help="Model output dir")

    ap.add_argument("--base-model", default="microsoft/deberta-v3-base")
    ap.add_argument("--init-model", default="", help="Optional model dir/id to initialize from (e.g. distilled scorer)")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    val = Path(str(args.val)) if str(args.val).strip() else None
    test = Path(str(args.test)) if str(args.test).strip() else None
    init = str(args.init_model).strip() or None

    summary = train_preference_scorer(
        train_pairs_path=Path(str(args.train)),
        val_pairs_path=val,
        test_pairs_path=test,
        out_dir=Path(str(args.out_dir)),
        base_model=str(args.base_model),
        init_model=init,
        doc_type=str(args.doc_type),
        normalize_text=bool(normalize_text),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        seed=int(args.seed),
        device=str(args.device).strip() or None,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

