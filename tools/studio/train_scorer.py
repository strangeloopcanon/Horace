from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.text_normalize import normalize_for_studio


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
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


def _pearsonr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    xv = x - xm
    yv = y - ym
    denom = float(np.sqrt(np.sum(xv * xv) * np.sum(yv * yv)))
    if denom <= 0:
        return None
    return float(np.sum(xv * yv) / denom)


def _best_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon
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
    # Heuristic for common remote-code model families.
    return "qwen" in s.lower()


def _ensure_pad_token(tokenizer) -> bool:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return False
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        tokenizer.pad_token = eos
        return True
    try:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return True
    except Exception:
        return False


def _default_lora_target_modules(model) -> List[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [m for m in preferred if m in present]
    if targets:
        return targets
    bertish = ["query", "key", "value", "dense"]
    return [m for m in bertish if m in present]


def _default_lora_modules_to_save(model) -> List[str]:
    out: List[str] = []
    for attr in ("score", "classifier", "classification_head"):
        if hasattr(model, attr):
            out.append(attr)
    return out


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
) -> Optional[float]:
    s = str(src or "")
    if s in pos_set:
        return 1.0
    if neg_set:
        return 0.0 if s in neg_set else None
    return 0.0


class _JsonlTextDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        *,
        doc_type: str,
        normalize_text: bool,
        positive_sources: Sequence[str],
        negative_sources: Optional[Sequence[str]],
        label_key: str,
    ):
        self._items: List[Tuple[str, float, dict]] = []
        self.doc_type = str(doc_type)
        self.normalize_text = bool(normalize_text)
        self.label_key = str(label_key)
        pos_set = {str(x) for x in positive_sources if str(x).strip()}
        neg_set = {str(x) for x in (negative_sources or []) if str(x).strip()} if negative_sources else None

        for r in rows:
            text = str(r.get("text") or "")
            if not text.strip():
                continue

            label = r.get(self.label_key)
            y: Optional[float] = None
            if isinstance(label, (int, float)) and math.isfinite(float(label)):
                y = float(label)
            else:
                y = _label_from_source(
                    str(r.get("source") or ""),
                    pos_set=pos_set,
                    neg_set=neg_set,
                )
            if y is None:
                continue

            t, _ = normalize_for_studio(text, doc_type=self.doc_type, enabled=self.normalize_text)
            meta = {"sample_id": r.get("sample_id"), "source": r.get("source"), "group_id": r.get("group_id")}
            self._items.append((t, float(y), meta))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t, y, meta = self._items[idx]
        return {"text": t, "label": float(y), "meta": meta}


def _collate(tokenizer, batch: List[Dict[str, Any]], *, max_length: int) -> Dict[str, Any]:
    texts = [x["text"] for x in batch]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=int(max_length),
        padding=True,
        return_tensors="pt",
    )
    labels = torch.tensor([float(x["label"]) for x in batch], dtype=torch.float32)
    enc["labels"] = labels
    enc["meta"] = [x["meta"] for x in batch]
    return enc


@dataclass(frozen=True)
class TrainSummary:
    out_dir: str
    base_model: str
    doc_type: str
    label_key: str
    positive_sources: List[str]
    negative_sources: Optional[List[str]]
    max_length: int
    train_rows: int
    val_rows: int
    device: str
    steps: int
    best_val_loss: Optional[float]
    best_val_auc: Optional[float]


def _eval_loop(
    model,
    loader: DataLoader,
    *,
    device: str,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    all_y: List[float] = []
    all_p: List[float] = []
    all_src: List[str] = []
    bce = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            meta = batch.pop("meta")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.squeeze(-1)
            loss = bce(logits, labels)
            losses.append(float(loss.detach().cpu().item()))
            p = torch.sigmoid(logits.float()).detach().cpu().numpy()
            all_p.extend([float(x) for x in p.tolist()])
            all_y.extend([float(x) for x in labels.detach().cpu().numpy().tolist()])
            all_src.extend([str(m.get("source") or "") for m in meta])

    y = np.asarray(all_y, dtype=np.float64)
    p = np.asarray(all_p, dtype=np.float64)
    mse = float(np.mean((p - y) ** 2)) if y.size else float("nan")
    mae = float(np.mean(np.abs(p - y))) if y.size else float("nan")
    pearson = _pearsonr(p, y)

    # AUC is only meaningful for binary labels; we treat near-binary as binary if all labels are 0/1.
    uniq = set(float(x) for x in y.tolist()) if y.size else set()
    auc = None
    if uniq.issubset({0.0, 1.0}) and y.size:
        auc = _auc_roc((y >= 0.5).astype(np.int64), p.astype(np.float64))

    return {
        "n": int(y.size),
        "loss_mean": float(np.mean(losses)) if losses else None,
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
        "auc": auc,
    }


def train_scorer(
    *,
    train_path: Path,
    val_path: Optional[Path],
    test_path: Optional[Path],
    out_dir: Path,
    base_model: str,
    doc_type: str,
    normalize_text: bool,
    positive_sources: Sequence[str],
    negative_sources: Optional[Sequence[str]],
    label_key: str,
    max_length: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    lora_r: int = 0,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
    gradient_checkpointing: bool = False,
    bf16: bool = False,
    grad_accum_steps: int = 1,
    merge_lora: bool = False,
) -> TrainSummary:
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(_iter_jsonl(Path(train_path)))
    val_rows = list(_iter_jsonl(Path(val_path))) if val_path is not None else []
    test_rows = list(_iter_jsonl(Path(test_path))) if test_path is not None else []

    train_ds = _JsonlTextDataset(
        train_rows,
        doc_type=str(doc_type),
        normalize_text=bool(normalize_text),
        positive_sources=positive_sources,
        negative_sources=negative_sources,
        label_key=str(label_key),
    )
    val_ds = (
        _JsonlTextDataset(
            val_rows,
            doc_type=str(doc_type),
            normalize_text=bool(normalize_text),
            positive_sources=positive_sources,
            negative_sources=negative_sources,
            label_key=str(label_key),
        )
        if val_rows
        else None
    )
    test_ds = (
        _JsonlTextDataset(
            test_rows,
            doc_type=str(doc_type),
            normalize_text=bool(normalize_text),
            positive_sources=positive_sources,
            negative_sources=negative_sources,
            label_key=str(label_key),
        )
        if test_rows
        else None
    )

    model_id = str(base_model)
    trc = bool(trust_remote_code) or _needs_trust_remote_code(model_id)
    tok_kwargs = {"trust_remote_code": trc}
    try:
        tok = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True, **tok_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    pad_added = _ensure_pad_token(tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        ignore_mismatched_sizes=True,
        trust_remote_code=trc,
    )
    if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
        model.config.pad_token_id = tok.pad_token_id
    if pad_added:
        try:
            model.resize_token_embeddings(len(tok))
        except Exception:
            pass

    dev = _best_device(device)
    model.to(dev)
    if bool(gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    is_lora = int(lora_r) > 0
    if is_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("LoRA requested but `peft` is not installed") from e

        targets = list(lora_target_modules) if lora_target_modules else _default_lora_target_modules(model)
        if not targets:
            raise RuntimeError("Failed to infer LoRA target_modules; pass --lora-target explicitly")
        lcfg = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=targets,
            modules_to_save=_default_lora_modules_to_save(model) or None,
        )
        model = get_peft_model(model, lcfg)
        model.to(dev)

    collate = lambda b: _collate(tok, b, max_length=int(max_length))

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate) if test_ds else None

    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    bce = torch.nn.BCEWithLogitsLoss()

    best_val_loss: Optional[float] = None
    best_val_auc: Optional[float] = None
    best_epoch: Optional[int] = None

    steps = 0
    ga = max(1, int(grad_accum_steps))
    use_amp = bool(bf16) and dev == "cuda"
    optim.zero_grad(set_to_none=True)
    for epoch in range(max(1, int(epochs))):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for micro_step, batch in enumerate(pbar, start=1):
            labels = batch.pop("labels").to(dev)
            batch.pop("meta", None)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(**batch)
                logits = out.logits.squeeze(-1)
                loss = bce(logits, labels) / float(ga)
            loss.backward()
            if micro_step % ga == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
            steps += 1
            pbar.set_postfix({"loss": float(loss.detach().cpu().item()) * float(ga)})

        if val_loader is not None:
            metrics = _eval_loop(model, val_loader, device=dev)
            vloss = metrics.get("loss_mean")
            vauc = metrics.get("auc")
            if isinstance(vloss, (int, float)) and math.isfinite(float(vloss)):
                if best_val_loss is None or float(vloss) < float(best_val_loss):
                    best_val_loss = float(vloss)
                    best_val_auc = float(vauc) if isinstance(vauc, (int, float)) else None
                    best_epoch = int(epoch + 1)
                    save_model = model
                    if bool(merge_lora) and hasattr(model, "merge_and_unload"):
                        try:
                            save_model = model.merge_and_unload()
                        except Exception:
                            save_model = model
                    save_model.save_pretrained(out_dir)
                    tok.save_pretrained(out_dir)

    # If no val split was provided, always save the final model.
    if val_loader is None:
        save_model = model
        if bool(merge_lora) and hasattr(model, "merge_and_unload"):
            try:
                save_model = model.merge_and_unload()
            except Exception:
                save_model = model
        save_model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

    # Write a small report
    report: Dict[str, Any] = {
        "run_meta": {
            "train_path": str(train_path),
            "val_path": str(val_path) if val_path is not None else None,
            "test_path": str(test_path) if test_path is not None else None,
            "base_model": str(base_model),
            "doc_type": str(doc_type),
            "normalize_text": bool(normalize_text),
            "label_key": str(label_key),
            "positive_sources": sorted({str(x) for x in positive_sources}),
            "negative_sources": sorted({str(x) for x in (negative_sources or [])}) if negative_sources else None,
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "seed": int(seed),
            "device": str(dev),
            "trust_remote_code": bool(trc),
            "lora": (
                {
                    "enabled": bool(is_lora),
                    "r": int(lora_r),
                    "alpha": int(lora_alpha),
                    "dropout": float(lora_dropout),
                    "target_modules": list(lora_target_modules) if lora_target_modules else None,
                    "gradient_checkpointing": bool(gradient_checkpointing),
                    "bf16": bool(bf16),
                    "grad_accum_steps": int(ga),
                    "merge_lora": bool(merge_lora),
                }
                if bool(is_lora)
                else {"enabled": False}
            ),
            "torch_num_threads": int(torch.get_num_threads()),
            "hf_home": str(os.environ.get("HF_HOME") or ""),
        },
        "train": {"n": len(train_ds)},
        "val": None,
        "test": None,
        "best": {"epoch": best_epoch, "val_loss": best_val_loss, "val_auc": best_val_auc},
    }
    if val_loader is not None:
        report["val"] = _eval_loop(model, val_loader, device=dev)
    if test_loader is not None:
        report["test"] = _eval_loop(model, test_loader, device=dev)

    (out_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainSummary(
        out_dir=str(out_dir),
        base_model=str(base_model),
        doc_type=str(doc_type),
        label_key=str(label_key),
        positive_sources=sorted({str(x) for x in positive_sources}),
        negative_sources=sorted({str(x) for x in (negative_sources or [])}) if negative_sources else None,
        max_length=int(max_length),
        train_rows=len(train_ds),
        val_rows=len(val_ds) if val_ds is not None else 0,
        device=str(dev),
        steps=int(steps),
        best_val_loss=best_val_loss,
        best_val_auc=best_val_auc,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a single text→score scorer model on a Studio benchmark split JSONL.")
    ap.add_argument("--train", required=True, help="Train split JSONL (rows include text + source)")
    ap.add_argument("--val", default="", help="Optional val split JSONL")
    ap.add_argument("--test", default="", help="Optional test split JSONL")
    ap.add_argument("--out-dir", required=True, help="Output directory (HF save_pretrained)")

    ap.add_argument("--base-model", default="distilbert-base-uncased", help="HF model id (encoder recommended)")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--normalize-text", action="store_true", help="Apply Studio normalization before tokenization")
    ap.add_argument("--no-normalize-text", action="store_true", help="Disable normalization (debug only)")
    ap.add_argument("--label-key", default="label", help="If present in JSONL rows, use this float label in [0,1]")
    ap.add_argument("--trust-remote-code", action="store_true", help="Allow HF remote code for model/tokenizer")

    ap.add_argument("--lora-r", type=int, default=0, help="Enable LoRA with rank r (0 disables)")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", action="append", default=[], help="LoRA target module name (repeatable)")
    ap.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights into base model on save")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Use bf16 autocast on CUDA")
    ap.add_argument("--grad-accum-steps", type=int, default=1)

    ap.add_argument("--pos", action="append", default=[], help="Positive source label(s) (repeatable)")
    ap.add_argument("--neg", action="append", default=[], help="Negative source label(s) (repeatable); empty => all non-pos")

    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="", help="Override device: cpu|cuda|mps")
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    pos = tuple(args.pos or [])
    neg = tuple(args.neg or []) if (args.neg or []) else None
    if not pos:
        raise SystemExit("--pos is required (one or more positive sources)")

    lora_target = tuple(str(x) for x in (args.lora_target or []) if str(x).strip()) or None

    res = train_scorer(
        train_path=Path(str(args.train)),
        val_path=Path(str(args.val)) if str(args.val).strip() else None,
        test_path=Path(str(args.test)) if str(args.test).strip() else None,
        out_dir=Path(str(args.out_dir)),
        base_model=str(args.base_model),
        doc_type=str(args.doc_type),
        normalize_text=bool(normalize_text),
        positive_sources=pos,
        negative_sources=neg,
        label_key=str(args.label_key),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        seed=int(args.seed),
        device=str(args.device).strip() or None,
        trust_remote_code=bool(args.trust_remote_code),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_target_modules=lora_target,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        bf16=bool(args.bf16),
        grad_accum_steps=int(args.grad_accum_steps),
        merge_lora=bool(args.merge_lora),
    )
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
