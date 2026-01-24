from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tools.studio.text_normalize import normalize_for_studio
from tools.studio.score import rubric_category_weights
from tools.studio.train_scorer import (
    _auc_roc,
    _best_device,
    _default_lora_modules_to_save,
    _default_lora_target_modules,
    _ensure_pad_token,
    _needs_trust_remote_code,
    _pearsonr,
    _tqdm_disabled,
)


_DEFAULT_RUBRIC_CATEGORIES: Tuple[str, ...] = ("focus", "cadence", "cohesion", "alignment", "distinctiveness")


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


def _label_from_source(src: str, *, pos_set: set[str], neg_set: Optional[set[str]]) -> Optional[float]:
    s = str(src or "")
    if s in pos_set:
        return 1.0
    if neg_set:
        return 0.0 if s in neg_set else None
    return 0.0


def _safe_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


class _JsonlMultiTargetDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        *,
        doc_type: str,
        normalize_text: bool,
        positive_sources: Sequence[str],
        negative_sources: Optional[Sequence[str]],
        teacher_label_key: str,
        teacher_categories_key: str,
        rubric_categories: Sequence[str],
    ):
        self.doc_type = str(doc_type)
        self.normalize_text = bool(normalize_text)
        self.teacher_label_key = str(teacher_label_key)
        self.teacher_categories_key = str(teacher_categories_key)
        self.rubric_categories = tuple(str(c) for c in rubric_categories)

        self.head_names = ("greatness",) + tuple(f"rubric_{c}" for c in self.rubric_categories)
        self.n_heads = int(len(self.head_names))

        pos_set = {str(x) for x in positive_sources if str(x).strip()}
        neg_set = {str(x) for x in (negative_sources or []) if str(x).strip()} if negative_sources else None

        self._items: List[Tuple[str, np.ndarray, np.ndarray, dict]] = []
        for r in rows:
            text = str(r.get("text") or "")
            if not text.strip():
                continue

            y = np.zeros((self.n_heads,), dtype=np.float32)
            m = np.zeros((self.n_heads,), dtype=np.float32)

            great = _label_from_source(str(r.get("source") or ""), pos_set=pos_set, neg_set=neg_set)
            if great is not None:
                y[0] = float(great)
                m[0] = 1.0

            teacher_y = _safe_float(r.get(self.teacher_label_key))

            cats = r.get(self.teacher_categories_key)
            if isinstance(cats, dict):
                for i, cat in enumerate(self.rubric_categories):
                    v = _safe_float(cats.get(cat))
                    if v is None:
                        continue
                    y[1 + i] = float(max(0.0, min(1.0, v)))
                    m[1 + i] = 1.0

            if float(np.sum(m)) <= 0.0:
                continue

            t, _ = normalize_for_studio(text, doc_type=self.doc_type, enabled=self.normalize_text)
            meta = {
                "sample_id": r.get("sample_id"),
                "source": r.get("source"),
                "group_id": r.get("group_id"),
                "teacher_overall": float(max(0.0, min(1.0, teacher_y))) if teacher_y is not None else None,
            }
            self._items.append((t, y, m, meta))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t, y, m, meta = self._items[idx]
        return {"text": t, "targets": y, "mask": m, "meta": meta}


def _collate(tokenizer, batch: List[Dict[str, Any]], *, max_length: int) -> Dict[str, Any]:
    texts = [x["text"] for x in batch]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=int(max_length),
        padding=True,
        return_tensors="pt",
    )
    targets = torch.tensor(np.stack([x["targets"] for x in batch], axis=0), dtype=torch.float32)
    mask = torch.tensor(np.stack([x["mask"] for x in batch], axis=0), dtype=torch.float32)
    enc["targets"] = targets
    enc["mask"] = mask
    enc["meta"] = [x["meta"] for x in batch]
    return enc


def _find_output_linear(model, *, out_features: int) -> Optional[torch.nn.Linear]:
    best = None
    best_name = ""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and int(module.out_features) == int(out_features):
            best = module
            best_name = name
    if best is not None:
        return best
    for attr in ("score", "classifier", "out_proj"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Linear) and int(m.out_features) == int(out_features):
            return m
    if best_name:
        return best
    return None


def _copy_head0(init_model, new_model, *, new_out_features: int) -> bool:
    init_head = _find_output_linear(init_model, out_features=1)
    new_head = _find_output_linear(new_model, out_features=new_out_features)
    if init_head is None or new_head is None:
        return False
    try:
        with torch.no_grad():
            if init_head.weight.ndim == 2 and new_head.weight.ndim == 2:
                new_head.weight[0].copy_(init_head.weight[0])
            if getattr(init_head, "bias", None) is not None and getattr(new_head, "bias", None) is not None:
                new_head.bias[0].copy_(init_head.bias[0])
        return True
    except Exception:
        return False


def _eval_multitask(
    model,
    tokenizer,
    loader: DataLoader,
    *,
    device: str,
    head_names: Sequence[str],
    head_weights: np.ndarray,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    teacher_overall: List[Optional[float]] = []

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    w = torch.tensor(head_weights.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in loader:
            targets = batch.pop("targets").to(device)
            mask = batch.pop("mask").to(device)
            meta = batch.pop("meta", None)
            if isinstance(meta, list):
                for r in meta:
                    v = r.get("teacher_overall") if isinstance(r, dict) else None
                    teacher_overall.append(_safe_float(v))
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            loss_mat = bce(logits, targets)
            wm = mask * w
            denom = float(torch.clamp(wm.sum(), min=1.0).detach().cpu().item())
            loss = float(((loss_mat * wm).sum() / float(denom)).detach().cpu().item())
            losses.append(loss)

            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_masks.append(mask.detach().cpu().numpy())

    if not all_logits:
        return {"n": 0, "loss_mean": None}

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

    out: Dict[str, Any] = {"n": int(logits.shape[0]), "loss_mean": float(np.mean(losses)) if losses else None}

    if logits.shape[1] >= 1:
        m0 = masks[:, 0] > 0.5
        if int(np.sum(m0)) > 0:
            y = targets[m0, 0]
            p = probs[m0, 0]
            uniq = set(float(x) for x in y.tolist())
            if uniq.issubset({0.0, 1.0}) and int(np.sum(y == 1.0)) > 0 and int(np.sum(y == 0.0)) > 0:
                out["greatness"] = {
                    "auc": _auc_roc((y >= 0.5).astype(np.int64), p.astype(np.float64)),
                    "mean_pos": float(np.mean(p[y >= 0.5])) if int(np.sum(y >= 0.5)) > 0 else None,
                    "mean_neg": float(np.mean(p[y < 0.5])) if int(np.sum(y < 0.5)) > 0 else None,
                }

    # Derived rubric overall (from the rubric_* category heads), compared to the teacher's
    # overall label if present in the dataset rows.
    if logits.shape[1] >= 2 and teacher_overall and len(teacher_overall) == int(logits.shape[0]):
        yy = np.asarray([(np.nan if v is None else float(v)) for v in teacher_overall], dtype=np.float64)
        mm = np.isfinite(yy)
        if int(np.sum(mm)) > 0:
            cat_w = rubric_category_weights()
            idxs: List[int] = []
            ws: List[float] = []
            for j in range(1, logits.shape[1]):
                lab = str(head_names[j])
                if not lab.startswith("rubric_"):
                    continue
                cat = lab[len("rubric_") :]
                wf = float(cat_w.get(cat, 1.0))
                if not math.isfinite(wf) or wf <= 0:
                    continue
                idxs.append(int(j))
                ws.append(float(wf))
            if idxs and float(sum(ws)) > 0:
                wv = (np.asarray(ws, dtype=np.float64) / float(sum(ws))).reshape(1, -1)
                pp = (probs[:, idxs].astype(np.float64) * wv).sum(axis=1)
                y = yy[mm].astype(np.float64)
                p = pp[mm].astype(np.float64)
                out["rubric_overall_from_categories"] = {
                    "n": int(y.size),
                    "mse": float(np.mean((p - y) ** 2)),
                    "mae": float(np.mean(np.abs(p - y))),
                    "pearson": _pearsonr(p, y),
                }

    cats: Dict[str, Any] = {}
    for j in range(1, logits.shape[1]):
        m = masks[:, j] > 0.5
        if int(np.sum(m)) <= 0:
            continue
        y = targets[m, j].astype(np.float64)
        p = probs[m, j].astype(np.float64)
        cats[str(head_names[j])] = {
            "n": int(y.size),
            "mse": float(np.mean((p - y) ** 2)),
            "mae": float(np.mean(np.abs(p - y))),
            "pearson": _pearsonr(p, y),
        }
    if cats:
        out["rubric_categories"] = cats

    return out


def eval_multihead_scorer(
    *,
    model_path_or_id: str,
    samples_path: Path,
    positive_sources: Sequence[str],
    negative_sources: Optional[Sequence[str]],
    teacher_label_key: str,
    teacher_categories_key: str,
    rubric_categories: Sequence[str],
    doc_type: str,
    normalize_text: bool,
    max_length: int,
    batch_size: int,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    rows = list(_iter_jsonl(samples_path))
    ds = _JsonlMultiTargetDataset(
        rows,
        doc_type=str(doc_type),
        normalize_text=bool(normalize_text),
        positive_sources=positive_sources,
        negative_sources=negative_sources,
        teacher_label_key=str(teacher_label_key),
        teacher_categories_key=str(teacher_categories_key),
        rubric_categories=tuple(rubric_categories),
    )
    if len(ds) <= 0:
        return {"n": 0, "loss_mean": None}

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

    head_names = tuple(ds.head_names)
    w = np.ones((len(head_names),), dtype=np.float32)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, collate_fn=lambda b: _collate(tok, b, max_length=int(max_length)))
    return _eval_multitask(model, tok, loader, device=dev, head_names=head_names, head_weights=w)


@dataclass(frozen=True)
class MultiheadTrainSummary:
    out_dir: str
    base_model: str
    doc_type: str
    max_length: int
    train_rows: int
    val_rows: int
    device: str
    steps: int
    head_names: Tuple[str, ...]
    head_weights: Tuple[float, ...]
    best_val_loss: Optional[float]


def train_multihead_scorer(
    *,
    train_path: Path,
    val_path: Optional[Path],
    out_dir: Path,
    base_model: str,
    init_model_for_head0: Optional[str],
    doc_type: str,
    normalize_text: bool,
    positive_sources: Sequence[str],
    negative_sources: Optional[Sequence[str]],
    teacher_label_key: str,
    teacher_categories_key: str,
    rubric_categories: Sequence[str],
    head_weights: Sequence[float],
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
    freeze_backbone: bool = False,
    primary_weights: Optional[Dict[str, float]] = None,
) -> MultiheadTrainSummary:
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(_iter_jsonl(Path(train_path)))
    val_rows = list(_iter_jsonl(Path(val_path))) if val_path is not None else []

    train_ds = _JsonlMultiTargetDataset(
        train_rows,
        doc_type=str(doc_type),
        normalize_text=bool(normalize_text),
        positive_sources=positive_sources,
        negative_sources=negative_sources,
        teacher_label_key=str(teacher_label_key),
        teacher_categories_key=str(teacher_categories_key),
        rubric_categories=tuple(rubric_categories),
    )
    val_ds = (
        _JsonlMultiTargetDataset(
            val_rows,
            doc_type=str(doc_type),
            normalize_text=bool(normalize_text),
            positive_sources=positive_sources,
            negative_sources=negative_sources,
            teacher_label_key=str(teacher_label_key),
            teacher_categories_key=str(teacher_categories_key),
            rubric_categories=tuple(rubric_categories),
        )
        if val_rows
        else None
    )

    n_heads = int(train_ds.n_heads)
    head_names = tuple(train_ds.head_names)
    w = np.asarray(list(head_weights), dtype=np.float32)
    if w.size != n_heads:
        raise ValueError(f"head_weights length {w.size} must match n_heads={n_heads}")
    if not np.all(np.isfinite(w)):
        raise ValueError("head_weights must be finite floats")
    if float(np.sum(w)) <= 0:
        raise ValueError("head_weights sum must be >0")

    model_id = str(base_model)
    trc = bool(trust_remote_code) or _needs_trust_remote_code(model_id)
    tok_kwargs = {"trust_remote_code": trc}
    try:
        tok = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True, **tok_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    _ensure_pad_token(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=n_heads,
        ignore_mismatched_sizes=True,
        trust_remote_code=trc,
    )
    if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
        model.config.pad_token_id = tok.pad_token_id

    model.config.id2label = {int(i): str(name) for i, name in enumerate(head_names)}
    model.config.label2id = {str(name): int(i) for i, name in enumerate(head_names)}
    model.config.horace_primary = {
        "kind": "weighted_sum",
        "heads": dict(primary_weights) if isinstance(primary_weights, dict) else {"greatness": 1.0},
    }

    init_for_copy = (init_model_for_head0 or "").strip()
    copied = False
    if init_for_copy and n_heads > 1:
        try:
            init_model = AutoModelForSequenceClassification.from_pretrained(
                init_for_copy,
                num_labels=1,
                ignore_mismatched_sizes=True,
                trust_remote_code=trc,
            )
            copied = _copy_head0(init_model, model, new_out_features=n_heads)
            del init_model
        except Exception:
            copied = False
    model.config.horace_head0_copied = bool(copied)

    dev = _best_device(device)
    model.to(dev)

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

    if bool(freeze_backbone):
        for p in model.parameters():
            p.requires_grad = False
        head = _find_output_linear(model, out_features=n_heads)
        if head is None:
            raise RuntimeError("freeze_backbone requested but could not find classifier head module")
        for p in head.parameters():
            p.requires_grad = True
        model.config.horace_freeze_backbone = True

    # Gradient checkpointing is only useful when backpropagating through the backbone.
    # If we're training only a classifier head, enabling it can create noisy warnings and overhead.
    if bool(gradient_checkpointing) and (not bool(freeze_backbone)) and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    collate = lambda b: _collate(tok, b, max_length=int(max_length))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate) if val_ds else None

    trainable_params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    if not trainable_params:
        raise RuntimeError("No trainable parameters (check freeze_backbone / LoRA config)")
    optim = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    head_w = torch.tensor(w.reshape(1, -1), dtype=torch.float32, device=dev)

    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    steps = 0
    ga = max(1, int(grad_accum_steps))
    use_amp = bool(bf16) and dev == "cuda"

    optim.zero_grad(set_to_none=True)
    for epoch in range(max(1, int(epochs))):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False, disable=_tqdm_disabled())
        for micro_step, batch in enumerate(pbar, start=1):
            targets = batch.pop("targets").to(dev)
            mask = batch.pop("mask").to(dev)
            batch.pop("meta", None)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(**batch)
                logits = out.logits
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                loss_mat = bce(logits, targets)
                wm = mask * head_w
                denom = torch.clamp(wm.sum(), min=1.0)
                loss = (loss_mat * wm).sum() / denom
                loss = loss / float(ga)
            loss.backward()
            if micro_step % ga == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
            steps += 1
            if not _tqdm_disabled():
                pbar.set_postfix({"loss": float(loss.detach().cpu().item()) * float(ga)})

        if val_loader is not None:
            metrics = _eval_multitask(model, tok, val_loader, device=dev, head_names=head_names, head_weights=w.astype(np.float32))
            vloss = metrics.get("loss_mean")
            if isinstance(vloss, (int, float)) and math.isfinite(float(vloss)):
                if best_val_loss is None or float(vloss) < float(best_val_loss):
                    best_val_loss = float(vloss)
                    best_epoch = int(epoch + 1)
                    save_model = model
                    if bool(merge_lora) and hasattr(model, "merge_and_unload"):
                        try:
                            save_model = model.merge_and_unload()
                        except Exception:
                            save_model = model
                    save_model.save_pretrained(out_dir)
                    tok.save_pretrained(out_dir)

    if val_loader is None:
        save_model = model
        if bool(merge_lora) and hasattr(model, "merge_and_unload"):
            try:
                save_model = model.merge_and_unload()
            except Exception:
                save_model = model
        save_model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

    report: Dict[str, Any] = {
        "run_meta": {
            "train_path": str(train_path),
            "val_path": str(val_path) if val_path is not None else None,
            "base_model": str(base_model),
            "init_model_for_head0": str(init_for_copy) if init_for_copy else None,
            "doc_type": str(doc_type),
            "normalize_text": bool(normalize_text),
            "positive_sources": sorted({str(x) for x in positive_sources}),
            "negative_sources": sorted({str(x) for x in (negative_sources or [])}) if negative_sources else None,
            "teacher_label_key": str(teacher_label_key),
            "teacher_categories_key": str(teacher_categories_key),
            "rubric_categories": list(rubric_categories),
            "head_names": list(head_names),
            "head_weights": [float(x) for x in w.tolist()],
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "seed": int(seed),
            "device": str(dev),
            "trust_remote_code": bool(trc),
            "primary": model.config.horace_primary if hasattr(model.config, "horace_primary") else None,
            "head0_copied": bool(getattr(model.config, "horace_head0_copied", False)),
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
        "train": {"n": int(len(train_ds))},
        "val": None,
        "best": {"epoch": best_epoch, "val_loss": best_val_loss},
    }
    if val_loader is not None:
        report["val"] = _eval_multitask(model, tok, val_loader, device=dev, head_names=head_names, head_weights=w.astype(np.float32))

    (out_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return MultiheadTrainSummary(
        out_dir=str(out_dir),
        base_model=str(base_model),
        doc_type=str(doc_type),
        max_length=int(max_length),
        train_rows=int(len(train_ds)),
        val_rows=int(len(val_ds)) if val_ds is not None else 0,
        device=str(dev),
        steps=int(steps),
        head_names=head_names,
        head_weights=tuple(float(x) for x in w.tolist()),
        best_val_loss=best_val_loss,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a multi-head text→score scorer (greatness + rubric breakdown).")
    ap.add_argument("--train", required=True, help="Train split JSONL")
    ap.add_argument("--val", default="", help="Optional val split JSONL")
    ap.add_argument("--out-dir", required=True, help="Output directory (HF save_pretrained)")

    ap.add_argument("--base-model", default="distilbert-base-uncased", help="HF model id or model dir")
    ap.add_argument("--init-model-for-head0", default="", help="Optional 1-head model to copy into head0 before training")
    ap.add_argument("--doc-type", default="prose")
    ap.add_argument("--normalize-text", action="store_true")
    ap.add_argument("--no-normalize-text", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument("--teacher-label-key", default="label", help="Float rubric overall label key in [0,1]")
    ap.add_argument("--teacher-categories-key", default="teacher_categories_0_1", help="Dict of rubric category labels in [0,1]")
    ap.add_argument("--rubric-category", action="append", default=[], help="Rubric category name (repeatable)")

    ap.add_argument("--pos", action="append", default=[], help="Positive source label(s) (repeatable)")
    ap.add_argument("--neg", action="append", default=[], help="Negative source label(s) (repeatable); empty => all non-pos")

    ap.add_argument("--head-weight", action="append", default=[], help="Per-head loss weight (repeatable, must match heads)")
    ap.add_argument("--primary", default="", help='JSON dict like {"greatness":0.7,"rubric_overall":0.3}')

    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="")

    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", action="append", default=[])
    ap.add_argument("--merge-lora", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    args = ap.parse_args(argv)

    normalize_text = bool(args.normalize_text) or not bool(args.no_normalize_text)
    pos = tuple(args.pos or [])
    neg = tuple(args.neg or []) if (args.neg or []) else None
    if not pos:
        raise SystemExit("--pos is required")

    cats = tuple(str(x) for x in (args.rubric_category or []) if str(x).strip()) or _DEFAULT_RUBRIC_CATEGORIES
    head_names = ("greatness",) + tuple(f"rubric_{c}" for c in cats)

    w = [float(x) for x in (args.head_weight or [])] if (args.head_weight or []) else []
    if not w:
        w = [1.0] * len(head_names)
    if len(w) != len(head_names):
        raise SystemExit(f"--head-weight must appear {len(head_names)} times (got {len(w)})")

    primary = None
    if str(args.primary).strip():
        try:
            obj = json.loads(str(args.primary))
            primary = obj if isinstance(obj, dict) else None
        except Exception:
            primary = None

    lora_target = tuple(str(x) for x in (args.lora_target or []) if str(x).strip()) or None
    summary = train_multihead_scorer(
        train_path=Path(str(args.train)),
        val_path=Path(str(args.val)) if str(args.val).strip() else None,
        out_dir=Path(str(args.out_dir)),
        base_model=str(args.base_model),
        init_model_for_head0=str(args.init_model_for_head0).strip() or None,
        doc_type=str(args.doc_type),
        normalize_text=bool(normalize_text),
        positive_sources=pos,
        negative_sources=neg,
        teacher_label_key=str(args.teacher_label_key),
        teacher_categories_key=str(args.teacher_categories_key),
        rubric_categories=cats,
        head_weights=tuple(float(x) for x in w),
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
        primary_weights=primary,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
