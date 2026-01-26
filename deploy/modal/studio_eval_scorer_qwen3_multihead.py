"""
Modal job: evaluate a trained Qwen3 multihead scorer on the held-out test splits.

So what:
- The training job can be long-running and occasionally get interrupted.
- This runs evaluation in a fresh container (no training leftovers in GPU memory),
  and writes a durable JSON report to the horace-data volume.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-studio-eval-scorer-qwen3-multihead"
REPO_REMOTE_PATH = "/root/horace"


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists() and (p / "data").exists():
            return p
    return Path.cwd()


_LOCAL_REPO_ROOT = _local_repo_root()

data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("horace-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu121", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy>=1.24.0",
        "transformers>=4.40.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.2.0",
        "safetensors>=0.4.0",
    )
)
if (_LOCAL_REPO_ROOT / "tools").exists():
    image = image.add_local_dir(_LOCAL_REPO_ROOT / "tools", remote_path=f"{REPO_REMOTE_PATH}/tools")

app = modal.App(APP_NAME)


def _bootstrap_repo() -> None:
    import sys

    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)
    os.chdir(REPO_REMOTE_PATH)
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HORACE_TQDM_DISABLE", "1")


def _read_model_heads(model_dir: Path) -> Tuple[List[str], List[str]]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing model config.json: {cfg_path}")
    obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    id2label = obj.get("id2label") if isinstance(obj, dict) else None
    if not isinstance(id2label, dict):
        raise ValueError("config.json missing id2label mapping")
    labels: Dict[int, str] = {}
    for k, v in id2label.items():
        try:
            i = int(k)
        except Exception:
            continue
        labels[int(i)] = str(v)
    head_labels = [labels[i] for i in sorted(labels)]
    rubric_categories: List[str] = []
    marker_heads: List[str] = []
    for lab in head_labels:
        if lab.startswith("rubric_") and lab not in ("rubric_overall", "rubric_overall_from_categories"):
            rubric_categories.append(lab[len("rubric_") :])
        elif lab.startswith("marker_"):
            marker_heads.append(lab)
    if not rubric_categories:
        rubric_categories = ["focus", "cadence", "cohesion", "alignment", "distinctiveness"]
    return rubric_categories, marker_heads


@app.function(
    image=image,
    gpu="any",
    timeout=60 * 30,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
)
def eval_remote(cfg_json: str) -> str:
    _bootstrap_repo()
    cfg = json.loads(cfg_json)

    from tools.studio.train_multihead_scorer import eval_multihead_scorer

    started_unix = int(time.time())
    model_dir = Path(str(cfg["model_dir"]))
    run_dir = Path("/vol/reports/eval_runs") / f"{model_dir.name}_{started_unix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def _checkpoint(stage: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "stage": str(stage),
            "started_unix": int(started_unix),
            "now_unix": int(time.time()),
            "model_dir": str(model_dir),
        }
        if extra:
            payload["extra"] = extra
        _write_json(run_dir / "status.json", payload)
        try:
            data_vol.commit()
        except Exception:
            pass

    def _write_error(stage: str) -> None:
        payload: Dict[str, Any] = {
            "stage": str(stage),
            "started_unix": int(started_unix),
            "now_unix": int(time.time()),
            "model_dir": str(model_dir),
            "traceback": traceback.format_exc(),
        }
        _write_json(run_dir / "error.json", payload)
        try:
            data_vol.commit()
        except Exception:
            pass
        try:
            hf_cache_vol.commit()
        except Exception:
            pass

    _write_json(run_dir / "cfg.json", cfg)
    data_vol.commit()

    rubric_categories, marker_heads = _read_model_heads(model_dir)
    marker_baseline = str(cfg.get("marker_baseline") or "").strip()
    marker_head_mode = str(cfg.get("marker_head_mode") or "match_baseline")
    max_length = int(cfg.get("max_length") or 512)
    batch_size = int(cfg.get("batch_size") or 2)

    _checkpoint("eval_great_other", extra={"rubric_categories": rubric_categories, "marker_heads": marker_heads})
    try:
        eval_great_other = eval_multihead_scorer(
            model_path_or_id=str(model_dir),
            samples_path=Path(str(cfg["test_great_other"])),
            positive_sources=("great_author",),
            negative_sources=("other_author",),
            teacher_label_key="label",
            teacher_categories_key="teacher_categories_0_1",
            rubric_categories=tuple(rubric_categories),
            marker_heads=tuple(marker_heads),
            marker_baseline=marker_baseline or None,
            marker_head_mode=marker_head_mode,
            doc_type="prose",
            normalize_text=True,
            max_length=max_length,
            batch_size=batch_size,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
        )
    except Exception:
        _write_error("eval_great_other")
        raise

    _checkpoint("eval_teacher")
    try:
        eval_teacher = eval_multihead_scorer(
            model_path_or_id=str(model_dir),
            samples_path=Path(str(cfg["test_teacher"])),
            positive_sources=("great_author",),
            negative_sources=("other_author",),
            teacher_label_key="label",
            teacher_categories_key="teacher_categories_0_1",
            rubric_categories=tuple(rubric_categories),
            marker_heads=tuple(marker_heads),
            marker_baseline=marker_baseline or None,
            marker_head_mode=marker_head_mode,
            doc_type="prose",
            normalize_text=True,
            max_length=max_length,
            batch_size=batch_size,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
        )
    except Exception:
        _write_error("eval_teacher")
        raise

    result_obj = {
        "model_dir": str(model_dir),
        "rubric_categories": rubric_categories,
        "marker_heads": marker_heads,
        "marker_baseline": marker_baseline or None,
        "marker_head_mode": str(marker_head_mode),
        "eval_great_other_test": eval_great_other,
        "eval_teacher_test": eval_teacher,
    }
    _write_json(run_dir / "result.json", result_obj)
    _checkpoint("done")
    try:
        data_vol.commit()
    except Exception:
        pass
    try:
        hf_cache_vol.commit()
    except Exception:
        pass
    return json.dumps({"run_dir": str(run_dir), "result": result_obj}, ensure_ascii=False, indent=2)


@app.local_entrypoint()
def main(  # pragma: no cover
    model_dir: str = "/vol/models/scorer_qwen3_multihead_v10",
    test_great_other: str = "/vol/corpora/mixed_supervision_v2/splits/test_great_other.jsonl",
    test_teacher: str = "/vol/corpora/mixed_supervision_v2/splits/test_teacher.jsonl",
    marker_baseline: str = "/vol/baselines/gpt2_mixed_teacher_v1_rubricv3_512_docs.json",
    marker_head_mode: str = "match_baseline",
    max_length: int = 512,
    batch_size: int = 2,
) -> None:
    cfg = {
        "model_dir": str(model_dir),
        "test_great_other": str(test_great_other),
        "test_teacher": str(test_teacher),
        "marker_baseline": str(marker_baseline),
        "marker_head_mode": str(marker_head_mode),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
    }
    print(eval_remote.remote(json.dumps(cfg)))

