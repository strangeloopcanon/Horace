"""
Featurize pair data on Modal with GPU-accelerated Qwen3-1.7B.

Batches multiple pairs per container to amortize model loading cost.

Run:
    make setup-modal
    modal run deploy/modal/studio_featurize_pairs_v6.py

Or with custom paths:
    modal run deploy/modal/studio_featurize_pairs_v6.py \
        --pairs-dir data/pairs_v7 \
        --model-id Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-featurize-pairs-v6"
REPO_REMOTE_PATH = "/root/horace"

# Cost knobs
_DEFAULT_GPU = "T4"  # T4 (~$0.59/hr) is sufficient for 1.7B models; A10G (~$1.10/hr) was overkill
_DEFAULT_BATCH_SIZE = 10  # pairs per container invocation (amortises model load)
_DEFAULT_MAX_CONTAINERS = 10


def _local_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "tools").exists():
            return p
    return Path.cwd()


_LOCAL_REPO_ROOT = _local_repo_root()

data_vol = modal.Volume.from_name("horace-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("horace-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu121", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy>=1.24.0", "transformers>=4.40.0", "safetensors>=0.4.0", "scikit-learn>=1.3.0")
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


def _featurize_single(
    row: dict,
    *,
    model_id: str,
    max_input_tokens: int,
) -> dict:
    """Featurize one pair (both chosen and rejected texts). Assumes model is already loaded."""
    from tools.studio.analyze import analyze_text
    from tools.studio.preference_features import FEATURE_SCHEMA, extract_features

    chosen_text = str(row.get("chosen_text") or "")
    rejected_text = str(row.get("rejected_text") or "")

    t0 = time.monotonic()

    chosen_result = analyze_text(
        chosen_text,
        model_id=model_id,
        doc_type="prose",
        backend="auto",
        max_input_tokens=max_input_tokens,
        compute_cohesion=True,
        include_token_metrics=False,
    )
    chosen_metrics = chosen_result.get("doc_metrics") or {}
    chosen_vec = extract_features(chosen_metrics, chosen_text)
    chosen_feat = {name: float(chosen_vec[j]) for j, name in enumerate(FEATURE_SCHEMA)}

    rejected_result = analyze_text(
        rejected_text,
        model_id=model_id,
        doc_type="prose",
        backend="auto",
        max_input_tokens=max_input_tokens,
        compute_cohesion=True,
        include_token_metrics=False,
    )
    rejected_metrics = rejected_result.get("doc_metrics") or {}
    rejected_vec = extract_features(rejected_metrics, rejected_text)
    rejected_feat = {name: float(rejected_vec[j]) for j, name in enumerate(FEATURE_SCHEMA)}

    elapsed = time.monotonic() - t0

    out_row = dict(row)
    out_row["chosen_features"] = chosen_feat
    out_row["rejected_features"] = rejected_feat
    out_row["_featurize_sec"] = round(elapsed, 1)
    return out_row


@app.function(
    image=image,
    gpu=_DEFAULT_GPU,
    timeout=60 * 60,  # 1 hour per batch (batches can be large)
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
    max_containers=_DEFAULT_MAX_CONTAINERS,
)
def featurize_batch(
    batch_json: str,
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 1024,
) -> str:
    """Featurize a batch of pairs on GPU. Model loads once per batch.

    Input:  JSON array of pair objects.
    Output: JSON array of featurized pair objects.
    """
    _bootstrap_repo()

    rows = json.loads(batch_json)
    results = []
    for i, row in enumerate(rows):
        try:
            out = _featurize_single(row, model_id=model_id, max_input_tokens=max_input_tokens)
            results.append(out)
        except Exception as e:
            pid = str(row.get("pair_id") or f"idx-{i}")
            print(f"  ERROR on pair {pid}: {type(e).__name__}: {e}")

    hf_cache_vol.commit()
    return json.dumps(results, ensure_ascii=False)


# --- Backward-compatible single-pair entry point (kept for any external callers) ---

@app.function(
    image=image,
    gpu=_DEFAULT_GPU,
    timeout=60 * 30,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
    max_containers=_DEFAULT_MAX_CONTAINERS,
)
def featurize_one_pair(
    row_json: str,
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 1024,
) -> str:
    """Featurize a single pair on GPU. Returns JSON string of the result."""
    _bootstrap_repo()
    row = json.loads(row_json)
    out = _featurize_single(row, model_id=model_id, max_input_tokens=max_input_tokens)
    hf_cache_vol.commit()
    return json.dumps(out, ensure_ascii=False)


def _chunk(lst: list, size: int) -> List[list]:
    """Split a list into chunks of at most `size`."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


@app.local_entrypoint()
def main(
    pairs_dir: str = "",
    model_id: str = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 1024,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> None:
    # Support explicit --pairs-dir or default to data/pairs_v6
    local_pairs_dir = Path(pairs_dir) if pairs_dir.strip() else _LOCAL_REPO_ROOT / "data" / "pairs_v6"
    if not local_pairs_dir.is_absolute():
        local_pairs_dir = _LOCAL_REPO_ROOT / local_pairs_dir

    print(f"GPU: {_DEFAULT_GPU}, batch_size: {batch_size}, max_containers: {_DEFAULT_MAX_CONTAINERS}")

    for split in ("train", "val", "test"):
        split_path = local_pairs_dir / f"{split}.jsonl"
        out_path = local_pairs_dir / f"{split}_featurized.jsonl"

        if not split_path.exists():
            print(f"SKIP {split}: {split_path} not found")
            continue

        # Load pairs
        pairs: List[dict] = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))

        # Skip already-processed pairs (resume support)
        already_done: set = set()
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            pid = str(row.get("pair_id") or "")
                            if pid:
                                already_done.add(pid)
                        except json.JSONDecodeError:
                            pass

        todo = [p for p in pairs if str(p.get("pair_id") or "") not in already_done]

        if not todo:
            print(f"{split}: all {len(pairs)} pairs already featurized, skipping")
            continue

        batches = _chunk(todo, batch_size)
        print(
            f"{split}: {len(todo)} pairs to featurize ({len(already_done)} already done) "
            f"→ {len(batches)} batches of ≤{batch_size}"
        )

        batch_jsons = [json.dumps(b, ensure_ascii=False) for b in batches]
        t0 = time.monotonic()
        n_done = 0
        n_errors = 0

        mode = "a" if already_done else "w"
        with open(out_path, mode, encoding="utf-8") as fout:
            for batch_result_json in featurize_batch.map(
                batch_jsons,
                kwargs={"model_id": str(model_id), "max_input_tokens": int(max_input_tokens)},
            ):
                try:
                    batch_results = json.loads(batch_result_json)
                    for row in batch_results:
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_done += len(batch_results)
                except Exception as e:
                    print(f"  ERROR processing batch result: {e}")
                    n_errors += 1

                elapsed = time.monotonic() - t0
                rate = float(n_done) / max(elapsed, 0.01)
                remaining = (len(todo) - n_done) / max(rate, 0.001)
                print(
                    f"  [{split}] {n_done}/{len(todo)} done, "
                    f"{n_errors} batch errors, {rate:.2f} pairs/s, ETA {remaining/60:.1f}min"
                )

        elapsed = time.monotonic() - t0
        print(
            f"  {split} DONE: {n_done} featurized, {n_errors} batch errors, "
            f"{elapsed:.0f}s elapsed"
        )

    print("\nAll splits complete.")
