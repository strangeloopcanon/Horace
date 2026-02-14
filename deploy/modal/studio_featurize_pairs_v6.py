"""
Featurize v6 pair data on Modal with GPU-accelerated Qwen3-1.7B.

Uses modal.map() to process pairs in parallel across many GPU containers,
rather than sequentially in a single container.

Run:
    make setup-modal
    modal run deploy/modal/studio_featurize_pairs_v6.py

Or with custom paths:
    modal run deploy/modal/studio_featurize_pairs_v6.py \
        --pairs-dir /vol/pairs_v6 \
        --model-id Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _lazy_import_modal():
    try:
        import modal  # type: ignore

        return modal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Modal is not installed. Install with: pip install modal") from e


modal = _lazy_import_modal()

APP_NAME = "horace-featurize-pairs-v6"
REPO_REMOTE_PATH = "/root/horace"


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


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/vol": data_vol, "/cache/hf": hf_cache_vol},
    concurrency_limit=20,
)
def featurize_one_pair(
    row_json: str,
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 1024,
) -> str:
    """Featurize a single pair on GPU. Returns JSON string of the result."""
    _bootstrap_repo()

    from tools.studio.analyze import analyze_text
    from tools.studio.preference_features import FEATURE_SCHEMA, extract_features

    row = json.loads(row_json)
    pid = str(row.get("pair_id") or "")
    chosen_text = str(row.get("chosen_text") or "")
    rejected_text = str(row.get("rejected_text") or "")

    t0 = time.monotonic()

    # Analyze chosen text
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

    # Analyze rejected text
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

    hf_cache_vol.commit()
    return json.dumps(out_row, ensure_ascii=False)


@app.local_entrypoint()
def main(
    pairs_dir: str = "",
    model_id: str = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 1024,
) -> None:
    local_pairs_dir = _LOCAL_REPO_ROOT / "data" / "pairs_v6"

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

        print(f"{split}: {len(todo)} pairs to featurize ({len(already_done)} already done)")

        # Use modal.map for parallel processing
        pair_jsons = [json.dumps(p, ensure_ascii=False) for p in todo]
        t0 = time.monotonic()
        n_done = 0
        n_errors = 0

        mode = "a" if already_done else "w"
        with open(out_path, mode, encoding="utf-8") as fout:
            for result in featurize_one_pair.map(
                pair_jsons,
                kwargs={"model_id": str(model_id), "max_input_tokens": int(max_input_tokens)},
            ):
                try:
                    fout.write(result + "\n")
                    fout.flush()
                    n_done += 1
                except Exception as e:
                    print(f"  ERROR writing result: {e}")
                    n_errors += 1

                if (n_done + n_errors) % 20 == 0:
                    elapsed = time.monotonic() - t0
                    rate = float(n_done) / max(elapsed, 0.01)
                    remaining = (len(todo) - n_done - n_errors) / max(rate, 0.001)
                    print(
                        f"  [{split}] {n_done}/{len(todo)} done, "
                        f"{n_errors} errors, {rate:.2f}/s, ETA {remaining/60:.1f}min"
                    )

        elapsed = time.monotonic() - t0
        print(
            f"  {split} DONE: {n_done} featurized, {n_errors} errors, "
            f"{elapsed:.0f}s elapsed"
        )

    print("\nAll splits complete.")
