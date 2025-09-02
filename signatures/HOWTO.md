# HOWTO: Signatures CLI

## How to run

- Install dependencies (matplotlib newly added):

```bash
python3 -m pip install -r requirements.txt
```

- Basic classification using repo texts:
  - Option A (manifest): create a JSON/CSV/JSONL with rows: path, author[, title, year], then:

```bash
python3 signatures/cli.py --help
# Use PyTorch+HF (default if available) or MLX on Apple Silicon with --backend mlx
python3 signatures/cli.py --model gpt2 --manifest data/index.json --outdir reports/generated
python3 signatures/cli.py --backend mlx --model gpt2 --manifest data/index.json --outdir reports/generated
```

  - Option B (direct files): pair files with authors:

```bash
python3 signatures/cli.py --model gpt2 \
  --file data/novel/ernest_hemingway/the_old_man_and_the_sea.txt --author "Ernest Hemingway" \
  --file data/novel/f_scott_fitzgerald/the_great_gatsby.txt      --author "F Scott Fitzgerald" \
  --outdir reports/generated
```

- Add a trajectory plot (choose an author with multiple docs):

```bash
python3 signatures/cli.py --model gpt2 --manifest data/index.json \
  --trajectory-author "P G Wodehouse" --outdir reports/generated
```

- Speed/trial options:
  - Use smaller chunks: `--chunk-tokens 800`
  - Limit per-doc chunks: `--limit-chunks 2`
  - Skip plots (headless): `--no-plots`

## What the CLI does

- Token surprisal stream: Computes s_t = -log p(x_t|x_<t) and local entropy H_t via a HuggingFace causal LM (default gpt2). Handles long texts by sliding windows.
- Signature features per chunk:
  - Distribution shape: s-quantiles (5–95), tail index, Q90−Q50; normalized r = s/H quantiles (per idea.md).
  - Rhythm: ACF (lags 1–10), spectrum slope.
  - Burstiness/high-surprisal runs: run_mean, run_q90 (s ≥ 90th percentile).
- Classification:
  - Builds feature matrix, z-scores on train folds, nearest-centroid classification with leave-one-doc-out CV, reports chunk-level accuracy and confusion matrix.
- Early→late trajectory:
  - Aggregates per-doc average signature vectors for the chosen author, standardizes, projects to 2D with PCA (SVD), and connects points in chronological order (if year provided) or by title as a fallback.

## Outputs

- `reports/generated/signatures.jsonl`: per-chunk feature dictionaries with doc metadata.
- `reports/generated/classification_report.json`: accuracy, per-fold stats, labels, and confusion matrix.
- `reports/generated/classification_confusion_matrix.png`: confusion matrix plot.
- `reports/generated/trajectory_<author>.png`: early→late PCA trajectory plot (if requested).

## Notes

- Device: Automatically picks CUDA, then Apple Silicon MPS, then CPU.
- Years: For best “early→late” ordering, include year in the manifest (`data/index.json` doesn’t include it; plot will fall back to title order).
- Extensibility: The CLI focuses on the compact, robust features in idea.md. Dialogue/position/POS/depth-conditioned features can be added later once span/parse metadata is available.

## Requirements coverage

- New CLI provided and kept `signatures/sketch.py` unchanged: Done
- Accepts plain text paths (manifest or file/author): Done
- Computes idea.md signature features: Done (A/B core + runs)
- Author classification output: Done (JSON + confusion PNG)
- Early→late trajectory plot for chosen author: Done (PNG)
- Verified CLI help runs locally: PASS

## Try it

```bash
python3 signatures/cli.py --model gpt2 --manifest data/index.json --trajectory-author "P G Wodehouse" --outdir reports/generated
```
