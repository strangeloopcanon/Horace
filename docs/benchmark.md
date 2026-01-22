# Horace Studio — Benchmark Data & Held‑Out Evaluation

So what: if the product promise is “paste text → tell me if it matches top‑literature patterns”, we need a **frozen benchmark snapshot** and a **leakage‑safe split** so evals are trustworthy.

## What data we need (minimum viable, done now)

1) **Baseline corpus** (for “profile vs top literature” percentiles)
- Sources: public‑domain literary prose (Project Gutenberg excerpts; Standard Ebooks corpus).
- Output: `data/baselines/<model>_gutenberg_512_docs.json` (example: `data/baselines/gpt2_gutenberg_512_docs.json`).
- Note: for strictness, prefer baseline books that are **different** from benchmark “positive” books.

2) **Benchmark snapshot** (for “is it literary?” separation)
- Sources (license‑safe):
  - `gutenberg_excerpt` (positive proxy; public domain)
  - `wikipedia_summary` (good non‑literary prose; CC BY‑SA)
  - `wikinews_published` (news; CC)
  - `nasa_breaking_news` (gov writing; likely public domain)
  - `rfc_excerpt` (technical; open standards)
  - `gibberish_control` (low‑quality control)
- Required fields per row:
  - `sample_id` (unique)
  - `group_id` (document‑level id for leakage‑safe splitting)
  - `source`, `title`, `url`, `text`, `fetched_at_unix`, `meta`

3) **Leakage‑safe splits**
- Split at the **group/document level** (`group_id`), not by individual samples.
- Keep the split frozen once created.

## What’s implemented

- Small fixed set (quick regression): `data/eval_sets/studio_fixed_v1.jsonl`
- Group splitter: `python -m tools.studio.split_eval_set --samples ... --out-dir ...` (`tools/studio/split_eval_set.py`)
- Larger benchmark snapshot builder (downloads + splits): `python -m tools.studio.build_benchmark_set` (`tools/studio/build_benchmark_set.py`)
- Download caching: HTTP fetches are cached under `data/cache/http/` by default (gitignored). Set `HORACE_HTTP_NO_CACHE=1` to disable or `HORACE_HTTP_CACHE_DIR` to override.

## Recommended protocol (no tainted evals)

1) Build/freeze dataset (once per version)
- `make build-benchmark-set`

Makefile shortcuts (same protocol):
- `make eval-benchmark-train`
- `make eval-benchmark-val`
- `make eval-benchmark-test`
- `make train-calibrator-benchmark`

2) Score splits (produces reports with rubric outputs/features)
- Train: `python -m tools.studio.eval_set --samples data/benchmarks/studio_benchmark_v3/splits/train.jsonl --report-out reports/benchmark_train.json`
- Val: `python -m tools.studio.eval_set --samples data/benchmarks/studio_benchmark_v3/splits/val.jsonl --report-out reports/benchmark_val.json`
- Test: `python -m tools.studio.eval_set --samples data/benchmarks/studio_benchmark_v3/splits/test.jsonl --report-out reports/benchmark_test.json`

3) Train calibrator **only on train**
- `python -m tools.studio.train_calibrator --report reports/benchmark_train.json --out reports/calibrators/benchmark_calibrator.json --pos gutenberg_excerpt --neg wikipedia_summary --neg wikinews_published --neg nasa_breaking_news --neg rfc_excerpt --neg gibberish_control`

4) Evaluate calibrator on val/test
- `python -m tools.studio.eval_set --samples data/benchmarks/studio_benchmark_v3/splits/test.jsonl --calibrator reports/calibrators/benchmark_calibrator.json --report-out reports/benchmark_test_calibrated.json`

## Example results (v3 snapshot, GPT‑2, 512 tokens)

These are from a local run of `make train-calibrator-benchmark` (held‑out: calibrator trained on train split; evaluated on test).

- Raw rubric score AUC on test: `0.646`
- Calibrated score AUC on test: `0.879`

Reports:
- `reports/studio_benchmark_train_report.json`
- `reports/studio_benchmark_test_report.json`
- `reports/studio_benchmark_test_report_calibrated.json`

## Fixing “domain shortcut” (v4, within Gutenberg)

So what: v3 mixes domains (Wikipedia/news/RFC), so a model can “cheat” by learning domain cues. v4 makes the primary task **within‑domain**:
- Positives: `gutenberg_top_excerpt` (top downloaded)
- Negatives: `gutenberg_random_excerpt` (long‑tail by download rank) + `gutenberg_corrupt_*` (controlled degradations)

Build v4 snapshot:
```bash
make build-benchmark-v4
```

## Curated literary corpus (Standard Ebooks)

So what: Standard Ebooks is often a better “great writing” proxy than raw Gutenberg because the texts are curated/cleaned. We ingest it via EPUB extraction and sample readable windows.

Build snapshot (writes under `data/corpora/`, gitignored):
```bash
make build-standardebooks-corpus
```

Build on Modal (recommended for larger runs; writes to `/vol/corpora/standardebooks_corpus_v1`):
```bash
make modal-build-standardebooks-corpus
```

## Great authors vs others (raw → windows → teacher labels)

So what: to train a single fast scorer without losing interpretability, we want:
- an explicit **great-author vs other-author** anchor label (from an allowlist), and
- optional **rubric teacher labels** (slow, diagnostic) for breakdown supervision.

Data sources:
- Standard Ebooks (EPUB → cleaned text)
- Project Gutenberg (via official `pg_catalog.csv` for broader coverage)

Build locally (gitignored, resumable):
```bash
# 1) Download raw full texts and bucket into great_author/other_author
make download-gutenberg-raw GUTENBERG_ID_SOURCE=catalog
make download-standardebooks-raw

# 2) Sample leakage-safe fixed windows and merge into one corpus with splits
make build-mixed-windows-corpus

# 3) Build a baseline from great-author train groups and label splits with the rubric teacher
make label-mixed-windows LABEL_MAX_SAMPLES=0
```

Notes:
- Set caps like `GUTENBERG_MAX_BYTES=30000000000` (≈30GB) / `STD_EBOOKS_MAX_BYTES=...` to control disk usage.
- Allowlist: `configs/great_authors_v1.txt`.

## Modern prose corpus (RSS / essays / news)

So what: to avoid “literary == old public-domain prose” shortcuts, we ingest modern prose via RSS/Atom
feeds (default: The Conversation + ProPublica).

Build snapshot locally (writes under `data/corpora/`, gitignored):
```bash
make build-rss-corpus
```

Build on Modal (writes to `/vol/corpora/rss_corpus_v1`):
```bash
make modal-build-rss-corpus
```

## Single scorer model (text → score)

So what: the calibrator is useful, but still depends on token‑level analysis at inference. The scorer model is a **single HF model** that maps normalized text → a score (0–100).

Train locally (writes to `models/`, gitignored):
```bash
make train-scorer-v4
```

Train on Modal (GPU; writes to `/vol/models/scorer_v4` in your `horace-data` volume):
```bash
make modal-train-scorer-v4
```

## Distilled scorer model (rubric teacher → fast score)

So what: the most reliable “single model” path today is to **distill the existing rubric score** into an encoder model:
- Teacher: token-level analysis → rubric `overall_0_100` (slow, diagnostic)
- Student: HF encoder → `score_0_100` in one forward pass (fast, deployable)

Local smoke (labels a small subset, trains to `models/scorer_v4_distill_smoke/`):
```bash
make train-scorer-distill-v4-smoke
```

Modal (full run; writes to `/vol/models/scorer_v4_distilled`):
```bash
make modal-distill-scorer-v4
```

Standard Ebooks distillation (baseline built from the corpus train split; writes to `/vol/models/scorer_standardebooks_distilled`):
```bash
make modal-distill-scorer-standardebooks
```

Mixed distillation (Standard Ebooks + RSS + synthetic degradations; writes to `/vol/models/scorer_mixed_distilled`):
```bash
make modal-distill-scorer-mixed
```

Evaluate a trained scorer on the fixed set (Modal):
```bash
make modal-eval-trained-scorer TRAINED_SCORER_MODEL=/vol/models/scorer_mixed_distilled
```

## Next data we’ll need (later)

- A small human‑labeled set (100–300 samples) where “literary” isn’t synonymous with “old public‑domain prose”.
- Rewrite pairs (original → edited) for training the rewrite/rerank layer without changing meaning.
