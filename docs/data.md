# Horace Studio — Data Strategy (Literary Quality)

So what: “is this *good writing* by the standards of great literature?” is not a clean classifier problem. We need **license‑safe literary positives**, **within‑domain hard negatives**, and **held‑out splits** that can’t be tainted by duplicate works.

## Buckets we need

1) Literary positives (what we want to emulate)
- Public‑domain prose with strong editing/formatting.
- Primary use: build baseline distributions + train the fast scorer (via teacher distillation / ranking).

2) Within‑domain hard negatives (same content domain, worse writing)
- Controlled degradations (shuffle, flatten, repetition, etc).
- Later: rewrite ladders (original → slightly dulled → heavily dulled) to make “cadence/voice” learnable without domain shortcuts.

3) Non‑literary comparators (diagnostics, calibration)
- High-quality expository prose (e.g. Wikipedia Featured Articles).
- News/technical/legal (Wikinews/RFC/PEPs/court opinions) to check robustness and avoid rewarding “just long”.

## License-safe sources that fit “great writing”

- **Standard Ebooks** (public-domain texts, carefully produced). We ingest via EPUB → text windows.
  - Build snapshot: `make build-standardebooks-corpus` (writes under `data/corpora/`, gitignored).
- **RSS/Atom feeds for modern prose** (default: The Conversation + ProPublica).
  - Build snapshot: `make build-rss-corpus` (writes under `data/corpora/`, gitignored).
- **Project Gutenberg** (public domain, large). Useful for scale; quality is mixed, so prefer curated subsets for “great writers”.
- **Wikisource** (public domain; extraction is messier, but expands coverage beyond Gutenberg/SE).

## Local “raw stores” (for bigger runs, up to disk limits)

So what: for larger local datasets (tens of GB), store **full normalized texts** first, then sample windows for training/eval.

- Standard Ebooks raw store (EPUB → full text): `make download-standardebooks-raw`
  - Outputs: `data/corpora/standardebooks_raw_v1/books/`, `meta.jsonl`, `failures.jsonl`, `stats.json`
  - Note: Standard Ebooks rate limits aggressively; this repo uses retries + backoff via `HORACE_HTTP_RETRIES` and a conservative `STD_EBOOKS_SLEEP_S`.
- Gutenberg raw store (plain text → full text): `make download-gutenberg-raw`
  - Outputs: `data/corpora/gutenberg_raw_v1/books/`, `meta.jsonl`, `failures.jsonl`, `stats.json`
  - Scale tip: run in chunks by download rank:
    - `make download-gutenberg-raw GUTENBERG_START_INDEX=1 GUTENBERG_MAX_BOOKS=200`
    - `make download-gutenberg-raw GUTENBERG_START_INDEX=201 GUTENBERG_MAX_BOOKS=200`
    - …

Bucketing:
- Both raw downloaders assign `bucket ∈ {great_author, other_author}` using `configs/great_authors_v1.txt` (author allowlist).

## Window sampling (training/eval-ready corpora)

From a raw store, build a windowed corpus with leakage-safe splits:

```bash
python -m tools.studio.sample_windows_from_raw \
  --raw-dir data/corpora/gutenberg_raw_v1 \
  --out-dir data/corpora/gutenberg_great_windows_v1 \
  --bucket great_author \
  --windows-per-doc 20
```

This emits:
- `samples.jsonl` (windows)
- `splits/{train,val,test}.jsonl` (group-safe; no tainted eval)
- `stats.json`

## Mixed corpus (single train/val/test split across sources)

So what: if you plan to train one scorer on *multiple* sources, you need a single split across the union.

```bash
python -m tools.studio.build_mixed_windows_corpus \
  --out-dir data/corpora/mixed_windows_v1 \
  --input data/corpora/gutenberg_great_windows_v1/samples.jsonl \
  --input data/corpora/gutenberg_other_windows_v1/samples.jsonl \
  --input data/corpora/rss_corpus_v1/samples.jsonl
```

## Splits & leakage rules (non-negotiable)

- Split on **work-level group_id** (author/work), never by individual window.
- If a Standard Ebooks page links to a Gutenberg transcription (`gutenberg.org/ebooks/<id>`), use that id as the group id so we don’t leak the same work across corpora.

## Labels (beyond a binary classifier)

- Use the existing rubric as a **teacher** to produce dense supervision:
  - `teacher_overall_0_100` + `teacher_categories_0_1` (multi-head targets).
- Train the scorer as a **regressor / ranker**, not a “Gutenberg vs not” classifier.

## What we do next (once corpora exist)

1) Build corpus snapshot(s)
- `make build-standardebooks-corpus`
  - Modal (recommended for larger runs): `make modal-build-standardebooks-corpus`
- `make build-rss-corpus`
  - Modal: `make modal-build-rss-corpus`

2) Label on Modal (teacher = rubric; produces dense targets)
- Use `tools/studio/label_scorer_dataset.py` over train/val/test splits.

3) Train the fast scorer
- `tools/studio/train_scorer.py` with `--label-key label` (and later: multi-head using `teacher_categories_0_1`).

4) Evaluate without taint
- Held-out split metrics + cross-domain diagnostics (Wikipedia/RFC/news) to detect shortcut learning.

## Practical training recipe (current)

- Build baseline from **Standard Ebooks train split only** (no tainted eval).
- Label a **mixed** dataset (Standard Ebooks + RSS + synthetic degradations) with the rubric teacher.
- Train a single scorer model on the mixed labeled set (Modal: `make modal-distill-scorer-mixed`).
- Add **within-content cadence supervision**:
  - Build preference pairs from Standard Ebooks splits: original > dulled rewrite (meaning preserved; cadence flattened) and > deterministic corruptions.
  - Fine-tune the scorer with a pairwise ranking loss.
  - End-to-end Modal wrapper: `make modal-train-scorer-hybrid`.
