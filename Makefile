.PHONY: setup setup-modal modal-token check lint typecheck security test all clean run-ui run-api build-baseline build-baseline-web build-eval-set split-eval-set build-benchmark-set build-benchmark-v4 build-benchmark-v5 build-standardebooks-corpus download-standardebooks-raw download-gutenberg-raw sample-windows-great sample-windows-other build-rss-corpus build-antipattern-originals build-antipattern-pairs build-antipattern-pairs-openai-batch merge-antipattern-openai-batch build-ai-overfit-eval eval-ai-overfit eval-web eval-set eval-set-train eval-set-val eval-set-test eval-benchmark-train eval-benchmark-val eval-benchmark-test train-calibrator-benchmark train-calibrator-eval-set train-calibrator-eval-set-tainted train-scorer-v4 train-scorer-v5-antipattern label-benchmark-v4-smoke train-scorer-distill-v4-smoke modal-eval-web modal-eval-set modal-eval-trained-scorer modal-build-baseline-web modal-train-calibrator-web modal-train-calibrator-eval-set modal-train-scorer-v4 modal-distill-scorer-v4 modal-build-standardebooks-corpus modal-distill-scorer-standardebooks modal-build-rss-corpus modal-distill-scorer-mixed modal-train-scorer-hybrid
.PHONY: modal-score-urls

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
UV ?= uv
UVX ?= uvx
MODAL ?= $(PYTHON) -m modal
MODAL_RUN_FLAGS ?=
URL_SCORE_ARGS ?=
MODEL ?= gpt2
BENCH_DIR ?= data/benchmarks/studio_benchmark_v3
BENCH_DIR_V4 ?= data/benchmarks/studio_benchmark_v4
BENCH_DIR_V5 ?= data/benchmarks/studio_benchmark_v5
AI_OVERFIT_EVAL_DIR ?= data/eval_sets/ai_overfit_v1
STD_EBOOKS_DIR ?= data/corpora/standardebooks_corpus_v1
STD_EBOOKS_MAX_PAGES ?= 30
STD_EBOOKS_START_PAGE ?= 1
STD_EBOOKS_MAX_BOOKS ?= 240
STD_EBOOKS_EXCERPTS_PER_BOOK ?= 2
STD_EBOOKS_MAX_CHARS ?= 3800
STD_EBOOKS_MIN_CHARS ?= 900
STD_EBOOKS_SLEEP_S ?= 0.6
STD_EBOOKS_RAW_DIR ?= data/corpora/standardebooks_raw_v1
GUTENBERG_RAW_DIR ?= data/corpora/gutenberg_raw_v1
GUTENBERG_START_INDEX ?= 1
GUTENBERG_MAX_PAGES ?= 200
GUTENBERG_MAX_BOOKS ?= 3000
GUTENBERG_MAX_BYTES ?= 0
GUTENBERG_SLEEP_S ?= 0.1
GREAT_AUTHORS_FILE ?= configs/great_authors_v1.txt
GREAT_WINDOWS_DIR ?= data/corpora/gutenberg_great_windows_v1
OTHER_WINDOWS_DIR ?= data/corpora/gutenberg_other_windows_v1
WINDOWS_MAX_CHARS ?= 3800
WINDOWS_MIN_CHARS ?= 900
WINDOWS_PER_DOC ?= 8
RSS_DIR ?= data/corpora/rss_corpus_v1
RSS_FEEDS_JSON ?= configs/rss_feeds_v1.json
RSS_MAX_CHARS ?= 3800
RSS_MIN_CHARS ?= 900
RSS_MAX_ITEMS_PER_FEED ?= 80
RSS_EXCERPTS_PER_ITEM ?= 1
RSS_SLEEP_S ?= 0.0
TRAINED_SCORER_MODEL ?= /vol/models/scorer_standardebooks_distilled
REBUILD ?= 0
DISTILL_DIR_V4_SMOKE ?= data/benchmarks/studio_benchmark_v4_distill_smoke
ANTIPATTERN_DIR ?= data/antipattern
ANTIPATTERN_ORIGINALS ?= $(ANTIPATTERN_DIR)/originals_v1.jsonl
ANTIPATTERN_PAIRS ?= $(ANTIPATTERN_DIR)/pairs_v1.jsonl
ANTIPATTERN_NEGATIVES ?= $(ANTIPATTERN_DIR)/negatives_v1.jsonl
ANTIPATTERN_OPENAI_BATCH_REQ ?= $(ANTIPATTERN_DIR)/openai_batch_requests_v1.jsonl
ANTIPATTERN_OPENAI_BATCH_RES ?= $(ANTIPATTERN_DIR)/openai_batch_results_v1.jsonl
ANTIPATTERN_UNRESOLVED ?= $(ANTIPATTERN_DIR)/unresolved_jobs_v1.jsonl
ANTIPATTERN_INPUT_JSONL ?= data/corpora/standardebooks_corpus_v1/splits/train.jsonl data/corpora/gutenberg_great_windows_v1/splits/train.jsonl
ANTIPATTERN_INPUT_TEXT_DIRS ?= data/poem data/shortstory
ANTIPATTERN_PROVIDER_MODELS ?= openai:gpt-5 google:gemini-3-flash anthropic:claude-haiku-4.5
ANTIPATTERN_VARIANTS_PER_TIER ?= 1
ANTIPATTERN_MAX_ORIGINALS ?= 1000
ANTIPATTERN_MIN_GENERATED_CHARS ?= 220
ANTIPATTERN_MAX_GENERATED_CHARS ?= 4200
ANTIPATTERN_TEMPERATURE ?= 0.9
ANTIPATTERN_MAX_OUTPUT_TOKENS ?= 700
AI_OVERFIT_MODEL ?= models/scorer_v5_antipattern
LINT_FILES ?= tools/studio/rss.py tools/studio/url_safety.py tools/studio/model_security.py tools/studio/rewrite.py tools/studio/scorer_model.py tools/studio/analyze.py tools/studio/score.py tools/studio/baselines.py tools/studio/calibrator.py tools/studio/cadence_profile.py tools/studio/write_like.py tools/studio/critique.py tools/studio/meaning_lock.py tools/studio/span_patcher.py tools/studio/paragraph_cadence.py tools/reward.py tools/cli.py tools/train.py deploy/modal/score_urls_qwen3.py tests/test_rss.py tests/test_url_safety.py tests/test_model_security.py
TYPECHECK_FILES ?= tools/studio/rss.py tools/studio/url_safety.py tools/studio/model_security.py tools/studio/score.py tools/studio/baselines.py tools/studio/calibrator.py tools/cli.py
SECURITY_FILES ?= tools/studio/rss.py tools/studio/url_safety.py tools/studio/model_security.py deploy/modal/score_urls_qwen3.py

SETUP_SENTINEL := $(VENV)/.horace_setup

$(SETUP_SENTINEL): requirements.txt
	$(UV) venv $(VENV)
	$(UV) pip install -r requirements.txt --python $(PYTHON)
	@touch $(SETUP_SENTINEL)

setup: $(SETUP_SENTINEL)

setup-modal: setup
	$(UV) pip install modal --python $(PYTHON)
	@echo "Modal CLI: $(MODAL)"

modal-token: setup-modal
	$(MODAL) token new

check: setup lint typecheck security
	$(PYTHON) -m compileall tools deploy
	$(UV) pip check --python $(PYTHON)

lint: setup
	$(UVX) ruff check $(LINT_FILES)

typecheck: setup
	$(UVX) ty check $(TYPECHECK_FILES)

security: setup
	$(UVX) bandit -q -s B110,B311 $(SECURITY_FILES)

test: setup
	$(PYTHON) -m unittest -v

all: check test

run-ui: setup
	$(PYTHON) tools/studio_ui.py --host 127.0.0.1 --port 7861

run-api: setup
	$(PYTHON) -m tools.studio_api --host 127.0.0.1 --port 8000

build-baseline: setup
	$(PYTHON) -c "from tools.studio.baselines import build_baseline; build_baseline('$(MODEL)')"

build-baseline-web: setup
	$(PYTHON) -m tools.studio.build_baseline_web --model-id $(MODEL) --max-input-tokens 512 --n 200 --out-dir data/baselines

build-eval-set: setup
	test -f data/eval_sets/studio_fixed_v1.jsonl || $(PYTHON) -m tools.studio.build_eval_set --out data/eval_sets/studio_fixed_v1.jsonl

split-eval-set: build-eval-set
	test -f data/eval_sets/studio_fixed_v1_splits/manifest.json || $(PYTHON) -m tools.studio.split_eval_set --samples data/eval_sets/studio_fixed_v1.jsonl --out-dir data/eval_sets/studio_fixed_v1_splits

build-benchmark-set: setup
	@if [ "$(REBUILD)" = "1" ] || [ ! -f "$(BENCH_DIR)/samples.jsonl" ]; then \
		$(PYTHON) -m tools.studio.build_benchmark_set --out-dir $(BENCH_DIR); \
	fi

build-benchmark-v4: setup
	@if [ "$(REBUILD)" = "1" ] || [ ! -f "$(BENCH_DIR_V4)/samples.jsonl" ]; then \
		$(PYTHON) -m tools.studio.build_benchmark_v4 --out-dir $(BENCH_DIR_V4); \
	fi

build-benchmark-v5: setup
	@if [ "$(REBUILD)" = "1" ] || [ ! -f "$(BENCH_DIR_V5)/samples.jsonl" ]; then \
		$(PYTHON) -m tools.studio.build_benchmark_v5 --out-dir $(BENCH_DIR_V5) --antipattern-negatives $(ANTIPATTERN_NEGATIVES); \
	fi

build-standardebooks-corpus: setup
	@if [ "$(REBUILD)" = "1" ] || [ ! -f "$(STD_EBOOKS_DIR)/samples.jsonl" ]; then \
		$(PYTHON) -m tools.studio.build_standardebooks_corpus --out-dir $(STD_EBOOKS_DIR) --start-page $(STD_EBOOKS_START_PAGE) --max-pages $(STD_EBOOKS_MAX_PAGES) --max-books $(STD_EBOOKS_MAX_BOOKS) --excerpts-per-book $(STD_EBOOKS_EXCERPTS_PER_BOOK) --max-chars $(STD_EBOOKS_MAX_CHARS) --min-chars $(STD_EBOOKS_MIN_CHARS) --sleep-s $(STD_EBOOKS_SLEEP_S) --normalize-text; \
	fi

download-standardebooks-raw: setup
	HORACE_HTTP_RETRIES=3 HORACE_HTTP_RETRY_BASE_SLEEP_S=1.2 $(PYTHON) -m tools.studio.download_standardebooks_raw --out-dir $(STD_EBOOKS_RAW_DIR) --start-page $(STD_EBOOKS_START_PAGE) --max-pages $(STD_EBOOKS_MAX_PAGES) --max-books $(STD_EBOOKS_MAX_BOOKS) --sleep-s $(STD_EBOOKS_SLEEP_S) --normalize-text --great-authors $(GREAT_AUTHORS_FILE)

download-gutenberg-raw: setup
	HORACE_HTTP_RETRIES=2 HORACE_HTTP_RETRY_BASE_SLEEP_S=0.8 $(PYTHON) -m tools.studio.download_gutenberg_raw --out-dir $(GUTENBERG_RAW_DIR) --start-index $(GUTENBERG_START_INDEX) --max-pages $(GUTENBERG_MAX_PAGES) --max-books $(GUTENBERG_MAX_BOOKS) --max-bytes $(GUTENBERG_MAX_BYTES) --sleep-s $(GUTENBERG_SLEEP_S) --normalize-text --great-authors $(GREAT_AUTHORS_FILE) --keep all

sample-windows-great: setup
	$(PYTHON) -m tools.studio.sample_windows_from_raw --raw-dir $(GUTENBERG_RAW_DIR) --out-dir $(GREAT_WINDOWS_DIR) --bucket great_author --max-chars $(WINDOWS_MAX_CHARS) --min-chars $(WINDOWS_MIN_CHARS) --windows-per-doc $(WINDOWS_PER_DOC)

sample-windows-other: setup
	$(PYTHON) -m tools.studio.sample_windows_from_raw --raw-dir $(GUTENBERG_RAW_DIR) --out-dir $(OTHER_WINDOWS_DIR) --bucket other_author --max-chars $(WINDOWS_MAX_CHARS) --min-chars $(WINDOWS_MIN_CHARS) --windows-per-doc $(WINDOWS_PER_DOC)

build-rss-corpus: setup
	@if [ "$(REBUILD)" = "1" ] || [ ! -f "$(RSS_DIR)/samples.jsonl" ]; then \
		$(PYTHON) -m tools.studio.build_rss_corpus --out-dir $(RSS_DIR) --feeds-json $(RSS_FEEDS_JSON) --max-items-per-feed $(RSS_MAX_ITEMS_PER_FEED) --excerpts-per-item $(RSS_EXCERPTS_PER_ITEM) --max-chars $(RSS_MAX_CHARS) --min-chars $(RSS_MIN_CHARS) --sleep-s $(RSS_SLEEP_S) --normalize-text; \
	fi

eval-web: setup
	$(PYTHON) -m tools.studio.eval_web

eval-set: setup
	$(PYTHON) -m tools.studio.eval_set --samples data/eval_sets/studio_fixed_v1.jsonl --report-out reports/studio_eval_set_report.json

eval-set-train: split-eval-set
	$(PYTHON) -m tools.studio.eval_set --samples data/eval_sets/studio_fixed_v1_splits/train.jsonl --report-out reports/studio_eval_set_train_report.json

eval-set-val: split-eval-set
	$(PYTHON) -m tools.studio.eval_set --samples data/eval_sets/studio_fixed_v1_splits/val.jsonl --report-out reports/studio_eval_set_val_report.json

eval-set-test: split-eval-set
	$(PYTHON) -m tools.studio.eval_set --samples data/eval_sets/studio_fixed_v1_splits/test.jsonl --report-out reports/studio_eval_set_test_report.json

# Benchmark v3 (held-out, no leakage via group_id)
eval-benchmark-train: build-benchmark-set
	$(PYTHON) -m tools.studio.eval_set --samples $(BENCH_DIR)/splits/train.jsonl --report-out reports/studio_benchmark_train_report.json

eval-benchmark-val: build-benchmark-set
	$(PYTHON) -m tools.studio.eval_set --samples $(BENCH_DIR)/splits/val.jsonl --report-out reports/studio_benchmark_val_report.json

eval-benchmark-test: build-benchmark-set
	$(PYTHON) -m tools.studio.eval_set --samples $(BENCH_DIR)/splits/test.jsonl --report-out reports/studio_benchmark_test_report.json

train-calibrator-benchmark: eval-benchmark-train eval-benchmark-test
	$(PYTHON) -m tools.studio.train_calibrator --report reports/studio_benchmark_train_report.json --out reports/calibrators_benchmark/calibrator.json --pos gutenberg_excerpt --neg wikipedia_summary --neg wikinews_published --neg nasa_breaking_news --neg rfc_excerpt --neg gibberish_control
	$(PYTHON) -m tools.studio.eval_set --samples $(BENCH_DIR)/splits/test.jsonl --calibrator reports/calibrators_benchmark/calibrator.json --report-out reports/studio_benchmark_test_report_calibrated.json

# Held-out (no leakage): train on train split, report on test split.
train-calibrator-eval-set: eval-set-train eval-set-test
	$(PYTHON) -m tools.studio.train_calibrator --report reports/studio_eval_set_train_report.json --out reports/calibrators_eval_set/calibrator.json --pos gutenberg_excerpt --neg wikipedia_summary --neg wikinews_published --neg nasa_breaking_news --neg rfc_excerpt --neg gibberish_control
	$(PYTHON) -m tools.studio.eval_set --samples data/eval_sets/studio_fixed_v1_splits/test.jsonl --calibrator reports/calibrators_eval_set/calibrator.json --report-out reports/studio_eval_set_test_report_calibrated.json

# Debug only (tainted): trains on the same set it evaluates.
train-calibrator-eval-set-tainted: eval-set
	$(PYTHON) -m tools.studio.train_calibrator --report reports/studio_eval_set_report.json --out reports/calibrators_eval_set/calibrator_tainted.json --pos gutenberg_excerpt --neg wikipedia_summary --neg wikinews_published --neg nasa_breaking_news --neg rfc_excerpt --neg gibberish_control
	$(PYTHON) -m tools.studio.eval_set --samples data/eval_sets/studio_fixed_v1.jsonl --calibrator reports/calibrators_eval_set/calibrator_tainted.json --report-out reports/studio_eval_set_report_tainted_calibrated.json

# Single scorer model (text → score), trained within-domain on Gutenberg top vs long-tail (+ corruptions).
train-scorer-v4: build-benchmark-v4
	$(PYTHON) -m tools.studio.train_scorer --train $(BENCH_DIR_V4)/splits/train.jsonl --val $(BENCH_DIR_V4)/splits/val.jsonl --test $(BENCH_DIR_V4)/splits/test.jsonl --out-dir models/scorer_v4 --base-model distilbert-base-uncased --doc-type prose --normalize-text --pos gutenberg_top_excerpt --neg gutenberg_random_excerpt --neg gutenberg_corrupt_shuffle_sentences_global --neg gutenberg_corrupt_shuffle_paragraphs --neg gutenberg_corrupt_repeat_sentences --neg gutenberg_corrupt_flatten

train-scorer-v5-antipattern: build-benchmark-v5
	$(PYTHON) -m tools.studio.train_scorer --train $(BENCH_DIR_V5)/splits/train.jsonl --val $(BENCH_DIR_V5)/splits/val.jsonl --test $(BENCH_DIR_V5)/splits/test.jsonl --out-dir models/scorer_v5_antipattern --base-model distilbert-base-uncased --doc-type prose --normalize-text --pos gutenberg_top_excerpt --neg gutenberg_random_excerpt --neg gutenberg_corrupt_shuffle_sentences_global --neg gutenberg_corrupt_shuffle_paragraphs --neg gutenberg_corrupt_repeat_sentences --neg gutenberg_corrupt_flatten --neg llm_antipattern_write_like --neg llm_antipattern_continue_from --neg llm_antipattern_rewrite_from_memory

build-antipattern-originals: setup
	$(PYTHON) -m tools.studio.build_antipattern_originals $(foreach p,$(ANTIPATTERN_INPUT_JSONL),--input-jsonl $(p)) $(foreach d,$(ANTIPATTERN_INPUT_TEXT_DIRS),--input-text-dir $(d)) --out $(ANTIPATTERN_ORIGINALS) --max-samples $(ANTIPATTERN_MAX_ORIGINALS) --min-chars 800 --max-chars 3800 --windows-per-text-file 1 --normalize-text --skip-missing

build-antipattern-pairs: build-antipattern-originals
	$(PYTHON) -m tools.studio.build_antipattern_pairs --originals $(ANTIPATTERN_ORIGINALS) --out-pairs $(ANTIPATTERN_PAIRS) --out-negatives $(ANTIPATTERN_NEGATIVES) --out-unresolved-jobs $(ANTIPATTERN_UNRESOLVED) $(foreach pm,$(ANTIPATTERN_PROVIDER_MODELS),--provider-model $(pm)) --tier write_like --tier continue_from --tier rewrite_from_memory --variants-per-tier $(ANTIPATTERN_VARIANTS_PER_TIER) --temperature $(ANTIPATTERN_TEMPERATURE) --max-output-tokens $(ANTIPATTERN_MAX_OUTPUT_TOKENS) --min-generated-chars $(ANTIPATTERN_MIN_GENERATED_CHARS) --max-generated-chars $(ANTIPATTERN_MAX_GENERATED_CHARS) --run-online

build-antipattern-pairs-openai-batch: build-antipattern-originals
	$(PYTHON) -m tools.studio.build_antipattern_pairs --originals $(ANTIPATTERN_ORIGINALS) --out-pairs $(ANTIPATTERN_PAIRS) --out-negatives $(ANTIPATTERN_NEGATIVES) --openai-batch-requests-out $(ANTIPATTERN_OPENAI_BATCH_REQ) --out-unresolved-jobs $(ANTIPATTERN_UNRESOLVED) --provider-model openai:gpt-5 --tier write_like --tier continue_from --tier rewrite_from_memory --variants-per-tier $(ANTIPATTERN_VARIANTS_PER_TIER) --temperature $(ANTIPATTERN_TEMPERATURE) --max-output-tokens $(ANTIPATTERN_MAX_OUTPUT_TOKENS) --min-generated-chars $(ANTIPATTERN_MIN_GENERATED_CHARS) --max-generated-chars $(ANTIPATTERN_MAX_GENERATED_CHARS)

merge-antipattern-openai-batch: build-antipattern-originals
	$(PYTHON) -m tools.studio.build_antipattern_pairs --originals $(ANTIPATTERN_ORIGINALS) --out-pairs $(ANTIPATTERN_PAIRS) --out-negatives $(ANTIPATTERN_NEGATIVES) --openai-batch-results $(ANTIPATTERN_OPENAI_BATCH_RES) --provider-model openai:gpt-5 --tier write_like --tier continue_from --tier rewrite_from_memory --variants-per-tier $(ANTIPATTERN_VARIANTS_PER_TIER) --temperature $(ANTIPATTERN_TEMPERATURE) --max-output-tokens $(ANTIPATTERN_MAX_OUTPUT_TOKENS) --min-generated-chars $(ANTIPATTERN_MIN_GENERATED_CHARS) --max-generated-chars $(ANTIPATTERN_MAX_GENERATED_CHARS)

build-ai-overfit-eval: build-antipattern-pairs
	$(PYTHON) -m tools.studio.build_ai_overfit_eval --originals $(ANTIPATTERN_ORIGINALS) --negatives $(ANTIPATTERN_NEGATIVES) --out-dir $(AI_OVERFIT_EVAL_DIR) --author-holdout

eval-ai-overfit: build-ai-overfit-eval
	$(PYTHON) -m tools.studio.eval_scorer --model $(AI_OVERFIT_MODEL) --samples $(AI_OVERFIT_EVAL_DIR)/splits/test.jsonl --out reports/ai_overfit_eval_report.json --pos human_original --neg llm_antipattern_write_like --neg llm_antipattern_continue_from --neg llm_antipattern_rewrite_from_memory

# Distill the slow rubric score into a fast encoder (smoke-sized subset).
label-benchmark-v4-smoke: build-benchmark-v4
	$(PYTHON) -m tools.studio.label_scorer_dataset --in $(BENCH_DIR_V4)/splits/train.jsonl --out $(DISTILL_DIR_V4_SMOKE)/train.jsonl --max-samples 120 --teacher-model gpt2 --baseline-model gpt2_gutenberg_512 --doc-type prose --backend hf --max-input-tokens 256 --normalize-text
	$(PYTHON) -m tools.studio.label_scorer_dataset --in $(BENCH_DIR_V4)/splits/val.jsonl --out $(DISTILL_DIR_V4_SMOKE)/val.jsonl --max-samples 40 --teacher-model gpt2 --baseline-model gpt2_gutenberg_512 --doc-type prose --backend hf --max-input-tokens 256 --normalize-text
	$(PYTHON) -m tools.studio.label_scorer_dataset --in $(BENCH_DIR_V4)/splits/test.jsonl --out $(DISTILL_DIR_V4_SMOKE)/test.jsonl --max-samples 40 --teacher-model gpt2 --baseline-model gpt2_gutenberg_512 --doc-type prose --backend hf --max-input-tokens 256 --normalize-text

train-scorer-distill-v4-smoke: label-benchmark-v4-smoke
	$(PYTHON) -m tools.studio.train_scorer --train $(DISTILL_DIR_V4_SMOKE)/train.jsonl --val $(DISTILL_DIR_V4_SMOKE)/val.jsonl --test $(DISTILL_DIR_V4_SMOKE)/test.jsonl --out-dir models/scorer_v4_distill_smoke --base-model distilbert-base-uncased --doc-type prose --normalize-text --pos gutenberg_top_excerpt --label-key label

modal-eval-web: setup-modal
	$(MODAL) run deploy/modal/studio_eval_web.py --out-dir reports/studio_eval_modal

modal-eval-set: setup-modal
	$(MODAL) run deploy/modal/studio_eval_set.py --out-dir reports/studio_eval_set_modal

modal-eval-trained-scorer: setup-modal
	$(MODAL) run deploy/modal/studio_eval_trained_scorer.py --out-dir reports/studio_eval_trained_scorer_modal --model $(TRAINED_SCORER_MODEL) --samples data/eval_sets/studio_fixed_v1_splits/test.jsonl --pos gutenberg_excerpt

modal-score-urls: setup-modal
	$(MODAL) run $(MODAL_RUN_FLAGS) deploy/modal/score_urls_qwen3.py --urls "$(URLS)" $(URL_SCORE_ARGS)

modal-build-baseline-web: setup-modal
	$(MODAL) run deploy/modal/studio_build_baseline_web.py --out-dir reports/baselines

modal-train-calibrator-web: setup-modal
	$(MODAL) run deploy/modal/studio_train_calibrator_web.py --out-dir reports/calibrators

modal-train-calibrator-eval-set: setup-modal
	$(MODAL) run deploy/modal/studio_train_calibrator_eval_set.py --out-dir reports/calibrators_eval_set_modal

modal-train-scorer-v4: setup-modal
	$(MODAL) run deploy/modal/studio_train_scorer_v4.py --out-dir /vol/models/scorer_v4

modal-distill-scorer-v4: setup-modal
	$(MODAL) run deploy/modal/studio_distill_scorer_v4.py --out-dir /vol/models/scorer_v4_distilled

modal-build-standardebooks-corpus: setup-modal
	$(MODAL) run deploy/modal/studio_build_standardebooks_corpus.py --out-dir /vol/corpora/standardebooks_corpus_v1 --start-page $(STD_EBOOKS_START_PAGE) --max-pages $(STD_EBOOKS_MAX_PAGES) --max-books $(STD_EBOOKS_MAX_BOOKS) --excerpts-per-book $(STD_EBOOKS_EXCERPTS_PER_BOOK) --max-chars $(STD_EBOOKS_MAX_CHARS) --min-chars $(STD_EBOOKS_MIN_CHARS) --sleep-s $(STD_EBOOKS_SLEEP_S)

modal-distill-scorer-standardebooks: setup-modal
	$(MODAL) run deploy/modal/studio_distill_scorer_standardebooks.py --out-dir /vol/models/scorer_standardebooks_distilled --corpus-dir /vol/corpora/standardebooks_corpus_v1 --max-pages $(STD_EBOOKS_MAX_PAGES) --max-books $(STD_EBOOKS_MAX_BOOKS) --excerpts-per-book $(STD_EBOOKS_EXCERPTS_PER_BOOK) --max-chars $(STD_EBOOKS_MAX_CHARS) --min-chars $(STD_EBOOKS_MIN_CHARS) --sleep-s $(STD_EBOOKS_SLEEP_S)

modal-build-rss-corpus: setup-modal
	$(MODAL) run deploy/modal/studio_build_rss_corpus.py --out-dir /vol/corpora/rss_corpus_v1 --feeds-json $(RSS_FEEDS_JSON) --max-items-per-feed $(RSS_MAX_ITEMS_PER_FEED) --excerpts-per-item $(RSS_EXCERPTS_PER_ITEM) --max-chars $(RSS_MAX_CHARS) --min-chars $(RSS_MIN_CHARS) --sleep-s $(RSS_SLEEP_S)

modal-distill-scorer-mixed: setup-modal
	$(MODAL) run deploy/modal/studio_distill_scorer_mixed.py --out-dir /vol/models/scorer_mixed_distilled --standardebooks-dir /vol/corpora/standardebooks_corpus_v1 --rss-dir /vol/corpora/rss_corpus_v1 --mixed-corpus-dir /vol/corpora/mixed_corpus_v1

modal-train-scorer-hybrid: setup-modal
	$(MODAL) run deploy/modal/studio_train_scorer_hybrid.py --out-dir /vol/models/scorer_hybrid_v1

modal-train-scorer-qwen3-great-other: setup-modal
	$(MODAL) run deploy/modal/studio_train_scorer_qwen3_great_other.py --out-dir /vol/models/scorer_qwen3_great_other_v1 --base-model Qwen/Qwen3-1.7B

clean:
	rm -rf $(VENV) .pytest_cache __pycache__
