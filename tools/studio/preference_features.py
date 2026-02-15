"""Quality scorer: logistic classifier over cadence + taxonomy features.

v8 scorer. Trains P(good | features) directly on individual texts labelled as
human-original (good) or LLM-imitation (slop). Feature set selected by Cohen's d
effect sizes (see reports/feature_separation_v7.json); 33 features survive from
the original 52.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# Cadence features (from causal LM analysis via analyze_text)
# v8: dropped surprisal_masd (d=0.18), surprisal_acf2 (d=0.04),
#     surprisal_peak_period_tokens (d=0.12), surprisal_kurtosis (d=-0.15)
_CADENCE_FEATURES = [
    "high_surprise_rate_per_100",
    "ipi_mean",
    "ipi_cv",
    "surprisal_cv",
    "surprisal_acf1",
    "run_low_mean_len",
    "run_high_mean_len",
    "surprisal_skewness",
    "surprisal_iqr",
]

# Entropy / focus features
# v8: dropped entropy_sd (dead), rank_percentile_cv (d=-0.20),
#     rank_percentile_p10 (dead), perm_entropy (d=0.14)
_ENTROPY_FEATURES = [
    "entropy_mean",
    "nucleus_w_mean",
    "entropy_range_ratio",
]

# Texture features
# v8: dropped word_ttr (d=-0.97, redundant with word_hapax_ratio d=-1.27)
_TEXTURE_FEATURES = [
    "word_hapax_ratio",
    "adjacent_word_repeat_rate",
    "bigram_repeat_rate",
    "trigram_repeat_rate",
]

# Structure features
# v8: dropped cohesion_delta (d=0.03), norm_surprisal_mean (d=-0.04)
_STRUCTURE_FEATURES = [
    "sent_burst_cv",
    "para_burst_cv",
    "sent_len_cv",
    "para_len_cv",
    "hurst_rs",
]

# Surface features (from analyze_text surface metrics)
# v8: dropped syllables_per_word_mean (d=-0.68, redundant with word_len_mean d=-0.98),
#     sent_words_cv (d=1.45, redundant with sent_len_cv d=1.46)
_SURFACE_FEATURES = [
    "word_len_mean",
    "alpha_char_fraction",
    "punct_variety_per_1000_chars",
    "content_fraction",
]

# Taxonomy-derived features (computed from raw text, not from LM)
# v8: dropped comma_per_100w (d=-0.03), dash_per_100w (data artifact),
#     question_rate (d=0.17), discourse_marker_rate (d=0.19),
#     hedging_phrase_rate (d=-0.10), sentence_opener_entropy (d=0.09)
_TAXONOMY_FEATURES = [
    "semicolon_per_100w",
    "colon_per_100w",
    "exclamation_rate",
    "one_sent_para_rate",
    "function_word_rate",
    "parenthetical_rate",
    "slop_phrase_density",
    "paragraph_metric_cv",
]

FEATURE_SCHEMA: List[str] = (
    _CADENCE_FEATURES
    + _ENTROPY_FEATURES
    + _TEXTURE_FEATURES
    + _STRUCTURE_FEATURES
    + _SURFACE_FEATURES
    + _TAXONOMY_FEATURES
)

# ---------------------------------------------------------------------------
# Taxonomy feature extraction (cheap, string-based)
# ---------------------------------------------------------------------------

_FUNCTION_WORDS = frozenset(
    "a an the and but or nor for yet so if then else when while as since until "
    "before after although because though unless whereas in on at by to from with "
    "of into through during upon about between among is am are was were be been being "
    "have has had do does did will would shall should can could may might must "
    "i me my mine we us our ours you your yours he him his she her hers it its "
    "they them their theirs this that these those who whom whose which what "
    "not no very much many more most some any all each every both few several "
    "here there where how just also too only even still already again".split()
)

_DISCOURSE_MARKERS = frozenset(
    "however therefore moreover furthermore nevertheless nonetheless meanwhile "
    "consequently accordingly thus hence indeed actually basically essentially "
    "specifically particularly notably importantly significantly alternatively "
    "conversely similarly likewise regardless otherwise finally ultimately".split()
)

# Known LLM slop phrases (overused by chatbot-style models)
_SLOP_PHRASES = [
    "it's important to note", "it's worth noting", "it is important to note",
    "it is worth noting", "it should be noted", "it bears mentioning",
    "in today's world", "in this day and age", "at the end of the day",
    "when it comes to", "on the other hand", "in other words",
    "needless to say", "it goes without saying",
    "dive into", "dive deep", "diving into", "deep dive",
    "delve into", "delving into",
    "the landscape of", "the realm of", "the world of",
    "leverage", "leveraging", "leveraged",
    "tapestry", "tapestries",
    "multifaceted", "nuanced",
    "paradigm", "paradigm shift",
    "in conclusion", "to summarize", "to sum up",
    "let's explore", "let us explore", "we'll explore",
    "it's crucial", "it is crucial", "it's essential", "it is essential",
    "plays a crucial role", "plays a vital role", "plays a key role",
    "a testament to", "stands as a testament",
    "the importance of", "the significance of",
    "it's fascinating", "it is fascinating",
    "a myriad of", "a plethora of",
    "navigate the", "navigating the",
    "foster", "fostering", "fostered",
    "unlock", "unlocking", "unlocked",
    "empower", "empowering", "empowered",
    "in the realm of", "in the landscape of",
    "resonate with", "resonates with", "resonating with",
    "holistic", "holistic approach",
    "ever-evolving", "ever-changing", "ever-growing",
    "robust", "cutting-edge", "game-changer",
    "transformative", "groundbreaking",
]
_SLOP_PATTERNS = [re.compile(re.escape(p), re.IGNORECASE) for p in _SLOP_PHRASES]

# Hedging phrases (uncertainty markers overused by LLMs)
_HEDGING_PHRASES = [
    "to some extent", "to a certain extent", "to a degree",
    "it could be argued", "one could argue", "one might argue",
    "generally speaking", "broadly speaking", "loosely speaking",
    "in many ways", "in some ways", "in a sense",
    "more or less", "arguably", "presumably", "supposedly",
    "it seems", "it appears", "it would seem",
    "perhaps", "possibly", "potentially",
    "tends to", "tend to", "tended to",
    "in some cases", "in certain cases", "in many cases",
    "somewhat", "relatively", "fairly", "rather",
    "to varying degrees", "a certain degree of",
]
_HEDGING_PATTERNS = [re.compile(re.escape(p), re.IGNORECASE) for p in _HEDGING_PHRASES]

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_END_RE = re.compile(r"[.!?]+")


def compute_taxonomy_features(text: str) -> Dict[str, float]:
    """Compute cheap taxonomy-derived features from raw text."""
    t = (text or "").strip()
    if not t:
        return {k: 0.0 for k in _TAXONOMY_FEATURES}

    words = [m.group(0).lower() for m in _WORD_RE.finditer(t)]
    n_words = max(1, len(words))

    # Punctuation rates per 100 words
    comma_count = t.count(",")
    semicolon_count = t.count(";")
    dash_count = t.count("\u2014") + t.count("\u2013") + t.count(" - ")
    colon_count = t.count(":")
    paren_count = min(t.count("("), t.count(")"))

    # Sentence-end analysis
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    n_sents = max(1, len(sentences))
    question_count = sum(1 for s in sentences if s.rstrip().endswith("?"))
    exclamation_count = sum(1 for s in sentences if s.rstrip().endswith("!"))

    # Paragraph analysis
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    n_paras = max(1, len(paragraphs))
    one_sent_paras = 0
    for para in paragraphs:
        para_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if len(para_sents) <= 1:
            one_sent_paras += 1

    # Function words and discourse markers
    func_count = sum(1 for w in words if w in _FUNCTION_WORDS)
    disc_count = sum(1 for w in words if w in _DISCOURSE_MARKERS)

    # Slop phrase density: known LLM catchphrases per 100 words
    slop_count = sum(len(pat.findall(t)) for pat in _SLOP_PATTERNS)

    # Hedging phrase rate: uncertainty markers per 100 words
    hedge_count = sum(len(pat.findall(t)) for pat in _HEDGING_PATTERNS)

    # Sentence opener entropy: diversity of how sentences begin
    openers: list[str] = []
    for s in sentences:
        opener_words = _WORD_RE.findall(s.strip())[:3]
        if opener_words:
            openers.append(" ".join(w.lower() for w in opener_words))
    opener_entropy = 0.0
    if len(openers) >= 2:
        freq: dict[str, int] = {}
        for o in openers:
            freq[o] = freq.get(o, 0) + 1
        total = float(len(openers))
        opener_entropy = -sum(
            (c / total) * math.log(c / total) for c in freq.values() if c > 0
        )

    # Paragraph metric CV: uniformity of paragraph structure.
    # For each paragraph, compute average sentence word count.
    # Then CV of those averages across paragraphs. LLM paragraphs are more uniform.
    para_avg_sent_lens: list[float] = []
    for para in paragraphs:
        para_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if para_sents:
            sent_wcs = [len(_WORD_RE.findall(s)) for s in para_sents]
            avg = float(sum(sent_wcs)) / max(1, len(sent_wcs))
            para_avg_sent_lens.append(avg)
    para_metric_cv = 0.0
    if len(para_avg_sent_lens) >= 2:
        _pm_mean = sum(para_avg_sent_lens) / len(para_avg_sent_lens)
        _pm_var = sum((x - _pm_mean) ** 2 for x in para_avg_sent_lens) / len(para_avg_sent_lens)
        _pm_std = _pm_var ** 0.5
        para_metric_cv = _pm_std / max(_pm_mean, 1e-12)

    per_100w = 100.0 / float(n_words)
    return {
        "comma_per_100w": float(comma_count) * per_100w,
        "semicolon_per_100w": float(semicolon_count) * per_100w,
        "dash_per_100w": float(dash_count) * per_100w,
        "colon_per_100w": float(colon_count) * per_100w,
        "question_rate": float(question_count) / float(n_sents),
        "exclamation_rate": float(exclamation_count) / float(n_sents),
        "one_sent_para_rate": float(one_sent_paras) / float(n_paras),
        "function_word_rate": float(func_count) / float(n_words),
        "discourse_marker_rate": float(disc_count) / float(n_words),
        "parenthetical_rate": float(paren_count) * per_100w,
        # LLM-slop detection
        "slop_phrase_density": float(slop_count) * per_100w,
        "hedging_phrase_rate": float(hedge_count) * per_100w,
        "sentence_opener_entropy": opener_entropy,
        "paragraph_metric_cv": para_metric_cv,
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(doc_metrics: Dict[str, Any], text: str) -> np.ndarray:
    """Extract the canonical feature vector from analyze_text output + raw text.

    Returns a 1-D numpy array of length len(FEATURE_SCHEMA).
    Missing or None values are replaced with 0.0.
    """
    taxonomy = compute_taxonomy_features(text)
    merged = {**doc_metrics, **taxonomy}

    vec = np.zeros(len(FEATURE_SCHEMA), dtype=np.float64)
    for i, key in enumerate(FEATURE_SCHEMA):
        val = merged.get(key)
        if val is not None and isinstance(val, (int, float)) and math.isfinite(float(val)):
            vec[i] = float(val)
    return vec


def extract_features_from_text(
    text: str,
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    doc_type: str = "prose",
    backend: str = "auto",
    max_input_tokens: int = 1024,
) -> np.ndarray:
    """Convenience: run analyze_text then extract features."""
    from tools.studio.analyze import analyze_text

    result = analyze_text(
        text,
        model_id=model_id,
        doc_type=doc_type,
        backend=backend,
        max_input_tokens=max_input_tokens,
        compute_cohesion=True,
        include_token_metrics=False,
    )
    doc_metrics = result.get("doc_metrics") or {}
    return extract_features(doc_metrics, text)


# ---------------------------------------------------------------------------
# Feature preference model (linear Bradley-Terry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainReport:
    n_train: int
    n_val: int
    train_accuracy: float
    val_accuracy: Optional[float]
    l2_reg: float
    feature_names: List[str]
    weights: List[float]
    bias: float


class FeaturePreferenceModel:
    """Logistic quality classifier over cadence + taxonomy features.

    v8: P(good | features) = sigmoid(w . z(features) + bias)
    where z() is per-feature z-score standardization.

    Supports two training modes:
    - train(): pairwise Bradley-Terry (legacy, v6/v7)
    - train_direct(): direct binary classification (v8+)

    Feature standardization prevents any single feature from dominating
    purely due to scale differences.
    """

    def __init__(self) -> None:
        self.weights = np.zeros(len(FEATURE_SCHEMA), dtype=np.float64)
        self.bias: float = 0.0
        self.feature_names: List[str] = list(FEATURE_SCHEMA)
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        # Per-feature standardization: z = (x - feat_mean) / feat_std
        # Learned from training data (all chosen + rejected features).
        self.feat_mean = np.zeros(len(FEATURE_SCHEMA), dtype=np.float64)
        self.feat_std = np.ones(len(FEATURE_SCHEMA), dtype=np.float64)
        # Score calibration: raw scores are mapped via sigmoid((raw - center) / scale) * 100.
        # center/scale are learned from the training distribution so that
        # typical rejected text ≈ 25, median ≈ 50, typical chosen text ≈ 75.
        self.score_center: float = 0.0
        self.score_scale: float = 1.0
        self._fitted = False

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        """Apply per-feature z-score standardization."""
        return (features - self.feat_mean) / self.feat_std

    def train(
        self,
        train_features: List[Tuple[np.ndarray, np.ndarray]],
        val_features: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        *,
        l2_reg: float = 1.0,
    ) -> TrainReport:
        """Train from lists of (chosen_features, rejected_features) tuples.

        Uses sklearn LogisticRegression on standardized feature diffs.
        Features are z-scored using statistics from all training texts
        (both chosen and rejected) so that no feature dominates by scale.
        """
        from sklearn.linear_model import LogisticRegression

        # Learn per-feature standardization from ALL training texts
        all_features = np.array(
            [f for pair in train_features for f in pair], dtype=np.float64
        )
        self.feat_mean = np.mean(all_features, axis=0)
        self.feat_std = np.std(all_features, axis=0)
        self.feat_std[self.feat_std < 1e-12] = 1.0

        # Standardize all features first
        train_std = [
            (self._standardize(c), self._standardize(r))
            for c, r in train_features
        ]

        # Build diff matrix with both orientations for valid 2-class logistic regression.
        # chosen - rejected -> label 1, rejected - chosen -> label 0.
        pos_diffs = [c - r for c, r in train_std]
        neg_diffs = [r - c for c, r in train_std]
        diffs = np.array(pos_diffs + neg_diffs, dtype=np.float64)
        labels = np.array(
            [1] * len(pos_diffs) + [0] * len(neg_diffs), dtype=np.int32
        )

        clf = LogisticRegression(
            C=float(l2_reg),
            fit_intercept=True,
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(diffs, labels)

        # Weights are in standardized feature space (by design — this is what
        # prevents scale-driven dominance). We keep them in this space and
        # always standardize inputs before scoring.
        self.weights = clf.coef_[0].astype(np.float64)
        self.bias = float(clf.intercept_[0])
        self.feature_names = list(FEATURE_SCHEMA)
        self._fitted = True

        # Compute accuracies (using original features — score() handles standardization)
        train_acc = self._pairwise_accuracy(train_features)
        val_acc = self._pairwise_accuracy(val_features) if val_features else None

        # Compute reference stats from "chosen" side (in original space for feedback)
        chosen_features = np.array([c for c, _ in train_features], dtype=np.float64)
        self.reference_stats = {}
        for i, name in enumerate(FEATURE_SCHEMA):
            col = chosen_features[:, i]
            finite = col[np.isfinite(col)]
            if finite.size > 0:
                self.reference_stats[name] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "p25": float(np.percentile(finite, 25)),
                    "median": float(np.median(finite)),
                    "p75": float(np.percentile(finite, 75)),
                }

        # Compute score calibration from the raw score distribution.
        # Center on overall median; scale so that the typical chosen/rejected
        # range spans the sensitive region of the sigmoid.
        chosen_raw = np.array([self.score(c) for c, _ in train_features])
        rejected_raw = np.array([self.score(r) for _, r in train_features])
        all_raw = np.concatenate([chosen_raw, rejected_raw])
        self.score_center = float(np.median(all_raw))
        # Scale so p10-p90 of all scores maps roughly to sigmoid 0.1-0.9
        # (sigmoid(±2.2) ≈ 0.1/0.9), so scale = (p90 - p10) / 4.4
        p10 = float(np.percentile(all_raw, 10))
        p90 = float(np.percentile(all_raw, 90))
        spread = max(p90 - p10, 0.1)
        self.score_scale = spread / 4.4

        return TrainReport(
            n_train=len(train_features),
            n_val=len(val_features) if val_features else 0,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            l2_reg=float(l2_reg),
            feature_names=list(FEATURE_SCHEMA),
            weights=[float(w) for w in self.weights],
            bias=float(self.bias),
        )

    def train_direct(
        self,
        features: List[np.ndarray],
        labels: List[int],
        *,
        l2_reg: float = 1.0,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train a direct classifier: P(good | features).

        Each sample is a feature vector with label 1 (good/human) or 0 (slop/LLM).
        Logistic regression output probability IS the quality score.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        X = np.array(features, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        names = list(feature_names or FEATURE_SCHEMA)

        # Learn standardization from all training texts
        self.feat_mean = np.mean(X, axis=0)
        self.feat_std = np.std(X, axis=0)
        self.feat_std[self.feat_std < 1e-12] = 1.0

        Z = (X - self.feat_mean) / self.feat_std

        clf = LogisticRegression(
            C=float(l2_reg),
            fit_intercept=True,
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(Z, y)

        self.weights = clf.coef_[0].astype(np.float64)
        self.bias = float(clf.intercept_[0])
        self.feature_names = names
        self._fitted = True

        # Direct classifier: sigmoid IS the probability, no calibration needed
        self.score_center = 0.0
        self.score_scale = 1.0

        # Reference stats from "good" texts (label=1) for feedback
        good_mask = y == 1
        good_X = X[good_mask]
        self.reference_stats = {}
        for i, name in enumerate(names):
            col = good_X[:, i]
            finite = col[np.isfinite(col)]
            if finite.size > 0:
                self.reference_stats[name] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "p25": float(np.percentile(finite, 25)),
                    "median": float(np.median(finite)),
                    "p75": float(np.percentile(finite, 75)),
                }

        # Metrics
        probs = clf.predict_proba(Z)[:, 1]
        preds = (probs >= 0.5).astype(int)
        accuracy = float(np.mean(preds == y))
        auc = float(roc_auc_score(y, probs))

        return {
            "accuracy": accuracy,
            "auc": auc,
            "n_samples": int(len(y)),
            "n_good": int(good_mask.sum()),
            "n_slop": int((~good_mask).sum()),
            "l2_reg": float(l2_reg),
            "feature_names": names,
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
        }

    def _pairwise_accuracy(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        correct = 0
        for chosen, rejected in pairs:
            s_c = self.score(chosen)
            s_r = self.score(rejected)
            if s_c > s_r:
                correct += 1
        return float(correct) / max(1, len(pairs))

    def score(self, features: np.ndarray) -> float:
        """Raw score (unbounded). Higher = better writing.

        Input features are in original space; standardization is applied internally.
        """
        z = self._standardize(features)
        return float(np.dot(self.weights, z) + self.bias)

    def score_0_100(self, features: np.ndarray) -> int:
        """Map raw score to 0-100 via calibrated sigmoid.

        Uses center/scale learned from training distribution so scores
        spread naturally: typical rejected ≈ 25, median ≈ 50, typical chosen ≈ 75.
        """
        raw = self.score(features)
        scaled = (raw - self.score_center) / max(self.score_scale, 1e-6)
        prob = 1.0 / (1.0 + math.exp(-scaled))
        return max(0, min(100, int(round(prob * 100))))

    def feature_contributions(self, features: np.ndarray) -> Dict[str, float]:
        """Per-feature contribution to the score: w_i * x_i."""
        contribs: Dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            contribs[name] = float(self.weights[i] * features[i])
        return contribs

    def feature_gaps(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Compare each feature to the reference distribution. Sorted by leverage.

        Leverage = |weight| * |feature - reference_median|.
        High leverage = biggest potential improvement.
        """
        gaps: List[Dict[str, Any]] = []
        for i, name in enumerate(self.feature_names):
            ref = self.reference_stats.get(name)
            if ref is None:
                continue
            value = float(features[i])
            median = float(ref["median"])
            weight = float(self.weights[i])
            gap = value - median
            leverage = abs(weight) * abs(gap)
            direction = "increase" if weight > 0 and gap < 0 else "decrease" if weight < 0 and gap < 0 else "on_track"
            if weight > 0 and gap > 0:
                direction = "on_track"
            if weight < 0 and gap > 0:
                direction = "decrease"

            gaps.append({
                "feature": name,
                "value": value,
                "reference_median": median,
                "reference_p25": float(ref["p25"]),
                "reference_p75": float(ref["p75"]),
                "weight": weight,
                "contribution": float(weight * value),
                "gap": gap,
                "leverage": leverage,
                "direction": direction,
            })
        gaps.sort(key=lambda g: g["leverage"], reverse=True)
        return gaps

    def save(self, path: Path) -> None:
        """Save model to JSON."""
        data = {
            "feature_names": self.feature_names,
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "reference_stats": self.reference_stats,
            "feature_standardization": {
                "mean": [float(m) for m in self.feat_mean],
                "std": [float(s) for s in self.feat_std],
            },
            "score_calibration": {
                "center": float(self.score_center),
                "scale": float(self.score_scale),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeaturePreferenceModel":
        """Load model from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        model = cls()
        model.feature_names = data["feature_names"]
        model.weights = np.array(data["weights"], dtype=np.float64)
        model.bias = float(data["bias"])
        model.reference_stats = data.get("reference_stats") or {}
        # Feature standardization (v7+). Defaults to identity (mean=0, std=1)
        # for backward compatibility with v6 models.
        std_data = data.get("feature_standardization") or {}
        n_feats = len(model.feature_names)
        model.feat_mean = np.array(
            std_data.get("mean", [0.0] * n_feats), dtype=np.float64
        )
        model.feat_std = np.array(
            std_data.get("std", [1.0] * n_feats), dtype=np.float64
        )
        model.feat_std[model.feat_std < 1e-12] = 1.0
        cal = data.get("score_calibration") or {}
        model.score_center = float(cal.get("center", 0.0))
        model.score_scale = float(cal.get("scale", 1.0))
        model._fitted = True
        return model


# ---------------------------------------------------------------------------
# Feature feedback (human-readable suggestions)
# ---------------------------------------------------------------------------

_FEATURE_TEMPLATES: Dict[str, str] = {
    "high_surprise_rate_per_100": "Your spike rate is {value:.1f} per 100 tokens. Strong prose typically ranges {p25:.0f}\u2013{p75:.0f}.",
    "ipi_mean": "Average distance between surprise peaks is {value:.0f} tokens. Reference range: {p25:.0f}\u2013{p75:.0f}.",
    "ipi_cv": "Surprise peak spacing variation (CV) is {value:.2f}. More varied rhythm ({p25:.2f}\u2013{p75:.2f}) reads better.",
    "surprisal_cv": "Overall surprise variation is {value:.2f}. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "word_hapax_ratio": "Unique-word ratio is {value:.3f}. Reference: {p25:.3f}\u2013{p75:.3f}.",
    "sent_burst_cv": "Sentence-level surprise burstiness is {value:.2f}. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "entropy_mean": "Mean entropy is {value:.2f}. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "function_word_rate": "Function word rate is {value:.3f}. Reference: {p25:.3f}\u2013{p75:.3f}.",
    "semicolon_per_100w": "Semicolon usage is {value:.2f} per 100 words. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "slop_phrase_density": "Slop/clich\u00e9 phrase density is {value:.1f} per 100 words. Strong prose stays under {p75:.1f}.",
    "surprisal_skewness": "Surprisal skewness is {value:.2f}. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "para_len_cv": "Paragraph length variation (CV) is {value:.2f}. Strong prose shows {p25:.2f}\u2013{p75:.2f}.",
    "sent_len_cv": "Sentence length variation (CV) is {value:.2f}. Strong prose shows {p25:.2f}\u2013{p75:.2f}.",
    "paragraph_metric_cv": "Paragraph structure variation is {value:.2f}. More variety ({p25:.2f}\u2013{p75:.2f}) reads less formulaic.",
    "one_sent_para_rate": "One-sentence paragraph rate is {value:.2f}. Good prose: {p25:.2f}\u2013{p75:.2f}.",
    "content_fraction": "Content-word fraction is {value:.3f}. Reference: {p25:.3f}\u2013{p75:.3f}.",
    "word_len_mean": "Average word length is {value:.1f} characters. Reference: {p25:.1f}\u2013{p75:.1f}.",
}


def generate_feedback(
    model: FeaturePreferenceModel,
    features: np.ndarray,
    *,
    max_suggestions: int = 5,
) -> List[Dict[str, Any]]:
    """Generate actionable writing suggestions from feature gaps."""
    gaps = model.feature_gaps(features)
    suggestions: List[Dict[str, Any]] = []

    for gap in gaps[:max_suggestions]:
        name = gap["feature"]
        if gap["direction"] == "on_track":
            continue
        template = _FEATURE_TEMPLATES.get(name)
        ref = model.reference_stats.get(name, {})
        if template and ref:
            message = template.format(
                value=gap["value"],
                p25=ref.get("p25", 0),
                p75=ref.get("p75", 0),
            )
        else:
            message = f"{name} is {gap['value']:.3f} (reference median: {gap['reference_median']:.3f})."

        suggestions.append({
            "feature": name,
            "message": message,
            "direction": gap["direction"],
            "leverage": gap["leverage"],
            "value": gap["value"],
            "reference_median": gap["reference_median"],
        })

    return suggestions
