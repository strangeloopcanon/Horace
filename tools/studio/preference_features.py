"""Preference-calibrated scorer: linear Bradley-Terry over cadence + taxonomy features.

This is the v6 scorer. It learns which measurable text properties predict human
preference from pairwise data (human original > LLM imitation), then uses those
learned weights to score any text and explain what to improve.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# Cadence features (from causal LM analysis via analyze_text)
_CADENCE_FEATURES = [
    "high_surprise_rate_per_100",
    "ipi_mean",
    "ipi_cv",
    "surprisal_cv",
    "surprisal_masd",
    "surprisal_acf1",
    "surprisal_acf2",
    "surprisal_peak_period_tokens",
    "run_low_mean_len",
    "run_high_mean_len",
]

# Entropy / focus features
_ENTROPY_FEATURES = [
    "entropy_mean",
    "entropy_sd",
    "nucleus_w_mean",
    "rank_percentile_mean",
    "perm_entropy",
]

# Texture features
_TEXTURE_FEATURES = [
    "word_ttr",
    "word_hapax_ratio",
    "adjacent_word_repeat_rate",
    "bigram_repeat_rate",
    "trigram_repeat_rate",
]

# Structure features
_STRUCTURE_FEATURES = [
    "cohesion_delta",
    "sent_burst_cv",
    "para_burst_cv",
    "sent_len_cv",
    "para_len_cv",
    "hurst_rs",
    "norm_surprisal_mean",
]

# Surface features (from analyze_text surface metrics)
_SURFACE_FEATURES = [
    "word_len_mean",
    "syllables_per_word_mean",
    "alpha_char_fraction",
    "sent_words_cv",
    "punct_variety_per_1000_chars",
    "content_fraction",
]

# Taxonomy-derived features (computed from raw text, not from LM)
_TAXONOMY_FEATURES = [
    "comma_per_100w",
    "semicolon_per_100w",
    "dash_per_100w",
    "colon_per_100w",
    "question_rate",
    "exclamation_rate",
    "one_sent_para_rate",
    "function_word_rate",
    "discourse_marker_rate",
    "parenthetical_rate",
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
    """Linear Bradley-Terry model over cadence features.

    P(A > B) = sigmoid(w . features_A + bias - w . features_B - bias)
             = sigmoid(w . (features_A - features_B))

    Trained via logistic regression on feature diffs with L2 regularization.
    """

    def __init__(self) -> None:
        self.weights = np.zeros(len(FEATURE_SCHEMA), dtype=np.float64)
        self.bias: float = 0.0
        self.feature_names: List[str] = list(FEATURE_SCHEMA)
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False

    def train(
        self,
        train_features: List[Tuple[np.ndarray, np.ndarray]],
        val_features: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        *,
        l2_reg: float = 1.0,
    ) -> TrainReport:
        """Train from lists of (chosen_features, rejected_features) tuples.

        Uses sklearn LogisticRegression on feature diffs.
        """
        from sklearn.linear_model import LogisticRegression

        # Build diff matrix with both orientations for valid 2-class logistic regression.
        # chosen - rejected -> label 1, rejected - chosen -> label 0.
        pos_diffs = [c - r for c, r in train_features]
        neg_diffs = [r - c for c, r in train_features]
        diffs = np.array(pos_diffs + neg_diffs, dtype=np.float64)
        labels = np.array(
            [1] * len(pos_diffs) + [0] * len(neg_diffs), dtype=np.int32
        )

        # Standardize features for stable training
        self._train_mean = np.mean(diffs, axis=0)
        self._train_std = np.std(diffs, axis=0)
        self._train_std[self._train_std < 1e-12] = 1.0

        diffs_norm = (diffs - self._train_mean) / self._train_std

        clf = LogisticRegression(
            C=float(l2_reg),
            fit_intercept=True,
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(diffs_norm, labels)

        # Store weights in original feature space
        self.weights = (clf.coef_[0] / self._train_std).astype(np.float64)
        self.bias = float(clf.intercept_[0]) - float(np.sum(clf.coef_[0] * self._train_mean / self._train_std))
        self.feature_names = list(FEATURE_SCHEMA)
        self._fitted = True

        # Compute accuracies
        train_acc = self._pairwise_accuracy(train_features)
        val_acc = self._pairwise_accuracy(val_features) if val_features else None

        # Compute reference stats from "chosen" side
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

    def _pairwise_accuracy(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        correct = 0
        for chosen, rejected in pairs:
            s_c = float(np.dot(self.weights, chosen) + self.bias)
            s_r = float(np.dot(self.weights, rejected) + self.bias)
            if s_c > s_r:
                correct += 1
        return float(correct) / max(1, len(pairs))

    def score(self, features: np.ndarray) -> float:
        """Raw score (unbounded). Higher = better writing."""
        return float(np.dot(self.weights, features) + self.bias)

    def score_0_100(self, features: np.ndarray) -> int:
        """Map raw score to 0-100 via sigmoid scaling."""
        raw = self.score(features)
        prob = 1.0 / (1.0 + math.exp(-raw))
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
    "cohesion_delta": "Shuffling your text changes log-prob by {value:.3f}. Stronger narrative structure shows {p25:.3f}\u2013{p75:.3f}.",
    "word_ttr": "Vocabulary diversity (TTR) is {value:.3f}. Reference: {p25:.3f}\u2013{p75:.3f}.",
    "word_hapax_ratio": "Unique-word ratio is {value:.3f}. Reference: {p25:.3f}\u2013{p75:.3f}.",
    "sent_burst_cv": "Sentence-level surprise burstiness is {value:.2f}. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "entropy_mean": "Mean entropy is {value:.2f}. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "function_word_rate": "Function word rate is {value:.3f}. Reference: {p25:.3f}\u2013{p75:.3f}.",
    "comma_per_100w": "Comma density is {value:.1f} per 100 words. Reference: {p25:.1f}\u2013{p75:.1f}.",
    "semicolon_per_100w": "Semicolon usage is {value:.2f} per 100 words. Reference: {p25:.2f}\u2013{p75:.2f}.",
    "dash_per_100w": "Dash usage is {value:.1f} per 100 words. Reference: {p25:.1f}\u2013{p75:.1f}.",
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
