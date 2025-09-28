from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

from rapidfuzz import fuzz

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

TASK_FILE = Path("data/tasks/gepa_writing_tasks.json")


def _count_words(text: str) -> int:
    return len((text or "").split())


def _count_sentences(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len([s for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()])


def _count_lines(text: str) -> int:
    return len([ln for ln in (text or "").splitlines() if ln.strip()])


import re


def writing_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    target = (gold.answer or "").strip()
    generated = (pred.answer or "").strip()

    min_words = gold.get("min_words", 0)
    max_words = gold.get("max_words", 10_000)
    word_count = _count_words(generated)
    center = (min_words + max_words) / 2
    if min_words == max_words:
        len_score = 1.0 if word_count == min_words else 0.0
    else:
        span = max_words - min_words
        if min_words <= word_count <= max_words:
            len_score = 1.0
        else:
            len_score = max(0.0, 1.0 - abs(word_count - center) / (span / 2 or 1))

    motifs = gold.get("motifs", []) or []
    motif_hits = []
    motif_misses = []
    for motif in motifs:
        score = fuzz.partial_ratio(motif.lower(), generated.lower())
        if score >= 80:
            motif_hits.append(motif)
        else:
            motif_misses.append(motif)
    motif_score = len(motif_hits) / max(1, len(motifs))

    overlap = fuzz.token_set_ratio(target.lower(), generated.lower()) / 100.0

    doc_type = gold.get("doc_type", "prose")
    if doc_type == "poem":
        gold_lines = _count_lines(target)
        gen_lines = _count_lines(generated)
        required = max(2, round(0.6 * gold_lines))
        if gen_lines >= required:
            structure_score = 1.0
        elif gen_lines >= max(1, required // 2):
            structure_score = 0.4
        else:
            structure_score = 0.0
    else:
        sentence_goal = max(2, min(6, _count_sentences(target)))
        structure_score = 1.0 if _count_sentences(generated) >= sentence_goal else 0.3

    weights = dict(length=0.25, motifs=0.30, overlap=0.30, structure=0.15)
    total = (
        weights["length"] * len_score
        + weights["motifs"] * motif_score
        + weights["overlap"] * overlap
        + weights["structure"] * structure_score
    )
    total = max(0.0, min(1.0, float(total)))

    feedback = [
        f"Score={total:.2f}",
        f"Words={word_count} (target {min_words}-{max_words})",
        f"Motifs={len(motif_hits)}/{len(motifs)}",
        f"Overlap={overlap:.2f}",
    ]
    if motif_misses:
        feedback.append("Missing motifs: " + ", ".join(motif_misses))
    if not generated:
        feedback.append("Output empty")
    return ScoreWithFeedback(score=total, feedback=" | ".join(feedback))


class WriteSample(dspy.Signature):
    instruction = dspy.InputField(desc="Author-specific instruction")
    context = dspy.InputField(desc="Stylistic context and motifs")
    answer = dspy.OutputField(desc="Generated passage")
    reflection = dspy.OutputField(desc="Short self-review of adherence and quality")


class Writer(dspy.Module):
    def __init__(self, instruction_seed: str):
        super().__init__()
        self.step = dspy.Predict(WriteSample, instructions=instruction_seed)

    def forward(self, instruction: str, context: str) -> dspy.Prediction:
        out = self.step(instruction=instruction, context=context)
        return dspy.Prediction(answer=out.answer, reflection=getattr(out, "reflection", ""))


def get_seed_prompt() -> str:
    return (
        "You are a meticulous literary mimic.\n"
        "Follow every component of the instruction and stylistic cues.\n"
        "Guidelines:\n"
        "- Stay within the requested word count window.\n"
        "- Mirror the narrative voice, tone, and structure implied by the context.\n"
        "- Work each motif into the piece naturally; avoid simple name-dropping.\n"
        "- Keep factual claims inside the provided context; never invent outside references.\n"
        "- Prefer concrete sensory detail over generic statements.\n"
        "- For poems, preserve line-based cadence and rhyme patterns signaled in the context.\n"
        "- For prose, produce flowing paragraphs with sentence rhythms reflecting the author.\n"
        "Output format (JSON only): {\"answer\": \"...\", \"reflection\": \"Brief self-review covering word count fit, motif coverage, style/cadence adherence, and any fixes to improve.\"}."
    )


def _load_tasks():
    data = json.loads(TASK_FILE.read_text(encoding="utf-8"))
    examples = []
    for item in data:
        examples.append(dspy.Example(**item).with_inputs("instruction", "context"))
    return examples


def build_datasets(train_count: int, val_count: int, seed: int) -> Tuple[list, list]:
    tasks = _load_tasks()
    if len(tasks) < train_count + val_count:
        raise ValueError("Not enough tasks for requested split")
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    shuffled = [tasks[i] for i in indices]
    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    return train, val


def build_student(seed_instruction: str) -> dspy.Module:
    return Writer(seed_instruction)


def metric():
    return writing_metric


def prediction_to_dict(pred: dspy.Prediction) -> dict:
    return {
        "answer": getattr(pred, "answer", ""),
        "reflection": getattr(pred, "reflection", ""),
    }


def reference_from_example(example: dspy.Example) -> str:
    return getattr(example, "answer", "")
