#!/usr/bin/env python3
"""Run GEPA prompt optimization for the Gemma 3 4B writer on Horace tasks."""

import argparse
import json
import os
import random
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Sequence
from types import SimpleNamespace

from dotenv import load_dotenv
from rapidfuzz import fuzz

import dspy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from google import genai
from google.genai import types as genai_types

from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback


def _count_words(text: str) -> int:
    return len(re.findall(r"\\b\\w+\\b", text or ""))


def _count_sentences(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len([s for s in re.split(r"(?<=[.!?])\\s+", stripped) if s.strip()])


def _count_lines(text: str) -> int:
    return len([ln for ln in (text or "").splitlines() if ln.strip()])


def writing_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
) -> dict:
    """Return a GEPA-compatible metric with scalar score + textual feedback."""
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

    structure_score = 1.0
    doc_type = gold.get("doc_type", "prose")
    if doc_type == "poem":
        gold_lines = _count_lines(target)
        gen_lines = _count_lines(generated)
        required = max(2, round(0.6 * gold_lines))
        structure_score = 1.0 if gen_lines >= required else 0.4 if gen_lines >= max(1, required // 2) else 0.0
    else:
        sentence_goal = max(2, min(6, _count_sentences(target)))
        structure_score = 1.0 if _count_sentences(generated) >= sentence_goal else 0.3

    weights = {
        "length": 0.25,
        "motif": 0.30,
        "overlap": 0.30,
        "structure": 0.15,
    }
    total = (
        weights["length"] * len_score
        + weights["motif"] * motif_score
        + weights["overlap"] * overlap
        + weights["structure"] * structure_score
    )
    total = max(0.0, min(1.0, float(total)))

    feedback_bits = [
        f"Score={total:.2f}",
        f"Words={word_count} (target {min_words}-{max_words})",
        f"Motifs={len(motif_hits)}/{len(motifs)}",
        f"Overlap={overlap:.2f}",
    ]
    if motif_misses:
        feedback_bits.append("Missing motifs: " + ", ".join(motif_misses))
    if not generated.strip():
        feedback_bits.append("Output empty or whitespace only")
    return ScoreWithFeedback(score=total, feedback=" | ".join(feedback_bits))


class WriteSample(dspy.Signature):
    """Given instruction/context, produce an answer."""

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
        return dspy.Prediction(answer=out.answer)


def _messages_to_prompt(messages: list[dict[str, str]] | None) -> str:
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role = msg.get("role", "user").strip().upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _select_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


class LocalGemmaLM(dspy.BaseLM):
    """Minimal local wrapper around a Hugging Face Gemma model."""

    def __init__(self, model_name: str, temperature: float = 0.0, max_new_tokens: int = 220):
        super().__init__(model=f"local/{model_name}", model_type="chat", temperature=temperature, max_tokens=max_new_tokens, cache=False)
        self.model_name = model_name
        self.device, self.dtype = _select_device_and_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        device_map = "auto" if self.device.type != "cpu" else None
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if device_map is None:
            self.hf_model.to(self.device)
        self.hf_model.eval()

    def forward(self, prompt=None, messages=None, **kwargs):
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.0))
        max_new_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 320))
        if messages:
            if hasattr(self.tokenizer, "apply_chat_template"):
                input_tensors = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                prompt_text = _messages_to_prompt(messages)
                if not prompt_text:
                    raise ValueError("Prompt text is required for generation")
                input_tensors = self.tokenizer(prompt_text, return_tensors="pt")
        else:
            prompt_text = prompt or ""
            if not prompt_text:
                raise ValueError("Prompt text is required for generation")
            input_tensors = self.tokenizer(prompt_text, return_tensors="pt")

        if isinstance(input_tensors, torch.Tensor):
            input_ids = input_tensors.to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            input_tensors = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
        input_length = input_tensors["input_ids"].shape[-1]
        do_sample = temperature > 0
        generate_kwargs = dict(
            **input_tensors,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = max(0.1, temperature)
        with torch.no_grad():
            generated = self.hf_model.generate(**generate_kwargs)
        gen_tokens = generated[0][input_length:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        payload = json.dumps({"answer": text})
        choice = SimpleNamespace(
            index=0,
            finish_reason="stop",
            message=SimpleNamespace(role="assistant", content=payload),
        )
        usage = {
            "prompt_tokens": int(input_length),
            "completion_tokens": int(gen_tokens.shape[0]),
            "total_tokens": int(input_length + gen_tokens.shape[0]),
        }
        response = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=self.model_name,
        )
        return response


class AIStudioGemmaLM(dspy.BaseLM):
    """Wrapper around Google AI Studio's Gemma API."""

    def __init__(self, model_name: str = "gemma-3-4b-it", temperature: float = 0.7, max_new_tokens: int = 512):
        super().__init__(model=f"aistudio/{model_name}", model_type="chat", temperature=temperature, max_tokens=max_new_tokens, cache=False)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment; cannot use AI Studio Gemma.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.generation_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )

    def forward(self, prompt=None, messages=None, **kwargs):
        temperature = kwargs.get("temperature", self.generation_config.temperature)
        max_tokens = kwargs.get("max_tokens", self.generation_config.max_output_tokens)
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if messages:
            def _map_role(r: str) -> str:
                r = (r or "user").lower()
                if r in ("user", "model"):
                    return r
                if r in ("assistant", "system"):
                    return "model" if r == "assistant" else "user"
                return "user"

            contents = []
            for msg in messages:
                role = _map_role(msg.get("role", "user"))
                text = msg.get("content", "")
                contents.append(genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=text)]))
        else:
            text = prompt or ""
            if not text:
                raise ValueError("Prompt text is required for generation")
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=text)],
                )
            ]

        # Stream to reduce latency and avoid large payload issues
        text_parts: list[str] = []
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        ):
            if getattr(chunk, "text", None):
                text_parts.append(chunk.text)
        raw_text = "".join(text_parts)

        # Try to extract a clean JSON answer if present
        text_output = raw_text
        try:
            import re, json as _json
            fenced = re.findall(r"```(?:json)?\n(.*?)```", raw_text, flags=re.S)
            if fenced:
                cand = fenced[0].strip()
                # validate JSON
                _json.loads(cand)
                text_output = cand
            else:
                m = re.search(r"\{[^{}]*\"answer\"[^{}]*\}", raw_text, flags=re.S)
                if m:
                    cand = m.group(0)
                    _json.loads(cand)
                    text_output = cand
        except Exception:
            pass
        choice = SimpleNamespace(
            index=0,
            finish_reason="stop",
            message=SimpleNamespace(role="assistant", content=text_output),
        )
        usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        return SimpleNamespace(choices=[choice], usage=usage, model=self.model_name)

def load_tasks(path: Path) -> List[dspy.Example]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for item in raw:
        example = dspy.Example(**item).with_inputs("instruction", "context")
        examples.append(example)
    return examples


def split_tasks(tasks: Sequence[dspy.Example], train_count: int, val_count: int, seed: int) -> tuple[List[dspy.Example], List[dspy.Example]]:
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    chosen = [tasks[i] for i in indices]
    train = chosen[:train_count]
    val = chosen[train_count:train_count + val_count]
    return train, val


def summarise_scores(entries: List[dict]) -> dict:
    if not entries:
        return {}
    scores = [e["score"] for e in entries]
    return {
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def run_eval(module: Writer, dataset: Sequence[dspy.Example]) -> List[dict]:
    rows = []
    for ex in dataset:
        pred = module(instruction=ex.instruction, context=ex.context)
        metric = writing_metric(ex, pred)
        prediction_text = pred.answer or ""
        reflection_text = getattr(pred, "reflection", "") or ""
        rows.append(
            {
                "id": ex.get("id"),
                "author": ex.get("author"),
                "title": ex.get("title"),
                "doc_type": ex.get("doc_type"),
                "score": metric["score"],
                "feedback": metric["feedback"],
                "prediction": prediction_text,
                "reflection": reflection_text,
                "reference": ex.answer,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a writing prompt with GEPA and Gemma 3 4B.")
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/gepa_writing_tasks.json"), help="Path to curated tasks JSON.")
    parser.add_argument("--train", type=int, default=12, help="Number of training tasks.")
    parser.add_argument("--val", type=int, default=6, help="Number of validation tasks.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for task split.")
    parser.add_argument("--max-metric-calls", type=int, default=120, help="Budget for GEPA metric calls.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/gepa_gemma3/outputs"), help="Directory to store results.")
    parser.add_argument("--reflection-minibatch", type=int, default=4, help="Reflection minibatch size for GEPA.")
    parser.add_argument("--candidate-strategy", default="pareto", choices=["pareto", "topk", "top1"], help="Candidate selection strategy.")
    parser.add_argument("--instruction-seed", type=str, default=None, help="Optional override for the initial instruction seed string.")
    return parser.parse_args()


def default_instruction_seed() -> str:
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


def ensure_lms(task_model: str, reflect_model: str) -> tuple[str, dspy.LM]:
    resolved_task_model = task_model
    if task_model.startswith("local:"):
        local_name = task_model.split("local:", 1)[1]
        lm = LocalGemmaLM(local_name)
        resolved_task_model = f"local/{local_name}"
    elif task_model.startswith("aistudio/") or task_model in {"gemma-3-4b-it", "google/gemma-3-4b-it"}:
        model_name = task_model.split("/", 1)[1] if "/" in task_model else task_model
        lm = AIStudioGemmaLM(model_name)
        resolved_task_model = f"aistudio/{model_name}"
    elif task_model.startswith("openrouter/"):
        local_name = task_model.split("/", 1)[1]
        if os.getenv("OPENROUTER_API_KEY"):
            lm = dspy.LM(task_model)
        elif os.getenv("GEMINI_API_KEY"):
            print("[warn] OPENROUTER_API_KEY not found. Redirecting to Google AI Studio for", local_name)
            lm = AIStudioGemmaLM(local_name)
            resolved_task_model = f"aistudio/{local_name}"
        else:
            print("[warn] OPENROUTER_API_KEY not found. Falling back to local Hugging Face model", local_name)
            lm = LocalGemmaLM(local_name)
            resolved_task_model = f"local/{local_name}"
    else:
        lm = dspy.LM(task_model)
    dspy.configure(lm=lm)
    return resolved_task_model, lm


def main() -> None:
    load_dotenv()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if not args.tasks.exists():
        raise FileNotFoundError(f"Tasks file not found: {args.tasks}")

    tasks = load_tasks(args.tasks)
    if len(tasks) < args.train + args.val:
        raise ValueError("Not enough tasks to satisfy train/val split")

    trainset, valset = split_tasks(tasks, args.train, args.val, args.seed)

    default_task_model = "openrouter/google/gemma-3-4b-it"
    task_model = os.getenv("TASK_MODEL", default_task_model)
    if task_model == "google/gemma-3-4b-it":
        task_model = default_task_model
    reflect_model = os.getenv("REFLECT_MODEL", os.getenv("REFLECT_MODEL_ID", "openai/gpt-4.1-mini"))
    task_model, _ = ensure_lms(task_model, reflect_model)
    reflect_lm = dspy.LM(reflect_model)

    seed_instruction = args.instruction_seed or default_instruction_seed()

    student = Writer(seed_instruction)
    gepa = dspy.GEPA(
        metric=writing_metric,
        reflection_minibatch_size=args.reflection_minibatch,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflect_lm,
        candidate_selection_strategy=args.candidate_strategy,
        track_stats=True,
        track_best_outputs=True,
    )

    optimized = gepa.compile(student, trainset=trainset, valset=valset)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    optimized_prompt = optimized.step.signature.instructions
    (run_dir / "optimized_prompt.txt").write_text(optimized_prompt, encoding="utf-8")
    # Also persist the seed prompt alongside for before/after comparison.
    (run_dir / "seed_prompt.txt").write_text(seed_instruction, encoding="utf-8")

    train_eval = run_eval(optimized, trainset)
    val_eval = run_eval(optimized, valset)

    summary = {
        "timestamp": timestamp,
        "task_model": task_model,
        "reflect_model": reflect_model,
        "train_task_count": len(trainset),
        "val_task_count": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "optimized_prompt_path": str(run_dir / "optimized_prompt.txt"),
        "train_scores": summarise_scores(train_eval),
        "val_scores": summarise_scores(val_eval),
        "train_details": train_eval,
        "val_details": val_eval,
        "seed_instruction": seed_instruction,
        "optimized_instruction": optimized_prompt,
    }

    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write a short usage note into the run directory.
    usage_md = "\n".join([
        "# Using the Evolved Prompt",
        "",
        "- System/Instruction prompt: see `optimized_prompt.txt`.",
        "- Input format: the model receives an `instruction` and a `context` string.",
        "- Output format: return a JSON object with a single key `answer`, e.g. `{\"answer\": \"...\"}`.",
        "- Word count: keep within the task's specified window (typically 180–210 words).",
        "- Motifs: include all motifs verbatim from the context.",
    ])
    (run_dir / "USAGE.md").write_text(usage_md, encoding="utf-8")

    # Maintain a convenience copy of artifacts under `experiments/gepa_gemma3/latest/`.
    latest_dir = Path("experiments/gepa_gemma3/latest")
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("seed_prompt.txt", "optimized_prompt.txt", "results.json", "report.md", "USAGE.md"):
        src = run_dir / fname
        if src.exists():
            (latest_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    markdown_lines = [
        "# GEPA Run Summary",
        f"- Timestamp: {timestamp}",
        f"- Task model: `{task_model}`",
        f"- Reflect model: `{reflect_model}`",
        f"- Train tasks: {len(trainset)}",
        f"- Validation tasks: {len(valset)}",
        f"- Max metric calls: {args.max_metric_calls}",
        "",
        "## Optimized Prompt",
        "```",
        optimized_prompt,
        "```",
        "",
        "## Validation Outputs",
    ]
    for row in val_eval:
        markdown_lines.extend(
            [
                f"### {row['author']} — {row['title']} ({row['doc_type']})",
                f"**Score:** {row['score']:.3f}",
                f"**Feedback:** {row['feedback']}",
                "**Prediction:**",
                "```",
                row["prediction"].strip(),
                "```",
            ]
        )
    (run_dir / "report.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print("=== Optimized Prompt ===")
    print(optimized_prompt)
    print(f"\nResults stored in {run_dir}")


if __name__ == "__main__":
    main()
