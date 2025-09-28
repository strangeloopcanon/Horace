from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import dspy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from google import genai
from google.genai import types as genai_types


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


def _select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


class LocalGemmaLM(dspy.BaseLM):
    """Minimal wrapper around a Hugging Face Gemma checkpoint."""

    def __init__(self, model_name: str, temperature: float = 0.0, max_new_tokens: int = 220):
        super().__init__(
            model=f"local/{model_name}",
            model_type="chat",
            temperature=temperature,
            max_tokens=max_new_tokens,
            cache=False,
        )
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

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
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
        super().__init__(
            model=f"aistudio/{model_name}",
            model_type="chat",
            temperature=temperature,
            max_tokens=max_new_tokens,
            cache=False,
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment; cannot use AI Studio Gemma.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.generation_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
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

        text_parts: list[str] = []
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        ):
            if getattr(chunk, "text", None):
                text_parts.append(chunk.text)
        raw_text = "".join(text_parts)

        text_output = raw_text
        try:
            fenced = re.findall(r"```(?:json)?\n(.*?)```", raw_text, flags=re.S)
            if fenced:
                candidate = fenced[0].strip()
                json.loads(candidate)
                text_output = candidate
            else:
                match = re.search(r"\{[^{}]*\"answer\"[^{}]*\}", raw_text, flags=re.S)
                if match:
                    candidate = match.group(0)
                    json.loads(candidate)
                    text_output = candidate
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
