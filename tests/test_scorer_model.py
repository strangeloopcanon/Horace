from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.studio.scorer_model import score_with_scorer


class _FakeTokenizer:
    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _FakeModel:
    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        self.config = SimpleNamespace(num_labels=int(logits.shape[-1]) if logits.ndim >= 2 else 1)

    def __call__(self, **_kwargs):
        return SimpleNamespace(logits=self._logits)


class TestScorerModelHeads(unittest.TestCase):
    def test_binary_two_logit_head_uses_positive_softmax_probability(self) -> None:
        logits = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
        fake = (_FakeTokenizer(), _FakeModel(logits), "cpu")
        with patch("tools.studio.scorer_model.load_scorer", return_value=fake):
            out = score_with_scorer("hello", model_path_or_id="dummy")

        # softmax([-1, 1])[1] ~= 0.8808
        self.assertAlmostEqual(out.prob_0_1, 0.8808, places=3)
        self.assertAlmostEqual(out.score_0_100, 88.08, places=2)

    def test_multiclass_head_fails_fast(self) -> None:
        logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        fake = (_FakeTokenizer(), _FakeModel(logits), "cpu")
        with patch("tools.studio.scorer_model.load_scorer", return_value=fake):
            with self.assertRaises(ValueError):
                score_with_scorer("hello", model_path_or_id="dummy")


if __name__ == "__main__":
    unittest.main()
