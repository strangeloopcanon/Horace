from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tools.studio import analyze as studio_analyze
from tools.studio.model_security import ModelSource


class _StubBackend:
    model_id = "stub-model"

    def max_context(self) -> int:
        return 1024

    def vocab_size(self) -> int:
        return 50_000

    def token_str(self, _token_id: int) -> str:
        return "tok"

    def tokenize(self, text: str):
        n = 5
        offs = [(i, min(i + 1, len(text))) for i in range(n)]
        return {"input_ids": [101, 102, 103, 104, 105], "offset_mapping": offs}

    def metrics_for_input_ids(self, input_ids, k: int, nucleus_p: float):
        n = max(0, len(input_ids) - 1)
        return {
            "p_true": np.full((n,), 0.5, dtype=np.float32),
            "logp": np.full((n,), -1.0, dtype=np.float32),
            "H": np.full((n,), 1.0, dtype=np.float32),
            "eff": np.full((n,), 2.0, dtype=np.float32),
            "rk": np.full((n,), 10, dtype=np.int32),
            "w": np.full((n,), 5, dtype=np.int32),
        }


class TestStudioAnalyzeModelPolicy(unittest.TestCase):
    def test_get_backend_resolves_model_source_before_loading(self) -> None:
        studio_analyze._BACKEND_CACHE.clear()
        sentinel = object()
        with patch(
            "tools.studio.analyze.resolve_model_source",
            return_value=ModelSource(source_id="gpt2", revision=None, is_local=False),
        ) as mock_resolve:
            with patch("tools.studio.analyze.pick_backend", return_value=sentinel) as mock_pick:
                out = studio_analyze._get_backend("org/untrusted", backend="auto")

        self.assertIs(out, sentinel)
        mock_resolve.assert_called_once_with("org/untrusted", purpose="analysis model")
        mock_pick.assert_called_once_with("gpt2", prefer_mlx=True, backend="auto")

    def test_get_backend_forces_hf_for_pinned_revision(self) -> None:
        studio_analyze._BACKEND_CACHE.clear()
        sentinel = object()
        with patch(
            "tools.studio.analyze.resolve_model_source",
            return_value=ModelSource(source_id="org/model", revision="abc123", is_local=False),
        ):
            with patch("tools.studio.analyze.pick_backend", return_value=sentinel) as mock_pick:
                out = studio_analyze._get_backend("org/model@abc123", backend="auto")

        self.assertIs(out, sentinel)
        mock_pick.assert_called_once_with("org/model@abc123", prefer_mlx=True, backend="hf")


class TestStudioAnalyzeDocTypeSegmentation(unittest.TestCase):
    def test_poem_uses_poem_paragraph_units(self) -> None:
        with patch("tools.studio.analyze._get_backend", return_value=_StubBackend()):
            with patch("tools.studio.analyze.paragraph_units", return_value=[(0, 4)]) as mock_paras:
                studio_analyze.analyze_text(
                    "rose\n\nthorn",
                    doc_type="poem",
                    normalize_text=False,
                    compute_cohesion=False,
                    max_input_tokens=16,
                )

        self.assertTrue(any(call.args[1] == "poem" for call in mock_paras.call_args_list))


if __name__ == "__main__":
    unittest.main()
