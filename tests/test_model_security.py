from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.studio.model_security import resolve_model_source, resolve_trust_remote_code, split_model_revision


class TestModelSecurity(unittest.TestCase):
    def test_split_model_revision_parses_remote_reference(self) -> None:
        repo, rev = split_model_revision("org/custom-model@abc123")
        self.assertEqual(repo, "org/custom-model")
        self.assertEqual(rev, "abc123")

    def test_split_model_revision_preserves_local_like_identifier(self) -> None:
        repo, rev = split_model_revision("model@v2")
        self.assertEqual(repo, "model@v2")
        self.assertIsNone(rev)

    def test_allows_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model"
            model_dir.mkdir(parents=True)
            source = resolve_model_source(str(model_dir), purpose="test model")
            self.assertTrue(source.is_local)
            self.assertIsNone(source.revision)

    def test_allows_allowlisted_remote_model(self) -> None:
        source = resolve_model_source("gpt2", purpose="test model")
        self.assertFalse(source.is_local)
        self.assertEqual(source.source_id, "gpt2")
        self.assertIsNone(source.revision)

    def test_allows_pinned_remote_model(self) -> None:
        source = resolve_model_source("org/custom-model@abc123def", purpose="test model")
        self.assertFalse(source.is_local)
        self.assertEqual(source.source_id, "org/custom-model")
        self.assertEqual(source.revision, "abc123def")

    def test_rejects_unpinned_unallowlisted_remote_model(self) -> None:
        with self.assertRaises(ValueError):
            resolve_model_source("org/custom-model", purpose="test model")

    def test_remote_code_disabled_by_default(self) -> None:
        source = resolve_model_source("Qwen/Qwen2.5-0.5B-Instruct", purpose="test model")
        with patch.dict(os.environ, {}, clear=True):
            out = resolve_trust_remote_code(source, requested=True, purpose="test model")
        self.assertFalse(out)

    def test_remote_code_can_be_enabled(self) -> None:
        source = resolve_model_source("Qwen/Qwen2.5-0.5B-Instruct", purpose="test model")
        with patch.dict(os.environ, {"HORACE_ALLOW_REMOTE_CODE": "1"}, clear=False):
            out = resolve_trust_remote_code(source, requested=True, purpose="test model")
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main()
