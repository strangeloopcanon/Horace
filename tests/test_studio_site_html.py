"""Regression tests for the embedded Horace Studio HTML."""

import unittest

from tools.studio.site import STUDIO_HTML


class TestStudioSiteHtml(unittest.TestCase):
    def test_experimental_tabs_hidden_by_default(self) -> None:
        self.assertIn('data-tab="rewrite" style="display:none;"', STUDIO_HTML)
        self.assertIn('data-tab="match" style="display:none;"', STUDIO_HTML)
        self.assertIn('id="panel-rewrite" class="panel" style="display:none;"', STUDIO_HTML)
        self.assertIn('id="panel-match" class="panel" style="display:none;"', STUDIO_HTML)

    def test_theme_storage_access_uses_safe_wrapper(self) -> None:
        self.assertIn("function safeStorageGet(", STUDIO_HTML)
        self.assertIn("function safeStorageSet(", STUDIO_HTML)
        self.assertIn("const savedTheme = safeStorageGet('horace-theme', 'dark');", STUDIO_HTML)
        self.assertNotIn("const savedTheme = localStorage.getItem('horace-theme')", STUDIO_HTML)

    def test_match_join_uses_escaped_newline_literal(self) -> None:
        self.assertIn("out = prompt.trimEnd() + '\\n\\n' + continuation.trimStart();", STUDIO_HTML)
        self.assertNotIn("out = prompt.trimEnd() + '\n\n' + continuation.trimStart();", STUDIO_HTML)

    def test_score_summary_join_uses_runtime_newline(self) -> None:
        self.assertIn("const lineBreak = String.fromCharCode(10);", STUDIO_HTML)
        self.assertIn("detailLines.join(lineBreak)", STUDIO_HTML)
        self.assertNotIn("detailLines.join('\\n')", STUDIO_HTML)

    def test_wrong_scorer_path_is_blocked_client_side(self) -> None:
        self.assertIn("const scorerModelPathInvalid = Boolean(scorerModelPath && looksLikeWrongScorerModel(scorerModelPath));", STUDIO_HTML)
        self.assertIn("const scorerModelPathSafe = scorerModelPathInvalid ? '' : scorerModelPath;", STUDIO_HTML)
        self.assertIn("scorer_model_path: scorerModelPathSafe", STUDIO_HTML)

    def test_authenticity_penalty_defaults_on(self) -> None:
        self.assertIn("const DEFAULT_APPLY_ANTIPATTERN_PENALTY = true;", STUDIO_HTML)
        self.assertIn("headline score includes authenticity adjustment", STUDIO_HTML)

    def test_headline_score_uses_primary_overall(self) -> None:
        self.assertIn("const overallScore = Number(", STUDIO_HTML)
        self.assertIn("primary.overall_0_100", STUDIO_HTML)
        self.assertIn("Math.round(overallScore || 0)", STUDIO_HTML)
        self.assertIn("Quality score (before authenticity)", STUDIO_HTML)


if __name__ == "__main__":
    unittest.main()
