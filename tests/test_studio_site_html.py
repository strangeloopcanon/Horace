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


if __name__ == "__main__":
    unittest.main()
