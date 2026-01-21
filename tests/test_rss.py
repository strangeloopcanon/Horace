import unittest


class TestRSS(unittest.TestCase):
    def test_parse_atom_feed(self):
        from tools.studio.rss import parse_feed

        xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example</title>
  <entry>
    <title>One</title>
    <link rel="alternate" href="https://example.com/one"/>
    <published>2026-01-19T17:39:34Z</published>
    <content type="html">&lt;p&gt;Hello&lt;/p&gt;</content>
  </entry>
</feed>
"""
        entries = parse_feed(xml)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.title, "One")
        self.assertEqual(e.url, "https://example.com/one")
        self.assertIn("<p>", e.content_html)
        self.assertIsInstance(e.published_unix, int)

    def test_parse_rss_feed(self):
        from tools.studio.rss import parse_feed

        xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Example</title>
    <item>
      <title>Two</title>
      <link>https://example.com/two</link>
      <pubDate>Sat, 17 Jan 2026 02:57:53 GMT</pubDate>
      <content:encoded><![CDATA[<p>Hi</p>]]></content:encoded>
    </item>
  </channel>
</rss>
"""
        entries = parse_feed(xml)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.title, "Two")
        self.assertEqual(e.url, "https://example.com/two")
        self.assertIn("<p>", e.content_html)
        self.assertIsInstance(e.published_unix, int)


if __name__ == "__main__":
    unittest.main()

