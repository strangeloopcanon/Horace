from __future__ import annotations

import io
import zipfile
import unittest

from tools.studio.epub_extract import extract_epub_text


def _make_epub_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""",
        )
        z.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookId" version="3.0">
  <manifest>
    <item id="titlepage" href="text/titlepage.xhtml" media-type="application/xhtml+xml"/>
    <item id="chap1" href="text/chapter-1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chap2" href="text/chapter-2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="titlepage"/>
    <itemref idref="chap1"/>
    <itemref idref="chap2"/>
  </spine>
</package>
""",
        )
        z.writestr(
            "OEBPS/text/titlepage.xhtml",
            """<html><body><p>Front Matter</p></body></html>""",
        )
        z.writestr(
            "OEBPS/text/chapter-1.xhtml",
            """<html><body><p>Hello</p></body></html>""",
        )
        z.writestr(
            "OEBPS/text/chapter-2.xhtml",
            """<html><body><p>World</p></body></html>""",
        )
    return buf.getvalue()


class TestEpubExtract(unittest.TestCase):
    def test_extract_epub_text_uses_spine_and_skips_front_matter(self) -> None:
        epub = _make_epub_bytes()
        text = extract_epub_text(epub)

        self.assertIn("Hello", text)
        self.assertIn("World", text)
        self.assertNotIn("Front Matter", text)
        self.assertLess(text.find("Hello"), text.find("World"))

