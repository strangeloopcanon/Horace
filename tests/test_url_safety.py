from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from tools.studio.url_safety import validate_public_http_url


class TestUrlSafety(unittest.TestCase):
    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(ValueError):
            validate_public_http_url("file:///etc/passwd")

    def test_rejects_localhost(self) -> None:
        with self.assertRaises(ValueError):
            validate_public_http_url("http://localhost/admin")

    def test_rejects_private_ip_literal(self) -> None:
        with self.assertRaises(ValueError):
            validate_public_http_url("http://127.0.0.1:8000/")

    def test_rejects_private_dns_resolution(self) -> None:
        fake_info = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.12", 443))]
        with patch("socket.getaddrinfo", return_value=fake_info):
            with self.assertRaises(ValueError):
                validate_public_http_url("https://example.com/path")

    def test_accepts_public_ip_literal(self) -> None:
        out = validate_public_http_url("https://1.1.1.1/example")
        self.assertEqual(out, "https://1.1.1.1/example")

    def test_accepts_public_dns_resolution(self) -> None:
        fake_info = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))]
        with patch("socket.getaddrinfo", return_value=fake_info):
            out = validate_public_http_url("https://example.com/path")
        self.assertEqual(out, "https://example.com/path")


if __name__ == "__main__":
    unittest.main()
