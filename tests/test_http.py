from __future__ import annotations

import unittest

from discourse_ai_bot.http import _resolve_url


class HttpTests(unittest.TestCase):
    def test_resolve_url_supports_relative_paths(self) -> None:
        resolved = _resolve_url("https://forum.example.com/", "/site.json")

        self.assertEqual(resolved, "https://forum.example.com/site.json")

    def test_resolve_url_preserves_absolute_urls(self) -> None:
        resolved = _resolve_url("https://forum.example.com/", "https://other.example.com/path")

        self.assertEqual(resolved, "https://other.example.com/path")
