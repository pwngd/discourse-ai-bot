from __future__ import annotations

import unittest

from discourse_ai_bot.http import HttpError
from discourse_ai_bot.presence import DiscoursePresenceAdapter


class PresenceAdapterTests(unittest.TestCase):
    def test_adapter_builds_configured_headers(self) -> None:
        adapter = DiscoursePresenceAdapter(
            discourse_host="https://forum.example.com",
            cookie="session=abc; theme=dark",
            client_id="client123",
            origin="https://forum.example.com",
            user_agent="Mozilla/5.0 Test",
            extra_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
        )

        headers = adapter.http.default_headers
        self.assertEqual(headers["Cookie"], "session=abc; theme=dark")
        self.assertEqual(headers["User-Agent"], "Mozilla/5.0 Test")
        self.assertEqual(headers["Accept-Language"], "en-US,en;q=0.9")
        self.assertEqual(headers["Accept-Encoding"], "gzip, deflate, br")
        self.assertEqual(headers["Origin"], "https://forum.example.com")

    def test_presence_update_fetches_csrf_and_uses_session_headers(self) -> None:
        adapter = DiscoursePresenceAdapter(
            discourse_host="https://forum.example.com",
            cookie="session=abc; theme=dark",
            client_id="client123",
            origin="https://forum.example.com",
        )
        calls: list[tuple[str, str, dict[str, object]]] = []

        class FakeHttp:
            def request_json(self, method: str, path_or_url: str, **kwargs: object) -> object:
                calls.append((method, path_or_url, kwargs))
                if path_or_url == "/session/csrf.json":
                    return {"csrf": "csrf-token"}
                return {"success": "OK"}

        adapter.http = FakeHttp()  # type: ignore[assignment]

        adapter.present("/discourse-presence/reply/123")

        self.assertEqual(calls[0][1], "/session/csrf.json")
        self.assertEqual(calls[1][1], "/presence/update")
        self.assertEqual(calls[1][2]["headers"]["X-CSRF-Token"], "csrf-token")

    def test_presence_update_retries_on_bad_csrf(self) -> None:
        adapter = DiscoursePresenceAdapter(
            discourse_host="https://forum.example.com",
            cookie="session=abc; theme=dark",
            client_id="client123",
            origin="https://forum.example.com",
        )
        calls: list[tuple[str, str, dict[str, object]]] = []
        responses: list[object] = [
            {"csrf": "stale-token"},
            HttpError(403, "https://forum.example.com/presence/update", '["BAD CSRF"]'),
            {"csrf": "fresh-token"},
            {"success": "OK"},
        ]

        class FakeHttp:
            def request_json(self, method: str, path_or_url: str, **kwargs: object) -> object:
                calls.append((method, path_or_url, kwargs))
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        adapter.http = FakeHttp()  # type: ignore[assignment]

        adapter.leave("/discourse-presence/reply/123")

        self.assertEqual(calls[2][1], "/session/csrf.json")
        self.assertEqual(calls[3][2]["headers"]["X-CSRF-Token"], "fresh-token")
