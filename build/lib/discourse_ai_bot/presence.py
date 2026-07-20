from __future__ import annotations

from discourse_ai_bot.http import HttpError, JsonHttpClient


class NullPresenceAdapter:
    enabled = False

    def present(self, channel: str) -> None:
        return None

    def leave(self, channel: str) -> None:
        return None


class DiscoursePresenceAdapter:
    enabled = True

    def __init__(
        self,
        *,
        discourse_host: str,
        cookie: str,
        client_id: str,
        origin: str,
        user_agent: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.client_id = client_id
        self.origin = origin.rstrip("/")
        self._csrf_token: str | None = None
        default_headers = {
            "Cookie": cookie,
            "Origin": origin,
            "Referer": f"{origin}/",
            "Discourse-Logged-In": "true",
            "Discourse-Present": "true",
            "X-Requested-With": "XMLHttpRequest",
        }
        if user_agent:
            default_headers["User-Agent"] = user_agent
        if extra_headers:
            default_headers.update(extra_headers)
        self.http = JsonHttpClient(
            discourse_host,
            default_headers=default_headers,
            timeout_seconds=timeout_seconds,
        )

    def present(self, channel: str) -> None:
        self._request_presence_update(
            [
                ("client_id", self.client_id),
                ("present_channels[]", channel),
            ]
        )

    def leave(self, channel: str) -> None:
        self._request_presence_update(
            [
                ("client_id", self.client_id),
                ("leave_channels[]", channel),
            ]
        )

    def _request_presence_update(self, form_body: list[tuple[str, str]]) -> None:
        def make_request() -> None:
            self.http.request_json(
                "POST",
                "/presence/update",
                headers=self._session_headers(),
                form_body=form_body,
            )

        try:
            make_request()
        except HttpError as exc:
            if exc.status_code != 403 or "BAD CSRF" not in exc.body:
                raise
            self.get_csrf_token(force_refresh=True)
            make_request()

    def get_csrf_token(self, *, force_refresh: bool = False) -> str:
        if self._csrf_token and not force_refresh:
            return self._csrf_token

        response = self.http.request_json("GET", "/session/csrf.json")
        if not isinstance(response, dict) or not isinstance(response.get("csrf"), str):
            raise ValueError("Discourse /session/csrf.json did not return a csrf token for presence.")
        self._csrf_token = response["csrf"]
        return self._csrf_token

    def _session_headers(self) -> dict[str, str]:
        return {
            "X-CSRF-Token": self.get_csrf_token(),
            "X-Requested-With": "XMLHttpRequest",
            "Origin": self.origin,
            "Referer": f"{self.origin}/",
        }
