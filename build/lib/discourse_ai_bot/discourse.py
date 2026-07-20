from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from discourse_ai_bot.http import HttpError, JsonHttpClient
from discourse_ai_bot.models import Notification


class DiscourseClient:
    def __init__(
        self,
        host: str,
        *,
        auth_mode: str,
        token: str | None = None,
        username: str | None = None,
        cookie: str | None = None,
        user_agent: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        default_headers: dict[str, str] = {}
        if auth_mode == "api_key":
            if not token or not username:
                raise ValueError("token and username are required for api_key auth mode.")
            default_headers["Api-Key"] = token
            default_headers["Api-Username"] = username
        elif auth_mode == "session_cookie":
            if not cookie:
                raise ValueError("cookie is required for session_cookie auth mode.")
        else:
            raise ValueError("auth_mode must be 'api_key' or 'session_cookie'.")
        if cookie:
            default_headers["Cookie"] = cookie
        if user_agent:
            default_headers["User-Agent"] = user_agent
        if extra_headers:
            default_headers.update(extra_headers)
        self.http = JsonHttpClient(
            host,
            default_headers=default_headers,
            timeout_seconds=timeout_seconds,
        )
        self.notification_type_map: dict[int, str] = {}
        self.auth_mode = auth_mode
        self.host = host.rstrip("/")
        self._csrf_token: str | None = None

    def set_notification_type_map(self, mapping: dict[int, str]) -> None:
        self.notification_type_map = dict(mapping)

    def get_site_info(self) -> dict[str, Any]:
        response = self.http.request_json("GET", "/site.json")
        return response if isinstance(response, dict) else {}

    def list_categories(self) -> list[dict[str, Any]]:
        site_info = self.get_site_info()
        categories = site_info.get("categories", [])
        return [item for item in categories if isinstance(item, dict)]

    def get_user(self, username: str) -> dict[str, Any]:
        response = self.http.request_json("GET", f"/u/{username}.json")
        return response if isinstance(response, dict) else {}

    def get_current_session(self) -> dict[str, Any]:
        response = self.http.request_json("GET", "/session/current.json")
        return response if isinstance(response, dict) else {}

    def get_topic(self, topic_id: int, *, post_number: int | None = None) -> dict[str, Any]:
        path = f"/t/{topic_id}.json" if post_number is None else f"/t/{topic_id}/{post_number}.json"
        response = self.http.request_json("GET", path)
        return response if isinstance(response, dict) else {}

    def get_topic_posts(self, topic_id: int, post_ids: list[int]) -> dict[str, Any]:
        query = urlencode([("post_ids[]", post_id) for post_id in post_ids], doseq=True)
        response = self.http.request_json(
            "GET",
            f"/t/{topic_id}/posts.json?{query}",
        )
        return response if isinstance(response, dict) else {}

    def list_latest_topics(self, *, per_page: int = 5, page: int | None = None) -> dict[str, Any]:
        query = {"per_page": per_page}
        if page is not None:
            query["page"] = page
        response = self.http.request_json("GET", f"/latest.json?{urlencode(query)}")
        return response if isinstance(response, dict) else {}

    def list_category_topics(self, *, slug: str, category_id: int) -> dict[str, Any]:
        response = self.http.request_json("GET", f"/c/{slug}/{category_id}.json")
        return response if isinstance(response, dict) else {}

    def get_post(self, post_id: int) -> dict[str, Any]:
        response = self.http.request_json("GET", f"/posts/{post_id}.json")
        return response if isinstance(response, dict) else {}

    def list_notifications(self, *, paginate: bool = True) -> list[Notification]:
        notifications: list[Notification] = []
        page: str | None = "/notifications.json"
        while page:
            payload = self.http.request_json("GET", page)
            if not isinstance(payload, dict):
                break
            for item in payload.get("notifications", []):
                if isinstance(item, dict):
                    notifications.append(
                        Notification.from_payload(item, type_name_map=self.notification_type_map)
                    )
            if not paginate:
                break
            next_page = payload.get("load_more_notifications")
            page = next_page if isinstance(next_page, str) and next_page.strip() else None
        return notifications

    def mark_notification_read(self, notification_id: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if notification_id is not None:
            payload["id"] = notification_id
        if self.auth_mode == "session_cookie":
            response = self._request_with_session_auth(
                "PUT",
                "/notifications/mark-read.json",
                form_body=payload,
            )
        else:
            response = self.http.request_json(
                "PUT",
                "/notifications/mark-read.json",
                json_body=payload,
            )
        return response if isinstance(response, dict) else {}

    def create_post(
        self,
        *,
        raw: str,
        topic_id: int | None = None,
        reply_to_post_number: int | None = None,
        title: str | None = None,
        category: int | None = None,
        target_recipients: str | None = None,
        archetype: str | None = None,
        auto_track: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "raw": raw,
            "auto_track": auto_track,
        }
        if topic_id is not None:
            payload["topic_id"] = topic_id
        if reply_to_post_number is not None:
            payload["reply_to_post_number"] = reply_to_post_number
        if title is not None:
            payload["title"] = title
        if category is not None:
            payload["category"] = category
        if target_recipients is not None:
            payload["target_recipients"] = target_recipients
        if archetype is not None:
            payload["archetype"] = archetype

        if self.auth_mode == "session_cookie":
            response = self._request_with_session_auth(
                "POST",
                "/posts.json",
                form_body=payload,
            )
        else:
            response = self.http.request_json("POST", "/posts.json", json_body=payload)
        return response if isinstance(response, dict) else {}

    def create_topic(self, *, title: str, raw: str, category: int | None = None) -> dict[str, Any]:
        return self.create_post(title=title, raw=raw, category=category)

    def upload_file(
        self,
        file_path: str | Path,
        *,
        upload_type: str = "composer",
        synchronous: bool = True,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        payload: dict[str, Any] = {
            "type": upload_type,
            "synchronous": "true" if synchronous else "false",
        }
        if user_id is not None:
            payload["user_id"] = user_id

        content_type, _ = mimetypes.guess_type(path.name)
        file_tuple = (
            path.name,
            path.read_bytes(),
            content_type or "application/octet-stream",
        )

        if self.auth_mode == "session_cookie":
            response = self._request_with_session_auth(
                "POST",
                "/uploads.json",
                multipart_body=payload,
                multipart_files={"file": file_tuple},
            )
        else:
            response = self.http.request_json(
                "POST",
                "/uploads.json",
                multipart_body=payload,
                multipart_files={"file": file_tuple},
            )
        return response if isinstance(response, dict) else {}

    def record_topic_timings(
        self,
        *,
        topic_id: int,
        timings: dict[int, int],
        topic_time: int,
        referer: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "topic_id": topic_id,
            "topic_time": topic_time,
        }
        for post_number, duration in timings.items():
            payload[f"timings[{post_number}]"] = duration

        headers = {
            "Accept": "*/*",
            "X-SILENCE-LOGGER": "true",
            "Discourse-Background": "true",
            "Discourse-Logged-In": "true",
            "Discourse-Present": "true",
        }
        if referer:
            headers["Referer"] = referer

        if self.auth_mode == "session_cookie":
            response = self._request_with_session_auth(
                "POST",
                "/topics/timings",
                form_body=payload,
                extra_headers=headers,
            )
        else:
            response = self.http.request_json(
                "POST",
                "/topics/timings",
                headers=headers,
                form_body=payload,
            )
        return response if isinstance(response, dict) else {}

    def resolve_post_url(self, post_url: str) -> dict[str, int | None]:
        path = urlsplit(post_url).path or post_url
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 2 and segments[0] == "p" and segments[1].isdigit():
            post = self.get_post(int(segments[1]))
            return {
                "topic_id": int(post["topic_id"]),
                "post_id": int(post["id"]),
                "post_number": int(post["post_number"]),
            }

        if segments and segments[0] == "t":
            numeric_indices = [
                (index, segment)
                for index, segment in enumerate(segments[1:], start=1)
                if segment.isdigit()
            ]
            if not numeric_indices:
                raise ValueError(f"Unsupported Discourse topic URL: {post_url}")

            topic_index, topic_segment = numeric_indices[0]
            topic_id = int(topic_segment)
            post_number = None
            if len(segments) > topic_index + 1 and segments[topic_index + 1].isdigit():
                post_number = int(segments[topic_index + 1])

            return {
                "topic_id": topic_id,
                "post_id": None,
                "post_number": post_number,
            }

        raise ValueError(f"Unsupported Discourse post URL: {post_url}")

    def resolve_category_url(self, category_url: str) -> dict[str, str | int]:
        path = urlsplit(category_url).path or category_url
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 2 and segments[0] == "c":
            numeric_segments = [segment for segment in segments[1:] if segment.isdigit()]
            if not numeric_segments:
                raise ValueError(f"Unsupported Discourse category URL: {category_url}")
            slug_segments = [segment for segment in segments[1:] if not segment.isdigit()]
            return {
                "slug": slug_segments[-1] if slug_segments else "",
                "category_id": int(numeric_segments[-1]),
            }
        raise ValueError(f"Unsupported Discourse category URL: {category_url}")

    def get_csrf_token(self, *, force_refresh: bool = False) -> str:
        if self.auth_mode != "session_cookie":
            raise ValueError("CSRF tokens are only used for session_cookie auth mode.")
        if self._csrf_token and not force_refresh:
            return self._csrf_token

        response = self.http.request_json("GET", "/session/csrf.json")
        if not isinstance(response, dict) or not isinstance(response.get("csrf"), str):
            raise ValueError("Discourse /session/csrf.json did not return a csrf token.")
        self._csrf_token = response["csrf"]
        return self._csrf_token

    def _request_with_session_auth(
        self,
        method: str,
        path_or_url: str,
        *,
        form_body: dict[str, Any] | None = None,
        multipart_body: dict[str, Any] | None = None,
        multipart_files: dict[str, tuple[str, bytes, str]] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        if self.auth_mode != "session_cookie":
            return self.http.request_json(
                method,
                path_or_url,
                headers=extra_headers,
                form_body=form_body,
                multipart_body=multipart_body,
                multipart_files=multipart_files,
            )

        def make_request() -> Any:
            return self.http.request_json(
                method,
                path_or_url,
                headers={**self._session_headers(), **(extra_headers or {})},
                form_body=form_body,
                multipart_body=multipart_body,
                multipart_files=multipart_files,
            )

        try:
            return make_request()
        except HttpError as exc:
            if exc.status_code != 403 or "BAD CSRF" not in exc.body:
                raise
            self.get_csrf_token(force_refresh=True)
            return make_request()

    def _session_headers(self) -> dict[str, str]:
        return {
            "X-CSRF-Token": self.get_csrf_token(),
            "X-Requested-With": "XMLHttpRequest",
            "Origin": self.host,
            "Referer": f"{self.host}/",
        }
