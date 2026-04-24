from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discourse_ai_bot.discourse import DiscourseClient
from discourse_ai_bot.http import HttpError


class FakeTransport:
    def __init__(self, responses: dict[tuple[str, str], object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def request_json(self, method: str, path_or_url: str, **kwargs: object) -> object:
        self.calls.append((method, path_or_url, kwargs))
        return self.responses[(method, path_or_url)]


class SequenceTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def request_json(self, method: str, path_or_url: str, **kwargs: object) -> object:
        self.calls.append((method, path_or_url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DiscourseClientTests(unittest.TestCase):
    def test_client_applies_optional_headers_to_all_requests(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
            cookie="session=abc; theme=dark",
            user_agent="Mozilla/5.0 Test",
            extra_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

        self.assertEqual(client.http.default_headers["Api-Key"], "token")
        self.assertEqual(client.http.default_headers["Api-Username"], "bot")
        self.assertEqual(client.http.default_headers["Cookie"], "session=abc; theme=dark")
        self.assertEqual(client.http.default_headers["User-Agent"], "Mozilla/5.0 Test")
        self.assertEqual(
            client.http.default_headers["Accept-Language"],
            "en-US,en;q=0.9",
        )

    def test_client_supports_session_cookie_auth(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="session_cookie",
            cookie="session=abc; theme=dark",
        )

        self.assertEqual(client.http.default_headers["Cookie"], "session=abc; theme=dark")
        self.assertNotIn("Api-Key", client.http.default_headers)

    def test_notification_pagination_uses_load_more_path(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )
        client.http = FakeTransport(
            {
                ("GET", "/notifications.json"): {
                    "notifications": [
                        {
                            "id": 1,
                            "notification_type": 1,
                            "read": False,
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "topic_id": 100,
                            "post_number": 2,
                            "data": {"username": "alice"},
                        }
                    ],
                    "load_more_notifications": "/notifications?page=2",
                },
                ("GET", "/notifications?page=2"): {
                    "notifications": [
                        {
                            "id": 2,
                            "notification_type": 6,
                            "read": False,
                            "created_at": "2026-01-01T00:01:00+00:00",
                            "topic_id": 101,
                            "post_number": 1,
                            "data": {"username": "bob"},
                        }
                    ]
                },
            }
        )
        client.set_notification_type_map({1: "mentioned", 6: "private_message"})

        notifications = client.list_notifications()

        self.assertEqual([item.notification_id for item in notifications], [1, 2])
        self.assertEqual(client.http.calls[1][1], "/notifications?page=2")

    def test_notification_polling_can_stop_after_first_page(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )
        client.http = FakeTransport(
            {
                ("GET", "/notifications.json"): {
                    "notifications": [
                        {
                            "id": 1,
                            "notification_type": 1,
                            "read": False,
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "topic_id": 100,
                            "post_number": 2,
                            "data": {"username": "alice"},
                        }
                    ],
                    "load_more_notifications": "/notifications?page=2",
                }
            }
        )
        client.set_notification_type_map({1: "mentioned"})

        notifications = client.list_notifications(paginate=False)

        self.assertEqual([item.notification_id for item in notifications], [1])
        self.assertEqual(len(client.http.calls), 1)

    def test_resolve_post_url_supports_direct_post_links(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )
        client.http = FakeTransport(
            {
                ("GET", "/posts/901.json"): {
                    "id": 901,
                    "topic_id": 100,
                    "post_number": 2,
                }
            }
        )

        target = client.resolve_post_url("https://forum.example.com/p/901")

        self.assertEqual(target["topic_id"], 100)
        self.assertEqual(target["post_id"], 901)
        self.assertEqual(target["post_number"], 2)

    def test_resolve_post_url_supports_topic_reply_links(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )

        target = client.resolve_post_url("https://forum.example.com/t/topic-title/422/63?u=admin")

        self.assertEqual(target["topic_id"], 422)
        self.assertIsNone(target["post_id"])
        self.assertEqual(target["post_number"], 63)

    def test_resolve_category_url_supports_category_links(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )

        target = client.resolve_category_url("https://forum.example.com/c/support/7")

        self.assertEqual(target["slug"], "support")
        self.assertEqual(target["category_id"], 7)

    def test_resolve_category_url_supports_subcategory_links(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )

        target = client.resolve_category_url("https://forum.example.com/c/parent/support/17")

        self.assertEqual(target["slug"], "support")
        self.assertEqual(target["category_id"], 17)

    def test_list_latest_topics_uses_documented_endpoint(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )
        client.http = FakeTransport(
            {
                ("GET", "/latest.json?per_page=5"): {
                    "topic_list": {"topics": [{"id": 100, "title": "Latest topic"}]}
                }
            }
        )

        payload = client.list_latest_topics(per_page=5)

        self.assertEqual(payload["topic_list"]["topics"][0]["id"], 100)

    def test_list_latest_topics_supports_page_parameter(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )
        client.http = FakeTransport(
            {
                ("GET", "/latest.json?per_page=30&page=1"): {
                    "topic_list": {"topics": [{"id": 101, "title": "Next page"}]}
                }
            }
        )

        payload = client.list_latest_topics(per_page=30, page=1)

        self.assertEqual(payload["topic_list"]["topics"][0]["id"], 101)

    def test_get_topic_posts_uses_specific_posts_endpoint(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )
        client.http = FakeTransport(
            {
                ("GET", "/t/100/posts.json?post_ids%5B%5D=901&post_ids%5B%5D=902"): {
                    "post_stream": {"posts": [{"id": 901, "post_number": 2, "username": "alice"}]}
                }
            }
        )

        payload = client.get_topic_posts(100, [901, 902])

        self.assertEqual(payload["post_stream"]["posts"][0]["id"], 901)

    def test_upload_file_uses_documented_uploads_endpoint(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="api_key",
            token="token",
            username="bot",
        )
        client.http = FakeTransport(
            {
                ("POST", "/uploads.json"): {
                    "id": 1,
                    "url": "/uploads/default/original/1X/friendly_wave.gif",
                    "short_url": "upload://friendly.gif",
                }
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "friendly_wave.gif"
            gif_path.write_bytes(b"GIF89a")
            payload = client.upload_file(gif_path)

        self.assertEqual(payload["url"], "/uploads/default/original/1X/friendly_wave.gif")
        method, path, kwargs = client.http.calls[0]
        self.assertEqual((method, path), ("POST", "/uploads.json"))
        self.assertEqual(kwargs["multipart_body"]["type"], "composer")
        self.assertIn("file", kwargs["multipart_files"])

    def test_session_cookie_create_post_fetches_csrf_and_uses_form_post(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="session_cookie",
            cookie="session=abc; theme=dark",
        )
        client.http = SequenceTransport(
            [
                {"csrf": "csrf-token"},
                {"id": 999, "topic_id": 100, "post_number": 3},
            ]
        )

        response = client.create_post(raw="Hello", topic_id=100, reply_to_post_number=2)

        self.assertEqual(response["id"], 999)
        self.assertEqual(client.http.calls[0][1], "/session/csrf.json")
        method, path, kwargs = client.http.calls[1]
        self.assertEqual((method, path), ("POST", "/posts.json"))
        self.assertEqual(kwargs["headers"]["X-CSRF-Token"], "csrf-token")
        self.assertEqual(kwargs["form_body"]["topic_id"], 100)
        self.assertEqual(kwargs["form_body"]["reply_to_post_number"], 2)

    def test_session_cookie_create_post_retries_once_on_bad_csrf(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="session_cookie",
            cookie="session=abc; theme=dark",
        )
        client.http = SequenceTransport(
            [
                {"csrf": "stale-token"},
                HttpError(403, "https://forum.example.com/posts.json", '["BAD CSRF"]'),
                {"csrf": "fresh-token"},
                {"id": 1001, "topic_id": 100, "post_number": 4},
            ]
        )

        response = client.create_post(raw="Hello again", topic_id=100)

        self.assertEqual(response["id"], 1001)
        self.assertEqual(client.http.calls[2][1], "/session/csrf.json")
        self.assertEqual(client.http.calls[3][2]["headers"]["X-CSRF-Token"], "fresh-token")

    def test_record_topic_timings_uses_form_payload_and_session_headers(self) -> None:
        client = DiscourseClient(
            "https://forum.example.com",
            auth_mode="session_cookie",
            cookie="session=abc; theme=dark",
        )
        client.http = SequenceTransport(
            [
                {"csrf": "csrf-token"},
                {"success": "OK"},
            ]
        )

        response = client.record_topic_timings(
            topic_id=4580953,
            timings={172: 2033, 173: 2033},
            topic_time=3033,
            referer="https://forum.example.com/t/example/4580953/173",
        )

        self.assertEqual(response["success"], "OK")
        method, path, kwargs = client.http.calls[1]
        self.assertEqual((method, path), ("POST", "/topics/timings"))
        self.assertEqual(kwargs["form_body"]["topic_id"], 4580953)
        self.assertEqual(kwargs["form_body"]["timings[172]"], 2033)
        self.assertEqual(kwargs["headers"]["Referer"], "https://forum.example.com/t/example/4580953/173")
        self.assertEqual(kwargs["headers"]["Discourse-Present"], "true")
