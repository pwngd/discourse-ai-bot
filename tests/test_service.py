from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from discourse_ai_bot.gifs import GifOption
from discourse_ai_bot.models import AutonomousSelection, ModelDecision, Notification
from discourse_ai_bot.presence import NullPresenceAdapter
from discourse_ai_bot.service import BotService
from discourse_ai_bot.settings import Settings
from discourse_ai_bot.storage import BotStorage


@dataclass
class FakeClock:
    current: datetime

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current = self.current + timedelta(seconds=seconds)


class FakeDiscourseClient:
    def __init__(self, notifications: list[Notification], *, fail_create_post: bool = False) -> None:
        self.notifications = notifications
        self.fail_create_post = fail_create_post
        self.notification_type_map: dict[int, str] = {}
        self.notification_paginate_calls: list[bool] = []
        self.created_posts: list[dict[str, object]] = []
        self.mark_read_calls: list[int] = []
        self.site_info = {
            "notification_types": {
                "mentioned": 1,
                "replied": 2,
                "private_message": 6,
            },
            "categories": [
                {"id": 10, "slug": "restricted", "parent_category_id": None},
                {"id": 11, "slug": "child-restricted", "parent_category_id": 10},
                {"id": 20, "slug": "support", "parent_category_id": None},
            ],
        }
        self.session_payload = {"current_user": {"id": 77, "username": "bot", "name": "Bot"}}
        self.user_payload = {"user": {"id": 77, "username": "bot", "name": "Bot"}}
        self.topics = {
            100: {
                "title": "Topic 100",
                "slug": "topic-100",
                "archetype": "regular",
                "post_stream": {
                    "posts": [
                        {
                            "id": 900,
                            "topic_id": 100,
                            "post_number": 1,
                            "username": "alice",
                            "cooked": "<p>First post</p>",
                        },
                        {
                            "id": 901,
                            "topic_id": 100,
                            "post_number": 2,
                            "username": "alice",
                            "cooked": "<p>Hey @bot</p>",
                        },
                    ]
                },
            },
            200: {
                "title": "Private chat",
                "slug": "private-chat",
                "archetype": "private_message",
                "post_stream": {
                    "posts": [
                        {
                            "id": 1000,
                            "topic_id": 200,
                            "post_number": 1,
                            "username": "alice",
                            "cooked": "<p>Private ping</p>",
                        }
                    ]
                },
            },
            300: {
                "title": "Latest support question",
                "slug": "latest-support-question",
                "archetype": "regular",
                "post_stream": {
                    "posts": [
                        {
                            "id": 1100,
                            "topic_id": 300,
                            "post_number": 1,
                            "username": "charlie",
                            "cooked": "<p>Does anyone know how to configure this?</p>",
                        }
                    ]
                },
            },
            301: {
                "title": "Allowed followup question",
                "slug": "allowed-followup-question",
                "archetype": "regular",
                "post_stream": {
                    "posts": [
                        {
                            "id": 1101,
                            "topic_id": 301,
                            "post_number": 1,
                            "username": "dana",
                            "cooked": "<p>Can someone explain the supported path?</p>",
                        }
                    ]
                },
            },
        }
        self.posts = {
            901: {
                "id": 901,
                "topic_id": 100,
                "post_number": 2,
                "username": "alice",
                "cooked": "<p>Hey @bot</p>",
            },
            1000: {
                "id": 1000,
                "topic_id": 200,
                "post_number": 1,
                "username": "alice",
                "cooked": "<p>Private ping</p>",
            },
            1100: {
                "id": 1100,
                "topic_id": 300,
                "post_number": 1,
                "username": "charlie",
                "cooked": "<p>Does anyone know how to configure this?</p>",
            },
            1101: {
                "id": 1101,
                "topic_id": 301,
                "post_number": 1,
                "username": "dana",
                "cooked": "<p>Can someone explain the supported path?</p>",
            },
        }
        self.upload_calls: list[dict[str, object]] = []
        self.latest_topic_calls: list[dict[str, int | None]] = []
        self.latest_topics_payload = {
            "topic_list": {
                "topics": [
                    {
                        "id": 300,
                        "title": "Latest support question",
                        "slug": "latest-support-question",
                        "highest_post_number": 1,
                        "last_poster_username": "charlie",
                        "category_id": 20,
                    }
                ]
            }
        }
        self.latest_topic_pages: dict[int | None, dict[str, object]] | None = None

    def set_notification_type_map(self, mapping: dict[int, str]) -> None:
        self.notification_type_map = mapping

    def get_site_info(self) -> dict[str, object]:
        return self.site_info

    def get_user(self, username: str) -> dict[str, object]:
        return self.user_payload

    def get_current_session(self) -> dict[str, object]:
        return self.session_payload

    def list_notifications(self, *, paginate: bool = True) -> list[Notification]:
        self.notification_paginate_calls.append(paginate)
        return list(self.notifications)

    def list_latest_topics(self, *, per_page: int = 5, page: int | None = None) -> dict[str, object]:
        self.latest_topic_calls.append({"per_page": per_page, "page": page})
        if self.latest_topic_pages is not None:
            return self.latest_topic_pages.get(page, {"topic_list": {"topics": []}})
        return self.latest_topics_payload

    def get_topic(self, topic_id: int, *, post_number: int | None = None) -> dict[str, object]:
        return self.topics[topic_id]

    def get_post(self, post_id: int) -> dict[str, object]:
        return self.posts[post_id]

    def resolve_post_url(self, post_url: str) -> dict[str, int | None]:
        if post_url.endswith("/p/901"):
            return {"topic_id": 100, "post_id": 901, "post_number": 2}
        if "/t/" in post_url and post_url.endswith("/2"):
            return {"topic_id": 100, "post_id": None, "post_number": 2}
        if post_url == "https://forum.example.com/t/latest-support-question/300/1":
            return {"topic_id": 300, "post_id": None, "post_number": 1}
        if post_url == "https://forum.example.com/t/allowed-followup-question/301/1":
            return {"topic_id": 301, "post_id": None, "post_number": 1}
        raise ValueError("unsupported post url")

    def create_post(self, **kwargs: object) -> dict[str, object]:
        if self.fail_create_post:
            raise RuntimeError("post failed")
        self.created_posts.append(kwargs)
        return {"id": len(self.created_posts) + 5000, **kwargs}

    def upload_file(self, file_path: object, **kwargs: object) -> dict[str, object]:
        self.upload_calls.append({"file_path": file_path, **kwargs})
        return {"id": 1, "url": "/uploads/default/original/1X/friendly_wave.gif"}

    def mark_notification_read(self, notification_id: int | None = None) -> dict[str, object]:
        if notification_id is not None:
            self.mark_read_calls.append(notification_id)
        return {"success": "OK"}


class FakeOllamaClient:
    def __init__(
        self,
        decisions: list[ModelDecision | Exception],
        *,
        autonomous_selections: list[AutonomousSelection | Exception] | None = None,
    ) -> None:
        self.decisions = decisions
        self.autonomous_selections = autonomous_selections or []
        self.calls: list[dict[str, object]] = []

    def decide(self, **kwargs: object) -> ModelDecision:
        self.calls.append(kwargs)
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision

    def compose_manual_reply(self, **kwargs: object) -> ModelDecision:
        self.calls.append(kwargs)
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision

    def compose_autonomous_reply(self, **kwargs: object) -> ModelDecision:
        self.calls.append(kwargs)
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision

    def select_autonomous_reply_target(self, **kwargs: object) -> AutonomousSelection:
        self.calls.append(kwargs)
        selection = self.autonomous_selections.pop(0)
        if isinstance(selection, Exception):
            raise selection
        return selection


class FakeDocsContext:
    def __init__(self, content: str) -> None:
        self.content = content

    def format_for_prompt(self, *, max_chars: int) -> str:
        return self.content[:max_chars]


class FakeRobloxDocsClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def context_for_text(self, text: str) -> FakeDocsContext | None:
        self.queries.append(text)
        if "Part.Shape" not in text:
            return None
        return FakeDocsContext("Verified Roblox API docs context:\n- class: Part\n  Relevant members:\n    - Part.Shape")


class FailingPresenceAdapter(NullPresenceAdapter):
    enabled = True

    def present(self, channel: str) -> None:
        raise RuntimeError("presence failed")

    def leave(self, channel: str) -> None:
        raise RuntimeError("leave failed")


def make_notification(notification_id: int, notification_type: int, topic_id: int, post_number: int) -> Notification:
    return Notification(
        notification_id=notification_id,
        notification_type=notification_type,
        type_name=None,
        read=False,
        created_at="2026-01-01T00:00:00+00:00",
        topic_id=topic_id,
        post_number=post_number,
        slug="topic",
        data={"username": "alice"},
    )


class ServiceTests(unittest.TestCase):
    def make_settings(self, database_path: str, **overrides: object) -> Settings:
        base = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="api_key",
            discourse_token="token",
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie=None,
            discourse_user_agent=None,
            discourse_extra_headers={},
            system_prompt="Prompt",
            bot_db_path=database_path,
            bot_poll_interval_seconds=5,
            bot_response_delay_min_seconds=0,
            bot_response_delay_max_seconds=0,
            bot_max_context_posts=4,
            bot_mark_read_on_skip=True,
            bot_allowed_triggers=("mentioned", "replied", "private_message"),
            ollama_options={},
            ollama_keep_alive="5m",
            ollama_timeout_seconds=30,
            bot_typing_mode="none",
            discourse_presence_cookie=None,
            discourse_presence_client_id=None,
            discourse_presence_origin="https://forum.example.com",
            discourse_presence_user_agent=None,
            discourse_presence_extra_headers={},
            discourse_presence_reply_channel_template="/discourse-presence/reply/{topic_id}",
        )
        values = base.__dict__ | overrides
        return Settings(**values)

    def test_mention_reply_posts_and_marks_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(1, 1, 100, 2)])
            ollama = FakeOllamaClient([ModelDecision("reply", "Thanks for the ping.", "Direct ask")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                randomizer=random.Random(0),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(len(discourse.created_posts), 1)
            self.assertEqual(discourse.created_posts[0]["reply_to_post_number"], 2)
            self.assertEqual(discourse.mark_read_calls, [1])
            self.assertTrue(storage.is_handled(1))

    def test_notification_decision_failure_is_warning_and_not_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(15, 1, 100, 2)])
            ollama = FakeOllamaClient([RuntimeError("decision timed out twice")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            with self.assertLogs(service.logger, level="WARNING") as logs:
                service.run_once()

            self.assertEqual(discourse.created_posts, [])
            self.assertFalse(storage.is_handled(15))
            self.assertTrue(any("Failed to evaluate notification 15" in line for line in logs.output))
            self.assertFalse(any(line.startswith("ERROR") for line in logs.output))

    def test_skip_marks_notification_handled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(2, 1, 100, 2)])
            ollama = FakeOllamaClient([ModelDecision("skip", "", "No reply needed")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(discourse.created_posts, [])
            self.assertEqual(discourse.mark_read_calls, [2])
            self.assertTrue(storage.is_handled(2))

    def test_roblox_docs_context_is_attached_for_coding_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(13, 1, 100, 2)])
            discourse.topics[100]["post_stream"]["posts"][1]["cooked"] = (
                "<p>In Luau, is <code>Part.Shape</code> an Enum.PartType?</p>"
            )
            ollama = FakeOllamaClient([ModelDecision("skip", "", "Docs answer not needed")])
            docs = FakeRobloxDocsClient()
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(Path(temp_dir) / "bot.sqlite3"),
                    bot_roblox_docs_enabled=True,
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
                roblox_docs_client=docs,
            )

            service.run_once()

            self.assertEqual(len(docs.queries), 1)
            self.assertIn("Part.Shape", docs.queries[0])
            self.assertIn("roblox_docs_context", ollama.calls[0])
            self.assertIn("Verified Roblox API docs context", ollama.calls[0]["roblox_docs_context"])

    def test_roblox_docs_context_is_not_attached_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(14, 1, 100, 2)])
            discourse.topics[100]["post_stream"]["posts"][1]["cooked"] = (
                "<p>In Luau, is <code>Part.Shape</code> an Enum.PartType?</p>"
            )
            ollama = FakeOllamaClient([ModelDecision("skip", "", "No docs")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertIsNone(ollama.calls[0]["roblox_docs_context"])

    def test_private_message_replies_into_topic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(3, 6, 200, 1)])
            ollama = FakeOllamaClient([ModelDecision("reply", "I'll take a look.", "PM needs answer")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(discourse.created_posts[0]["topic_id"], 200)

    def test_failed_post_is_retried_later(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([make_notification(4, 1, 100, 2)], fail_create_post=True)
            ollama = FakeOllamaClient([ModelDecision("reply", "Answer", "Need reply")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(database_path)),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            pending = storage.get_pending_job(4)
            self.assertIsNotNone(pending)
            self.assertEqual(pending.attempts, 1)
            self.assertFalse(storage.is_handled(4))

    def test_duplicate_notification_is_only_processed_once(self) -> None:
        notification = make_notification(5, 1, 100, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([notification, notification])
            ollama = FakeOllamaClient([ModelDecision("reply", "Single answer", "Need reply")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(len(discourse.created_posts), 1)

    def test_presence_failures_do_not_block_reply(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(6, 1, 100, 2)])
            ollama = FakeOllamaClient([ModelDecision("reply", "Still sending.", "Need reply")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(Path(temp_dir) / "bot.sqlite3"),
                    bot_typing_mode="presence_update",
                    discourse_presence_cookie="cookie=value",
                    discourse_presence_client_id="client123",
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=FailingPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(len(discourse.created_posts), 1)

    def test_reply_can_append_uploaded_gif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(9, 1, 100, 2)])
            ollama = FakeOllamaClient(
                [ModelDecision("reply", "Thanks for the ping.", "Direct ask", gif_id="friendly_wave")]
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                randomizer=random.Random(0),
                now_fn=clock.now,
            )
            gif_path = Path(temp_dir) / "friendly_wave.gif"
            gif_path.write_bytes(b"GIF89a")

            class FakeCatalog:
                def list_options(self) -> list[GifOption]:
                    return [GifOption("friendly_wave", gif_path, "friendly wave")]

                def get(self, gif_id: str | None) -> GifOption | None:
                    return GifOption("friendly_wave", gif_path, "friendly wave") if gif_id == "friendly_wave" else None

            service.gif_catalog = FakeCatalog()  # type: ignore[assignment]
            service.run_once()

            self.assertEqual(len(discourse.upload_calls), 1)
            self.assertIn("![friendly wave](https://forum.example.com/uploads/default/original/1X/friendly_wave.gif)", discourse.created_posts[0]["raw"])

    def test_reply_can_append_uploaded_gif_from_short_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")

            class ShortUrlDiscourseClient(FakeDiscourseClient):
                def upload_file(self, file_path: object, **kwargs: object) -> dict[str, object]:
                    self.upload_calls.append({"file_path": file_path, **kwargs})
                    return {"id": 1, "short_url": "upload://friendly_wave.gif"}

            discourse = ShortUrlDiscourseClient([make_notification(10, 1, 100, 2)])
            ollama = FakeOllamaClient(
                [ModelDecision("reply", "Thanks for the ping.", "Direct ask", gif_id="friendly_wave")]
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                randomizer=random.Random(0),
                now_fn=clock.now,
            )
            gif_path = Path(temp_dir) / "friendly_wave.gif"
            gif_path.write_bytes(b"GIF89a")

            class FakeCatalog:
                def list_options(self) -> list[GifOption]:
                    return [GifOption("friendly_wave", gif_path, "friendly wave")]

                def get(self, gif_id: str | None) -> GifOption | None:
                    return GifOption("friendly_wave", gif_path, "friendly wave") if gif_id == "friendly_wave" else None

            service.gif_catalog = FakeCatalog()  # type: ignore[assignment]
            service.run_once()

            self.assertEqual(len(discourse.upload_calls), 1)
            self.assertIn("![friendly wave](upload://friendly_wave.gif)", discourse.created_posts[0]["raw"])

    def test_session_cookie_mode_bootstraps_from_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([])
            ollama = FakeOllamaClient([])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(Path(temp_dir) / "bot.sqlite3"),
                    discourse_auth_mode="session_cookie",
                    discourse_token=None,
                    discourse_username=None,
                    discourse_cookie="session=abc; other=xyz",
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            identity = service.bootstrap()

            self.assertEqual(identity.username, "bot")
            self.assertEqual(identity.user_id, 77)

    def test_manual_ai_command_generates_and_posts_reply(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            command_id = storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/901",
                user_request="Tell them we are looking into it.",
                created_at="2026-01-01T00:00:00+00:00",
            )
            discourse = FakeDiscourseClient([])
            ollama = FakeOllamaClient([ModelDecision("reply", "We're checking now.", "Operator request")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(database_path)),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            commands = storage.list_manual_commands()
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0].command_id, command_id)
            self.assertEqual(commands[0].status, "completed")
            self.assertEqual(discourse.created_posts[0]["topic_id"], 100)
            self.assertEqual(discourse.created_posts[0]["reply_to_post_number"], 2)
            self.assertEqual(discourse.created_posts[0]["raw"], "We're checking now.")

    def test_manual_ai_command_uses_exact_reply_number_from_topic_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            storage.enqueue_manual_command(
                post_url="https://forum.example.com/t/topic-100/100/2",
                user_request="Reply to the exact linked reply.",
                created_at="2026-01-01T00:00:00+00:00",
            )
            discourse = FakeDiscourseClient([])
            ollama = FakeOllamaClient([ModelDecision("reply", "Exact reply target.", "Operator request")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(database_path)),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(discourse.created_posts[0]["reply_to_post_number"], 2)

    def test_autonomous_reply_scan_generates_reply_for_selected_latest_post(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([])
            selected_url = "https://forum.example.com/t/latest-support-question/300/1"
            ollama = FakeOllamaClient(
                [ModelDecision("reply", "The supported path is to configure it here.", "Autonomous reply")],
                autonomous_selections=[
                    AutonomousSelection("reply", selected_url, 0.91, "Clear unanswered setup question")
                ],
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(database_path),
                    bot_autonomous_reply_enabled=True,
                    bot_autonomous_reply_interval_seconds=60,
                    bot_autonomous_reply_latest_count=3,
                    bot_autonomous_reply_min_confidence=0.75,
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            commands = storage.list_manual_commands()
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0].status, "completed")
            self.assertEqual(commands[0].post_url, selected_url)
            self.assertIn("Clear unanswered setup question", commands[0].user_request)
            self.assertTrue(commands[0].user_request.startswith("AUTONOMOUS_REPLY_SELECTION:"))
            stats = storage.stats_summary()
            self.assertEqual(stats["autonomous_queued"], 1)
            self.assertEqual(discourse.created_posts[0]["raw"], "The supported path is to configure it here.")

    def test_autonomous_reply_command_uses_autonomous_composer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            storage.enqueue_manual_command(
                post_url="https://forum.example.com/t/topic-100/100/2",
                user_request="AUTONOMOUS_REPLY_SELECTION: Good place to join the thread.",
                created_at="2026-01-01T00:00:00+00:00",
            )
            discourse = FakeDiscourseClient([])
            ollama = FakeOllamaClient([ModelDecision("reply", "That is not how this works.", "Autonomous reply")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(database_path)),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(discourse.created_posts[0]["raw"], "That is not how this works.")
            self.assertIn("selection_reason", ollama.calls[0])
            self.assertNotIn("user_request", ollama.calls[0])

    def test_autonomous_reply_selection_failure_forces_first_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([])
            selected_url = "https://forum.example.com/t/latest-support-question/300/1"
            ollama = FakeOllamaClient(
                [ModelDecision("reply", "Forced fallback reply.", "Autonomous reply")],
                autonomous_selections=[
                    RuntimeError("thinking never finished"),
                ],
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(database_path),
                    bot_autonomous_reply_enabled=True,
                    bot_autonomous_reply_interval_seconds=60,
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            commands = storage.list_manual_commands()
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0].status, "completed")
            self.assertEqual(commands[0].post_url, selected_url)
            self.assertIn("forcing a reply", commands[0].user_request)
            autonomous_calls = [call for call in ollama.calls if "candidates" in call]
            self.assertEqual(len(autonomous_calls), 1)
            self.assertEqual(discourse.created_posts[0]["raw"], "Forced fallback reply.")

    def test_autonomous_reply_selection_matches_formatted_post_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([])
            selected_url = "https://forum.example.com/t/latest-support-question/300/1"
            formatted_url = f"[selected]({selected_url}?u=bot)"
            ollama = FakeOllamaClient(
                [ModelDecision("reply", "Matched despite formatting.", "Autonomous reply")],
                autonomous_selections=[
                    AutonomousSelection("reply", formatted_url, 0.91, "Formatted link")
                ],
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(database_path),
                    bot_autonomous_reply_enabled=True,
                    bot_autonomous_reply_interval_seconds=60,
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            commands = storage.list_manual_commands()
            self.assertEqual(commands[0].status, "completed")
            self.assertEqual(commands[0].post_url, selected_url)
            self.assertEqual(discourse.created_posts[0]["raw"], "Matched despite formatting.")

    def test_autonomous_reply_generation_retries_until_reply_is_scheduled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([])
            selected_url = "https://forum.example.com/t/latest-support-question/300/1"
            ollama = FakeOllamaClient(
                [
                    RuntimeError("autonomous composer timed out"),
                    ModelDecision("reply", "Generated on retry.", "Autonomous reply"),
                ],
                autonomous_selections=[
                    AutonomousSelection("reply", selected_url, 0.91, "Good place to answer")
                ],
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(database_path),
                    bot_autonomous_reply_enabled=True,
                    bot_autonomous_reply_interval_seconds=60,
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()
            commands = storage.list_manual_commands()
            self.assertEqual(commands[0].status, "queued")
            self.assertEqual(commands[0].attempts, 1)
            self.assertEqual(discourse.created_posts, [])

            clock.advance(31)
            service.run_once()

            commands = storage.list_manual_commands()
            self.assertEqual(commands[0].status, "completed")
            self.assertEqual(discourse.created_posts[0]["raw"], "Generated on retry.")

    def test_autonomous_reply_scan_skips_post_with_pending_notification_reply(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([make_notification(12, 1, 100, 2)])
            discourse.latest_topics_payload = {
                "topic_list": {
                    "topics": [
                        {
                            "id": 100,
                            "title": "Topic 100",
                            "slug": "topic-100",
                            "highest_post_number": 2,
                            "last_poster_username": "alice",
                            "category_id": 20,
                        }
                    ]
                }
            }
            ollama = FakeOllamaClient([ModelDecision("reply", "Already handling it.", "Notification reply")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(database_path),
                    bot_autonomous_reply_enabled=True,
                    bot_autonomous_reply_interval_seconds=60,
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            self.assertEqual(len(discourse.created_posts), 1)
            self.assertTrue(storage.has_bot_reply_target(topic_id=100, reply_to_post_number=2))
            autonomous_calls = [call for call in ollama.calls if "candidates" in call]
            self.assertEqual(autonomous_calls, [])

    def test_autonomous_reply_scan_skips_candidates_when_model_declines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([])
            ollama = FakeOllamaClient(
                [],
                autonomous_selections=[
                    AutonomousSelection("skip", None, 0.2, "Nothing needs a proactive reply")
                ],
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(database_path),
                    bot_autonomous_reply_enabled=True,
                    bot_autonomous_reply_interval_seconds=1,
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()
            clock.advance(2)
            service.run_once()

            self.assertEqual(storage.list_manual_commands(), [])
            self.assertEqual(storage.stats_summary()["autonomous_skipped"], 1)
            autonomous_calls = [
                call for call in ollama.calls if "candidates" in call
            ]
            self.assertEqual(len(autonomous_calls), 1)

    def test_autonomous_reply_scan_skips_blocked_categories_and_loads_more_latest_topics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            discourse = FakeDiscourseClient([])
            discourse.latest_topic_pages = {
                None: {
                    "topic_list": {
                        "topics": [
                            {
                                "id": 300,
                                "title": "Blocked child topic",
                                "slug": "latest-support-question",
                                "highest_post_number": 1,
                                "last_poster_username": "charlie",
                                "category_id": 11,
                            }
                        ]
                    }
                },
                1: {
                    "topic_list": {
                        "topics": [
                            {
                                "id": 301,
                                "title": "Allowed followup question",
                                "slug": "allowed-followup-question",
                                "highest_post_number": 1,
                                "last_poster_username": "dana",
                                "category_id": 20,
                            }
                        ]
                    }
                },
            }
            selected_url = "https://forum.example.com/t/allowed-followup-question/301/1"
            ollama = FakeOllamaClient(
                [ModelDecision("reply", "Use the supported path.", "Autonomous reply")],
                autonomous_selections=[
                    AutonomousSelection("reply", selected_url, 0.9, "Allowed category question")
                ],
            )
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(
                    str(database_path),
                    bot_autonomous_reply_enabled=True,
                    bot_autonomous_reply_latest_count=1,
                    bot_autonomous_reply_blocked_category_urls=(
                        "https://forum.example.com/c/restricted/10",
                    ),
                ),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            service.run_once()

            commands = storage.list_manual_commands()
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0].status, "completed")
            self.assertEqual(commands[0].post_url, selected_url)
            self.assertEqual(discourse.created_posts[0]["raw"], "Use the supported path.")
            autonomous_calls = [call for call in ollama.calls if "candidates" in call]
            self.assertEqual(len(autonomous_calls[0]["candidates"]), 1)
            self.assertEqual(autonomous_calls[0]["candidates"][0].topic_id, 301)
            self.assertEqual(
                [call["page"] for call in discourse.latest_topic_calls],
                [None, 1],
            )

    def test_inspect_notifications_uses_single_page_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(7, 1, 100, 2)])
            ollama = FakeOllamaClient([])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            snapshots = service.inspect_notifications()

            self.assertEqual(len(snapshots), 1)
            self.assertEqual(discourse.notification_paginate_calls[-1], False)

    def test_inspect_stats_reports_runtime_and_storage_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            storage.record_handled(
                1,
                action="reply",
                reason="done",
                handled_at="2026-01-01T00:00:00+00:00",
            )
            storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/1",
                user_request="reply",
                created_at="2026-01-01T00:00:00+00:00",
            )
            discourse = FakeDiscourseClient([])
            ollama = FakeOllamaClient([])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(database_path)),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                now_fn=clock.now,
            )

            stats = service.inspect_stats()

            self.assertEqual(stats["identity"]["username"], "bot")
            self.assertEqual(stats["runtime"]["model"], "qwen3")
            self.assertEqual(stats["storage"]["handled_total"], 1)
            self.assertEqual(stats["storage"]["manual_total"], 1)

    def test_recent_activity_tracks_last_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            discourse = FakeDiscourseClient([make_notification(8, 1, 100, 2)])
            ollama = FakeOllamaClient([ModelDecision("reply", "Thanks.", "Direct ask")])
            clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
            service = BotService(
                settings=self.make_settings(str(Path(temp_dir) / "bot.sqlite3")),
                discourse_client=discourse,
                ollama_client=ollama,
                storage=storage,
                presence_adapter=NullPresenceAdapter(),
                randomizer=random.Random(0),
                now_fn=clock.now,
            )

            service.run_once()
            recent = service.inspect_recent_activity(limit=5)

            self.assertGreaterEqual(len(recent), 3)
            self.assertIn("Bootstrapped bot identity", recent[0]["message"])
            self.assertTrue(any("Polled 1 notifications" in item["message"] for item in recent))
            self.assertTrue(any("Posted reply for notification 8" in item["message"] for item in recent))
