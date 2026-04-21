from __future__ import annotations

import threading
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from discourse_ai_bot.cli import (
    _InteractiveState,
    _TerminalUI,
    _autoread_worker_loop,
    _build_autoread_plan,
    _handle_clear_command,
    _handle_config_command,
    _handle_slash_command,
    _interactive_shell,
    _read_topic_via_api,
    _simulate_autoread_post,
    _stop_autoread_if_running,
    _runtime_config_snapshot,
    _build_private_chat_system_prompt,
    _handle_interactive_input_safe,
    _parse_send_command,
    _resolve_message_to_manual_command,
    _retry_after_seconds,
    _trim_private_chat_messages,
    build_parser,
    main,
)
from discourse_ai_bot.http import HttpError
from discourse_ai_bot.settings import Settings
from discourse_ai_bot.storage import BotStorage


class CliTests(unittest.TestCase):
    def test_parser_allows_running_without_explicit_command(self) -> None:
        parser = build_parser()

        args = parser.parse_args([])

        self.assertIsNone(args.command)

    def test_parse_send_command_supports_pipe_syntax(self) -> None:
        post_url, request = _parse_send_command(
            '/send https://forum.example.com/p/123 | Reply briefly and ask for logs.'
        )

        self.assertEqual(post_url, "https://forum.example.com/p/123")
        self.assertEqual(request, "Reply briefly and ask for logs.")

    def test_parse_send_command_supports_quoted_args(self) -> None:
        post_url, request = _parse_send_command(
            '/send "https://forum.example.com/p/123" "Reply briefly and ask for logs."'
        )

        self.assertEqual(post_url, "https://forum.example.com/p/123")
        self.assertEqual(request, "Reply briefly and ask for logs.")

    def test_resolve_message_treats_plain_text_as_request(self) -> None:
        post_url, request = _resolve_message_to_manual_command("Reply briefly and ask for logs.")

        self.assertIsNone(post_url)
        self.assertEqual(request, "Reply briefly and ask for logs.")

    def test_resolve_message_treats_url_as_post_url(self) -> None:
        post_url, request = _resolve_message_to_manual_command("https://forum.example.com/p/123")

        self.assertEqual(post_url, "https://forum.example.com/p/123")
        self.assertIsNone(request)

    def test_resolve_message_supports_one_line_shorthand(self) -> None:
        post_url, request = _resolve_message_to_manual_command(
            "https://forum.example.com/p/123 | Reply briefly and ask for logs."
        )

        self.assertEqual(post_url, "https://forum.example.com/p/123")
        self.assertEqual(request, "Reply briefly and ask for logs.")

    def test_retry_after_seconds_parses_discourse_rate_limit_message(self) -> None:
        retry_after = _retry_after_seconds(
            "Slow down, you're making too many requests.\nPlease retry again in 4 seconds.\n"
        )

        self.assertEqual(retry_after, 4.0)

    def test_interactive_input_safe_catches_exceptions(self) -> None:
        with patch("discourse_ai_bot.cli._handle_interactive_input", side_effect=RuntimeError("boom")):
            should_exit = _handle_interactive_input_safe(
                raw="/notifications",
                settings=None,  # type: ignore[arg-type]
                discourse=None,  # type: ignore[arg-type]
                ollama=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                service=None,  # type: ignore[arg-type]
                service_lock=threading.Lock(),
                state=_InteractiveState(),
            )

        self.assertFalse(should_exit)

    def test_interactive_shell_stops_cleanly_on_keyboard_interrupt(self) -> None:
        settings = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="session_cookie",
            discourse_token=None,
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie="session=abc",
        )

        class FakeService:
            def run_once(self) -> None:
                return None

        ui = _TerminalUI()
        with (
            patch("discourse_ai_bot.cli._collect_health", return_value={"discourse_host": "x", "discourse_username": "bot", "user_id": 1, "ollama_model": "qwen3", "typing_mode": "none"}),
            patch.object(ui, "prompt", side_effect=KeyboardInterrupt),
        ):
            result = _interactive_shell(
                settings,
                discourse=object(),  # type: ignore[arg-type]
                ollama=object(),  # type: ignore[arg-type]
                storage=object(),  # type: ignore[arg-type]
                service=FakeService(),  # type: ignore[arg-type]
            )

        self.assertEqual(result, 0)

    def test_main_returns_zero_on_keyboard_interrupt_from_run_command(self) -> None:
        settings = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="session_cookie",
            discourse_token=None,
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie="session=abc",
        )

        class FakeService:
            def run_forever(self) -> None:
                raise KeyboardInterrupt

        with (
            patch("discourse_ai_bot.cli.load_settings", return_value=settings),
            patch("discourse_ai_bot.cli.DiscourseClient", return_value=object()),
            patch("discourse_ai_bot.cli.OllamaClient", return_value=object()),
            patch("discourse_ai_bot.cli.BotStorage", return_value=object()),
            patch("discourse_ai_bot.cli._build_presence_adapter", return_value=object()),
            patch("discourse_ai_bot.cli.BotService", return_value=FakeService()),
        ):
            result = main(["run"])

        self.assertEqual(result, 0)

    def test_trim_private_chat_messages_keeps_recent_context_only(self) -> None:
        messages = [{"role": "user", "content": str(index)} for index in range(20)]

        trimmed = _trim_private_chat_messages(messages, max_messages=6)

        self.assertEqual([item["content"] for item in trimmed], ["14", "15", "16", "17", "18", "19"])

    def test_build_private_chat_system_prompt_marks_private_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            settings = Settings(
                discourse_host="https://forum.example.com",
                discourse_auth_mode="session_cookie",
                discourse_token=None,
                discourse_username="bot",
                ollama_host="http://localhost:11434",
                ollama_model="qwen3",
                discourse_cookie="session=abc",
            )

            prompt = _build_private_chat_system_prompt(
                settings,
                {"discourse_host": "https://forum.example.com", "ollama_model": "qwen3", "typing_mode": "none"},
                storage,
            )

        self.assertIn("private operator chat", prompt.lower())
        self.assertIn("must not be treated as forum content", prompt.lower())
        self.assertIn("Pending manual commands", prompt)

    def test_runtime_config_snapshot_reports_current_values(self) -> None:
        settings = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="session_cookie",
            discourse_token=None,
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie="session=abc",
            bot_poll_interval_seconds=9,
            bot_response_delay_min_seconds=1,
            bot_response_delay_max_seconds=3,
            bot_max_context_posts=5,
            bot_mark_read_on_skip=False,
        )

        snapshot = _runtime_config_snapshot(settings)

        self.assertEqual(snapshot["poll_interval_seconds"], 9)
        self.assertEqual(snapshot["response_delay_min_seconds"], 1)
        self.assertEqual(snapshot["response_delay_max_seconds"], 3)
        self.assertEqual(snapshot["autoread_post_time_seconds"], 120.0)
        self.assertEqual(snapshot["autoread_post_time_label"], "2m")
        self.assertEqual(snapshot["max_context_posts"], 5)
        self.assertFalse(snapshot["mark_read_on_skip"])

    def test_handle_config_command_updates_delay(self) -> None:
        settings = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="session_cookie",
            discourse_token=None,
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie="session=abc",
        )

        class FakeResolver:
            max_posts = 8

        class FakeService:
            context_resolver = FakeResolver()

        should_exit = _handle_config_command(
            tokens=["/config", "delay", "2", "4"],
            settings=settings,
            service=FakeService(),  # type: ignore[arg-type]
        )

        self.assertFalse(should_exit)
        self.assertEqual(settings.bot_response_delay_min_seconds, 2.0)
        self.assertEqual(settings.bot_response_delay_max_seconds, 4.0)

    def test_handle_config_command_updates_context_and_service_resolver(self) -> None:
        settings = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="session_cookie",
            discourse_token=None,
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie="session=abc",
        )

        class FakeResolver:
            max_posts = 8

        class FakeService:
            context_resolver = FakeResolver()

        service = FakeService()
        should_exit = _handle_config_command(
            tokens=["/config", "context", "12"],
            settings=settings,
            service=service,  # type: ignore[arg-type]
        )

        self.assertFalse(should_exit)
        self.assertEqual(settings.bot_max_context_posts, 12)
        self.assertEqual(service.context_resolver.max_posts, 12)

    def test_handle_config_command_updates_autoread_time(self) -> None:
        settings = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="session_cookie",
            discourse_token=None,
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie="session=abc",
        )

        class FakeResolver:
            max_posts = 8

        class FakeService:
            context_resolver = FakeResolver()

        should_exit = _handle_config_command(
            tokens=["/config", "autoread-time", "1h"],
            settings=settings,
            service=FakeService(),  # type: ignore[arg-type]
        )

        self.assertFalse(should_exit)
        self.assertEqual(settings.bot_autoread_post_time_seconds, 3600.0)

    def test_handle_clear_command_clears_queue(self) -> None:
        class FakeService:
            def clear_queue(self) -> dict[str, int]:
                return {
                    "manual_commands_deleted": 3,
                    "pending_replies_deleted": 2,
                }

        state = _InteractiveState()
        should_exit = _handle_clear_command(
            tokens=["/clear", "queue"],
            service=FakeService(),  # type: ignore[arg-type]
            service_lock=threading.Lock(),
            state=state,
        )

        self.assertFalse(should_exit)

    def test_handle_clear_command_resets_db_and_private_chat_messages(self) -> None:
        class FakeService:
            def reset_database(self) -> dict[str, int]:
                return {
                    "handled_notifications_deleted": 4,
                    "manual_commands_deleted": 3,
                    "pending_replies_deleted": 2,
                }

        state = _InteractiveState(private_chat_messages=[{"role": "user", "content": "hello"}])
        should_exit = _handle_clear_command(
            tokens=["/clear", "db"],
            service=FakeService(),  # type: ignore[arg-type]
            service_lock=threading.Lock(),
            state=state,
        )

        self.assertFalse(should_exit)
        self.assertEqual(state.private_chat_messages, [])

    def test_terminal_ui_uses_placeholder_text_for_prompt(self) -> None:
        ui = _TerminalUI()

        class FakeSession:
            def __init__(self) -> None:
                self.calls: list[tuple[object, object]] = []

            def prompt(self, message: object, **kwargs: object) -> str:
                self.calls.append((message, kwargs.get("placeholder")))
                return "typed"

        fake_session = FakeSession()
        ui.session = fake_session  # type: ignore[assignment]

        result = ui.prompt(mode="bot")

        self.assertEqual(result, "typed")
        self.assertEqual(fake_session.calls[0][0], "")
        self.assertEqual(fake_session.calls[0][1], [("class:prompt", ">>> Send command or message here")])

    def test_terminal_ui_uses_private_chat_placeholder(self) -> None:
        ui = _TerminalUI()

        class FakeSession:
            def __init__(self) -> None:
                self.calls: list[tuple[object, object]] = []

            def prompt(self, message: object, **kwargs: object) -> str:
                self.calls.append((message, kwargs.get("placeholder")))
                return "typed"

        fake_session = FakeSession()
        ui.session = fake_session  # type: ignore[assignment]

        result = ui.prompt(mode="chat")

        self.assertEqual(result, "typed")
        self.assertEqual(fake_session.calls[0][0], "")
        self.assertEqual(fake_session.calls[0][1], [("class:chatprompt", "Private chat here")])

    def test_summarize_command_calls_ollama_with_recent_activity(self) -> None:
        class FakeService:
            def inspect_stats(self) -> dict[str, object]:
                return {
                    "identity": {"username": "bot", "user_id": 1},
                    "runtime": {"model": "qwen3", "typing_mode": "none"},
                    "storage": {"handled_total": 1},
                }

            def inspect_recent_activity(self, *, limit: int = 10) -> list[dict[str, str]]:
                return [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "level": "info",
                        "message": "Did a thing.",
                    }
                ]

        class FakeOllama:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def summarize_activity(self, **kwargs: object) -> str:
                self.calls.append(kwargs)
                return "Summary sentence."

        settings = Settings(
            discourse_host="https://forum.example.com",
            discourse_auth_mode="session_cookie",
            discourse_token=None,
            discourse_username="bot",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3",
            discourse_cookie="session=abc",
        )
        ollama = FakeOllama()

        should_exit = _handle_slash_command(
            raw="/summarize",
            settings=settings,
            discourse=None,  # type: ignore[arg-type]
            ollama=ollama,  # type: ignore[arg-type]
            storage=None,  # type: ignore[arg-type]
            service=FakeService(),  # type: ignore[arg-type]
            service_lock=threading.Lock(),
            state=_InteractiveState(),
        )

        self.assertFalse(should_exit)
        self.assertEqual(ollama.calls[0]["model"], "qwen3")

    def test_build_autoread_plan_defaults_to_latest_topics(self) -> None:
        class FakeDiscourse:
            def list_categories(self) -> list[dict[str, object]]:
                return [{"id": 1, "slug": "support"}, {"id": 2, "slug": "bug-reports"}]

            def list_latest_topics(self, *, per_page: int = 5) -> dict[str, object]:
                return {
                    "topic_list": {
                        "topics": [
                            {"id": 100, "title": "One"},
                            {"id": 101, "title": "Two"},
                        ]
                    }
                }

            def list_category_topics(self, *, slug: str, category_id: int) -> dict[str, object]:
                return {
                    "topic_list": {
                        "topics": [
                            {"id": 200 + category_id, "title": f"{slug}-{category_id}"},
                        ]
                    }
                }

        plan = _build_autoread_plan(disourse=FakeDiscourse(), target=None)  # type: ignore[arg-type]

        self.assertEqual(plan["source"], "automatic_discovery")
        self.assertEqual(plan["categories_count"], 2)
        self.assertEqual(len(plan["topics"]), 4)

    def test_read_topic_via_api_fetches_missing_posts(self) -> None:
        class FakeDiscourse:
            host = "https://forum.example.com"

            def get_topic(self, topic_id: int) -> dict[str, object]:
                return {
                    "title": "Topic 100",
                    "slug": "topic-100",
                    "post_stream": {
                        "stream": [900, 901, 902],
                        "posts": [
                            {"id": 900, "post_number": 1, "username": "alice"},
                        ],
                    },
                }

            def get_topic_posts(self, topic_id: int, post_ids: list[int]) -> dict[str, object]:
                return {
                    "post_stream": {
                        "posts": [
                            {"id": 901, "post_number": 2, "username": "bob"},
                            {"id": 902, "post_number": 3, "username": "carol"},
                        ]
                    }
                }

            def record_topic_timings(self, **kwargs: object) -> dict[str, object]:
                return {"success": "OK"}

        with patch("discourse_ai_bot.cli.time.sleep"):
            result = _read_topic_via_api(disourse=FakeDiscourse(), topic_id=100)  # type: ignore[arg-type]

        self.assertEqual(result["posts_read"], 3)
        self.assertEqual(result["authors"], ["alice", "bob", "carol"])

    def test_read_topic_via_api_reports_per_post_progress(self) -> None:
        class FakeDiscourse:
            host = "https://forum.example.com"

            def get_topic(self, topic_id: int) -> dict[str, object]:
                return {
                    "title": "Topic 100",
                    "slug": "topic-100",
                    "post_stream": {
                        "stream": [900, 901],
                        "posts": [
                            {"id": 900, "post_number": 1, "username": "alice"},
                            {"id": 901, "post_number": 2, "username": "bob"},
                        ],
                    },
                }

            def record_topic_timings(self, **kwargs: object) -> dict[str, object]:
                return {"success": "OK"}

        progress: list[tuple[str, int, int]] = []
        with patch("discourse_ai_bot.cli.time.sleep"):
            _read_topic_via_api(
                disourse=FakeDiscourse(),  # type: ignore[arg-type]
                topic_id=100,
                progress_callback=lambda title, current, total: progress.append((title, current, total)),
            )

        self.assertEqual(
            progress,
            [("Topic 100", 1, 2), ("Topic 100", 2, 2)],
        )

    def test_read_topic_via_api_flushes_timings_batches(self) -> None:
        class FakeDiscourse:
            host = "https://forum.example.com"

            def __init__(self) -> None:
                self.timing_calls: list[dict[str, object]] = []

            def get_topic(self, topic_id: int) -> dict[str, object]:
                return {
                    "title": "Topic 100",
                    "slug": "topic-100",
                    "post_stream": {
                        "stream": [900, 901],
                        "posts": [
                            {"id": 900, "post_number": 1, "username": "alice"},
                            {"id": 901, "post_number": 2, "username": "bob"},
                        ],
                    },
                }

            def record_topic_timings(self, **kwargs: object) -> dict[str, object]:
                self.timing_calls.append(kwargs)
                return {"success": "OK"}

        discourse = FakeDiscourse()
        with patch("discourse_ai_bot.cli.time.sleep"):
            result = _read_topic_via_api(
                disourse=discourse,  # type: ignore[arg-type]
                topic_id=100,
            )

        self.assertEqual(result["posts_read"], 2)
        self.assertEqual(len(discourse.timing_calls), 2)
        self.assertEqual(discourse.timing_calls[0]["topic_id"], 100)
        self.assertEqual(discourse.timing_calls[0]["timings"], {1: 120000})
        self.assertEqual(discourse.timing_calls[0]["topic_time"], 120000)
        self.assertEqual(discourse.timing_calls[1]["timings"], {2: 120000})
        self.assertEqual(discourse.timing_calls[1]["topic_time"], 120000)

    def test_read_topic_via_api_falls_back_to_individual_post_fetches(self) -> None:
        class FakeDiscourse:
            host = "https://forum.example.com"

            def get_topic(self, topic_id: int) -> dict[str, object]:
                return {
                    "title": "Topic title",
                    "slug": "topic-title",
                    "post_stream": {
                        "stream": [901, 902],
                        "posts": [],
                    },
                }

            def get_topic_posts(self, topic_id: int, post_ids: list[int]) -> dict[str, object]:
                raise HttpError(
                    status_code=403,
                    url=f"https://forum.example.com/t/{topic_id}/posts.json",
                    body="forbidden",
                )

            def get_post(self, post_id: int) -> dict[str, object]:
                return {
                    "id": post_id,
                    "post_number": 1 if post_id == 901 else 2,
                    "username": "alice" if post_id == 901 else "bob",
                }

            def record_topic_timings(self, **kwargs: object) -> dict[str, object]:
                return {"success": "OK"}

        with patch("discourse_ai_bot.cli.time.sleep"):
            result = _read_topic_via_api(disourse=FakeDiscourse(), topic_id=100)  # type: ignore[arg-type]

        self.assertEqual(result["posts_read"], 2)
        self.assertEqual(result["authors"], ["alice", "bob"])

    def test_simulate_autoread_post_splits_long_duration_into_120_second_chunks(self) -> None:
        class FakeDiscourse:
            host = "https://forum.example.com"

            def __init__(self) -> None:
                self.timing_calls: list[dict[str, object]] = []

            def record_topic_timings(self, **kwargs: object) -> dict[str, object]:
                self.timing_calls.append(kwargs)
                return {"success": "OK"}

        discourse = FakeDiscourse()
        with patch("discourse_ai_bot.cli.time.sleep"):
            stopped = _simulate_autoread_post(
                disourse=discourse,  # type: ignore[arg-type]
                topic_id=100,
                topic_slug="topic-100",
                post_number=3,
                stop_event=None,
                post_time_seconds=360.0,
            )

        self.assertFalse(stopped)
        self.assertEqual(len(discourse.timing_calls), 3)
        self.assertEqual(
            [call["topic_time"] for call in discourse.timing_calls],
            [120000, 120000, 120000],
        )
        self.assertEqual(
            [call["timings"] for call in discourse.timing_calls],
            [{3: 120000}, {3: 120000}, {3: 120000}],
        )

    def test_stop_autoread_if_running_clears_state(self) -> None:
        stop_event = threading.Event()
        worker = threading.Thread(target=stop_event.wait, args=(0.1,), daemon=True)
        worker.start()
        state = _InteractiveState(
            autoread_stop_event=stop_event,
            autoread_thread=worker,
            autoread_target="https://forum.example.com/t/example/100",
        )

        _stop_autoread_if_running(state, reason="test interrupt")

        self.assertIsNone(state.autoread_stop_event)
        self.assertIsNone(state.autoread_thread)
        self.assertIsNone(state.autoread_target)

    def test_autoread_worker_loop_logs_and_survives_topic_errors(self) -> None:
        class FakeDiscourse:
            def list_categories(self) -> list[dict[str, object]]:
                return []

            def list_latest_topics(self, *, per_page: int = 5) -> dict[str, object]:
                return {"topic_list": {"topics": [{"id": 100, "title": "Broken topic"}]}}

            def get_topic(self, topic_id: int) -> dict[str, object]:
                raise RuntimeError("topic fetch failed")

        stop_event = threading.Event()
        waits = {"count": 0}

        def fake_wait(_event: threading.Event | None, _seconds: float) -> bool:
            waits["count"] += 1
            if waits["count"] >= 1:
                stop_event.set()
                return True
            return False

        ui = _TerminalUI()
        with (
            patch("discourse_ai_bot.cli._wait_for_autoread", side_effect=fake_wait),
            patch.object(ui, "print_muted") as print_muted,
        ):
            _autoread_worker_loop(FakeDiscourse(), None, stop_event, 120.0)  # type: ignore[arg-type]

        self.assertTrue(
            any("Starting Broken topic" in str(call.args[0]) for call in print_muted.call_args_list)
        )
