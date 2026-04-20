from __future__ import annotations

import threading
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from discourse_ai_bot.cli import (
    _InteractiveState,
    _TerminalUI,
    _handle_clear_command,
    _handle_config_command,
    _handle_slash_command,
    _runtime_config_snapshot,
    _build_private_chat_system_prompt,
    _handle_interactive_input_safe,
    _parse_send_command,
    _resolve_message_to_manual_command,
    _retry_after_seconds,
    _trim_private_chat_messages,
    build_parser,
)
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
