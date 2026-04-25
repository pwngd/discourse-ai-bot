from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from discourse_ai_bot.settings import load_settings


class SettingsTests(unittest.TestCase):
    def test_loads_dotenv_from_current_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / ".env").write_text(
                "\n".join(
                    [
                        "DISCOURSE_HOST=https://forum.example.com",
                        "DISCOURSE_AUTH_MODE=api_key",
                        "DISCOURSE_TOKEN=token",
                        "DISCOURSE_USERNAME=bot",
                        "BOT_OLLAMA_HOST=http://localhost:11434",
                        "OLLAMA_MODEL=qwen3",
                    ]
                ),
                encoding="utf-8",
            )
            with _pushd(temp_path):
                settings = load_settings(env={})
        self.assertEqual(settings.discourse_host, "https://forum.example.com")
        self.assertEqual(settings.ollama_model, "qwen3")

    def test_system_prompt_file_overrides_inline_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "prompt.txt"
            prompt_path.write_text("Prompt from file", encoding="utf-8")
            settings = load_settings(
                {
                    "DISCOURSE_HOST": "https://forum.example.com",
                    "DISCOURSE_AUTH_MODE": "api_key",
                    "DISCOURSE_TOKEN": "token",
                    "DISCOURSE_USERNAME": "bot",
                    "BOT_OLLAMA_HOST": "http://localhost:11434",
                    "OLLAMA_MODEL": "qwen3",
                    "BOT_SYSTEM_PROMPT": "Inline prompt",
                    "BOT_SYSTEM_PROMPT_FILE": str(prompt_path),
                }
            )
        self.assertEqual(settings.system_prompt, "Prompt from file")

    def test_presence_mode_requires_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with _pushd(Path(temp_dir)):
                with self.assertRaises(ValueError):
                    load_settings(
                        {
                            "DISCOURSE_HOST": "https://forum.example.com",
                            "DISCOURSE_AUTH_MODE": "session_cookie",
                            "BOT_OLLAMA_HOST": "http://localhost:11434",
                            "OLLAMA_MODEL": "qwen3",
                            "BOT_TYPING_MODE": "presence_update",
                        }
                    )

    def test_presence_cookie_alias_and_headers_are_parsed(self) -> None:
        settings = load_settings(
            {
                "DISCOURSE_HOST": "https://forum.example.com",
                "DISCOURSE_AUTH_MODE": "session_cookie",
                "BOT_OLLAMA_HOST": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen3",
                "BOT_TYPING_MODE": "presence_update",
                "DISCOURSE_COOKIE_STRING": "session=abc; other=xyz",
                "DISCOURSE_USER_AGENT": "Mozilla/5.0 Test",
                "DISCOURSE_EXTRA_HEADERS_JSON": '{"Accept-Language":"en-US,en;q=0.9"}',
            }
        )
        self.assertEqual(settings.discourse_cookie, "session=abc; other=xyz")
        self.assertEqual(settings.discourse_user_agent, "Mozilla/5.0 Test")
        self.assertEqual(settings.discourse_auth_mode, "session_cookie")
        self.assertEqual(
            settings.discourse_extra_headers,
            {"Accept-Language": "en-US,en;q=0.9"},
        )
        self.assertEqual(settings.discourse_presence_cookie, "session=abc; other=xyz")
        self.assertEqual(settings.discourse_presence_user_agent, "Mozilla/5.0 Test")
        self.assertEqual(
            settings.discourse_presence_extra_headers,
            {"Accept-Language": "en-US,en;q=0.9"},
        )

    def test_csv_and_json_options_are_parsed(self) -> None:
        settings = load_settings(
            {
                "DISCOURSE_HOST": "https://forum.example.com",
                "DISCOURSE_AUTH_MODE": "api_key",
                "DISCOURSE_TOKEN": "token",
                "DISCOURSE_USERNAME": "bot",
                "BOT_OLLAMA_HOST": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen3",
                "BOT_ALLOWED_TRIGGERS": "mentioned, private_message",
                "OLLAMA_OPTIONS_JSON": '{"temperature": 0.2}',
            }
        )
        self.assertEqual(settings.bot_allowed_triggers, ("mentioned", "private_message"))
        self.assertEqual(settings.ollama_options, {"temperature": 0.2})

    def test_autoread_post_time_supports_human_duration(self) -> None:
        settings = load_settings(
            {
                "DISCOURSE_HOST": "https://forum.example.com",
                "DISCOURSE_AUTH_MODE": "api_key",
                "DISCOURSE_TOKEN": "token",
                "DISCOURSE_USERNAME": "bot",
                "BOT_OLLAMA_HOST": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen3",
                "BOT_AUTOREAD_POST_TIME": "1m",
            }
        )
        self.assertEqual(settings.bot_autoread_post_time_seconds, 60.0)

    def test_autonomous_reply_settings_are_parsed(self) -> None:
        settings = load_settings(
            {
                "DISCOURSE_HOST": "https://forum.example.com",
                "DISCOURSE_AUTH_MODE": "api_key",
                "DISCOURSE_TOKEN": "token",
                "DISCOURSE_USERNAME": "bot",
                "BOT_OLLAMA_HOST": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen3",
                "BOT_AUTONOMOUS_REPLY_ENABLED": "true",
                "BOT_AUTONOMOUS_REPLY_INTERVAL": "2m",
                "BOT_AUTONOMOUS_REPLY_LATEST_COUNT": "7",
                "BOT_AUTONOMOUS_REPLY_MIN_CONFIDENCE": "0.8",
                "BOT_AUTONOMOUS_REPLY_BLOCKED_CATEGORY_URLS": "https://forum.example.com/c/staff/4, https://forum.example.com/c/private/5",
            }
        )
        self.assertTrue(settings.bot_autonomous_reply_enabled)
        self.assertEqual(settings.bot_autonomous_reply_interval_seconds, 120.0)
        self.assertEqual(settings.bot_autonomous_reply_latest_count, 7)
        self.assertEqual(settings.bot_autonomous_reply_min_confidence, 0.8)
        self.assertEqual(
            settings.bot_autonomous_reply_blocked_category_urls,
            ("https://forum.example.com/c/staff/4", "https://forum.example.com/c/private/5"),
        )

    def test_roblox_docs_settings_are_parsed(self) -> None:
        settings = load_settings(
            {
                "DISCOURSE_HOST": "https://forum.example.com",
                "DISCOURSE_AUTH_MODE": "api_key",
                "DISCOURSE_TOKEN": "token",
                "DISCOURSE_USERNAME": "bot",
                "BOT_OLLAMA_HOST": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen3",
                "BOT_ROBLOX_DOCS_ENABLED": "true",
                "BOT_ROBLOX_DOCS_SOURCE": "local",
                "BOT_ROBLOX_DOCS_LOCAL_PATH": "vendor/creator-docs",
                "BOT_ROBLOX_DOCS_REF": "main",
                "BOT_ROBLOX_DOCS_TIMEOUT_SECONDS": "4",
                "BOT_ROBLOX_DOCS_CACHE_TTL": "12h",
                "BOT_ROBLOX_DOCS_MAX_TERMS": "3",
                "BOT_ROBLOX_DOCS_MAX_RESULTS": "2",
                "BOT_ROBLOX_DOCS_MAX_CONTEXT_CHARS": "1500",
            }
        )
        self.assertTrue(settings.bot_roblox_docs_enabled)
        self.assertEqual(settings.bot_roblox_docs_source, "local")
        self.assertEqual(settings.bot_roblox_docs_local_path, "vendor/creator-docs")
        self.assertEqual(settings.bot_roblox_docs_ref, "main")
        self.assertEqual(settings.bot_roblox_docs_timeout_seconds, 4.0)
        self.assertEqual(settings.bot_roblox_docs_cache_ttl_seconds, 43200.0)
        self.assertEqual(settings.bot_roblox_docs_max_terms, 3)
        self.assertEqual(settings.bot_roblox_docs_max_results, 2)
        self.assertEqual(settings.bot_roblox_docs_max_context_chars, 1500)

    def test_session_cookie_mode_is_inferred_from_cookie_string(self) -> None:
        settings = load_settings(
            {
                "DISCOURSE_HOST": "https://forum.example.com",
                "DISCOURSE_COOKIE_STRING": "session=abc; other=xyz",
                "BOT_OLLAMA_HOST": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen3",
            }
        )
        self.assertEqual(settings.discourse_auth_mode, "session_cookie")


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
