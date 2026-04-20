from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from discourse_ai_bot.utils import parse_duration_seconds


DEFAULT_ALLOWED_TRIGGERS = ("mentioned", "replied", "private_message")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful Discourse assistant. Reply only when a useful response is warranted. "
    "It is always acceptable to skip."
)


@dataclass(frozen=True)
class Settings:
    discourse_host: str
    discourse_auth_mode: str
    discourse_token: str | None
    discourse_username: str | None
    ollama_host: str
    ollama_model: str
    discourse_cookie: str | None = None
    discourse_user_agent: str | None = None
    discourse_extra_headers: dict[str, str] = field(default_factory=dict)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    bot_db_path: str = "discourse_ai_bot.sqlite3"
    bot_poll_interval_seconds: float = 15.0
    bot_response_delay_min_seconds: float = 10.0
    bot_response_delay_max_seconds: float = 45.0
    bot_autoread_post_time_seconds: float = 120.0
    bot_max_context_posts: int = 8
    bot_mark_read_on_skip: bool = True
    bot_allowed_triggers: tuple[str, ...] = DEFAULT_ALLOWED_TRIGGERS
    ollama_options: dict[str, Any] = field(default_factory=dict)
    ollama_keep_alive: str | None = "5m"
    ollama_timeout_seconds: float = 120.0
    bot_typing_mode: str = "none"
    discourse_presence_cookie: str | None = None
    discourse_presence_client_id: str | None = None
    discourse_presence_origin: str | None = None
    discourse_presence_user_agent: str | None = None
    discourse_presence_extra_headers: dict[str, str] = field(default_factory=dict)
    discourse_presence_reply_channel_template: str = "/discourse-presence/reply/{topic_id}"
    presence_heartbeat_interval_seconds: float = 5.0

    @property
    def typing_enabled(self) -> bool:
        return self.bot_typing_mode == "presence_update"

    @property
    def database_path(self) -> Path:
        return Path(self.bot_db_path)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    values = _load_dotenv_file(Path.cwd() / ".env")
    values.update(os.environ)
    if env is not None:
        values.update(env)

    prompt = _load_system_prompt(values)
    delay_min = _parse_float(values, "BOT_RESPONSE_DELAY_MIN_SECONDS", default=10.0)
    delay_max = _parse_float(values, "BOT_RESPONSE_DELAY_MAX_SECONDS", default=45.0)
    if delay_min < 0 or delay_max < 0 or delay_min > delay_max:
        raise ValueError("BOT_RESPONSE_DELAY_MIN_SECONDS must be <= BOT_RESPONSE_DELAY_MAX_SECONDS.")

    typing_mode = values.get("BOT_TYPING_MODE", "none").strip() or "none"
    if typing_mode not in {"none", "presence_update"}:
        raise ValueError("BOT_TYPING_MODE must be 'none' or 'presence_update'.")

    host = _require(values, "DISCOURSE_HOST")
    ollama_host = _require(values, "BOT_OLLAMA_HOST")
    auth_mode = _resolve_auth_mode(values)
    discourse_cookie = values.get("DISCOURSE_COOKIE_STRING") or values.get("DISCOURSE_COOKIE")
    discourse_user_agent = _optional(values, "DISCOURSE_USER_AGENT")
    presence_origin = values.get("DISCOURSE_PRESENCE_ORIGIN") or _origin_from_url(host)
    presence_client_id = values.get("DISCOURSE_PRESENCE_CLIENT_ID") or uuid.uuid4().hex
    presence_cookie = (
        values.get("DISCOURSE_PRESENCE_COOKIE_STRING")
        or values.get("DISCOURSE_PRESENCE_COOKIE")
        or discourse_cookie
    )

    if typing_mode == "presence_update" and not presence_cookie:
        raise ValueError(
            "DISCOURSE_PRESENCE_COOKIE or DISCOURSE_PRESENCE_COOKIE_STRING is required when BOT_TYPING_MODE=presence_update."
        )

    discourse_token = _optional(values, "DISCOURSE_TOKEN")
    discourse_username = _optional(values, "DISCOURSE_USERNAME")
    if auth_mode == "api_key":
        if not discourse_token:
            raise ValueError("DISCOURSE_TOKEN is required when DISCOURSE_AUTH_MODE=api_key.")
        if not discourse_username:
            raise ValueError("DISCOURSE_USERNAME is required when DISCOURSE_AUTH_MODE=api_key.")
    elif auth_mode == "session_cookie":
        if not discourse_cookie:
            raise ValueError(
                "DISCOURSE_COOKIE or DISCOURSE_COOKIE_STRING is required when DISCOURSE_AUTH_MODE=session_cookie."
            )
    else:
        raise ValueError("DISCOURSE_AUTH_MODE must be 'api_key' or 'session_cookie'.")

    options_raw = values.get("OLLAMA_OPTIONS_JSON", "{}").strip() or "{}"
    try:
        ollama_options = json.loads(options_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OLLAMA_OPTIONS_JSON must be valid JSON.") from exc
    if not isinstance(ollama_options, dict):
        raise ValueError("OLLAMA_OPTIONS_JSON must decode to a JSON object.")

    normalized_discourse_headers = _parse_string_header_json(
        values.get("DISCOURSE_EXTRA_HEADERS_JSON", "{}").strip() or "{}",
        env_name="DISCOURSE_EXTRA_HEADERS_JSON",
    )
    normalized_presence_headers = {
        **normalized_discourse_headers,
        **_parse_string_header_json(
            values.get("DISCOURSE_PRESENCE_EXTRA_HEADERS_JSON", "{}").strip() or "{}",
            env_name="DISCOURSE_PRESENCE_EXTRA_HEADERS_JSON",
        ),
    }
    presence_user_agent = _optional(values, "DISCOURSE_PRESENCE_USER_AGENT") or discourse_user_agent

    settings = Settings(
        discourse_host=host,
        discourse_auth_mode=auth_mode,
        discourse_token=discourse_token,
        discourse_username=discourse_username,
        discourse_cookie=discourse_cookie,
        discourse_user_agent=discourse_user_agent,
        discourse_extra_headers=normalized_discourse_headers,
        ollama_host=ollama_host,
        ollama_model=_require(values, "OLLAMA_MODEL"),
        system_prompt=prompt,
        bot_db_path=values.get("BOT_DB_PATH", "discourse_ai_bot.sqlite3"),
        bot_poll_interval_seconds=_parse_float(values, "BOT_POLL_INTERVAL_SECONDS", default=15.0),
        bot_response_delay_min_seconds=delay_min,
        bot_response_delay_max_seconds=delay_max,
        bot_autoread_post_time_seconds=_parse_duration_env(
            values,
            "BOT_AUTOREAD_POST_TIME",
            default=120.0,
        ),
        bot_max_context_posts=_parse_int(values, "BOT_MAX_CONTEXT_POSTS", default=8),
        bot_mark_read_on_skip=_parse_bool(values, "BOT_MARK_READ_ON_SKIP", default=True),
        bot_allowed_triggers=_parse_csv(
            values.get("BOT_ALLOWED_TRIGGERS"),
            default=DEFAULT_ALLOWED_TRIGGERS,
        ),
        ollama_options=ollama_options,
        ollama_keep_alive=_optional(values, "OLLAMA_KEEP_ALIVE", default="5m"),
        ollama_timeout_seconds=_parse_float(values, "OLLAMA_TIMEOUT_SECONDS", default=120.0),
        bot_typing_mode=typing_mode,
        discourse_presence_cookie=presence_cookie,
        discourse_presence_client_id=presence_client_id,
        discourse_presence_origin=presence_origin,
        discourse_presence_user_agent=presence_user_agent,
        discourse_presence_extra_headers=normalized_presence_headers,
        discourse_presence_reply_channel_template=values.get(
            "DISCOURSE_PRESENCE_REPLY_CHANNEL_TEMPLATE",
            "/discourse-presence/reply/{topic_id}",
        ),
    )

    if settings.bot_max_context_posts <= 0:
        raise ValueError("BOT_MAX_CONTEXT_POSTS must be greater than 0.")
    if settings.bot_poll_interval_seconds <= 0:
        raise ValueError("BOT_POLL_INTERVAL_SECONDS must be greater than 0.")
    if settings.bot_autoread_post_time_seconds <= 0:
        raise ValueError("BOT_AUTOREAD_POST_TIME must be greater than 0.")
    if settings.ollama_timeout_seconds <= 0:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than 0.")
    return settings


def _load_system_prompt(values: dict[str, str]) -> str:
    prompt_file = values.get("BOT_SYSTEM_PROMPT_FILE", "").strip()
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    prompt = values.get("BOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
    return prompt or DEFAULT_SYSTEM_PROMPT


def _require(values: dict[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def _optional(values: dict[str, str], name: str, default: str | None = None) -> str | None:
    value = values.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or None


def _parse_bool(values: dict[str, str], name: str, *, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean.")


def _parse_float(values: dict[str, str], name: str, *, default: float) -> float:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _parse_int(values: dict[str, str], name: str, *, default: int) -> int:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _parse_duration_env(values: dict[str, str], name: str, *, default: float) -> float:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    return parse_duration_seconds(raw, field_name=name)


def _parse_csv(raw: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        return default
    return values


def _parse_string_header_json(raw: str, *, env_name: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{env_name} must decode to a JSON object.")

    normalized: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{env_name} keys must be non-empty strings.")
        if not isinstance(value, str):
            raise ValueError(f"{env_name} values must be strings.")
        normalized[key] = value
    return normalized


def _load_dotenv_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _resolve_auth_mode(values: dict[str, str]) -> str:
    explicit = values.get("DISCOURSE_AUTH_MODE")
    if explicit and explicit.strip():
        return explicit.strip()
    if _optional(values, "DISCOURSE_COOKIE_STRING") or _optional(values, "DISCOURSE_COOKIE"):
        return "session_cookie"
    return "api_key"


def _origin_from_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise ValueError("DISCOURSE_HOST must be a valid URL.")
    return f"{parts.scheme}://{parts.netloc}"
