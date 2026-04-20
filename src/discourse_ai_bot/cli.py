from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import logging
import re
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from discourse_ai_bot.discourse import DiscourseClient
from discourse_ai_bot.http import HttpError
from discourse_ai_bot.ollama import OllamaClient
from discourse_ai_bot.presence import DiscoursePresenceAdapter, NullPresenceAdapter
from discourse_ai_bot.service import BotService
from discourse_ai_bot.settings import Settings, load_settings
from discourse_ai_bot.storage import BotStorage
from discourse_ai_bot.utils import parse_duration_seconds

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, WordCompleter
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
    from prompt_toolkit.styles import Style as PromptStyle
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    PromptSession = None
    AutoSuggestFromHistory = None
    Completer = object  # type: ignore[assignment]
    Completion = None
    WordCompleter = None
    InMemoryHistory = None
    NoConsoleScreenBufferError = RuntimeError
    PromptStyle = None

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    Console = None
    RichHandler = None
    Markdown = None
    Panel = None
    Table = None


PROMPT_TEXT = ">>> Send command or message here "
AUTOREAD_DEFAULT_TOPIC_LIMIT = 5
AUTOREAD_POST_CHUNK_SIZE = 50
AUTOREAD_CYCLE_DELAY_SECONDS = 0.35
AUTOREAD_MAX_TIMING_SECONDS = 120.0
AUTOREAD_MAX_PARALLEL_TOPICS = 4
SLASH_COMMANDS = (
    "/help",
    "/health",
    "/stats",
    "/summarize",
    "/autoread",
    "/config",
    "/notifications",
    "/manual",
    "/clear",
    "/chat",
    "/chat-reset",
    "/bot",
    "/send",
    "/quit",
    "/exit",
)

PRIVATE_CHAT_POLICY = """
You are in a private operator chat inside the Discourse AI bot CLI.

Rules:
- This conversation is private to the operator and must not be treated as forum content.
- Do not treat these operator messages as notifications, posts, replies, or future automatic bot context.
- You may help with diagnostics, drafting, planning, and reasoning about the bot's current work.
- If the operator asks for a forum reply draft, make it clear it is a private draft unless they queue it separately.
- Keep responses useful, direct, and grounded in the current bot state summary.
""".strip()


@dataclass
class _InteractiveState:
    health_snapshot: dict[str, Any] | None = None
    private_chat_active: bool = False
    private_chat_messages: list[dict[str, str]] = field(default_factory=list)
    autoread_stop_event: threading.Event | None = None
    autoread_thread: threading.Thread | None = None
    autoread_target: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discourse AI bot")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")

    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser("run", help="Run the long-lived notification worker")
    subparsers.add_parser("healthcheck", help="Check Discourse and Ollama connectivity")
    subparsers.add_parser("list-notifications", help="List current notifications and bot state")
    subparsers.add_parser("list-manual-commands", help="List queued and completed manual AI reply commands")

    queue_parser = subparsers.add_parser(
        "queue-ai-reply",
        help="Queue an AI-generated reply to a specific Discourse post URL",
    )
    queue_parser.add_argument("--post-url", required=True, help="Discourse post or topic URL")
    queue_parser.add_argument(
        "--request",
        required=True,
        help="Instruction to send to Ollama for the reply",
    )

    reply_parser = subparsers.add_parser("reply", help="Post a manual reply for testing")
    reply_parser.add_argument("--topic-id", type=int, help="Topic ID to reply to")
    reply_parser.add_argument("--post-id", type=int, help="Fetch a post to infer topic/post number")
    reply_parser.add_argument(
        "--reply-to-post-number",
        type=int,
        help="Reply target post number inside the topic",
    )
    reply_parser.add_argument("--raw", required=True, help="Markdown body to post")

    topic_parser = subparsers.add_parser("post-topic", help="Create a test topic")
    topic_parser.add_argument("--title", required=True, help="Topic title")
    topic_parser.add_argument("--raw", required=True, help="First post body")
    topic_parser.add_argument("--category", type=int, help="Optional category ID")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))

    settings = load_settings()
    discourse = DiscourseClient(
        settings.discourse_host,
        auth_mode=settings.discourse_auth_mode,
        token=settings.discourse_token,
        username=settings.discourse_username,
        cookie=settings.discourse_cookie,
        user_agent=settings.discourse_user_agent,
        extra_headers=settings.discourse_extra_headers,
    )
    ollama = OllamaClient(
        settings.ollama_host,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    storage = BotStorage(settings.database_path)
    presence = _build_presence_adapter(settings)
    service = BotService(
        settings=settings,
        discourse_client=discourse,
        ollama_client=ollama,
        storage=storage,
        presence_adapter=presence,
    )

    try:
        if not args.command:
            return _interactive_shell(settings, discourse, ollama, storage, service)

        if args.command == "run":
            service.run_forever()
            return 0
        if args.command == "healthcheck":
            return _healthcheck(settings, discourse, ollama)
        if args.command == "list-notifications":
            service.bootstrap()
            print(json.dumps(service.inspect_notifications(), indent=2))
            return 0
        if args.command == "list-manual-commands":
            service.bootstrap()
            print(json.dumps(service.inspect_manual_commands(), indent=2))
            return 0
        if args.command == "queue-ai-reply":
            command_id = storage.enqueue_manual_command(
                post_url=args.post_url,
                user_request=args.request,
                created_at=_now_storage(),
            )
            print(json.dumps({"command_id": command_id, "status": "queued"}, indent=2))
            return 0
        if args.command == "reply":
            return _manual_reply(args, discourse)
        if args.command == "post-topic":
            response = discourse.create_topic(
                title=args.title,
                raw=args.raw,
                category=args.category,
            )
            print(json.dumps(response, indent=2))
            return 0
        parser.error(f"Unknown command: {args.command}")
        return 2
    except KeyboardInterrupt:
        ui = _TerminalUI()
        ui.print_blank()
        ui.print_status("Bot stopped")
        return 0


def _healthcheck(settings: Settings, discourse: DiscourseClient, ollama: OllamaClient) -> int:
    result = _collect_health(settings, discourse, ollama)
    print(json.dumps(result, indent=2))
    return 0


def _collect_health(
    settings: Settings,
    discourse: DiscourseClient,
    ollama: OllamaClient,
) -> dict[str, Any]:
    site = discourse.get_site_info()
    if settings.discourse_auth_mode == "session_cookie":
        session = discourse.get_current_session()
        current_user = session.get("current_user") if isinstance(session, dict) else None
        username = current_user.get("username") if isinstance(current_user, dict) else None
        user_id = current_user.get("id") if isinstance(current_user, dict) else None
    else:
        user = discourse.get_user(settings.discourse_username or "")
        current_user = user.get("user") if isinstance(user, dict) else None
        username = settings.discourse_username
        user_id = current_user.get("id") if isinstance(current_user, dict) else None
    model_info = ollama.healthcheck(settings.ollama_model)
    result = {
        "discourse_host": settings.discourse_host,
        "discourse_auth_mode": settings.discourse_auth_mode,
        "discourse_username": username,
        "site_categories": len(site.get("categories", [])) if isinstance(site, dict) else None,
        "user_id": user_id,
        "ollama_model": model_info["model"],
        "typing_mode": settings.bot_typing_mode,
    }
    return result


def _manual_reply(args: argparse.Namespace, discourse: DiscourseClient) -> int:
    topic_id = args.topic_id
    reply_to_post_number = args.reply_to_post_number
    if args.post_id is not None:
        post = discourse.get_post(args.post_id)
        topic_id = topic_id or int(post["topic_id"])
        reply_to_post_number = reply_to_post_number or int(post["post_number"])
    if topic_id is None:
        raise SystemExit("Either --topic-id or --post-id is required.")

    response = discourse.create_post(
        raw=args.raw,
        topic_id=topic_id,
        reply_to_post_number=reply_to_post_number,
    )
    print(json.dumps(response, indent=2))
    return 0


def _build_presence_adapter(settings: Settings) -> Any:
    if not settings.typing_enabled:
        return NullPresenceAdapter()
    return DiscoursePresenceAdapter(
        discourse_host=settings.discourse_host,
        cookie=settings.discourse_presence_cookie or "",
        client_id=settings.discourse_presence_client_id or "",
        origin=settings.discourse_presence_origin or settings.discourse_host,
        user_agent=settings.discourse_presence_user_agent,
        extra_headers=settings.discourse_presence_extra_headers,
    )


def _now_storage() -> str:
    from discourse_ai_bot.utils import datetime_to_storage, utc_now

    return datetime_to_storage(utc_now())


def _interactive_shell(
    settings: Settings,
    discourse: DiscourseClient,
    ollama: OllamaClient,
    storage: BotStorage,
    service: BotService,
) -> int:
    logger = logging.getLogger(__name__)
    ui = _TerminalUI()
    state = _InteractiveState()
    ui.print_banner("Bot initiated")
    try:
        health = _collect_health(settings, discourse, ollama)
        state.health_snapshot = health
        ui.print_health(health)
    except Exception as exc:
        logger.exception("Health check failed during interactive startup: %s", exc)
        ui.print_muted("Health: unavailable, see logs above.")
    ui.print_status("Polling for notifications")
    ui.print_muted("Use /help for commands.")

    stop_event = threading.Event()
    service_lock = threading.Lock()
    worker = threading.Thread(
        target=_polling_worker,
        args=(service, service_lock, settings, stop_event),
        name="discourse-ai-bot-worker",
        daemon=True,
    )
    worker.start()

    try:
        while True:
            try:
                raw = ui.prompt(mode="chat" if state.private_chat_active else "bot").strip()
            except EOFError:
                ui.print_blank()
                break
            except KeyboardInterrupt:
                ui.print_blank()
                break

            if not raw:
                continue

            _stop_autoread_if_running(state, reason="interrupted by operator input")

            should_exit = _handle_interactive_input_safe(
                raw=raw,
                settings=settings,
                discourse=discourse,
                ollama=ollama,
                storage=storage,
                service=service,
                service_lock=service_lock,
                state=state,
            )
            if should_exit:
                break
    finally:
        stop_event.set()
        _stop_autoread_if_running(state)
        worker.join(timeout=2.0)
        ui.print_status("Bot stopped")

    return 0


def _polling_worker(
    service: BotService,
    service_lock: threading.Lock,
    settings: Settings,
    stop_event: threading.Event,
) -> None:
    logger = logging.getLogger(__name__)
    while not stop_event.is_set():
        sleep_seconds = settings.bot_poll_interval_seconds
        try:
            with service_lock:
                service.run_once()
        except HttpError as exc:
            sleep_seconds = max(settings.bot_poll_interval_seconds, _retry_after_seconds(exc.body))
            logger.warning("Polling backed off for %.0f seconds after HTTP %s.", sleep_seconds, exc.status_code)
        except Exception as exc:
            logger.exception("Polling loop failed: %s", exc)
        stop_event.wait(sleep_seconds)


def _handle_interactive_input(
    *,
    raw: str,
    settings: Settings,
    discourse: DiscourseClient,
    ollama: OllamaClient,
    storage: BotStorage,
    service: BotService,
    service_lock: threading.Lock,
    state: _InteractiveState,
) -> bool:
    if raw.startswith("/"):
        return _handle_slash_command(
            raw=raw,
            settings=settings,
            discourse=discourse,
            ollama=ollama,
            storage=storage,
            service=service,
            service_lock=service_lock,
            state=state,
        )

    if state.private_chat_active:
        return _handle_private_chat_message(
            raw=raw,
            settings=settings,
            ollama=ollama,
            storage=storage,
            state=state,
        )

    post_url, request = _resolve_message_to_manual_command(raw)
    if post_url is None:
        post_url = _TerminalUI().prompt_followup("Post URL").strip()
    if request is None:
        request = _TerminalUI().prompt_followup("Message").strip()

    if not post_url or not request:
        _TerminalUI().print_muted("Manual reply queueing cancelled.")
        return False

    command_id = storage.enqueue_manual_command(
        post_url=post_url,
        user_request=request,
        created_at=_now_storage(),
    )
    _TerminalUI().print_success(f"Queued manual AI reply #{command_id} for {post_url}")
    return False


def _handle_interactive_input_safe(
    *,
    raw: str,
    settings: Settings,
    discourse: DiscourseClient,
    ollama: OllamaClient,
    storage: BotStorage,
    service: BotService,
    service_lock: threading.Lock,
    state: _InteractiveState,
) -> bool:
    logger = logging.getLogger(__name__)
    try:
        return _handle_interactive_input(
            raw=raw,
            settings=settings,
            discourse=discourse,
            ollama=ollama,
            storage=storage,
            service=service,
            service_lock=service_lock,
            state=state,
        )
    except Exception as exc:
        logger.exception("Interactive command failed: %s", exc)
        _TerminalUI().print_error("Command failed, see logs above.")
        return False


def _handle_slash_command(
    *,
    raw: str,
    settings: Settings,
    discourse: DiscourseClient,
    ollama: OllamaClient,
    storage: BotStorage,
    service: BotService,
    service_lock: threading.Lock,
    state: _InteractiveState,
) -> bool:
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        _TerminalUI().print_error(f"Invalid command syntax: {exc}")
        return False

    command = tokens[0].lower()
    if command in {"/quit", "/exit"}:
        return True
    if command == "/help":
        _TerminalUI().print_help()
        return False
    if command == "/health":
        _TerminalUI().print_json(_collect_health(settings, discourse, ollama))
        return False
    if command == "/stats":
        with service_lock:
            _TerminalUI().print_stats(service.inspect_stats())
        return False
    if command == "/summarize":
        with service_lock:
            runtime_snapshot = service.inspect_stats()
            recent_activity = service.inspect_recent_activity(limit=10)
        if not recent_activity:
            _TerminalUI().print_muted("No recent activity to summarize yet.")
            return False
        summary = ollama.summarize_activity(
            model=settings.ollama_model,
            runtime_snapshot=runtime_snapshot,
            activity_events=recent_activity,
            options=settings.ollama_options,
            keep_alive=settings.ollama_keep_alive,
        )
        _TerminalUI().print_summary(summary)
        return False
    if command == "/autoread":
        target = tokens[1] if len(tokens) > 1 else None
        if len(tokens) > 2:
            _TerminalUI().print_error("Usage: /autoread or /autoread <topic_or_category_url>")
            return False
        _start_autoread(state=state, settings=settings, discourse=discourse, target=target)
        return False
    if command == "/config":
        return _handle_config_command(tokens=tokens, settings=settings, service=service)
    if command == "/notifications":
        with service_lock:
            _TerminalUI().print_json(service.inspect_notifications(paginate=False))
        return False
    if command == "/manual":
        _TerminalUI().print_json(service.inspect_manual_commands())
        return False
    if command == "/clear":
        return _handle_clear_command(
            tokens=tokens,
            service=service,
            service_lock=service_lock,
            state=state,
        )
    if command == "/chat":
        if not state.private_chat_messages:
            state.private_chat_messages = []
        state.private_chat_active = True
        _TerminalUI().print_status("Private operator chat enabled")
        _TerminalUI().print_muted("Messages here stay in private CLI chat only. Use /bot to return.")
        return False
    if command == "/chat-reset":
        state.private_chat_messages.clear()
        state.private_chat_active = True
        _TerminalUI().print_status("Private operator chat context reset")
        return False
    if command == "/bot":
        state.private_chat_active = False
        _TerminalUI().print_status("Returned to normal bot mode")
        return False
    if command == "/send":
        post_url, request = _parse_send_command(raw)
        if post_url is None or request is None:
            _TerminalUI().print_error('Usage: /send "<post_url>" "<request>"')
            _TerminalUI().print_muted('Or: /send <post_url> | <request>')
            return False
        command_id = storage.enqueue_manual_command(
            post_url=post_url,
            user_request=request,
            created_at=_now_storage(),
        )
        _TerminalUI().print_success(f"Queued manual AI reply #{command_id} for {post_url}")
        return False

    _TerminalUI().print_error(f"Unknown command: {command}. Use /help.")
    return False


def _parse_send_command(raw: str) -> tuple[str | None, str | None]:
    stripped = raw.strip()
    if "|" in stripped:
        left, right = stripped.split("|", 1)
        left_tokens = shlex.split(left)
        if len(left_tokens) >= 2 and right.strip():
            return left_tokens[1].strip(), right.strip()
        return None, None

    tokens = shlex.split(stripped)
    if len(tokens) < 3:
        return None, None
    return tokens[1].strip(), " ".join(tokens[2:]).strip()


def _resolve_message_to_manual_command(raw: str) -> tuple[str | None, str | None]:
    stripped = raw.strip()
    if "|" in stripped:
        left, right = stripped.split("|", 1)
        left = left.strip()
        right = right.strip()
        if _looks_like_url(left) and right:
            return left, right
        if _looks_like_url(right) and left:
            return right, left

    if _looks_like_url(stripped):
        return stripped, None
    return None, stripped


def _looks_like_url(value: str) -> bool:
    parts = urlsplit(value.strip())
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def _retry_after_seconds(body: str) -> float:
    match = re.search(r"retry again in (\d+) seconds", body, flags=re.IGNORECASE)
    if not match:
        return 0.0
    return float(match.group(1))


def _handle_config_command(
    *,
    tokens: list[str],
    settings: Settings,
    service: BotService,
) -> bool:
    ui = _TerminalUI()
    if len(tokens) == 1 or (len(tokens) == 2 and tokens[1].lower() == "show"):
        ui.print_json(_runtime_config_snapshot(settings))
        return False

    subcommand = tokens[1].lower()
    if subcommand == "delay":
        if len(tokens) != 4:
            ui.print_error("Usage: /config delay <min_seconds> <max_seconds>")
            return False
        delay_min = float(tokens[2])
        delay_max = float(tokens[3])
        if delay_min < 0 or delay_max < 0 or delay_min > delay_max:
            raise ValueError("Delay values must be non-negative and min must be <= max.")
        object.__setattr__(settings, "bot_response_delay_min_seconds", delay_min)
        object.__setattr__(settings, "bot_response_delay_max_seconds", delay_max)
        ui.print_success(f"Updated reply delay to {delay_min:.2f}s - {delay_max:.2f}s")
        return False

    if subcommand == "poll":
        if len(tokens) != 3:
            ui.print_error("Usage: /config poll <seconds>")
            return False
        poll_seconds = float(tokens[2])
        if poll_seconds <= 0:
            raise ValueError("Poll interval must be greater than 0.")
        object.__setattr__(settings, "bot_poll_interval_seconds", poll_seconds)
        ui.print_success(f"Updated poll interval to {poll_seconds:.2f}s")
        return False

    if subcommand == "autoread-time":
        if len(tokens) != 3:
            ui.print_error("Usage: /config autoread-time <duration>")
            ui.print_muted("Examples: /config autoread-time 30s, /config autoread-time 1m, /config autoread-time 1h")
            return False
        post_time_seconds = parse_duration_seconds(tokens[2], field_name="autoread-time")
        if post_time_seconds <= 0:
            raise ValueError("AutoRead post time must be greater than 0.")
        object.__setattr__(settings, "bot_autoread_post_time_seconds", post_time_seconds)
        ui.print_success(f"Updated AutoRead per-post timing to {_format_duration(post_time_seconds)}")
        return False

    if subcommand == "context":
        if len(tokens) != 3:
            ui.print_error("Usage: /config context <post_count>")
            return False
        max_posts = int(tokens[2])
        if max_posts <= 0:
            raise ValueError("Context post count must be greater than 0.")
        object.__setattr__(settings, "bot_max_context_posts", max_posts)
        service.context_resolver.max_posts = max_posts
        ui.print_success(f"Updated context window to {max_posts} posts")
        return False

    if subcommand == "mark-read":
        if len(tokens) != 3:
            ui.print_error("Usage: /config mark-read <on|off>")
            return False
        value = tokens[2].lower()
        if value not in {"on", "off"}:
            raise ValueError("mark-read must be 'on' or 'off'.")
        enabled = value == "on"
        object.__setattr__(settings, "bot_mark_read_on_skip", enabled)
        ui.print_success(f"Updated mark-read-on-skip to {'on' if enabled else 'off'}")
        return False

    ui.print_error("Unknown /config subcommand. Use /config show.")
    return False


def _handle_clear_command(
    *,
    tokens: list[str],
    service: BotService,
    service_lock: threading.Lock,
    state: _InteractiveState,
) -> bool:
    ui = _TerminalUI()
    if len(tokens) != 2:
        ui.print_error("Usage: /clear <queue|db>")
        return False

    target = tokens[1].lower()
    with service_lock:
        if target == "queue":
            result = service.clear_queue()
            ui.print_success(
                "Cleared outbound queue "
                f"(manual_commands={result['manual_commands_deleted']}, "
                f"pending_replies={result['pending_replies_deleted']})"
            )
            return False
        if target == "db":
            result = service.reset_database()
            state.private_chat_messages.clear()
            ui.print_success(
                "Reset local bot database "
                f"(handled_notifications={result['handled_notifications_deleted']}, "
                f"manual_commands={result['manual_commands_deleted']}, "
                f"pending_replies={result['pending_replies_deleted']})"
            )
            return False

    ui.print_error("Unknown /clear target. Use /clear queue or /clear db.")
    return False


def _start_autoread(
    *,
    state: _InteractiveState,
    settings: Settings,
    discourse: DiscourseClient,
    target: str | None,
) -> None:
    ui = _TerminalUI()
    stop_event = threading.Event()
    state.autoread_stop_event = stop_event
    state.autoread_target = target
    thread = threading.Thread(
        target=_autoread_worker_loop,
        args=(discourse, target, stop_event, settings.bot_autoread_post_time_seconds),
        name="discourse-ai-bot-autoread",
        daemon=True,
    )
    state.autoread_thread = thread
    thread.start()
    label = target or "automatic discovery"
    ui.print_status(f"AutoRead started for {label}. Submit any command or message to interrupt it.")


def _stop_autoread_if_running(state: _InteractiveState, *, reason: str | None = None) -> None:
    thread = state.autoread_thread
    stop_event = state.autoread_stop_event
    if thread is None or stop_event is None:
        return
    stop_event.set()
    if thread.is_alive():
        thread.join(timeout=2.0)
    if reason:
        _TerminalUI().print_muted(f"AutoRead stopped: {reason}.")
    state.autoread_thread = None
    state.autoread_stop_event = None
    state.autoread_target = None


def _autoread_worker_loop(
    disourse: DiscourseClient,
    target: str | None,
    stop_event: threading.Event,
    post_time_seconds: float,
) -> None:
    ui = _TerminalUI()
    logger = logging.getLogger(__name__)
    cycle_number = 0
    while not stop_event.is_set():
        try:
            cycle_number += 1
            plan = _build_autoread_plan(disourse=disourse, target=target)
            results: list[dict[str, Any]] = []
            topic_items = list(plan["topics"])
            max_workers = max(1, min(AUTOREAD_MAX_PARALLEL_TOPICS, len(topic_items)))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="discourse-ai-bot-autoread-topic") as executor:
                future_to_topic: dict[object, tuple[int, str]] = {}
                for item in topic_items:
                    if stop_event.is_set():
                        break
                    topic_id = int(item["topic_id"])
                    title = str(item.get("title") or f"Topic {topic_id}")
                    posts_count = item.get("posts_count")
                    if isinstance(posts_count, int) and posts_count > 0:
                        ui.print_muted(
                            f"AutoRead cycle {cycle_number}: Starting {title} ({posts_count} planned posts)"
                        )
                    else:
                        ui.print_muted(
                            f"AutoRead cycle {cycle_number}: Starting {title}"
                        )
                    future = executor.submit(
                        _read_topic_via_api,
                        disourse=disourse,
                        topic_id=topic_id,
                        stop_event=stop_event,
                        post_time_seconds=post_time_seconds,
                    )
                    future_to_topic[future] = (topic_id, title)
                while future_to_topic and not stop_event.is_set():
                    done, _pending = wait(
                        list(future_to_topic.keys()),
                        timeout=0.25,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        continue
                    for future in done:
                        topic_id, _title = future_to_topic.pop(future)
                        try:
                            topic_result = future.result()
                        except Exception as exc:
                            logger.exception("AutoRead topic %s failed: %s", topic_id, exc)
                            ui.print_error(f"AutoRead topic {topic_id} failed, see logs above.")
                            continue
                        if topic_result["posts_read"] > 0:
                            ui.print_muted(
                                f"AutoRead cycle {cycle_number}: {topic_result['title']} "
                                f"({topic_result['posts_read']} posts)"
                            )
                            results.append(topic_result)
            if results:
                ui.print_autoread_summary(
                    {
                        "source": plan["source"],
                        "categories_count": plan["categories_count"],
                        "topics_requested": len(plan["topics"]),
                        "topics_read": len(results),
                        "total_posts_read": sum(int(item["posts_read"]) for item in results),
                        "topics": results,
                    }
                )
        except Exception as exc:
            logger.exception("AutoRead cycle failed: %s", exc)
            ui.print_error("AutoRead cycle failed, see logs above.")
        if _wait_for_autoread(stop_event, AUTOREAD_CYCLE_DELAY_SECONDS):
            break


def _build_autoread_plan(*, disourse: DiscourseClient, target: str | None) -> dict[str, Any]:
    categories = disourse.list_categories()
    if not target:
        latest = disourse.list_latest_topics(per_page=100)
        topics_by_id = {
            int(item["topic_id"]): item
            for item in _extract_topic_list(latest)
        }
        for category in categories:
            slug = category.get("slug")
            category_id = category.get("id")
            if not isinstance(slug, str) or not isinstance(category_id, int):
                continue
            payload = disourse.list_category_topics(slug=slug, category_id=category_id)
            for topic in _extract_topic_list(payload):
                topics_by_id[int(topic["topic_id"])] = topic
        topics = list(topics_by_id.values())
        return {
            "source": "automatic_discovery",
            "categories_count": len(categories),
            "topics": topics,
        }

    if _looks_like_url(target):
        try:
            category = disourse.resolve_category_url(target)
        except ValueError:
            category = None
        if category is not None:
            payload = disourse.list_category_topics(
                slug=str(category["slug"]),
                category_id=int(category["category_id"]),
            )
            topics = _extract_topic_list(payload)
            return {
                "source": f"category:{category['slug']}",
                "categories_count": len(categories),
                "topics": topics,
            }

        post_target = disourse.resolve_post_url(target)
        topic_id = int(post_target["topic_id"])
        topic = disourse.get_topic(topic_id, post_number=post_target.get("post_number"))
        return {
            "source": f"topic:{topic_id}",
            "categories_count": len(categories),
            "topics": [
                {
                    "topic_id": topic_id,
                    "title": topic.get("title"),
                    "posts_count": topic.get("posts_count"),
                }
            ],
        }

    raise ValueError("Usage: /autoread or /autoread <topic_or_category_url>")


def _read_topic_via_api(
    *,
    disourse: DiscourseClient,
    topic_id: int,
    stop_event: threading.Event | None = None,
    post_time_seconds: float = 120.0,
) -> dict[str, Any]:
    topic = disourse.get_topic(topic_id)
    title = str(topic.get("title", f"Topic {topic_id}"))
    topic_slug = str(topic.get("slug") or topic.get("topic_slug") or "")
    post_stream = topic.get("post_stream") if isinstance(topic, dict) else {}
    stream = post_stream.get("stream", []) if isinstance(post_stream, dict) else []
    loaded_posts = post_stream.get("posts", []) if isinstance(post_stream, dict) else []
    posts_by_id = {
        int(post["id"]): post
        for post in loaded_posts
        if isinstance(post, dict) and post.get("id") is not None
    }
    ordered_ids = [int(post_id) for post_id in stream if isinstance(post_id, int)]
    if not ordered_ids:
        ordered_ids = sorted(posts_by_id.keys())

    read_posts: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered_ids):
        current_id = ordered_ids[index]
        if current_id not in posts_by_id:
            chunk_ids = [
                post_id
                for post_id in ordered_ids[index : index + AUTOREAD_POST_CHUNK_SIZE]
                if post_id not in posts_by_id
            ]
            if chunk_ids:
                extra_posts = _load_topic_posts_with_fallback(
                    disourse=disourse,
                    topic_id=topic_id,
                    post_ids=chunk_ids,
                )
                for post in extra_posts:
                    if isinstance(post, dict) and post.get("id") is not None:
                        posts_by_id[int(post["id"])] = post
                continue

        post = posts_by_id.get(current_id)
        if isinstance(post, dict):
            post_number = int(post.get("post_number", len(read_posts) + 1))
            read_posts.append(
                {
                    "post_id": int(post["id"]),
                    "post_number": post_number,
                    "username": str(post.get("username", "")),
                }
            )
            if _simulate_autoread_post(
                disourse=disourse,
                topic_id=topic_id,
                topic_slug=topic_slug,
                post_number=post_number,
                stop_event=stop_event,
                post_time_seconds=post_time_seconds,
            ):
                break
        index += 1

    return {
        "topic_id": topic_id,
        "title": title,
        "posts_read": len(read_posts),
        "authors": sorted({item["username"] for item in read_posts if item["username"]}),
    }


def _wait_for_autoread(stop_event: threading.Event | None, seconds: float) -> bool:
    if stop_event is None:
        time.sleep(seconds)
        return False
    return stop_event.wait(seconds)


def _simulate_autoread_post(
    *,
    disourse: DiscourseClient,
    topic_id: int,
    topic_slug: str,
    post_number: int,
    stop_event: threading.Event | None,
    post_time_seconds: float,
) -> bool:
    remaining_seconds = max(0.0, float(post_time_seconds))
    if remaining_seconds <= 0:
        return False

    while remaining_seconds > 0:
        interval_seconds = min(remaining_seconds, AUTOREAD_MAX_TIMING_SECONDS)
        if _wait_for_autoread(stop_event, interval_seconds):
            return True
        _flush_autoread_timings_safely(
            disourse=disourse,
            topic_id=topic_id,
            topic_slug=topic_slug,
            batch=[
                {
                    "post_number": post_number,
                    "duration_ms": _autoread_duration_ms(interval_seconds),
                }
            ],
        )
        remaining_seconds -= interval_seconds
    return False


def _load_topic_posts_with_fallback(
    *,
    disourse: DiscourseClient,
    topic_id: int,
    post_ids: list[int],
) -> list[dict[str, Any]]:
    logger = logging.getLogger(__name__)
    try:
        payload = disourse.get_topic_posts(topic_id, post_ids)
    except HttpError as exc:
        logger.warning(
            "AutoRead bulk post fetch failed for topic %s (%s). Falling back to per-post reads.",
            topic_id,
            exc,
        )
        fallback_posts: list[dict[str, Any]] = []
        for post_id in post_ids:
            try:
                post = disourse.get_post(post_id)
            except HttpError as post_exc:
                logger.warning(
                    "AutoRead fallback post fetch failed for topic %s post %s: %s",
                    topic_id,
                    post_id,
                    post_exc,
                )
                continue
            if isinstance(post, dict) and post.get("id") is not None:
                fallback_posts.append(post)
        return fallback_posts

    extra_stream = payload.get("post_stream", []) if isinstance(payload, dict) else {}
    extra_posts = extra_stream.get("posts", []) if isinstance(extra_stream, dict) else []
    return [post for post in extra_posts if isinstance(post, dict)]


def _flush_autoread_timings_safely(
    *,
    disourse: DiscourseClient,
    topic_id: int,
    topic_slug: str,
    batch: list[dict[str, Any]],
) -> None:
    logger = logging.getLogger(__name__)
    try:
        _flush_autoread_timings(
            disourse=disourse,
            topic_id=topic_id,
            topic_slug=topic_slug,
            batch=batch,
        )
    except HttpError as exc:
        logger.warning("AutoRead timings update failed for topic %s: %s", topic_id, exc)


def _flush_autoread_timings(
    *,
    disourse: DiscourseClient,
    topic_id: int,
    topic_slug: str,
    batch: list[dict[str, Any]],
) -> None:
    if not batch:
        return
    last_post_number = int(batch[-1]["post_number"])
    referer = f"{disourse.host}/t/{topic_slug}/{topic_id}/{last_post_number}" if topic_slug else f"{disourse.host}/t/{topic_id}/{last_post_number}"
    timings = {int(item["post_number"]): int(item["duration_ms"]) for item in batch}
    disourse.record_topic_timings(
        topic_id=topic_id,
        timings=timings,
        topic_time=sum(timings.values()),
        referer=referer,
    )


def _autoread_duration_ms(post_time_seconds: float) -> int:
    return max(1, int(round(post_time_seconds * 1000)))


def _format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{int(seconds // 3600)}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{int(seconds // 60)}m"
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:.2f}s"


def _extract_topic_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    topic_list = payload.get("topic_list", []) if isinstance(payload, dict) else {}
    topics = topic_list.get("topics", []) if isinstance(topic_list, dict) else []
    extracted: list[dict[str, Any]] = []
    for item in topics:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        extracted.append(
            {
                "topic_id": int(item["id"]),
                "title": item.get("title"),
                "posts_count": item.get("posts_count"),
            }
        )
    return extracted


def _runtime_config_snapshot(settings: Settings) -> dict[str, Any]:
    return {
        "poll_interval_seconds": settings.bot_poll_interval_seconds,
        "response_delay_min_seconds": settings.bot_response_delay_min_seconds,
        "response_delay_max_seconds": settings.bot_response_delay_max_seconds,
        "autoread_post_time_seconds": settings.bot_autoread_post_time_seconds,
        "autoread_post_time_label": _format_duration(settings.bot_autoread_post_time_seconds),
        "max_context_posts": settings.bot_max_context_posts,
        "mark_read_on_skip": settings.bot_mark_read_on_skip,
        "typing_mode": settings.bot_typing_mode,
    }


def _handle_private_chat_message(
    *,
    raw: str,
    settings: Settings,
    ollama: OllamaClient,
    storage: BotStorage,
    state: _InteractiveState,
) -> bool:
    ui = _TerminalUI()
    state.private_chat_messages.append({"role": "user", "content": raw})
    state.private_chat_messages = _trim_private_chat_messages(state.private_chat_messages)
    ui.begin_private_stream()
    try:
        response = ollama.chat_stream(
            model=settings.ollama_model,
            system_prompt=_build_private_chat_system_prompt(settings, state.health_snapshot, storage),
            messages=state.private_chat_messages,
            on_chunk=ui.stream_private_chunk,
            options=settings.ollama_options,
            keep_alive=settings.ollama_keep_alive,
        )
    finally:
        ui.end_private_stream()
    state.private_chat_messages.append({"role": "assistant", "content": response})
    state.private_chat_messages = _trim_private_chat_messages(state.private_chat_messages)
    return False


def _build_private_chat_system_prompt(
    settings: Settings,
    health_snapshot: dict[str, Any] | None,
    storage: BotStorage,
) -> str:
    manual_commands = storage.list_manual_commands()
    pending_manual = sum(1 for item in manual_commands if item.status in {"queued", "scheduled"})
    completed_manual = sum(1 for item in manual_commands if item.status == "completed")
    recent_manual = manual_commands[-3:]
    recent_summary = "\n".join(
        f"- command_id={item.command_id}, status={item.status}, topic_id={item.topic_id}, reply_to_post_number={item.reply_to_post_number}"
        for item in recent_manual
    ) or "- none"
    health_lines = []
    if health_snapshot:
        health_lines = [
            f"Discourse host: {health_snapshot.get('discourse_host', '')}",
            f"Discourse auth mode: {health_snapshot.get('discourse_auth_mode', '')}",
            f"Discourse username: {health_snapshot.get('discourse_username', '')}",
            f"Ollama model: {health_snapshot.get('ollama_model', settings.ollama_model)}",
            f"Typing mode: {health_snapshot.get('typing_mode', settings.bot_typing_mode)}",
        ]
    else:
        health_lines = [
            f"Discourse host: {settings.discourse_host}",
            f"Ollama model: {settings.ollama_model}",
            f"Typing mode: {settings.bot_typing_mode}",
        ]
    return "\n".join(
        [
            settings.system_prompt.strip(),
            "",
            PRIVATE_CHAT_POLICY,
            "",
            "Current bot state summary:",
            *health_lines,
            f"Pending manual commands: {pending_manual}",
            f"Completed manual commands: {completed_manual}",
            "Recent manual commands:",
            recent_summary,
        ]
    )


def _trim_private_chat_messages(messages: list[dict[str, str]], *, max_messages: int = 12) -> list[dict[str, str]]:
    if len(messages) <= max_messages:
        return list(messages)
    return list(messages[-max_messages:])


def _configure_logging(level: int) -> None:
    if RichHandler is not None:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                RichHandler(
                    rich_tracebacks=True,
                    show_level=True,
                    show_path=False,
                    markup=False,
                )
            ],
        )
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class _CommandCompleter(Completer):  # type: ignore[misc]
    def __init__(self) -> None:
        self._slash_completer = (
            WordCompleter(SLASH_COMMANDS, ignore_case=True) if WordCompleter is not None else None
        )

    def get_completions(self, document: Any, complete_event: Any):  # pragma: no cover - exercised via prompt_toolkit
        if self._slash_completer is None:
            return
        text = document.text_before_cursor.lstrip()
        if text.startswith("/"):
            yield from self._slash_completer.get_completions(document, complete_event)
        elif document.text_before_cursor.strip():
            suggestion = " | "
            if suggestion.startswith(document.get_word_before_cursor(WORD=True) or ""):
                return


class _TerminalUI:
    _instance: "_TerminalUI | None" = None

    def __new__(cls) -> "_TerminalUI":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self) -> None:
        self.console = (
            Console()
            if Console is not None and getattr(sys.stdout, "isatty", lambda: False)()
            else None
        )
        self.history = InMemoryHistory() if InMemoryHistory is not None else None
        self._stream_active = False
        self.prompt_style = (
            PromptStyle.from_dict(
                {
                    "prompt": "#9ca3af",
                    "followup": "#9ca3af",
                    "chatprompt": "#9ca3af",
                }
            )
            if PromptStyle is not None
            else None
        )
        self.session = None
        if PromptSession is not None:
            try:
                self.session = PromptSession(
                    history=self.history,
                    auto_suggest=AutoSuggestFromHistory() if AutoSuggestFromHistory is not None else None,
                    completer=_CommandCompleter(),
                    complete_while_typing=True,
                    style=self.prompt_style,
                )
            except (NoConsoleScreenBufferError, EOFError, OSError):
                self.session = None

    def print_banner(self, title: str) -> None:
        if self.console is not None and Panel is not None:
            self.console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
            return
        print(f"{title}.")

    def print_health(self, health: dict[str, Any]) -> None:
        if self.console is not None and Table is not None:
            table = Table(show_header=False, box=None, pad_edge=False)
            table.add_column(style="bold cyan")
            table.add_column(style="white")
            table.add_row("Discourse", f"{health['discourse_host']} as {health['discourse_username']} (user_id={health['user_id']})")
            table.add_row("Ollama", str(health["ollama_model"]))
            table.add_row("Typing", str(health["typing_mode"]))
            self.console.print(table)
            return
        print(
            "Health: "
            f"Discourse={health['discourse_host']} as {health['discourse_username']} "
            f"(user_id={health['user_id']}), Ollama={health['ollama_model']}, "
            f"typing={health['typing_mode']}"
        )

    def print_status(self, text: str) -> None:
        if self.console is not None:
            self.console.print(f"[bold green]{text}[/bold green]")
            return
        print(text)

    def print_success(self, text: str) -> None:
        if self.console is not None:
            self.console.print(f"[bold green]{text}[/bold green]")
            return
        print(text)

    def print_error(self, text: str) -> None:
        if self.console is not None:
            self.console.print(f"[bold red]{text}[/bold red]")
            return
        print(text)

    def print_muted(self, text: str) -> None:
        if self.console is not None:
            self.console.print(f"[grey62]{text}[/grey62]")
            return
        print(text)

    def print_json(self, value: Any) -> None:
        text = json.dumps(value, indent=2)
        if self.console is not None:
            self.console.print_json(text)
            return
        print(text)

    def print_stats(self, stats: dict[str, Any]) -> None:
        if self.console is not None and Table is not None and Panel is not None:
            identity = stats.get("identity", {})
            runtime = stats.get("runtime", {})
            storage = stats.get("storage", {})

            summary_table = Table(show_header=False, box=None, pad_edge=False)
            summary_table.add_column(style="bold cyan")
            summary_table.add_column(style="white")
            summary_table.add_row("Bot", f"{identity.get('username') or 'unknown'} (user_id={identity.get('user_id')})")
            summary_table.add_row("Model", str(runtime.get("model", "")))
            summary_table.add_row("Typing", str(runtime.get("typing_mode", "")))
            summary_table.add_row(
                "Polling",
                f"{runtime.get('poll_interval_seconds', 0)}s every cycle",
            )
            summary_table.add_row(
                "Delay",
                f"{runtime.get('delay_min_seconds', 0)}s - {runtime.get('delay_max_seconds', 0)}s",
            )
            if runtime.get("autoread_post_time_seconds") is not None:
                summary_table.add_row(
                    "AutoRead",
                    _format_duration(float(runtime.get("autoread_post_time_seconds", 0))),
                )

            queue_table = Table(title="Queue", header_style="bold cyan")
            queue_table.add_column("Metric", style="bold white")
            queue_table.add_column("Value", justify="right", style="green")
            queue_table.add_row("Pending notification replies", str(storage.get("pending_replies", 0)))
            queue_table.add_row("Notification reply errors", str(storage.get("pending_reply_errors", 0)))
            queue_table.add_row("Manual queued", str(storage.get("manual_queued", 0)))
            queue_table.add_row("Manual scheduled", str(storage.get("manual_scheduled", 0)))
            queue_table.add_row("Manual completed", str(storage.get("manual_completed", 0)))
            queue_table.add_row("Manual errors", str(storage.get("manual_errors", 0)))

            handled_table = Table(title="Handled", header_style="bold cyan")
            handled_table.add_column("Metric", style="bold white")
            handled_table.add_column("Value", justify="right", style="magenta")
            handled_table.add_row("Total handled", str(storage.get("handled_total", 0)))
            handled_table.add_row("Replies sent", str(storage.get("handled_replied", 0)))
            handled_table.add_row("Skipped", str(storage.get("handled_skipped", 0)))
            handled_table.add_row("Manual total", str(storage.get("manual_total", 0)))

            self.console.print(Panel.fit(summary_table, title="[bold cyan]Bot Stats[/bold cyan]", border_style="cyan"))
            self.console.print(queue_table)
            self.console.print(handled_table)
            return

        print("Bot Stats")
        print(f"Bot: {stats.get('identity', {}).get('username')} (user_id={stats.get('identity', {}).get('user_id')})")
        print(f"Model: {stats.get('runtime', {}).get('model')}")
        print(f"Typing: {stats.get('runtime', {}).get('typing_mode')}")
        print(f"Polling: {stats.get('runtime', {}).get('poll_interval_seconds')}s")
        print(
            "Delay: "
            f"{stats.get('runtime', {}).get('delay_min_seconds')}s - "
            f"{stats.get('runtime', {}).get('delay_max_seconds')}s"
        )
        if stats.get("runtime", {}).get("autoread_post_time_seconds") is not None:
            print(
                "AutoRead: "
                f"{_format_duration(float(stats.get('runtime', {}).get('autoread_post_time_seconds', 0)))}"
            )
        print(f"Pending notification replies: {stats.get('storage', {}).get('pending_replies')}")
        print(f"Notification reply errors: {stats.get('storage', {}).get('pending_reply_errors')}")
        print(f"Manual queued: {stats.get('storage', {}).get('manual_queued')}")
        print(f"Manual scheduled: {stats.get('storage', {}).get('manual_scheduled')}")
        print(f"Manual completed: {stats.get('storage', {}).get('manual_completed')}")
        print(f"Manual errors: {stats.get('storage', {}).get('manual_errors')}")
        print(f"Total handled: {stats.get('storage', {}).get('handled_total')}")
        print(f"Replies sent: {stats.get('storage', {}).get('handled_replied')}")
        print(f"Skipped: {stats.get('storage', {}).get('handled_skipped')}")

    def print_summary(self, text: str) -> None:
        if self.console is not None and Panel is not None:
            body = Markdown(text) if Markdown is not None else text
            self.console.print(Panel(body, title="[bold cyan]Recent Summary[/bold cyan]", border_style="cyan"))
            return
        print(text)

    def print_autoread_summary(self, result: dict[str, Any]) -> None:
        if self.console is not None and Table is not None and Panel is not None:
            summary = Table(show_header=False, box=None, pad_edge=False)
            summary.add_column(style="bold cyan")
            summary.add_column(style="white")
            summary.add_row("Source", str(result.get("source", "")))
            summary.add_row("Categories", str(result.get("categories_count", 0)))
            summary.add_row("Topics read", str(result.get("topics_read", 0)))
            summary.add_row("Posts read", str(result.get("total_posts_read", 0)))
            self.console.print(Panel.fit(summary, title="[bold cyan]AutoRead[/bold cyan]", border_style="cyan"))

            topics_table = Table(title="Topics", header_style="bold cyan")
            topics_table.add_column("Topic", style="bold white")
            topics_table.add_column("Posts", justify="right", style="green")
            topics_table.add_column("Authors", style="grey78")
            for topic in result.get("topics", []):
                authors = ", ".join(topic.get("authors", [])[:5])
                topics_table.add_row(
                    str(topic.get("title", topic.get("topic_id"))),
                    str(topic.get("posts_read", 0)),
                    authors or "-",
                )
            self.console.print(topics_table)
            return

        print(f"AutoRead source: {result.get('source')}")
        print(f"Categories: {result.get('categories_count')}")
        print(f"Topics read: {result.get('topics_read')}")
        print(f"Posts read: {result.get('total_posts_read')}")
        for topic in result.get("topics", []):
            print(f"- {topic.get('title')} ({topic.get('posts_read')} posts)")

    def print_blank(self) -> None:
        if self.console is not None:
            self.console.print("")
            return
        print()

    def print_help(self) -> None:
        if self.console is not None and Table is not None:
            table = Table(title="Commands", header_style="bold cyan")
            table.add_column("Command", style="bold white")
            table.add_column("Description", style="grey78")
            table.add_row("/help", "Show available commands")
            table.add_row("/health", "Show current Discourse and Ollama health")
            table.add_row("/stats", "Show a live local bot stats dashboard")
            table.add_row("/summarize", "Summarize the bot's last 5-10 recent actions")
            table.add_row("/autoread [url]", "Read latest topics or all posts in a specific topic/category via API")
            table.add_row("/config", "Show or change runtime settings")
            table.add_row("/notifications", "Show the latest notification page")
            table.add_row("/manual", "Show queued and completed manual AI commands")
            table.add_row("/clear queue", "Clear pending notification replies and queued manual commands")
            table.add_row("/clear db", "Reset all local bot database state")
            table.add_row("/chat", "Enter private operator chat with Ollama")
            table.add_row("/chat-reset", "Clear private operator chat history")
            table.add_row("/bot", "Return from private chat to bot mode")
            table.add_row('/send "<post_url>" "<request>"', "Queue a manual AI reply")
            table.add_row("/quit", "Exit the bot shell")
            self.console.print(table)
            config_table = Table(title="Runtime Config", header_style="bold cyan")
            config_table.add_column("Command", style="bold white")
            config_table.add_column("Effect", style="grey78")
            config_table.add_row("/config", "Show current runtime settings")
            config_table.add_row("/config delay <min> <max>", "Change reply delay for future jobs")
            config_table.add_row("/config poll <seconds>", "Change polling interval")
            config_table.add_row("/config autoread-time <duration>", "Change AutoRead per-post timing like 30s, 1m, or 1h")
            config_table.add_row("/config context <count>", "Change recent-post context size")
            config_table.add_row("/config mark-read <on|off>", "Toggle marking skipped notifications read")
            self.console.print(config_table)
            self.console.print("[grey62]Plain text mode: type a message, a post URL, or '<post_url> | <message>'.[/grey62]")
            return

        print("Commands:")
        print("/help")
        print("/health")
        print("/stats")
        print("/summarize")
        print("/autoread [url]")
        print("/config")
        print("/notifications")
        print("/manual")
        print("/clear queue")
        print("/clear db")
        print("/chat")
        print("/chat-reset")
        print("/bot")
        print('/send "<post_url>" "<request>"')
        print("/quit")
        print("Runtime config:")
        print("/config")
        print("/config delay <min> <max>")
        print("/config poll <seconds>")
        print("/config autoread-time <duration>")
        print("/config context <count>")
        print("/config mark-read <on|off>")

    def print_private_reply(self, text: str) -> None:
        if self.console is not None and Panel is not None:
            self.console.print(Panel.fit(text, title="[bold magenta]Private Chat[/bold magenta]", border_style="magenta"))
            return
        print(text)

    def prompt(self, *, mode: str = "bot") -> str:
        placeholder_text = PROMPT_TEXT.strip() if mode != "chat" else "Private chat here"
        if self.session is not None:
            style_name = "class:prompt" if mode != "chat" else "class:chatprompt"
            return self.session.prompt(
                "",
                placeholder=[(style_name, placeholder_text)],
            )
        fallback_prompt = "> " if mode != "chat" else "chat> "
        return input(fallback_prompt)

    def prompt_followup(self, label: str) -> str:
        if self.session is not None:
            return self.session.prompt([("class:followup", f"{label}> ")])
        return input(f"{label}> ")

    def begin_private_stream(self) -> None:
        self._stream_active = True
        if self.console is not None:
            self.console.print("[bold magenta]AI[/bold magenta]: ", end="")
            return
        print("AI: ", end="", flush=True)

    def stream_private_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        if self.console is not None:
            self.console.print(chunk, style="magenta", end="", highlight=False)
            return
        print(chunk, end="", flush=True)

    def end_private_stream(self) -> None:
        if not self._stream_active:
            return
        if self.console is not None:
            self.console.print("")
        else:
            print()
        self._stream_active = False
