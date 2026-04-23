from __future__ import annotations

import logging
import random
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from discourse_ai_bot.classifier import DEFAULT_NOTIFICATION_TYPES, NotificationClassifier
from discourse_ai_bot.context import ContextResolver
from discourse_ai_bot.gifs import GifCatalog, GifOption
from discourse_ai_bot.models import BotIdentity, ClassifiedNotification, Notification
from discourse_ai_bot.settings import Settings
from discourse_ai_bot.utils import datetime_to_storage, parse_datetime, utc_now


class BotService:
    def __init__(
        self,
        *,
        settings: Settings,
        discourse_client: object,
        ollama_client: object,
        storage: object,
        presence_adapter: object,
        logger: logging.Logger | None = None,
        randomizer: random.Random | None = None,
        now_fn: Callable[[], Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.discourse = discourse_client
        self.ollama = ollama_client
        self.storage = storage
        self.presence = presence_adapter
        self.logger = logger or logging.getLogger(__name__)
        self.randomizer = randomizer or random.Random()
        self.now_fn = now_fn or utc_now
        self.sleep_fn = sleep_fn
        self.identity: BotIdentity | None = None
        self.classifier: NotificationClassifier | None = None
        self.context_resolver = ContextResolver(
            self.discourse,
            max_posts=settings.bot_max_context_posts,
        )
        self.gif_catalog = GifCatalog(Path.cwd() / "gifs")
        self.activity_events: deque[dict[str, str]] = deque(maxlen=100)

    def bootstrap(self) -> BotIdentity:
        if self.identity is not None and self.classifier is not None:
            return self.identity

        site_info = self.discourse.get_site_info()
        raw_mapping = site_info.get("notification_types", {})
        type_map = _reverse_notification_types(raw_mapping)
        if not type_map:
            type_map = DEFAULT_NOTIFICATION_TYPES
        self.discourse.set_notification_type_map(type_map)

        if self.settings.discourse_auth_mode == "session_cookie":
            session_payload = self.discourse.get_current_session()
            user = session_payload.get("current_user") if isinstance(session_payload, dict) else None
            if not isinstance(user, dict) and self.settings.discourse_username:
                user_payload = self.discourse.get_user(self.settings.discourse_username)
                user = user_payload.get("user") if isinstance(user_payload, dict) else None
        else:
            user_payload = self.discourse.get_user(self.settings.discourse_username or "")
            user = user_payload.get("user") if isinstance(user_payload, dict) else None
        if not isinstance(user, dict):
            raise ValueError(
                "Discourse user lookup did not return a user object. "
                "For session_cookie auth, ensure the cookie belongs to a logged-in user or set DISCOURSE_USERNAME."
            )

        self.identity = BotIdentity(
            user_id=int(user.get("id", 0)),
            username=str(user["username"]),
            name=user.get("name"),
        )
        self.classifier = NotificationClassifier(
            self.identity.username,
            allowed_triggers=self.settings.bot_allowed_triggers,
            notification_types=type_map,
        )
        self._record_activity(
            f"Bootstrapped bot identity as {self.identity.username} (user_id={self.identity.user_id})."
        )
        return self.identity

    def run_once(self) -> None:
        self.bootstrap()
        notifications = self.discourse.list_notifications(paginate=False)
        self._record_activity(f"Polled {len(notifications)} notifications.")
        self._evaluate_notifications(notifications)
        self._evaluate_manual_commands()
        self._refresh_presence()
        self._refresh_manual_presence()
        self._process_due_jobs()
        self._process_due_manual_commands()

    def run_forever(self) -> None:
        self.bootstrap()
        while True:
            self.run_once()
            self.sleep_fn(self.settings.bot_poll_interval_seconds)

    def inspect_notifications(self, *, paginate: bool = False) -> list[dict[str, Any]]:
        self.bootstrap()
        assert self.classifier is not None
        snapshots: list[dict[str, Any]] = []
        for notification in self.discourse.list_notifications(paginate=paginate):
            candidate = self.classifier.classify(notification)
            if self.storage.is_handled(notification.notification_id):
                state = "handled"
            elif self.storage.get_pending_job(notification.notification_id):
                state = "pending"
            elif candidate:
                state = "candidate"
            else:
                state = "ignored"

            snapshots.append(
                {
                    "notification_id": notification.notification_id,
                    "type": notification.type_name or notification.notification_type,
                    "state": state,
                    "read": notification.read,
                    "topic_id": notification.topic_id,
                    "post_number": notification.post_number,
                    "actor_username": notification.actor_username,
                }
            )
        return snapshots

    def inspect_manual_commands(self) -> list[dict[str, Any]]:
        return [
            {
                "command_id": command.command_id,
                "status": command.status,
                "post_url": command.post_url,
                "user_request": command.user_request,
                "topic_id": command.topic_id,
                "reply_to_post_number": command.reply_to_post_number,
                "due_at": command.due_at,
                "attempts": command.attempts,
                "last_error": command.last_error,
                "response_post_id": command.response_post_id,
            }
            for command in self.storage.list_manual_commands()
        ]

    def clear_queue(self) -> dict[str, int]:
        result = self.storage.clear_queue()
        self._record_activity(
            "Cleared outbound queue "
            f"(manual_commands={result['manual_commands_deleted']}, "
            f"pending_replies={result['pending_replies_deleted']})."
        )
        return result

    def reset_database(self) -> dict[str, int]:
        result = self.storage.reset_database()
        self.activity_events.clear()
        self._record_activity(
            "Reset local database "
            f"(handled_notifications={result['handled_notifications_deleted']}, "
            f"manual_commands={result['manual_commands_deleted']}, "
            f"pending_replies={result['pending_replies_deleted']})."
        )
        return result

    def inspect_stats(self) -> dict[str, Any]:
        self.bootstrap()
        summary = self.storage.stats_summary()
        return {
            "identity": {
                "user_id": self.identity.user_id if self.identity else None,
                "username": self.identity.username if self.identity else None,
                "name": self.identity.name if self.identity else None,
            },
            "runtime": {
                "poll_interval_seconds": self.settings.bot_poll_interval_seconds,
                "delay_min_seconds": self.settings.bot_response_delay_min_seconds,
                "delay_max_seconds": self.settings.bot_response_delay_max_seconds,
                "autoread_post_time_seconds": self.settings.bot_autoread_post_time_seconds,
                "typing_mode": self.settings.bot_typing_mode,
                "model": self.settings.ollama_model,
            },
            "storage": summary,
        }

    def inspect_recent_activity(self, *, limit: int = 10) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        return list(self.activity_events)[-limit:]

    def _evaluate_notifications(self, notifications: list[Notification]) -> None:
        assert self.classifier is not None
        notifications = sorted(
            notifications,
            key=lambda item: (parse_datetime(item.created_at) or utc_now(), item.notification_id),
        )
        for notification in notifications:
            if notification.read:
                continue
            if self.storage.is_handled(notification.notification_id):
                continue
            if self.storage.get_pending_job(notification.notification_id):
                continue

            classified = self.classifier.classify(notification)
            if classified is None:
                continue
            self._handle_candidate(classified)

    def _handle_candidate(self, classified: ClassifiedNotification) -> None:
        now = self.now_fn()
        try:
            context = self.context_resolver.resolve(classified)
            decision = self.ollama.decide(
                model=self.settings.ollama_model,
                system_prompt=self.settings.system_prompt,
                identity=self.identity,
                context=context,
                available_gifs=self.gif_catalog.list_options(),
                options=self.settings.ollama_options,
                keep_alive=self.settings.ollama_keep_alive,
            )
        except Exception as exc:
            self._record_activity(
                f"Failed to evaluate notification {classified.notification.notification_id}: {exc}",
                level="warning",
            )
            self.logger.exception(
                "Failed to evaluate notification %s: %s",
                classified.notification.notification_id,
                exc,
            )
            return

        notification_id = classified.notification.notification_id
        if decision.action == "skip":
            self.storage.record_handled(
                notification_id,
                action="skip",
                reason=decision.reason,
                handled_at=datetime_to_storage(now),
            )
            if self.settings.bot_mark_read_on_skip:
                self._mark_read_best_effort(notification_id)
            self._record_activity(
                f"Skipped notification {notification_id}: {decision.reason}"
            )
            self.logger.info("Skipped notification %s: %s", notification_id, decision.reason)
            return

        delay_seconds = self.randomizer.uniform(
            self.settings.bot_response_delay_min_seconds,
            self.settings.bot_response_delay_max_seconds,
        )
        due_at = now + timedelta(seconds=delay_seconds)
        reply_to_post_number = context.reply_to_post_number or classified.notification.post_number
        presence_channel = None
        if self.settings.typing_enabled:
            presence_channel = self.settings.discourse_presence_reply_channel_template.format(
                topic_id=context.topic_id
            )

        inserted = self.storage.enqueue_job(
            notification_id=notification_id,
            topic_id=context.topic_id,
            reply_to_post_number=reply_to_post_number,
            raw=decision.reply_markdown,
            gif_id=decision.gif_id,
            decision_reason=decision.reason,
            due_at=datetime_to_storage(due_at),
            created_at=datetime_to_storage(now),
            presence_channel=presence_channel,
        )
        if inserted:
            self._record_activity(
                f"Queued reply for notification {notification_id} with {delay_seconds:.2f}s delay."
            )
            self.logger.info(
                "Queued reply for notification %s with %.2f second delay.",
                notification_id,
                delay_seconds,
            )

    def _refresh_presence(self) -> None:
        if not getattr(self.presence, "enabled", False):
            return
        now = self.now_fn()
        threshold = self.settings.presence_heartbeat_interval_seconds
        for job in self.storage.list_pending_jobs():
            if not job.presence_channel:
                continue
            due_at = parse_datetime(job.due_at)
            if due_at is None or due_at <= now:
                continue
            last_presence_at = parse_datetime(job.last_presence_at)
            if last_presence_at and (now - last_presence_at).total_seconds() < threshold:
                continue
            try:
                self.presence.present(job.presence_channel)
                self.storage.update_job_presence(
                    job.notification_id,
                    datetime_to_storage(now),
                )
            except Exception as exc:
                self._record_activity(
                    f"Presence heartbeat failed for notification {job.notification_id}: {exc}",
                    level="warning",
                )
                self.logger.warning(
                    "Presence heartbeat failed for notification %s: %s",
                    job.notification_id,
                    exc,
                )

    def _refresh_manual_presence(self) -> None:
        if not getattr(self.presence, "enabled", False):
            return
        now = self.now_fn()
        threshold = self.settings.presence_heartbeat_interval_seconds
        for command in self.storage.list_scheduled_manual_commands():
            if not command.presence_channel:
                continue
            due_at = parse_datetime(command.due_at)
            if due_at is None or due_at <= now:
                continue
            last_presence_at = parse_datetime(command.last_presence_at)
            if last_presence_at and (now - last_presence_at).total_seconds() < threshold:
                continue
            try:
                self.presence.present(command.presence_channel)
                self.storage.update_manual_command_presence(
                    command.command_id,
                    datetime_to_storage(now),
                )
            except Exception as exc:
                self._record_activity(
                    f"Presence heartbeat failed for manual command {command.command_id}: {exc}",
                    level="warning",
                )
                self.logger.warning(
                    "Presence heartbeat failed for manual command %s: %s",
                    command.command_id,
                    exc,
                )

    def _process_due_jobs(self) -> None:
        now = self.now_fn()
        due_jobs = self.storage.list_due_jobs(datetime_to_storage(now))
        for job in due_jobs:
            self._process_job(job, now)

    def _process_due_manual_commands(self) -> None:
        now = self.now_fn()
        due_commands = self.storage.list_due_manual_commands(datetime_to_storage(now))
        for command in due_commands:
            self._process_manual_command_send(command, now)

    def _evaluate_manual_commands(self) -> None:
        now = self.now_fn()
        for command in self.storage.list_ready_manual_commands(datetime_to_storage(now)):
            self._prepare_manual_command(command, now)

    def _prepare_manual_command(self, command: Any, now: Any) -> None:
        try:
            target = self.discourse.resolve_post_url(command.post_url)
            context = self.context_resolver.resolve_topic(
                notification_id=-command.command_id,
                trigger="manual_request",
                actor_username=None,
                topic_id=int(target["topic_id"]),
                post_number=target.get("post_number"),
                post_id=target.get("post_id"),
            )
            decision = self.ollama.compose_manual_reply(
                model=self.settings.ollama_model,
                system_prompt=self.settings.system_prompt,
                identity=self.identity,
                context=context,
                user_request=command.user_request,
                available_gifs=self.gif_catalog.list_options(),
                options=self.settings.ollama_options,
                keep_alive=self.settings.ollama_keep_alive,
            )
        except Exception as exc:
            attempts = command.attempts + 1
            next_available = now + timedelta(seconds=self._backoff_seconds(attempts))
            self.storage.reschedule_manual_command_generation(
                command.command_id,
                available_at=datetime_to_storage(next_available),
                attempts=attempts,
                last_error=str(exc),
            )
            self._record_activity(
                f"Failed to prepare manual command {command.command_id} (attempt {attempts}): {exc}",
                level="warning",
            )
            self.logger.warning(
                "Failed to prepare manual command %s: %s",
                command.command_id,
                exc,
            )
            return

        delay_seconds = self.randomizer.uniform(
            self.settings.bot_response_delay_min_seconds,
            self.settings.bot_response_delay_max_seconds,
        )
        due_at = now + timedelta(seconds=delay_seconds)
        presence_channel = None
        if self.settings.typing_enabled:
            presence_channel = self.settings.discourse_presence_reply_channel_template.format(
                topic_id=context.topic_id
            )

        self.storage.schedule_manual_command(
            command.command_id,
            topic_id=context.topic_id,
            reply_to_post_number=(
                int(target["post_number"]) if target.get("post_number") is not None else context.reply_to_post_number
            ),
            raw=decision.reply_markdown,
            gif_id=decision.gif_id,
            ollama_reason=decision.reason,
            due_at=datetime_to_storage(due_at),
            presence_channel=presence_channel,
        )
        self._record_activity(
            f"Queued manual command {command.command_id} with {delay_seconds:.2f}s delay."
        )
        self.logger.info(
            "Queued manual command %s with %.2f second delay.",
            command.command_id,
            delay_seconds,
        )

    def _process_job(self, job: Any, now: Any) -> None:
        if getattr(self.presence, "enabled", False) and job.presence_channel:
            try:
                self.presence.present(job.presence_channel)
            except Exception as exc:
                self._record_activity(
                    f"Final presence heartbeat failed for notification {job.notification_id}: {exc}",
                    level="warning",
                )
                self.logger.warning(
                    "Final presence heartbeat failed for notification %s: %s",
                    job.notification_id,
                    exc,
                )

        try:
            final_raw = self._build_post_body(job.raw, job.gif_id)
            response = self.discourse.create_post(
                raw=final_raw,
                topic_id=job.topic_id,
                reply_to_post_number=job.reply_to_post_number,
            )
            response_post_id = int(response["id"]) if "id" in response else None
            self.storage.record_handled(
                job.notification_id,
                action="reply",
                reason=job.decision_reason,
                response_post_id=response_post_id,
                handled_at=datetime_to_storage(now),
            )
            self._mark_read_best_effort(job.notification_id)
            self._record_activity(f"Posted reply for notification {job.notification_id}.")
            self.logger.info("Posted reply for notification %s.", job.notification_id)
        except Exception as exc:
            attempts = job.attempts + 1
            next_due = now + timedelta(seconds=self._backoff_seconds(attempts))
            self.storage.reschedule_job(
                job.notification_id,
                due_at=datetime_to_storage(next_due),
                attempts=attempts,
                last_error=str(exc),
            )
            self._record_activity(
                f"Failed to post reply for notification {job.notification_id} (attempt {attempts}): {exc}",
                level="warning",
            )
            self.logger.warning(
                "Failed to post reply for notification %s (attempt %s): %s",
                job.notification_id,
                attempts,
                exc,
            )
        finally:
            if getattr(self.presence, "enabled", False) and job.presence_channel:
                try:
                    self.presence.leave(job.presence_channel)
                except Exception as exc:
                    self._record_activity(
                        f"Presence leave failed for notification {job.notification_id}: {exc}",
                        level="warning",
                    )
                    self.logger.warning(
                        "Presence leave failed for notification %s: %s",
                        job.notification_id,
                        exc,
                    )

    def _process_manual_command_send(self, command: Any, now: Any) -> None:
        if getattr(self.presence, "enabled", False) and command.presence_channel:
            try:
                self.presence.present(command.presence_channel)
            except Exception as exc:
                self._record_activity(
                    f"Final presence heartbeat failed for manual command {command.command_id}: {exc}",
                    level="warning",
                )
                self.logger.warning(
                    "Final presence heartbeat failed for manual command %s: %s",
                    command.command_id,
                    exc,
                )

        try:
            final_raw = self._build_post_body(command.raw or "", command.gif_id)
            response = self.discourse.create_post(
                raw=final_raw,
                topic_id=command.topic_id,
                reply_to_post_number=command.reply_to_post_number,
            )
            response_post_id = int(response["id"]) if "id" in response else None
            self.storage.complete_manual_command(
                command.command_id,
                response_post_id=response_post_id,
                completed_at=datetime_to_storage(now),
            )
            self._record_activity(f"Posted manual reply for command {command.command_id}.")
            self.logger.info("Posted manual reply for command %s.", command.command_id)
        except Exception as exc:
            attempts = command.attempts + 1
            next_due = now + timedelta(seconds=self._backoff_seconds(attempts))
            self.storage.reschedule_manual_command_send(
                command.command_id,
                due_at=datetime_to_storage(next_due),
                attempts=attempts,
                last_error=str(exc),
            )
            self._record_activity(
                f"Failed to post manual reply for command {command.command_id} (attempt {attempts}): {exc}",
                level="warning",
            )
            self.logger.warning(
                "Failed to post manual reply for command %s (attempt %s): %s",
                command.command_id,
                attempts,
                exc,
            )
        finally:
            if getattr(self.presence, "enabled", False) and command.presence_channel:
                try:
                    self.presence.leave(command.presence_channel)
                except Exception as exc:
                    self._record_activity(
                        f"Presence leave failed for manual command {command.command_id}: {exc}",
                        level="warning",
                    )
                    self.logger.warning(
                        "Presence leave failed for manual command %s: %s",
                        command.command_id,
                        exc,
                    )

    def _mark_read_best_effort(self, notification_id: int) -> None:
        try:
            self.discourse.mark_notification_read(notification_id)
        except Exception as exc:
            self._record_activity(
                f"Unable to mark notification {notification_id} as read: {exc}",
                level="warning",
            )
            self.logger.warning(
                "Unable to mark notification %s as read: %s",
                notification_id,
                exc,
            )

    @staticmethod
    def _backoff_seconds(attempts: int) -> float:
        return min(3600.0, 30.0 * (2 ** max(attempts - 1, 0)))

    def _build_post_body(self, raw: str, gif_id: str | None) -> str:
        base = raw.strip()
        if not gif_id:
            return base

        option = self.gif_catalog.get(gif_id)
        if option is None:
            self._record_activity(
                f"GIF '{gif_id}' was requested but not found. Sending text only.",
                level="warning",
            )
            self.logger.warning("GIF '%s' was requested but not found. Sending text only.", gif_id)
            return base

        upload_url = self._upload_gif(option)
        if upload_url is None:
            return base
        return f"{base}\n\n![{option.alt_text}]({upload_url})".strip()

    def _upload_gif(self, option: GifOption) -> str | None:
        try:
            response = self.discourse.upload_file(option.path, upload_type="composer", synchronous=True)
        except Exception as exc:
            self._record_activity(
                f"Failed to upload GIF '{option.gif_id}': {exc}. Sending text only.",
                level="warning",
            )
            self.logger.warning("Failed to upload GIF '%s': %s", option.gif_id, exc)
            return None

        upload_reference = self._normalize_upload_reference(response)
        if upload_reference is None:
            self._record_activity(
                f"Upload response for GIF '{option.gif_id}' did not include a URL. Sending text only.",
                level="warning",
            )
            self.logger.warning(
                "Upload response for GIF '%s' did not include a usable URL.",
                option.gif_id,
            )
            return None
        return upload_reference

    def _normalize_upload_reference(self, response: object) -> str | None:
        if not isinstance(response, dict):
            return None

        for key in ("short_url", "url", "short_path"):
            raw_value = response.get(key)
            if not isinstance(raw_value, str):
                continue
            candidate = raw_value.strip()
            if not candidate:
                continue
            if candidate.startswith("upload://"):
                return candidate
            if candidate.startswith(("http://", "https://")):
                return candidate
            if candidate.startswith("//"):
                scheme = urlsplit(self.settings.discourse_host).scheme or "https"
                return f"{scheme}:{candidate}"
            return f"{self.settings.discourse_host.rstrip('/')}/{candidate.lstrip('/')}"
        return None

    def _record_activity(self, message: str, *, level: str = "info") -> None:
        self.activity_events.append(
            {
                "timestamp": datetime_to_storage(self.now_fn()),
                "level": level,
                "message": message,
            }
        )


def _reverse_notification_types(raw_mapping: object) -> dict[int, str]:
    if not isinstance(raw_mapping, dict):
        return {}
    reversed_mapping: dict[int, str] = {}
    for name, value in raw_mapping.items():
        try:
            reversed_mapping[int(value)] = str(name)
        except (TypeError, ValueError):
            continue
    return reversed_mapping
