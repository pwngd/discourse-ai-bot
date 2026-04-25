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
from discourse_ai_bot.models import (
    AutonomousCandidate,
    AutonomousSelection,
    BotIdentity,
    ClassifiedNotification,
    Notification,
    TopicContext,
    TopicPost,
)
from discourse_ai_bot.roblox_docs import RobloxDocsClient, RobloxDocsError
from discourse_ai_bot.settings import Settings
from discourse_ai_bot.utils import datetime_to_storage, parse_datetime, strip_html, utc_now


AUTONOMOUS_LATEST_PAGE_SIZE = 30
AUTONOMOUS_LATEST_MAX_PAGES = 5
AUTONOMOUS_REPLY_REQUEST_PREFIX = "AUTONOMOUS_REPLY_SELECTION:"


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
        roblox_docs_client: object | None = None,
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
        self.roblox_docs = (
            roblox_docs_client
            if roblox_docs_client is not None
            else _build_roblox_docs_client(settings)
        )
        self.activity_events: deque[dict[str, str]] = deque(maxlen=100)
        self._last_autonomous_reply_scan_at: Any | None = None
        self.category_parent_ids: dict[int, int | None] = {}
        self.autonomous_blocked_category_ids: set[int] = set()

    def bootstrap(self) -> BotIdentity:
        if self.identity is not None and self.classifier is not None:
            return self.identity

        site_info = self.discourse.get_site_info()
        raw_mapping = site_info.get("notification_types", {})
        type_map = _reverse_notification_types(raw_mapping)
        if not type_map:
            type_map = DEFAULT_NOTIFICATION_TYPES
        self.discourse.set_notification_type_map(type_map)
        self.category_parent_ids = _category_parent_map(site_info.get("categories", []))
        self.autonomous_blocked_category_ids = _resolve_blocked_category_ids(
            self.settings.bot_autonomous_reply_blocked_category_urls,
            site_info.get("categories", []),
            self.discourse,
            logger=self.logger,
        )

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
        self._evaluate_autonomous_reply_candidates()
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
                "autonomous_reply_enabled": self.settings.bot_autonomous_reply_enabled,
                "autonomous_reply_interval_seconds": self.settings.bot_autonomous_reply_interval_seconds,
                "autonomous_reply_latest_count": self.settings.bot_autonomous_reply_latest_count,
                "autonomous_reply_min_confidence": self.settings.bot_autonomous_reply_min_confidence,
                "autonomous_reply_blocked_categories": len(self.autonomous_blocked_category_ids),
                "roblox_docs_enabled": self.settings.bot_roblox_docs_enabled,
                "roblox_docs_source": self.settings.bot_roblox_docs_source,
                "roblox_docs_local_path": self.settings.bot_roblox_docs_local_path,
                "roblox_docs_ref": self.settings.bot_roblox_docs_ref,
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
            roblox_docs_context = self._roblox_docs_context(context)
            decision = self.ollama.decide(
                model=self.settings.ollama_model,
                system_prompt=self.settings.system_prompt,
                identity=self.identity,
                context=context,
                available_gifs=self.gif_catalog.list_options(),
                roblox_docs_context=roblox_docs_context,
                options=self.settings.ollama_options,
                keep_alive=self.settings.ollama_keep_alive,
            )
        except Exception as exc:
            self._record_activity(
                f"Failed to evaluate notification {classified.notification.notification_id}: {exc}",
                level="warning",
            )
            self.logger.warning(
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

    def _evaluate_autonomous_reply_candidates(self) -> None:
        if not self.settings.bot_autonomous_reply_enabled:
            return
        if self.identity is None:
            return

        now = self.now_fn()
        if (
            self._last_autonomous_reply_scan_at is not None
            and (now - self._last_autonomous_reply_scan_at).total_seconds()
            < self.settings.bot_autonomous_reply_interval_seconds
        ):
            return
        try:
            candidates = self._collect_autonomous_candidates()
        except Exception as exc:
            self._record_activity(
                f"Failed to collect autonomous reply candidates: {exc}",
                level="warning",
            )
            self.logger.warning("Failed to collect autonomous reply candidates: %s", exc)
            return

        if not candidates:
            self._last_autonomous_reply_scan_at = now
            self._record_activity("Autonomous reply scan found no new latest-post candidates.")
            return

        selection = self._select_autonomous_reply_target_or_force(candidates)
        if selection is None:
            return

        if selection.action == "skip":
            for candidate in candidates:
                self.storage.record_autonomous_target(
                    post_url=candidate.post_url,
                    topic_id=candidate.topic_id,
                    post_number=candidate.post_number,
                    status="skipped",
                    reason=selection.reason,
                    recorded_at=datetime_to_storage(now),
                )
            self._last_autonomous_reply_scan_at = now
            self._record_activity(
                "Autonomous reply scan skipped "
                f"{len(candidates)} candidates: {selection.reason}"
            )
            return

        selected = next(
            (candidate for candidate in candidates if candidate.post_url == selection.post_url),
            None,
        )
        if selected is None:
            self._record_activity(
                f"Autonomous selection returned an unknown post URL: {selection.post_url}",
                level="warning",
            )
            self.logger.warning(
                "Autonomous selection returned an unknown post URL: %s",
                selection.post_url,
            )
            return

        if selection.confidence < self.settings.bot_autonomous_reply_min_confidence:
            self.storage.record_autonomous_target(
                post_url=selected.post_url,
                topic_id=selected.topic_id,
                post_number=selected.post_number,
                status="skipped",
                reason=(
                    f"Confidence {selection.confidence:.2f} below "
                    f"{self.settings.bot_autonomous_reply_min_confidence:.2f}: {selection.reason}"
                ),
                recorded_at=datetime_to_storage(now),
            )
            self._last_autonomous_reply_scan_at = now
            self._record_activity(
                "Autonomous reply target skipped for low confidence "
                f"({selection.confidence:.2f}): {selected.post_url}"
            )
            return

        command_id = self.storage.enqueue_manual_command(
            post_url=selected.post_url,
            user_request=_autonomous_reply_request(selection.reason),
            created_at=datetime_to_storage(now),
        )
        self.storage.record_autonomous_target(
            post_url=selected.post_url,
            topic_id=selected.topic_id,
            post_number=selected.post_number,
            status="queued",
            reason=selection.reason,
            recorded_at=datetime_to_storage(now),
            command_id=command_id,
        )
        self._record_activity(
            "Autonomous reply queued manual command "
            f"{command_id} for {selected.post_url} (confidence {selection.confidence:.2f})."
        )
        self._last_autonomous_reply_scan_at = now
        command = self.storage.get_manual_command(command_id)
        if command is not None:
            self._prepare_manual_command(command, now)

    def _select_autonomous_reply_target_or_force(
        self,
        candidates: list[AutonomousCandidate],
    ) -> AutonomousSelection | None:
        try:
            return self.ollama.select_autonomous_reply_target(
                model=self.settings.ollama_model,
                system_prompt=self.settings.system_prompt,
                identity=self.identity,
                candidates=candidates,
                min_confidence=self.settings.bot_autonomous_reply_min_confidence,
                options=self.settings.ollama_options,
                keep_alive=self.settings.ollama_keep_alive,
            )
        except Exception as exc:
            selected = candidates[0] if candidates else None
            if selected is None:
                self._record_activity(
                    f"Failed to select autonomous reply target and no fallback candidate is available: {exc}",
                    level="warning",
                )
                self.logger.warning(
                    "Failed to select autonomous reply target and no fallback candidate is available: %s",
                    exc,
                )
                return None
            reason = (
                "Ollama failed to select a target after one chance, "
                f"so the bot is forcing a reply to the first eligible latest post: {exc}"
            )
            self._record_activity(reason, level="warning")
            self.logger.warning("%s", reason)
            return AutonomousSelection(
                action="reply",
                post_url=selected.post_url,
                confidence=1.0,
                reason=reason,
            )

    def _collect_autonomous_candidates(self) -> list[AutonomousCandidate]:
        candidates: list[AutonomousCandidate] = []
        skipped_blocked = 0
        for page in range(AUTONOMOUS_LATEST_MAX_PAGES):
            payload = self.discourse.list_latest_topics(
                per_page=AUTONOMOUS_LATEST_PAGE_SIZE,
                page=page if page > 0 else None,
            )
            topic_list = payload.get("topic_list") if isinstance(payload, dict) else None
            topics = topic_list.get("topics", []) if isinstance(topic_list, dict) else []
            if not topics:
                break

            for topic in topics:
                if not isinstance(topic, dict):
                    continue
                candidate = self._topic_to_autonomous_candidate(topic)
                if candidate == "blocked":
                    skipped_blocked += 1
                    continue
                if candidate is None:
                    continue
                candidates.append(candidate)
                if len(candidates) >= self.settings.bot_autonomous_reply_latest_count:
                    if skipped_blocked:
                        self._record_activity(
                            f"Autonomous reply scan skipped {skipped_blocked} latest topics in blocked categories."
                        )
                    return candidates
        if skipped_blocked:
            self._record_activity(
                f"Autonomous reply scan skipped {skipped_blocked} latest topics in blocked categories."
            )
        return candidates

    def _topic_to_autonomous_candidate(self, topic: dict[str, object]) -> AutonomousCandidate | str | None:
        topic_id = _optional_int(topic.get("id"))
        post_number = _latest_topic_post_number(topic)
        if topic_id is None or post_number is None:
            return None

        category_id = _optional_int(topic.get("category_id"))
        if _category_is_blocked(
            category_id,
            blocked_ids=self.autonomous_blocked_category_ids,
            parent_ids=self.category_parent_ids,
        ):
            return "blocked"

        actor_username = _optional_str(topic.get("last_poster_username"))
        if self.identity and _same_username(actor_username, self.identity.username):
            return None

        slug = _optional_str(topic.get("slug")) or "topic"
        post_url = _topic_post_url(
            self.settings.discourse_host,
            slug=slug,
            topic_id=topic_id,
            post_number=post_number,
        )
        if self.storage.has_manual_command_for_post_url(post_url):
            return None
        if self.storage.is_autonomous_target_seen(post_url):
            return None
        if self.storage.has_bot_reply_target(
            topic_id=topic_id,
            reply_to_post_number=post_number,
        ):
            return None

        try:
            context = self.context_resolver.resolve_topic(
                notification_id=0,
                trigger="autonomous_scan",
                actor_username=actor_username,
                topic_id=topic_id,
                post_number=post_number,
                slug=slug,
            )
        except Exception as exc:
            self._record_activity(
                f"Unable to resolve autonomous candidate {post_url}: {exc}",
                level="warning",
            )
            self.logger.warning(
                "Unable to resolve autonomous candidate %s: %s",
                post_url,
                exc,
            )
            return None

        if self.identity and context.target_post and _same_username(
            context.target_post.username,
            self.identity.username,
        ):
            return None

        return AutonomousCandidate(
            post_url=post_url,
            topic_id=topic_id,
            post_number=post_number,
            actor_username=actor_username,
            context=context,
        )

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
            roblox_docs_context = self._roblox_docs_context(
                context,
                extra_text=command.user_request,
            )
            selection_reason = _autonomous_reply_selection_reason(command.user_request)
            if selection_reason is None:
                decision = self.ollama.compose_manual_reply(
                    model=self.settings.ollama_model,
                    system_prompt=self.settings.system_prompt,
                    identity=self.identity,
                    context=context,
                    user_request=command.user_request,
                    available_gifs=self.gif_catalog.list_options(),
                    roblox_docs_context=roblox_docs_context,
                    options=self.settings.ollama_options,
                    keep_alive=self.settings.ollama_keep_alive,
                )
            else:
                decision = self.ollama.compose_autonomous_reply(
                    model=self.settings.ollama_model,
                    system_prompt=self.settings.system_prompt,
                    identity=self.identity,
                    context=context,
                    selection_reason=selection_reason,
                    available_gifs=self.gif_catalog.list_options(),
                    roblox_docs_context=roblox_docs_context,
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

    def _roblox_docs_context(
        self,
        context: TopicContext,
        *,
        extra_text: str = "",
    ) -> str | None:
        if self.roblox_docs is None:
            return None
        context_for_text = getattr(self.roblox_docs, "context_for_text", None)
        if not callable(context_for_text):
            return None

        query_text = _roblox_docs_query_text(context, extra_text=extra_text)
        try:
            docs_context = context_for_text(query_text)
        except RobloxDocsError as exc:
            self._record_activity(
                f"Roblox docs lookup failed: {exc}",
                level="warning",
            )
            self.logger.warning("Roblox docs lookup failed: %s", exc)
            return None
        except Exception as exc:
            self._record_activity(
                f"Roblox docs lookup failed unexpectedly: {exc}",
                level="warning",
            )
            self.logger.warning("Roblox docs lookup failed unexpectedly: %s", exc)
            return None
        if docs_context is None:
            return None

        format_for_prompt = getattr(docs_context, "format_for_prompt", None)
        if not callable(format_for_prompt):
            return None
        prompt_context = format_for_prompt(
            max_chars=self.settings.bot_roblox_docs_max_context_chars,
        )
        if prompt_context:
            self._record_activity("Attached Roblox API docs context to a coding question.")
        return prompt_context or None

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


def _build_roblox_docs_client(settings: Settings) -> RobloxDocsClient | None:
    if not settings.bot_roblox_docs_enabled:
        return None
    return RobloxDocsClient(
        ref=settings.bot_roblox_docs_ref,
        timeout_seconds=settings.bot_roblox_docs_timeout_seconds,
        cache_ttl_seconds=settings.bot_roblox_docs_cache_ttl_seconds,
        max_terms=settings.bot_roblox_docs_max_terms,
        max_results=settings.bot_roblox_docs_max_results,
        max_context_chars=settings.bot_roblox_docs_max_context_chars,
        source=settings.bot_roblox_docs_source,
        local_path=settings.bot_roblox_docs_local_path,
    )


def _roblox_docs_query_text(context: TopicContext, *, extra_text: str = "") -> str:
    parts = [
        f"Topic title: {context.topic_title}",
        f"Trigger: {context.trigger}",
        extra_text,
    ]
    if context.target_post is not None:
        parts.append(_topic_post_text(context.target_post))
    for post in context.recent_posts:
        if context.target_post is not None and post.post_number == context.target_post.post_number:
            continue
        parts.append(_topic_post_text(post))
    return "\n\n".join(part for part in parts if part.strip())


def _topic_post_text(post: TopicPost) -> str:
    return post.raw or strip_html(post.cooked) or ""


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


def _latest_topic_post_number(topic: dict[str, object]) -> int | None:
    for key in ("highest_post_number", "posts_count"):
        value = _optional_int(topic.get(key))
        if value is not None and value > 0:
            return value
    return None


def _category_parent_map(categories_payload: object) -> dict[int, int | None]:
    if not isinstance(categories_payload, list):
        return {}
    parent_ids: dict[int, int | None] = {}
    for item in categories_payload:
        if not isinstance(item, dict):
            continue
        category_id = _optional_int(item.get("id"))
        if category_id is None:
            continue
        parent_ids[category_id] = _optional_int(item.get("parent_category_id"))
    return parent_ids


def _resolve_blocked_category_ids(
    category_urls: tuple[str, ...],
    categories_payload: object,
    discourse: object,
    *,
    logger: logging.Logger,
) -> set[int]:
    if not category_urls:
        return set()

    categories = [item for item in categories_payload if isinstance(item, dict)] if isinstance(categories_payload, list) else []
    by_slug = _category_ids_by_slug(categories)
    blocked_ids: set[int] = set()
    for category_url in category_urls:
        category_id = _category_id_from_url(category_url)
        if category_id is None:
            category_id = _category_id_from_slug_url(category_url, by_slug)
        if category_id is None:
            try:
                resolved = discourse.resolve_category_url(category_url)
            except Exception as exc:
                logger.warning(
                    "Unable to resolve autonomous blocked category URL %s: %s",
                    category_url,
                    exc,
                )
                continue
            category_id = _optional_int(resolved.get("category_id")) if isinstance(resolved, dict) else None
        if category_id is None:
            logger.warning("Unable to resolve autonomous blocked category URL %s.", category_url)
            continue
        blocked_ids.add(category_id)
    return blocked_ids


def _category_ids_by_slug(categories: list[dict[str, object]]) -> dict[str, list[int]]:
    by_slug: dict[str, list[int]] = {}
    for category in categories:
        category_id = _optional_int(category.get("id"))
        slug = _optional_str(category.get("slug"))
        if category_id is None or slug is None:
            continue
        by_slug.setdefault(slug, []).append(category_id)
    return by_slug


def _category_id_from_url(category_url: str) -> int | None:
    segments = _category_url_segments(category_url)
    if not segments:
        return None
    numeric_segments = [_optional_int(segment) for segment in segments[1:]]
    numeric_segments = [segment for segment in numeric_segments if segment is not None]
    return numeric_segments[-1] if numeric_segments else None


def _category_id_from_slug_url(category_url: str, by_slug: dict[str, list[int]]) -> int | None:
    segments = _category_url_segments(category_url)
    if not segments:
        return None
    slug_segments = [segment for segment in segments[1:] if _optional_int(segment) is None]
    if not slug_segments:
        return None
    matches = by_slug.get(slug_segments[-1], [])
    return matches[0] if len(matches) == 1 else None


def _category_url_segments(category_url: str) -> list[str]:
    path = urlsplit(category_url).path or category_url
    segments = [segment for segment in path.split("/") if segment]
    if not segments or segments[0] != "c":
        return []
    return segments


def _category_is_blocked(
    category_id: int | None,
    *,
    blocked_ids: set[int],
    parent_ids: dict[int, int | None],
) -> bool:
    if category_id is None or not blocked_ids:
        return False
    seen: set[int] = set()
    current: int | None = category_id
    while current is not None and current not in seen:
        if current in blocked_ids:
            return True
        seen.add(current)
        current = parent_ids.get(current)
    return False


def _topic_post_url(host: str, *, slug: str, topic_id: int, post_number: int) -> str:
    return f"{host.rstrip('/')}/t/{slug}/{topic_id}/{post_number}"


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _same_username(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return left.strip().casefold() == right.strip().casefold()


def _autonomous_reply_request(reason: str) -> str:
    return f"{AUTONOMOUS_REPLY_REQUEST_PREFIX} {reason.strip()}"


def _autonomous_reply_selection_reason(user_request: str) -> str | None:
    if not user_request.startswith(AUTONOMOUS_REPLY_REQUEST_PREFIX):
        return None
    reason = user_request[len(AUTONOMOUS_REPLY_REQUEST_PREFIX) :].strip()
    return reason or "The latest-post scanner selected this post as worth joining."
