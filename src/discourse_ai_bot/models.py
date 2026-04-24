from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BotIdentity:
    user_id: int
    username: str
    name: str | None = None


@dataclass(frozen=True)
class Notification:
    notification_id: int
    notification_type: int
    type_name: str | None
    read: bool
    created_at: str | None
    topic_id: int | None
    post_number: int | None
    slug: str | None
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        type_name_map: dict[int, str] | None = None,
    ) -> "Notification":
        raw_data = payload.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        notification_type = int(payload["notification_type"])
        type_name = type_name_map.get(notification_type) if type_name_map else None
        return cls(
            notification_id=int(payload["id"]),
            notification_type=notification_type,
            type_name=type_name,
            read=bool(payload.get("read", False)),
            created_at=payload.get("created_at"),
            topic_id=_coerce_optional_int(payload.get("topic_id")),
            post_number=_coerce_optional_int(payload.get("post_number")),
            slug=payload.get("slug"),
            data=data,
        )

    @property
    def actor_username(self) -> str | None:
        for key in (
            "username",
            "display_username",
            "mentioned_by_username",
            "invited_by_username",
            "original_username",
        ):
            value = self.data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None


@dataclass(frozen=True)
class TopicPost:
    post_id: int | None
    topic_id: int
    post_number: int
    username: str
    cooked: str | None
    raw: str | None
    created_at: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any], topic_id: int | None = None) -> "TopicPost":
        return cls(
            post_id=_coerce_optional_int(payload.get("id")),
            topic_id=topic_id or int(payload["topic_id"]),
            post_number=int(payload["post_number"]),
            username=str(payload.get("username", "")),
            cooked=payload.get("cooked"),
            raw=payload.get("raw"),
            created_at=payload.get("created_at"),
        )


@dataclass(frozen=True)
class ClassifiedNotification:
    notification: Notification
    trigger: str
    actor_username: str | None


@dataclass(frozen=True)
class TopicContext:
    notification_id: int
    trigger: str
    actor_username: str | None
    topic_id: int
    topic_title: str
    topic_slug: str | None
    topic_archetype: str | None
    target_post: TopicPost | None
    recent_posts: tuple[TopicPost, ...]

    @property
    def reply_to_post_number(self) -> int | None:
        return self.target_post.post_number if self.target_post else None


@dataclass(frozen=True)
class ModelDecision:
    action: str
    reply_markdown: str
    reason: str
    gif_id: str | None = None


@dataclass(frozen=True)
class AutonomousCandidate:
    post_url: str
    topic_id: int
    post_number: int
    actor_username: str | None
    context: TopicContext


@dataclass(frozen=True)
class AutonomousSelection:
    action: str
    post_url: str | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class PendingJob:
    notification_id: int
    topic_id: int
    reply_to_post_number: int | None
    raw: str
    gif_id: str | None
    decision_reason: str
    due_at: str
    attempts: int
    last_error: str | None
    created_at: str
    presence_channel: str | None
    last_presence_at: str | None


@dataclass(frozen=True)
class ManualCommand:
    command_id: int
    post_url: str
    user_request: str
    status: str
    created_at: str
    available_at: str
    topic_id: int | None
    reply_to_post_number: int | None
    raw: str | None
    gif_id: str | None
    ollama_reason: str | None
    due_at: str | None
    attempts: int
    last_error: str | None
    presence_channel: str | None
    last_presence_at: str | None
    response_post_id: int | None
    completed_at: str | None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
