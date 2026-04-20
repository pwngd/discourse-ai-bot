from __future__ import annotations

from discourse_ai_bot.models import ClassifiedNotification, Notification


DEFAULT_NOTIFICATION_TYPES = {
    1: "mentioned",
    2: "replied",
    3: "quoted",
    4: "edited",
    5: "liked",
    6: "private_message",
    7: "invited_to_private_message",
    8: "invitee_accepted",
    9: "posted",
    10: "moved_post",
    11: "linked",
    12: "granted_badge",
    13: "invited_to_topic",
    14: "custom",
    15: "group_mentioned",
    16: "group_message_summary",
    17: "watching_first_post",
    18: "topic_reminder",
    19: "liked_consolidated",
    20: "post_approved",
    21: "code_review_commit_approved",
    22: "membership_request_accepted",
    23: "membership_request_consolidated",
    24: "bookmark_reminder",
    25: "reaction",
    26: "votes_released",
    27: "event_reminder",
    28: "event_invitation",
    29: "chat_mention",
    30: "chat_message",
    31: "chat_invitation",
    33: "chat_quoted",
    34: "assigned",
    36: "watching_category_or_tag",
    37: "new_features",
    38: "admin_problems",
    39: "linked_consolidated",
    40: "chat_watched_thread",
    41: "upcoming_change_available",
    42: "upcoming_change_automatically_promoted",
    43: "boost",
    800: "following",
    801: "following_created_topic",
    802: "following_replied",
    900: "circles_activity",
}


class NotificationClassifier:
    def __init__(
        self,
        bot_username: str,
        *,
        allowed_triggers: tuple[str, ...],
        notification_types: dict[int, str] | None = None,
    ) -> None:
        self.bot_username = bot_username.lower()
        self.allowed_triggers = {trigger.lower() for trigger in allowed_triggers}
        self.notification_types = notification_types or DEFAULT_NOTIFICATION_TYPES

    def classify(self, notification: Notification) -> ClassifiedNotification | None:
        type_name = notification.type_name or self.notification_types.get(notification.notification_type)
        if not type_name or type_name not in self.allowed_triggers:
            return None

        actor_username = notification.actor_username
        if actor_username and actor_username.lower() == self.bot_username:
            return None

        return ClassifiedNotification(
            notification=notification,
            trigger=type_name,
            actor_username=actor_username,
        )
