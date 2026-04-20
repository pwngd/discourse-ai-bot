from __future__ import annotations

from discourse_ai_bot.models import ClassifiedNotification, TopicContext, TopicPost
from discourse_ai_bot.utils import take_last


POSSIBLE_POST_ID_KEYS = (
    "post_id",
    "original_post_id",
    "topic_post_id",
    "reply_post_id",
)


class ContextResolver:
    def __init__(self, discourse_client: object, *, max_posts: int) -> None:
        self.discourse = discourse_client
        self.max_posts = max_posts

    def resolve(self, classified: ClassifiedNotification) -> TopicContext:
        notification = classified.notification
        if notification.topic_id is None:
            raise ValueError(
                f"Notification {notification.notification_id} does not include a topic_id."
            )

        return self.resolve_topic(
            notification_id=notification.notification_id,
            trigger=classified.trigger,
            actor_username=classified.actor_username,
            topic_id=notification.topic_id,
            post_number=notification.post_number,
            slug=notification.slug,
            data=notification.data,
        )

    def resolve_topic(
        self,
        *,
        notification_id: int,
        trigger: str,
        actor_username: str | None,
        topic_id: int,
        post_number: int | None = None,
        post_id: int | None = None,
        slug: str | None = None,
        data: dict[str, object] | None = None,
    ) -> TopicContext:
        data = data or {}

        topic = self.discourse.get_topic(topic_id, post_number=post_number)
        post_stream = topic.get("post_stream") if isinstance(topic, dict) else None
        posts_payload = post_stream.get("posts", []) if isinstance(post_stream, dict) else []
        posts = [
            TopicPost.from_payload(payload, topic_id=topic_id)
            for payload in posts_payload
            if isinstance(payload, dict)
        ]
        posts.sort(key=lambda item: item.post_number)

        target_post = None
        if post_number is not None:
            target_post = next(
                (post for post in posts if post.post_number == post_number),
                None,
            )

        inferred_post_id = post_id
        if inferred_post_id is None:
            inferred_post_id = self._extract_post_id(data)
        if target_post is None and inferred_post_id is not None:
            fetched_post = TopicPost.from_payload(
                self.discourse.get_post(inferred_post_id),
                topic_id=topic_id,
            )
            target_post = fetched_post
            if all(post.post_number != fetched_post.post_number for post in posts):
                posts.append(fetched_post)
                posts.sort(key=lambda item: item.post_number)

        if target_post is None and post_number is not None:
            target_post = next(
                (post for post in posts if post.post_number == post_number),
                None,
            )

        if target_post is None and posts:
            target_post = posts[-1]

        recent_posts = take_last(posts, self.max_posts)
        if target_post and all(post.post_number != target_post.post_number for post in recent_posts):
            if len(recent_posts) >= self.max_posts:
                recent_posts = recent_posts[1:]
            recent_posts.append(target_post)
            recent_posts.sort(key=lambda item: item.post_number)

        return TopicContext(
            notification_id=notification_id,
            trigger=trigger,
            actor_username=actor_username,
            topic_id=topic_id,
            topic_title=str(topic.get("title") or f"Topic {topic_id}"),
            topic_slug=topic.get("slug") or slug,
            topic_archetype=topic.get("archetype"),
            target_post=target_post,
            recent_posts=tuple(recent_posts),
        )

    @staticmethod
    def _extract_post_id(data: dict[str, object]) -> int | None:
        for key in POSSIBLE_POST_ID_KEYS:
            value = data.get(key)
            if value is None or value == "":
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None
