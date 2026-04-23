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

        topic = self.discourse.get_topic(topic_id)
        topic_metadata = topic if isinstance(topic, dict) else {}
        posts = self._extract_posts(topic_metadata, topic_id)
        stream_post_ids = self._extract_stream_post_ids(topic_metadata)

        target_post = None
        if post_number is not None:
            target_post = self._find_post_by_number(posts, post_number)

        if target_post is None and post_number is not None:
            focused_topic = self.discourse.get_topic(topic_id, post_number=post_number)
            if isinstance(focused_topic, dict) and not topic_metadata:
                topic_metadata = focused_topic
            posts = self._merge_posts(posts, self._extract_posts(focused_topic, topic_id))
            target_post = self._find_post_by_number(posts, post_number)

        inferred_post_id = post_id
        if inferred_post_id is None:
            inferred_post_id = self._extract_post_id(data)
        if target_post is None and inferred_post_id is not None:
            fetched_posts = self._load_posts_with_fallback(topic_id, [inferred_post_id])
            posts = self._merge_posts(posts, fetched_posts)
            target_post = self._find_post_by_id(posts, inferred_post_id)

        required_post_ids = self._required_post_ids(stream_post_ids, target_post)
        missing_post_ids = [
            candidate_id
            for candidate_id in required_post_ids
            if self._find_post_by_id(posts, candidate_id) is None
        ]
        if missing_post_ids:
            posts = self._merge_posts(
                posts,
                self._load_posts_with_fallback(topic_id, missing_post_ids),
            )

        if target_post is None and post_number is not None:
            target_post = self._find_post_by_number(posts, post_number)

        if target_post is None and posts:
            target_post = posts[-1]

        recent_posts = self._select_recent_posts(posts, stream_post_ids)
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
            topic_title=str(topic_metadata.get("title") or f"Topic {topic_id}"),
            topic_slug=topic_metadata.get("slug") or slug,
            topic_archetype=topic_metadata.get("archetype"),
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

    @staticmethod
    def _extract_posts(topic_payload: dict[str, object], topic_id: int) -> list[TopicPost]:
        post_stream = topic_payload.get("post_stream") if isinstance(topic_payload, dict) else None
        posts_payload = post_stream.get("posts", []) if isinstance(post_stream, dict) else []
        posts = [
            TopicPost.from_payload(payload, topic_id=topic_id)
            for payload in posts_payload
            if isinstance(payload, dict)
        ]
        posts.sort(key=lambda item: item.post_number)
        return posts

    @staticmethod
    def _extract_stream_post_ids(topic_payload: dict[str, object]) -> list[int]:
        post_stream = topic_payload.get("post_stream") if isinstance(topic_payload, dict) else None
        stream_payload = post_stream.get("stream", []) if isinstance(post_stream, dict) else []
        stream_post_ids: list[int] = []
        for value in stream_payload:
            try:
                stream_post_ids.append(int(value))
            except (TypeError, ValueError):
                continue
        return stream_post_ids

    @staticmethod
    def _merge_posts(*post_groups: list[TopicPost]) -> list[TopicPost]:
        merged: dict[int, TopicPost] = {}
        for posts in post_groups:
            for post in posts:
                merged[post.post_number] = post
        ordered = list(merged.values())
        ordered.sort(key=lambda item: item.post_number)
        return ordered

    def _load_posts_with_fallback(self, topic_id: int, post_ids: list[int]) -> list[TopicPost]:
        if not post_ids:
            return []

        payload: dict[str, object] | None = None
        get_topic_posts = getattr(self.discourse, "get_topic_posts", None)
        if callable(get_topic_posts):
            try:
                response = get_topic_posts(topic_id, post_ids)
            except Exception:
                payload = None
            else:
                payload = response if isinstance(response, dict) else {}
        if payload is not None:
            loaded_posts = self._extract_posts(payload, topic_id)
            if loaded_posts:
                return loaded_posts

        fallback_posts: list[TopicPost] = []
        for candidate_id in post_ids:
            try:
                payload = self.discourse.get_post(candidate_id)
            except Exception:
                continue
            if isinstance(payload, dict):
                fallback_posts.append(TopicPost.from_payload(payload, topic_id=topic_id))
        fallback_posts.sort(key=lambda item: item.post_number)
        return fallback_posts

    def _required_post_ids(
        self,
        stream_post_ids: list[int],
        target_post: TopicPost | None,
    ) -> list[int]:
        if self.max_posts <= 0:
            return []
        required = list(stream_post_ids[-self.max_posts :]) if stream_post_ids else []
        if target_post and target_post.post_id is not None and target_post.post_id not in required:
            required.append(target_post.post_id)
        return required

    def _select_recent_posts(
        self,
        posts: list[TopicPost],
        stream_post_ids: list[int],
    ) -> list[TopicPost]:
        if self.max_posts <= 0:
            return []
        if not stream_post_ids:
            return take_last(posts, self.max_posts)

        posts_by_id = {post.post_id: post for post in posts if post.post_id is not None}
        recent_posts = [
            posts_by_id[post_id]
            for post_id in stream_post_ids[-self.max_posts :]
            if post_id in posts_by_id
        ]
        if recent_posts:
            recent_posts.sort(key=lambda item: item.post_number)
            return recent_posts
        return take_last(posts, self.max_posts)

    @staticmethod
    def _find_post_by_number(posts: list[TopicPost], post_number: int) -> TopicPost | None:
        return next((post for post in posts if post.post_number == post_number), None)

    @staticmethod
    def _find_post_by_id(posts: list[TopicPost], post_id: int) -> TopicPost | None:
        return next((post for post in posts if post.post_id == post_id), None)
