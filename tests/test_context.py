from __future__ import annotations

import unittest

from discourse_ai_bot.context import ContextResolver


class ContextResolverTests(unittest.TestCase):
    def test_resolve_topic_fetches_recent_posts_from_topic_stream(self) -> None:
        class FakeDiscourse:
            def __init__(self) -> None:
                self.topic_posts_calls: list[list[int]] = []

            def get_topic(self, topic_id: int, *, post_number: int | None = None) -> dict[str, object]:
                if post_number is None:
                    return {
                        "title": "Deep topic",
                        "slug": "deep-topic",
                        "post_stream": {
                            "stream": [101, 102, 103, 104, 105],
                            "posts": [
                                {
                                    "id": 101,
                                    "topic_id": topic_id,
                                    "post_number": 1,
                                    "username": "alpha",
                                    "cooked": "<p>one</p>",
                                },
                                {
                                    "id": 102,
                                    "topic_id": topic_id,
                                    "post_number": 2,
                                    "username": "bravo",
                                    "cooked": "<p>two</p>",
                                },
                            ],
                        },
                    }
                return {
                    "post_stream": {
                        "posts": [
                            {
                                "id": 104,
                                "topic_id": topic_id,
                                "post_number": 4,
                                "username": "delta",
                                "cooked": "<p>four</p>",
                            }
                        ]
                    }
                }

            def get_topic_posts(self, topic_id: int, post_ids: list[int]) -> dict[str, object]:
                self.topic_posts_calls.append(list(post_ids))
                pool = {
                    103: {
                        "id": 103,
                        "topic_id": topic_id,
                        "post_number": 3,
                        "username": "charlie",
                        "cooked": "<p>three</p>",
                    },
                    104: {
                        "id": 104,
                        "topic_id": topic_id,
                        "post_number": 4,
                        "username": "delta",
                        "cooked": "<p>four</p>",
                    },
                    105: {
                        "id": 105,
                        "topic_id": topic_id,
                        "post_number": 5,
                        "username": "echo",
                        "cooked": "<p>five</p>",
                    },
                }
                return {
                    "post_stream": {
                        "posts": [pool[post_id] for post_id in post_ids]
                    }
                }

            def get_post(self, post_id: int) -> dict[str, object]:
                raise AssertionError(f"Unexpected per-post fallback for {post_id}")

        discourse = FakeDiscourse()
        resolver = ContextResolver(discourse, max_posts=3)

        context = resolver.resolve_topic(
            notification_id=1,
            trigger="mentioned",
            actor_username="delta",
            topic_id=500,
            post_number=4,
        )

        self.assertEqual([post.post_number for post in context.recent_posts], [3, 4, 5])
        self.assertIsNotNone(context.target_post)
        self.assertEqual(context.target_post.post_number, 4)
        self.assertEqual(discourse.topic_posts_calls, [[103, 105]])

