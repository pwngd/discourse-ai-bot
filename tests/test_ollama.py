from __future__ import annotations

import unittest

from discourse_ai_bot.models import BotIdentity, TopicContext, TopicPost
from discourse_ai_bot.ollama import OllamaClient, OllamaResponseError


class FakeTransport:
    def __init__(self, responses: dict[tuple[str, str], object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.stream_calls: list[tuple[str, str, dict[str, object]]] = []
        self.stream_responses: dict[tuple[str, str], list[object]] = {}

    def request_json(self, method: str, path_or_url: str, **kwargs: object) -> object:
        self.calls.append((method, path_or_url, kwargs))
        return self.responses[(method, path_or_url)]

    def stream_json_lines(self, method: str, path_or_url: str, **kwargs: object):
        self.stream_calls.append((method, path_or_url, kwargs))
        for item in self.stream_responses[(method, path_or_url)]:
            yield item


class OllamaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TopicContext(
            notification_id=10,
            trigger="mentioned",
            actor_username="alice",
            topic_id=99,
            topic_title="Question",
            topic_slug="question",
            topic_archetype=None,
            target_post=TopicPost(
                post_id=555,
                topic_id=99,
                post_number=2,
                username="alice",
                cooked="<p>Hello there</p>",
                raw=None,
            ),
            recent_posts=(),
        )
        self.identity = BotIdentity(user_id=1, username="bot")

    def test_decide_parses_valid_json_response(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client.http = FakeTransport(
            {
                ("POST", "/chat"): {
                    "message": {
                        "content": '{"action":"reply","reply_markdown":"Thanks!","reason":"Direct mention"}'
                    }
                }
            }
        )
        decision = client.decide(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
            options={"temperature": 0},
            keep_alive="5m",
        )
        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.reply_markdown, "Thanks!")
        method, path, kwargs = client.http.calls[0]
        self.assertEqual((method, path), ("POST", "/chat"))
        self.assertFalse(kwargs["json_body"]["stream"])

    def test_decide_rejects_invalid_action(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client.http = FakeTransport(
            {
                ("POST", "/chat"): {
                    "message": {
                        "content": '{"action":"maybe","reply_markdown":"","reason":"Nope"}'
                    }
                }
            }
            )
        with self.assertRaises(OllamaResponseError):
            client.decide(
                model="qwen3",
                system_prompt="Prompt",
                identity=self.identity,
                context=self.context,
            )

    def test_compose_manual_reply_requires_reply_action(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client.http = FakeTransport(
            {
                ("POST", "/chat"): {
                    "message": {
                        "content": '{"action":"reply","reply_markdown":"On it.","reason":"Operator request"}'
                    }
                }
            }
        )

        decision = client.compose_manual_reply(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
            user_request="Tell them I will handle it.",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.reply_markdown, "On it.")

    def test_chat_returns_plain_content(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client.http = FakeTransport(
            {
                ("POST", "/chat"): {
                    "message": {
                        "content": "Private operator response."
                    }
                }
            }
        )

        content = client.chat(
            model="qwen3",
            system_prompt="Private prompt",
            messages=[{"role": "user", "content": "Hello"}],
        )

        self.assertEqual(content, "Private operator response.")

    def test_chat_stream_yields_incremental_content(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport({})
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"content": "Private "}, "done": False},
            {"message": {"content": "operator "}, "done": False},
            {"message": {"content": "response."}, "done": False},
            {"done": True},
        ]
        client.http = transport
        chunks: list[str] = []

        content = client.chat_stream(
            model="qwen3",
            system_prompt="Private prompt",
            messages=[{"role": "user", "content": "Hello"}],
            on_chunk=chunks.append,
        )

        self.assertEqual(content, "Private operator response.")
        self.assertEqual(chunks, ["Private ", "operator ", "response."])
        method, path, kwargs = transport.stream_calls[0]
        self.assertEqual((method, path), ("POST", "/chat"))
        self.assertTrue(kwargs["json_body"]["stream"])

    def test_summarize_activity_returns_markdown_summary(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client.http = FakeTransport(
            {
                ("POST", "/chat"): {
                    "message": {
                        "content": "Summary sentence.\n\n## Recent Activity\n- Did a thing."
                    }
                }
            }
        )

        summary = client.summarize_activity(
            model="qwen3",
            runtime_snapshot={
                "identity": {"username": "bot", "user_id": 1},
                "runtime": {"model": "qwen3", "typing_mode": "none"},
                "storage": {"handled_total": 1},
            },
            activity_events=[
                {"timestamp": "2026-01-01T00:00:00+00:00", "level": "info", "message": "Did a thing."}
            ],
        )

        self.assertIn("## Recent Activity", summary)
        method, path, kwargs = client.http.calls[0]
        self.assertEqual((method, path), ("POST", "/chat"))
        self.assertFalse(kwargs["json_body"]["stream"])
