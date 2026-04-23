from __future__ import annotations

import unittest

from discourse_ai_bot.models import BotIdentity, TopicContext, TopicPost
from discourse_ai_bot.ollama import (
    OllamaClient,
    OllamaResponseError,
    ThinkingEvent,
    _build_manual_request_prompt,
)


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
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {
                "message": {
                    "content": '{"action":"reply","reply_markdown":"Thanks!","reason":"Direct mention","gif_id":"friendly_wave"}'
                }
            },
            {"done": True},
        ]
        client.http = transport
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
        self.assertEqual(decision.gif_id, "friendly_wave")
        method, path, kwargs = transport.stream_calls[0]
        self.assertEqual((method, path), ("POST", "/chat"))
        self.assertTrue(kwargs["json_body"]["stream"])

    def test_decide_rejects_invalid_action(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"content": '{"action":"maybe","reply_markdown":"","reason":"Nope"}'}},
            {"done": True},
        ]
        client.http = transport
        with self.assertRaises(OllamaResponseError):
            client.decide(
                model="qwen3",
                system_prompt="Prompt",
                identity=self.identity,
                context=self.context,
            )

    def test_compose_manual_reply_requires_reply_action(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"content": '{"action":"reply","reply_markdown":"On it.","reason":"Operator request"}'}},
            {"done": True},
        ]
        client.http = transport

        decision = client.compose_manual_reply(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
            user_request="Tell them I will handle it.",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.reply_markdown, "On it.")

    def test_manual_request_prompt_marks_gif_as_required_when_requested(self) -> None:
        prompt = _build_manual_request_prompt(
            self.identity,
            self.context,
            "Reply with a gif if it fits.",
            available_gifs=[],
        )

        self.assertIn("Operator GIF requirement: required", prompt)

    def test_manual_request_prompt_marks_gif_as_optional_when_not_requested(self) -> None:
        prompt = _build_manual_request_prompt(
            self.identity,
            self.context,
            "Reply briefly and disagree.",
            available_gifs=[],
        )

        self.assertIn("Operator GIF requirement: optional.", prompt)

    def test_chat_returns_plain_content(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"content": "Private operator response."}},
            {"done": True},
        ]
        client.http = transport

        content = client.chat(
            model="qwen3",
            system_prompt="Private prompt",
            messages=[{"role": "user", "content": "Hello"}],
        )

        self.assertEqual(content, "Private operator response.")

    def test_chat_stream_yields_incremental_content(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
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

    def test_decide_emits_global_thinking_events(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion", "thinking"],
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"thinking": "Inspect context. "}},
            {"message": {"thinking": "Draft response. "}},
            {"message": {"content": '{"action":"reply","reply_markdown":"Done.","reason":"Helpful"}'}},
            {"done": True},
        ]
        client.http = transport
        events: list[ThinkingEvent] = []
        client.set_thinking_callback(events.append)

        decision = client.decide(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
        )

        self.assertEqual(decision.reply_markdown, "Done.")
        self.assertEqual(
            [(event.kind, event.chunk) for event in events],
            [
                ("start", ""),
                ("chunk", "Inspect context. "),
                ("chunk", "Draft response. "),
                ("end", ""),
            ],
        )

    def test_supports_thinking_uses_show_model_capabilities(self) -> None:
        client = OllamaClient("http://localhost:11434")
        client.http = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion", "thinking"],
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                }
            }
        )

        self.assertTrue(client.supports_thinking("qwen3"))
        method, path, kwargs = client.http.calls[0]
        self.assertEqual((method, path), ("POST", "/show"))
        self.assertEqual(kwargs["json_body"], {"model": "qwen3"})

    def test_chat_stream_yields_reasoning_chunks_for_thinking_models(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion", "thinking"],
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"thinking": "Plan "}, "done": False},
            {"message": {"thinking": "steps. "}, "done": False},
            {"message": {"content": "Final answer."}, "done": False},
            {"done": True},
        ]
        client.http = transport
        chunks: list[str] = []
        thinking_chunks: list[str] = []

        content = client.chat_stream(
            model="qwen3",
            system_prompt="Private prompt",
            messages=[{"role": "user", "content": "Hello"}],
            on_chunk=chunks.append,
            on_thinking_chunk=thinking_chunks.append,
        )

        self.assertEqual(content, "Final answer.")
        self.assertEqual(chunks, ["Final answer."])
        self.assertEqual(thinking_chunks, ["Plan ", "steps. "])
        method, path, kwargs = transport.stream_calls[0]
        self.assertEqual((method, path), ("POST", "/chat"))
        self.assertTrue(kwargs["json_body"]["stream"])
        self.assertTrue(kwargs["json_body"]["think"])

    def test_summarize_activity_returns_markdown_summary(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"content": "Summary sentence.\n\n## Recent Activity\n- Did a thing."}},
            {"done": True},
        ]
        client.http = transport

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
        method, path, kwargs = transport.stream_calls[0]
        self.assertEqual((method, path), ("POST", "/chat"))
        self.assertTrue(kwargs["json_body"]["stream"])
