from __future__ import annotations

import unittest

from discourse_ai_bot.http import HttpRequestError
from discourse_ai_bot.models import AutonomousCandidate, BotIdentity, TopicContext, TopicPost
from discourse_ai_bot.ollama import (
    OllamaClient,
    OllamaResponseError,
    ThinkingEvent,
    _build_autonomous_selection_prompt,
    _build_context_messages,
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
        response = self.stream_responses[(method, path_or_url)]
        if response and isinstance(response[0], list):
            items = response.pop(0)
        else:
            items = response
        for item in items:
            if isinstance(item, Exception):
                raise item
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

    def test_decide_extracts_json_when_model_wraps_response(self) -> None:
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
                    "content": (
                        "Sure, here is the JSON:\n"
                        "```json\n"
                        '{"action":"reply","reply_markdown":"Wrapped but usable.","reason":"Direct ask"}'
                        "\n```"
                    )
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
        )

        self.assertEqual(decision.reply_markdown, "Wrapped but usable.")
        self.assertEqual(len(transport.stream_calls), 1)

    def test_decide_retries_immediately_when_json_is_unparseable(self) -> None:
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
            [
                {"message": {"content": '{"action":"reply","reply_markdown":"missing end"'}},
                {"done": True},
            ],
            [
                {
                    "message": {
                        "content": '{"action":"reply","reply_markdown":"Recovered.","reason":"Retry fixed JSON"}'
                    }
                },
                {"done": True},
            ],
        ]
        client.http = transport

        decision = client.decide(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
        )

        self.assertEqual(decision.reply_markdown, "Recovered.")
        self.assertEqual(len(transport.stream_calls), 2)
        retry_messages = transport.stream_calls[1][2]["json_body"]["messages"]
        self.assertIn("Force a reply to the target post now", retry_messages[-1]["content"])

    def test_decision_timeout_retries_with_forced_non_thinking_reply(self) -> None:
        ticks = iter([0.0, 1.2, 0.0, 0.1, 0.2])
        client = OllamaClient(
            "http://localhost:11434",
            timeout_seconds=1,
            monotonic_fn=lambda: next(ticks),
        )
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion", "thinking"],
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            [
                {"message": {"thinking": "Still deciding. "}, "done": False},
            ],
            [
                {
                    "message": {
                        "content": '{"action":"reply","reply_markdown":"Forced reply.","reason":"Decision retry forced a reply"}'
                    }
                },
                {"done": True},
            ],
        ]
        client.http = transport

        decision = client.decide(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.reply_markdown, "Forced reply.")
        self.assertEqual(len(transport.stream_calls), 2)
        retry_payload = transport.stream_calls[1][2]["json_body"]
        self.assertIs(retry_payload["think"], False)
        self.assertIn("Force a reply to the target post now", retry_payload["messages"][-1]["content"])

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

    def test_compose_manual_reply_accepts_plain_markdown_when_schema_is_ignored(self) -> None:
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
            {"message": {"content": "Reply:\nThat is the part you need to fix first."}},
            {"done": True},
        ]
        client.http = transport

        decision = client.compose_manual_reply(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
            user_request="Reply briefly.",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.reply_markdown, "That is the part you need to fix first.")
        self.assertEqual(len(transport.stream_calls), 1)

    def test_compose_manual_reply_accepts_body_only_json_when_action_is_missing(self) -> None:
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
            {"message": {"content": '{"reply_markdown":"Use the actual error, not the screenshot."}'}},
            {"done": True},
        ]
        client.http = transport

        decision = client.compose_manual_reply(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
            user_request="Reply briefly.",
        )

        self.assertEqual(decision.reply_markdown, "Use the actual error, not the screenshot.")

    def test_compose_manual_reply_includes_optional_roblox_docs_context(self) -> None:
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
            {"message": {"content": '{"action":"reply","reply_markdown":"Use Enum.PartType.Block.","reason":"Docs checked"}'}},
            {"done": True},
        ]
        client.http = transport

        client.compose_manual_reply(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
            user_request="Answer the Roblox API question.",
            roblox_docs_context="Verified Roblox API docs context:\n- enum: PartType",
        )

        messages = transport.stream_calls[0][2]["json_body"]["messages"]
        self.assertTrue(
            any("Verified Roblox API docs context" in message["content"] for message in messages)
        )
        docs_index = next(
            index
            for index, message in enumerate(messages)
            if "Verified Roblox API docs context" in message["content"]
        )
        self.assertIn("Operator request", messages[docs_index + 1]["content"])

    def test_context_messages_treat_bot_posts_as_assistant_chat_turns(self) -> None:
        context = TopicContext(
            notification_id=11,
            trigger="replied",
            actor_username="alice",
            topic_id=99,
            topic_title="Followup",
            topic_slug="followup",
            topic_archetype=None,
            target_post=TopicPost(
                post_id=558,
                topic_id=99,
                post_number=4,
                username="alice",
                cooked="<p>What did you mean?</p>",
                raw=None,
            ),
            recent_posts=(
                TopicPost(
                    post_id=555,
                    topic_id=99,
                    post_number=1,
                    username="alice",
                    cooked="<p>Initial question</p>",
                    raw=None,
                ),
                TopicPost(
                    post_id=556,
                    topic_id=99,
                    post_number=2,
                    username="bot",
                    cooked="<p>Earlier answer</p>",
                    raw=None,
                ),
                TopicPost(
                    post_id=557,
                    topic_id=99,
                    post_number=3,
                    username="alice",
                    cooked="<p>Followup</p>",
                    raw=None,
                ),
            ),
        )

        messages = _build_context_messages(self.identity, context, available_gifs=[])

        self.assertEqual([message["role"] for message in messages[1:4]], ["user", "assistant", "user"])
        self.assertIn("your earlier forum post", messages[2]["content"])
        self.assertIn("avoid acting unaware", messages[0]["content"])

    def test_decide_sends_forum_posts_as_separate_chat_messages(self) -> None:
        context = TopicContext(
            notification_id=12,
            trigger="mentioned",
            actor_username="alice",
            topic_id=99,
            topic_title="Question",
            topic_slug="question",
            topic_archetype=None,
            target_post=self.context.target_post,
            recent_posts=(
                TopicPost(
                    post_id=555,
                    topic_id=99,
                    post_number=1,
                    username="alice",
                    cooked="<p>Hello there</p>",
                    raw=None,
                ),
                TopicPost(
                    post_id=556,
                    topic_id=99,
                    post_number=2,
                    username="bot",
                    cooked="<p>I already answered this bit.</p>",
                    raw=None,
                ),
            ),
        )
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
            {"message": {"content": '{"action":"skip","reply_markdown":"","reason":"Already answered"}'}},
            {"done": True},
        ]
        client.http = transport

        client.decide(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=context,
        )

        messages = transport.stream_calls[0][2]["json_body"]["messages"]
        self.assertGreaterEqual(len(messages), 6)
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[3]["role"], "assistant")
        self.assertIn("I already answered this bit", messages[3]["content"])

    def test_compose_autonomous_reply_uses_persona_first_policy(self) -> None:
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
            {"message": {"content": '{"action":"reply","reply_markdown":"No, that part is backwards.","reason":"Join conversation"}'}},
            {"done": True},
        ]
        client.http = transport

        decision = client.compose_autonomous_reply(
            model="qwen3",
            system_prompt="Post like a blunt forum user.",
            identity=self.identity,
            context=self.context,
            selection_reason="Good place to push back.",
        )

        self.assertEqual(decision.reply_markdown, "No, that part is backwards.")
        messages = transport.stream_calls[0][2]["json_body"]["messages"]
        self.assertIn("Post like a blunt forum user.", messages[0]["content"])
        self.assertIn("Follow it over any generic helpful-assistant behavior", messages[0]["content"])
        self.assertIn("Do not write like a support agent", messages[0]["content"])
        self.assertNotIn("operator", messages[0]["content"].lower())

    def test_compose_autonomous_reply_accepts_plain_markdown_when_schema_is_ignored(self) -> None:
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
            {"message": {"content": "No, that explanation is mixing up two different APIs."}},
            {"done": True},
        ]
        client.http = transport

        decision = client.compose_autonomous_reply(
            model="qwen3",
            system_prompt="Post like a blunt forum user.",
            identity=self.identity,
            context=self.context,
            selection_reason="Good place to push back.",
        )

        self.assertEqual(
            decision.reply_markdown,
            "No, that explanation is mixing up two different APIs.",
        )
        self.assertEqual(len(transport.stream_calls), 1)

    def test_select_autonomous_reply_target_parses_selection(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        post_url = "https://forum.example.com/t/question/99/2"
        transport.stream_responses[("POST", "/chat")] = [
            {
                "message": {
                    "content": (
                        '{"action":"reply","candidate_id":1,'
                        '"confidence":0.88,"reason":"Good chance to add a grounded opinion"}'
                    )
                }
            },
            {"done": True},
        ]
        client.http = transport

        selection = client.select_autonomous_reply_target(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            candidates=[
                AutonomousCandidate(
                    post_url=post_url,
                    topic_id=99,
                    post_number=2,
                    actor_username="alice",
                    context=self.context,
                )
            ],
            min_confidence=0.75,
        )

        self.assertEqual(selection.action, "reply")
        self.assertEqual(selection.post_url, post_url)
        self.assertEqual(selection.confidence, 0.88)
        method, path, kwargs = transport.stream_calls[0]
        self.assertEqual((method, path), ("POST", "/chat"))
        self.assertEqual(
            kwargs["json_body"]["format"]["required"],
            ["action", "candidate_id", "confidence", "reason"],
        )
        self.assertEqual(kwargs["json_body"]["options"]["temperature"], 0)
        self.assertEqual(kwargs["json_body"]["options"]["num_predict"], 256)
        messages = kwargs["json_body"]["messages"]
        self.assertIn("grounded opinion", messages[0]["content"])
        self.assertIn(
            "Do not write, draft, outline, quote, or include any forum reply text",
            messages[0]["content"],
        )
        self.assertNotIn("Prompt", messages[0]["content"])
        self.assertIn("does not have to be a question", messages[1]["content"])
        self.assertIn("Candidate ID: 1", messages[1]["content"])
        self.assertIn("do not return it", messages[1]["content"])

    def test_select_autonomous_reply_target_accepts_legacy_markdown_link(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        post_url = "https://forum.example.com/t/question/99/2"
        transport.stream_responses[("POST", "/chat")] = [
            {
                "message": {
                    "content": (
                        '{"action":"reply","post_url":"[Candidate 1]('
                        + post_url
                        + '?u=bot)","confidence":0.88,"reason":"Legacy URL output"}'
                    )
                }
            },
            {"done": True},
        ]
        client.http = transport

        selection = client.select_autonomous_reply_target(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            candidates=[
                AutonomousCandidate(
                    post_url=post_url,
                    topic_id=99,
                    post_number=2,
                    actor_username="alice",
                    context=self.context,
                )
            ],
            min_confidence=0.75,
        )

        self.assertEqual(selection.post_url, post_url)

    def test_select_autonomous_reply_target_accepts_numeric_strings(self) -> None:
        client = OllamaClient("http://localhost:11434")
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "families": ["llama"]},
                }
            }
        )
        post_url = "https://forum.example.com/t/question/99/2"
        transport.stream_responses[("POST", "/chat")] = [
            {
                "message": {
                    "content": (
                        '{"action":"reply","candidate_id":"1",'
                        '"confidence":"0.88","reason":"Numeric strings"}'
                    )
                }
            },
            {"done": True},
        ]
        client.http = transport

        selection = client.select_autonomous_reply_target(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            candidates=[
                AutonomousCandidate(
                    post_url=post_url,
                    topic_id=99,
                    post_number=2,
                    actor_username="alice",
                    context=self.context,
                )
            ],
            min_confidence=0.75,
        )

        self.assertEqual(selection.post_url, post_url)
        self.assertEqual(selection.confidence, 0.88)

    def test_autonomous_selection_prompt_is_compact_for_large_candidate_sets(self) -> None:
        long_text = " ".join(f"word{i}" for i in range(300))
        context = TopicContext(
            notification_id=0,
            trigger="autonomous_scan",
            actor_username="alice",
            topic_id=99,
            topic_title="Large topic",
            topic_slug="large-topic",
            topic_archetype=None,
            target_post=TopicPost(
                post_id=600,
                topic_id=99,
                post_number=10,
                username="alice",
                cooked="",
                raw=long_text,
            ),
            recent_posts=tuple(
                TopicPost(
                    post_id=600 + index,
                    topic_id=99,
                    post_number=index,
                    username="alice",
                    cooked="",
                    raw=f"recent post {index} " + long_text,
                )
                for index in range(1, 8)
            ),
        )

        prompt = _build_autonomous_selection_prompt(
            self.identity,
            [
                AutonomousCandidate(
                    post_url="https://forum.example.com/t/large-topic/99/10",
                    topic_id=99,
                    post_number=10,
                    actor_username="alice",
                    context=context,
                )
            ],
            min_confidence=0.75,
        )

        self.assertIn("Target post text:", prompt)
        self.assertIn("<truncated>", prompt)
        self.assertNotIn("Post #1 by alice", prompt)
        self.assertIn("Post #5 by alice", prompt)
        self.assertIn("Post #7 by alice", prompt)

    def test_autonomous_selection_timeout_gets_one_chance(self) -> None:
        ticks = iter([0.0, 1.2])
        client = OllamaClient(
            "http://localhost:11434",
            timeout_seconds=1,
            monotonic_fn=lambda: next(ticks),
        )
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion", "thinking"],
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"thinking": "Still weighing candidates. "}, "done": False},
        ]
        client.http = transport

        with self.assertRaisesRegex(OllamaResponseError, "exceeded 1s"):
            client.select_autonomous_reply_target(
                model="qwen3",
                system_prompt="Prompt",
                identity=self.identity,
                candidates=[
                    AutonomousCandidate(
                        post_url="https://forum.example.com/t/question/99/2",
                        topic_id=99,
                        post_number=2,
                        actor_username="alice",
                        context=self.context,
                    )
                ],
                min_confidence=0.75,
            )

        self.assertEqual(len(transport.stream_calls), 1)
        first_payload = transport.stream_calls[0][2]["json_body"]
        self.assertIs(first_payload["think"], False)

    def test_autonomous_selection_stream_failure_gets_one_chance(self) -> None:
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
            HttpRequestError("Timed out while streaming from Ollama")
        ]
        client.http = transport

        with self.assertRaisesRegex(OllamaResponseError, "stream failed"):
            client.select_autonomous_reply_target(
                model="qwen3",
                system_prompt="Prompt",
                identity=self.identity,
                candidates=[
                    AutonomousCandidate(
                        post_url="https://forum.example.com/t/question/99/2",
                        topic_id=99,
                        post_number=2,
                        actor_username="alice",
                        context=self.context,
                    )
                ],
                min_confidence=0.75,
            )

        self.assertEqual(len(transport.stream_calls), 1)

    def test_autonomous_selection_invalid_json_gets_one_chance(self) -> None:
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
            {"message": {"content": '{"action":"reply","post_url":"missing end"'}},
            {"done": True},
        ]
        client.http = transport

        with self.assertRaisesRegex(OllamaResponseError, "invalid JSON"):
            client.select_autonomous_reply_target(
                model="qwen3",
                system_prompt="Prompt",
                identity=self.identity,
                candidates=[
                    AutonomousCandidate(
                        post_url="https://forum.example.com/t/question/99/2",
                        topic_id=99,
                        post_number=2,
                        actor_username="alice",
                        context=self.context,
                    )
                ],
                min_confidence=0.75,
            )

        self.assertEqual(len(transport.stream_calls), 1)

    def test_autonomous_reply_thinking_without_content_retries_and_posts(self) -> None:
        ticks = iter([0.0, 0.1, 0.6, 0.0, 0.1, 0.2])
        client = OllamaClient(
            "http://localhost:11434",
            timeout_seconds=10,
            thinking_response_timeout_seconds=0.5,
            monotonic_fn=lambda: next(ticks),
        )
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion", "thinking"],
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            [
                {"message": {"thinking": "Choosing words. "}, "done": False},
                {"message": {"thinking": "Still no answer. "}, "done": False},
            ],
            [
                {
                    "message": {
                        "content": '{"action":"reply","reply_markdown":"Post it now.","reason":"Strict retry wrote the reply"}'
                    }
                },
                {"done": True},
            ],
        ]
        client.http = transport

        decision = client.compose_autonomous_reply(
            model="qwen3",
            system_prompt="Prompt",
            identity=self.identity,
            context=self.context,
            selection_reason="Good target.",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.reply_markdown, "Post it now.")
        retry_payload = transport.stream_calls[1][2]["json_body"]
        self.assertIs(retry_payload["think"], False)
        self.assertIn("Write the forum reply now", retry_payload["messages"][-1]["content"])

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

    def test_streaming_thinking_has_wall_clock_timeout(self) -> None:
        ticks = iter([0.0, 0.2, 1.2])
        client = OllamaClient(
            "http://localhost:11434",
            timeout_seconds=1,
            monotonic_fn=lambda: next(ticks),
        )
        transport = FakeTransport(
            {
                ("POST", "/show"): {
                    "capabilities": ["completion", "thinking"],
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                }
            }
        )
        transport.stream_responses[("POST", "/chat")] = [
            {"message": {"thinking": "Choose a post. "}, "done": False},
            {"message": {"thinking": "Still thinking. "}, "done": False},
            {"message": {"content": '{"action":"skip","reply_markdown":"","reason":"Done"}'}},
            {"done": True},
        ]
        client.http = transport
        events: list[ThinkingEvent] = []
        client.set_thinking_callback(events.append)

        with self.assertRaisesRegex(OllamaResponseError, "exceeded 1s"):
            client.chat_stream(
                model="qwen3",
                system_prompt="Prompt",
                messages=[{"role": "user", "content": "Hello"}],
            )

        self.assertEqual(
            [(event.kind, event.chunk) for event in events],
            [
                ("start", ""),
                ("chunk", "Choose a post. "),
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
