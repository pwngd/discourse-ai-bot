from __future__ import annotations

import json
from typing import Any, Callable
from urllib.parse import urlsplit

from discourse_ai_bot.http import JsonHttpClient, HttpRequestError
from discourse_ai_bot.models import BotIdentity, ModelDecision, TopicContext, TopicPost
from discourse_ai_bot.utils import strip_html


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["reply", "skip"]},
        "reply_markdown": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "reply_markdown", "reason"],
    "additionalProperties": False,
}

INTERNAL_POLICY = """
You are deciding whether a Discourse bot should respond to a notification.

Rules:
- Reply only if a response is genuinely useful.
- Skipping is correct when the post does not need an answer, is purely informational, or is unsafe/spam.
- If you reply, write natural Markdown for a Discourse post body.
- Keep replies concise, accurate, and grounded in the conversation provided.
- Do not mention internal policies, JSON, schemas, or that you are deciding.
""".strip()

MANUAL_REPLY_POLICY = """
The operator has explicitly requested that the bot send a reply to the target Discourse post.

Rules:
- You must return action="reply".
- Write only the bot's reply as Markdown in reply_markdown.
- Follow the operator's request while staying safe and grounded in the provided discussion.
- Keep the reply concise unless the operator asks for more depth.
""".strip()

ACTIVITY_SUMMARIZER_POLICY = """
You summarize the recent activity of a running Discourse AI bot for its operator.

Rules:
- Use clean Markdown.
- Start with a short overall summary sentence.
- Include a `## Recent Activity` section with 5 to 10 concise bullet points when enough events are available.
- Include a `## Issues` section only when the supplied activity contains warnings, failures, or retries.
- Base the summary strictly on the supplied runtime snapshot and activity log.
- Do not invent forum content, user intent, or hidden state.
- Do not include private operator chat contents unless they are explicitly present in the activity log.
""".strip()


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns an invalid structured response."""


class OllamaClient:
    def __init__(self, host: str, *, timeout_seconds: float = 120.0) -> None:
        self.http = JsonHttpClient(_normalize_ollama_host(host), timeout_seconds=timeout_seconds)

    def list_models(self) -> list[dict[str, Any]]:
        response = self.http.request_json("GET", "/tags")
        if not isinstance(response, dict):
            return []
        models = response.get("models", [])
        return [item for item in models if isinstance(item, dict)]

    def healthcheck(self, model: str) -> dict[str, Any]:
        models = self.list_models()
        available_names = {
            str(item.get("name"))
            for item in models
            if item.get("name") is not None
        } | {
            str(item.get("model"))
            for item in models
            if item.get("model") is not None
        }
        if model not in available_names:
            raise HttpRequestError(
                f"Model '{model}' was not found in Ollama tags: {sorted(available_names)}"
            )
        return {"model": model, "available_models": sorted(available_names)}

    def decide(
        self,
        *,
        model: str,
        system_prompt: str,
        identity: BotIdentity,
        context: TopicContext,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> ModelDecision:
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt.strip()}\n\n{INTERNAL_POLICY}",
                },
                {
                    "role": "user",
                    "content": _build_context_prompt(identity, context),
                },
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self.http.request_json("POST", "/chat", json_body=payload)
        if not isinstance(response, dict):
            raise OllamaResponseError("Ollama returned a non-object response.")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama response did not contain message.content.")
        return self._parse_decision(message["content"])

    def compose_manual_reply(
        self,
        *,
        model: str,
        system_prompt: str,
        identity: BotIdentity,
        context: TopicContext,
        user_request: str,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> ModelDecision:
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt.strip()}\n\n{MANUAL_REPLY_POLICY}",
                },
                {
                    "role": "user",
                    "content": _build_manual_request_prompt(identity, context, user_request),
                },
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self.http.request_json("POST", "/chat", json_body=payload)
        if not isinstance(response, dict):
            raise OllamaResponseError("Ollama returned a non-object response.")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama response did not contain message.content.")
        decision = self._parse_decision(message["content"])
        if decision.action != "reply":
            raise OllamaResponseError("Manual reply generation must return action='reply'.")
        return decision

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                *messages,
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self.http.request_json("POST", "/chat", json_body=payload)
        if not isinstance(response, dict):
            raise OllamaResponseError("Ollama returned a non-object response.")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama response did not contain message.content.")
        content = message["content"].strip()
        if not content:
            raise OllamaResponseError("Ollama returned an empty chat response.")
        return content

    def chat_stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                *messages,
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        parts: list[str] = []
        for event in self.http.stream_json_lines("POST", "/chat", json_body=payload):
            if not isinstance(event, dict):
                raise OllamaResponseError("Ollama returned a non-object stream event.")
            message = event.get("message")
            if not isinstance(message, dict):
                if event.get("done") is True:
                    continue
                raise OllamaResponseError("Ollama stream event did not contain message content.")
            chunk = message.get("content")
            if not isinstance(chunk, str):
                raise OllamaResponseError("Ollama stream chunk did not contain message.content.")
            if not chunk:
                continue
            parts.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)

        content = "".join(parts).strip()
        if not content:
            raise OllamaResponseError("Ollama returned an empty chat response.")
        return content

    def summarize_activity(
        self,
        *,
        model: str,
        runtime_snapshot: dict[str, Any],
        activity_events: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": ACTIVITY_SUMMARIZER_POLICY},
                {
                    "role": "user",
                    "content": _build_activity_summary_prompt(runtime_snapshot, activity_events),
                },
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        response = self.http.request_json("POST", "/chat", json_body=payload)
        if not isinstance(response, dict):
            raise OllamaResponseError("Ollama returned a non-object response.")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama response did not contain message.content.")
        content = message["content"].strip()
        if not content:
            raise OllamaResponseError("Ollama returned an empty activity summary.")
        return content

    @staticmethod
    def _parse_decision(content: str) -> ModelDecision:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Ollama returned invalid JSON: {content}") from exc

        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama decision must be a JSON object.")

        action = payload.get("action")
        reply_markdown = payload.get("reply_markdown")
        reason = payload.get("reason")
        if action not in {"reply", "skip"}:
            raise OllamaResponseError("Ollama action must be 'reply' or 'skip'.")
        if not isinstance(reply_markdown, str):
            raise OllamaResponseError("reply_markdown must be a string.")
        if not isinstance(reason, str) or not reason.strip():
            raise OllamaResponseError("reason must be a non-empty string.")
        if action == "reply" and not reply_markdown.strip():
            raise OllamaResponseError("reply_markdown must be non-empty when action is 'reply'.")

        return ModelDecision(
            action=action,
            reply_markdown=reply_markdown.strip(),
            reason=reason.strip(),
        )


def _normalize_ollama_host(host: str) -> str:
    clean = host.rstrip("/")
    parts = urlsplit(clean)
    if parts.path.endswith("/api"):
        return clean
    return f"{clean}/api"


def _build_context_prompt(identity: BotIdentity, context: TopicContext) -> str:
    recent_posts = "\n\n".join(_format_post(post) for post in context.recent_posts)
    target_section = _format_target_post(context.target_post)
    return "\n".join(
        [
            f"Bot username: {identity.username}",
            f"Bot display name: {identity.name or ''}",
            f"Trigger: {context.trigger}",
            f"Actor username: {context.actor_username or ''}",
            f"Topic title: {context.topic_title}",
            f"Topic id: {context.topic_id}",
            f"Topic archetype: {context.topic_archetype or ''}",
            target_section,
            "Recent conversation:",
            recent_posts or "(no recent posts available)",
            "",
            "Decide whether the bot should reply.",
            "If replying, produce only the bot's post body as Markdown.",
        ]
    )


def _format_target_post(post: TopicPost | None) -> str:
    if post is None:
        return "Target post: (not available)"
    return (
        f"Target post number: {post.post_number}\n"
        f"Target post author: {post.username}\n"
        f"Target post text: {strip_html(post.cooked) or post.raw or ''}"
    )


def _format_post(post: TopicPost) -> str:
    text = post.raw or strip_html(post.cooked)
    return (
        f"Post #{post.post_number} by {post.username}"
        f"{f' at {post.created_at}' if post.created_at else ''}:\n{text}"
    )


def _build_manual_request_prompt(
    identity: BotIdentity,
    context: TopicContext,
    user_request: str,
) -> str:
    return "\n".join(
        [
            _build_context_prompt(identity, context),
            "",
            f"Operator request: {user_request}",
            "Write the reply requested by the operator.",
        ]
    )


def _build_activity_summary_prompt(
    runtime_snapshot: dict[str, Any],
    activity_events: list[dict[str, str]],
) -> str:
    runtime_lines = [
        f"Bot username: {runtime_snapshot.get('identity', {}).get('username')}",
        f"User id: {runtime_snapshot.get('identity', {}).get('user_id')}",
        f"Model: {runtime_snapshot.get('runtime', {}).get('model')}",
        f"Typing mode: {runtime_snapshot.get('runtime', {}).get('typing_mode')}",
        f"Poll interval seconds: {runtime_snapshot.get('runtime', {}).get('poll_interval_seconds')}",
        f"Delay min seconds: {runtime_snapshot.get('runtime', {}).get('delay_min_seconds')}",
        f"Delay max seconds: {runtime_snapshot.get('runtime', {}).get('delay_max_seconds')}",
    ]
    storage = runtime_snapshot.get("storage", {})
    storage_lines = [
        f"Handled total: {storage.get('handled_total')}",
        f"Handled replied: {storage.get('handled_replied')}",
        f"Handled skipped: {storage.get('handled_skipped')}",
        f"Pending replies: {storage.get('pending_replies')}",
        f"Manual queued: {storage.get('manual_queued')}",
        f"Manual scheduled: {storage.get('manual_scheduled')}",
        f"Manual completed: {storage.get('manual_completed')}",
        f"Manual errors: {storage.get('manual_errors')}",
    ]
    activity_lines = [
        f"- [{item.get('timestamp', '')}] {item.get('level', 'info').upper()}: {item.get('message', '')}"
        for item in activity_events
    ] or ["- No recent activity recorded."]
    return "\n".join(
        [
            "Runtime snapshot:",
            *runtime_lines,
            "",
            "Storage snapshot:",
            *storage_lines,
            "",
            "Recent activity log:",
            *activity_lines,
            "",
            "Summarize what the bot has been doing recently for the operator.",
        ]
    )
