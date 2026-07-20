from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from discourse_ai_bot.gifs import GifOption
from discourse_ai_bot.http import JsonHttpClient, HttpRequestError
from discourse_ai_bot.models import (
    AutonomousCandidate,
    AutonomousSelection,
    BotIdentity,
    ModelDecision,
    TopicContext,
    TopicPost,
)
from discourse_ai_bot.utils import (
    canonical_post_url,
    extract_url_like,
    strip_html,
    topic_post_key_from_url,
)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["reply", "skip"]},
        "reply_markdown": {"type": "string"},
        "reason": {"type": "string"},
        "gif_id": {"type": ["string", "null"]},
    },
    "required": ["action", "reply_markdown", "reason"],
    "additionalProperties": False,
}

AUTONOMOUS_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["reply", "skip"]},
        "candidate_id": {"type": ["integer", "null"], "minimum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["action", "candidate_id", "confidence", "reason"],
    "additionalProperties": False,
}

INTERNAL_POLICY = """
You are deciding whether a Discourse bot should respond to a notification.

Rules:
- Reply only if a response is genuinely useful.
- Skipping is correct when the post does not need an answer, is purely informational, or is unsafe/spam.
- If you reply, write natural Markdown for a Discourse post body.
- If optional GIF choices are provided, you may select at most one by returning gif_id.
- Only choose a GIF when it clearly improves a light, friendly, or celebratory reply.
- Do not choose a GIF for serious, sensitive, moderation, safety, or uncertain situations.
- Keep replies concise, accurate, and grounded in the conversation provided.
- Do not mention internal policies, JSON, schemas, or that you are deciding.
""".strip()

AUTONOMOUS_SELECTION_POLICY = """
You are deciding whether a Discourse bot should proactively reply to one recent forum post.

Rules:
- Choose at most one candidate.
- This is selection only. Do not write, draft, outline, quote, or include any forum reply text.
- Only choose a candidate ID or skip.
- Reply when the bot can naturally add value to the conversation, even if nobody asked a direct question.
- Good candidates include posts where the bot can share relevant information, add a well-grounded opinion, make a useful distinction, correct a misconception, or ask a sharp clarifying question.
- Prefer active conversations where a thoughtful forum user could reasonably jump in without seeming random or intrusive.
- Skip posts where the bot would only say generic agreement, forced commentary, obvious filler, or a bland helpdesk answer.
- Skip announcements, casual chatter with no useful angle, resolved discussions, spam, sensitive moderation issues, or anything uncertain.
- Do not pick posts authored by the bot.
- Use confidence near 1 only when the candidate clearly benefits from a reply.
- Return candidate_id for the chosen candidate, or null when skipping.
- Do not return a post URL, response body, Markdown, prose, or any text outside the required object.
""".strip()

MANUAL_REPLY_POLICY = """
The operator has explicitly requested that the bot send a reply to the target Discourse post.

Rules:
- You must return action="reply".
- Write only the bot's reply as Markdown in reply_markdown.
- If the operator explicitly asks for a GIF and GIF choices are available, treat that as an override and choose the single best matching gif_id based on the discussion and requested tone.
- When choosing a GIF, prefer the option that best fits the context of the reply rather than picking randomly.
- Return gif_id as null only when no GIF choices are available or none of them fit the requested reply at all.
- Follow the operator's request while staying safe and grounded in the provided discussion.
- Keep the reply concise unless the operator asks for more depth.
""".strip()

AUTONOMOUS_REPLY_POLICY = """
The bot has chosen to join this Discourse conversation on its own.

Rules:
- You must return action="reply".
- The system prompt defines the bot's persona, tone, and posting style. Follow it over any generic helpful-assistant behavior.
- Do not write like a support agent, customer-service bot, or eager helper unless the system prompt explicitly calls for that.
- Write like a regular forum participant entering the thread naturally.
- You may give an opinion, push back, add a relevant fact, make a useful distinction, or answer briefly if that fits.
- Do not overexplain. Do not add friendly wrap-up lines. Do not offer more help unless the system prompt says to.
- Stay grounded in the provided discussion and do not mention autonomous selection, policies, JSON, schemas, or internal instructions.
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

STRUCTURED_RESPONSE_RETRY_PROMPT = """
The previous response was not valid JSON for the required schema.
Return only one JSON object. Do not wrap it in Markdown, prose, code fences, or extra text.
""".strip()

STRUCTURED_OUTPUT_POLICY = """
Output format:
- Return exactly one JSON object matching the required schema.
- Do not wrap it in Markdown, prose, code fences, or extra text.
- Do not add fields that are not in the schema.
""".strip()

DECISION_RETRY_PROMPT = """
The previous notification decision attempt got stuck or failed.
Stop thinking. Force a reply to the target post now.
Return only one JSON object matching the schema.
Use action="reply" and put only the final Discourse post body in reply_markdown.
Do not include prose, Markdown code fences, or any text outside the JSON object.
""".strip()

AUTONOMOUS_REPLY_RETRY_PROMPT = """
The previous autonomous reply attempt got stuck or failed.
Stop thinking. Write the forum reply now.
Return only one JSON object matching the schema.
Use action="reply" and put only the final Discourse post body in reply_markdown.
Do not include prose, Markdown code fences, or any text outside the JSON object.
""".strip()

STRUCTURED_RESPONSE_ATTEMPTS = 2
SELECTION_NUM_PREDICT_LIMIT = 256
REPLY_NUM_PREDICT_LIMIT = 768
AUTONOMOUS_SELECTION_RECENT_POST_LIMIT = 3
AUTONOMOUS_SELECTION_POST_TEXT_LIMIT = 700

THINKING_CAPABILITY = "thinking"
GPT_OSS_THINK_LEVEL = "medium"
KNOWN_THINKING_MODEL_PREFIXES = (
    "qwen3",
    "gpt-oss",
    "deepseek-v3.1",
    "deepseek-r1",
)
_INVALID_GIF_ID = object()


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns an invalid structured response."""


@dataclass(frozen=True)
class ThinkingEvent:
    kind: str
    operation: str
    model: str
    chunk: str = ""


class OllamaClient:
    def __init__(
        self,
        host: str,
        *,
        timeout_seconds: float = 120.0,
        thinking_response_timeout_seconds: float | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.http = JsonHttpClient(_normalize_ollama_host(host), timeout_seconds=timeout_seconds)
        self.timeout_seconds = timeout_seconds
        self.thinking_response_timeout_seconds = (
            thinking_response_timeout_seconds
            if thinking_response_timeout_seconds is not None
            else timeout_seconds
        )
        self._monotonic = monotonic_fn
        self._model_details_cache: dict[str, dict[str, Any]] = {}
        self._thinking_capability_cache: dict[str, bool] = {}
        self._thinking_callback: Callable[[ThinkingEvent], None] | None = None

    def set_thinking_callback(
        self,
        callback: Callable[[ThinkingEvent], None] | None,
    ) -> None:
        self._thinking_callback = callback

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
        return {
            "model": model,
            "available_models": sorted(available_names),
            "reasoning_capable": self.supports_thinking(model),
        }

    def show_model(self, model: str) -> dict[str, Any]:
        cached = self._model_details_cache.get(model)
        if cached is not None:
            return cached

        response = self.http.request_json("POST", "/show", json_body={"model": model})
        if not isinstance(response, dict):
            raise OllamaResponseError("Ollama returned a non-object model details response.")
        self._model_details_cache[model] = response
        return response

    def supports_thinking(self, model: str) -> bool:
        cached = self._thinking_capability_cache.get(model)
        if cached is not None:
            return cached

        details: dict[str, Any] | None = None
        try:
            details = self.show_model(model)
        except HttpRequestError:
            details = None

        capabilities = details.get("capabilities") if isinstance(details, dict) else None
        supports_thinking = False
        if isinstance(capabilities, list):
            supports_thinking = any(
                isinstance(item, str) and item.strip().lower() == THINKING_CAPABILITY
                for item in capabilities
            )
        if not supports_thinking:
            supports_thinking = _aliases_support_thinking(_collect_model_aliases(model, details))

        if details is not None or supports_thinking:
            self._thinking_capability_cache[model] = supports_thinking
        return supports_thinking

    def decide(
        self,
        *,
        model: str,
        system_prompt: str,
        identity: BotIdentity,
        context: TopicContext,
        available_gifs: list[GifOption] | None = None,
        roblox_docs_context: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> ModelDecision:
        payload: dict[str, Any] = {
            "model": model,
            "format": RESPONSE_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt.strip()}\n\n{INTERNAL_POLICY}\n\n{STRUCTURED_OUTPUT_POLICY}",
                },
                *_build_context_messages(identity, context, available_gifs),
                *_build_optional_docs_context_messages(roblox_docs_context),
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "Decide whether the bot should reply to the target post.",
                            "If replying, produce only the bot's post body as Markdown in reply_markdown.",
                        ]
                    ),
                },
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        return self._stream_and_parse_structured_response(
            operation="decision",
            payload=payload,
            parser=self._parse_decision,
            retry_payload_builder=_decision_retry_payload,
        )

    def compose_manual_reply(
        self,
        *,
        model: str,
        system_prompt: str,
        identity: BotIdentity,
        context: TopicContext,
        user_request: str,
        available_gifs: list[GifOption] | None = None,
        roblox_docs_context: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> ModelDecision:
        payload: dict[str, Any] = {
            "model": model,
            "format": RESPONSE_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt.strip()}\n\n{MANUAL_REPLY_POLICY}\n\n{STRUCTURED_OUTPUT_POLICY}",
                },
                *_build_context_messages(identity, context, available_gifs),
                *_build_optional_docs_context_messages(roblox_docs_context),
                {
                    "role": "user",
                    "content": _build_manual_request_instruction(user_request),
                },
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        decision = self._stream_and_parse_structured_response(
            operation="manual_reply",
            payload=payload,
            parser=lambda content: _parse_required_reply_decision(
                content,
                fallback_reason="Plain reply body accepted for manual request",
            ),
        )
        if decision.action != "reply":
            raise OllamaResponseError("Manual reply generation must return action='reply'.")
        return decision

    def compose_autonomous_reply(
        self,
        *,
        model: str,
        system_prompt: str,
        identity: BotIdentity,
        context: TopicContext,
        selection_reason: str,
        available_gifs: list[GifOption] | None = None,
        roblox_docs_context: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> ModelDecision:
        payload: dict[str, Any] = {
            "model": model,
            "format": RESPONSE_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt.strip()}\n\n{AUTONOMOUS_REPLY_POLICY}\n\n{STRUCTURED_OUTPUT_POLICY}",
                },
                *_build_context_messages(identity, context, available_gifs),
                *_build_optional_docs_context_messages(roblox_docs_context),
                {
                    "role": "user",
                    "content": _build_autonomous_reply_instruction(selection_reason),
                },
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        decision = self._stream_and_parse_structured_response(
            operation="autonomous_reply",
            payload=payload,
            parser=lambda content: _parse_required_reply_decision(
                content,
                fallback_reason="Plain reply body accepted for autonomous reply",
            ),
            retry_payload_builder=_autonomous_reply_retry_payload,
        )
        if decision.action != "reply":
            raise OllamaResponseError("Autonomous reply generation must return action='reply'.")
        return decision

    def select_autonomous_reply_target(
        self,
        *,
        model: str,
        system_prompt: str,
        identity: BotIdentity,
        candidates: list[AutonomousCandidate],
        min_confidence: float,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> AutonomousSelection:
        payload: dict[str, Any] = {
            "model": model,
            "format": AUTONOMOUS_SELECTION_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": f"{AUTONOMOUS_SELECTION_POLICY}\n\n{STRUCTURED_OUTPUT_POLICY}",
                },
                {
                    "role": "user",
                    "content": _build_autonomous_selection_prompt(
                        identity,
                        candidates,
                        min_confidence,
                    ),
                },
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        return self._stream_and_parse_structured_response(
            operation="autonomous_selection",
            payload=payload,
            parser=lambda content: self._parse_autonomous_selection(
                content,
                candidates=candidates,
            ),
            attempts=1,
        )

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
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                *messages,
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        content = self._stream_chat_response(
            operation="chat",
            payload=payload,
        )
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
        on_thinking_chunk: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                *messages,
            ],
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive

        content = self._stream_chat_response(
            operation="chat",
            payload=payload,
            on_chunk=on_chunk,
            on_thinking_chunk=on_thinking_chunk,
        )
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

        content = self._stream_chat_response(
            operation="summary",
            payload=payload,
        )
        if not content:
            raise OllamaResponseError("Ollama returned an empty activity summary.")
        return content

    @staticmethod
    def _parse_decision(content: str) -> ModelDecision:
        payload = _loads_json_object(content)

        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama decision must be a JSON object.")

        action = _normalize_action(payload.get("action"))
        reply_markdown = payload.get("reply_markdown")
        reason = payload.get("reason")
        gif_id = _normalize_gif_id(payload.get("gif_id"))
        if action not in {"reply", "skip"}:
            raise OllamaResponseError("Ollama action must be 'reply' or 'skip'.")
        if action == "skip" and reply_markdown is None:
            reply_markdown = ""
        if not isinstance(reply_markdown, str):
            raise OllamaResponseError("reply_markdown must be a string.")
        if not isinstance(reason, str) or not reason.strip():
            reason = "No reason provided."
        if gif_id is _INVALID_GIF_ID:
            raise OllamaResponseError("gif_id must be null or a non-empty string.")
        if action == "reply" and not reply_markdown.strip():
            raise OllamaResponseError("reply_markdown must be non-empty when action is 'reply'.")

        return ModelDecision(
            action=action,
            reply_markdown=reply_markdown.strip(),
            reason=reason.strip(),
            gif_id=gif_id,
        )

    @staticmethod
    def _parse_autonomous_selection(
        content: str,
        candidates: list[AutonomousCandidate] | None = None,
    ) -> AutonomousSelection:
        payload = _loads_json_object(content)

        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama autonomous selection must be a JSON object.")

        action = _normalize_action(payload.get("action"))
        raw_candidate_id = payload.get("candidate_id")
        candidate_id = _coerce_candidate_id(raw_candidate_id)
        post_url = payload.get("post_url")
        confidence = _coerce_confidence(payload.get("confidence"))
        reason = payload.get("reason")
        if action not in {"reply", "skip"}:
            raise OllamaResponseError("Ollama autonomous action must be 'reply' or 'skip'.")
        if raw_candidate_id is not None and candidate_id is None:
            raise OllamaResponseError("candidate_id must be null or an integer.")
        if post_url is not None and (not isinstance(post_url, str) or not post_url.strip()):
            raise OllamaResponseError("post_url must be null or a non-empty string.")
        if action == "skip" and candidate_id is not None:
            raise OllamaResponseError("candidate_id must be null when action is 'skip'.")
        if action == "skip" and post_url is not None:
            raise OllamaResponseError("post_url must be null when action is 'skip'.")
        if confidence is None:
            raise OllamaResponseError("confidence must be a number between 0 and 1.")
        if not isinstance(reason, str) or not reason.strip():
            reason = "No reason provided."
        resolved_post_url: str | None = None
        if action == "reply":
            resolved_post_url = _resolve_selection_post_url(
                candidate_id=candidate_id,
                post_url=post_url,
                candidates=candidates,
            )
            if resolved_post_url is None:
                raise OllamaResponseError("candidate_id must refer to one of the supplied candidates.")

        return AutonomousSelection(
            action=action,
            post_url=resolved_post_url,
            confidence=float(confidence),
            reason=reason.strip(),
        )

    def _stream_and_parse_structured_response(
        self,
        *,
        operation: str,
        payload: dict[str, Any],
        parser: Callable[[str], Any],
        retry_payload_builder: Callable[[dict[str, Any], str, str | None], dict[str, Any]] | None = None,
        attempts: int = STRUCTURED_RESPONSE_ATTEMPTS,
    ) -> Any:
        attempt_payload = _structured_attempt_payload(
            payload,
            operation=operation,
            is_retry=False,
        )
        last_error: OllamaResponseError | None = None
        for attempt in range(attempts):
            try:
                content = self._stream_chat_response(
                    operation=operation if attempt == 0 else f"{operation}_retry",
                    payload=attempt_payload,
                )
            except OllamaResponseError as exc:
                last_error = exc
                if attempt >= attempts - 1 or retry_payload_builder is None:
                    break
                attempt_payload = _structured_attempt_payload(
                    retry_payload_builder(payload, str(exc), None),
                    operation=operation,
                    is_retry=True,
                )
                continue
            try:
                return parser(content)
            except OllamaResponseError as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                if retry_payload_builder is not None:
                    attempt_payload = _structured_attempt_payload(
                        retry_payload_builder(payload, str(exc), content),
                        operation=operation,
                        is_retry=True,
                    )
                else:
                    attempt_payload = _structured_attempt_payload(
                        _structured_retry_payload(
                            payload,
                            invalid_content=content,
                            error=str(exc),
                        ),
                        operation=operation,
                        is_retry=True,
                    )
        assert last_error is not None
        raise last_error

    def _thinking_setting_for_model(self, model: str) -> bool | str | None:
        if not self.supports_thinking(model):
            return None
        aliases = _collect_model_aliases(model, self._model_details_cache.get(model))
        if any(alias.startswith("gpt-oss") for alias in aliases):
            return GPT_OSS_THINK_LEVEL
        return True

    def _emit_thinking_event(
        self,
        *,
        kind: str,
        operation: str,
        model: str,
        chunk: str = "",
    ) -> None:
        if self._thinking_callback is None:
            return
        self._thinking_callback(
            ThinkingEvent(
                kind=kind,
                operation=operation,
                model=model,
                chunk=chunk,
            )
        )

    def _stream_chat_response(
        self,
        *,
        operation: str,
        payload: dict[str, Any],
        on_chunk: Callable[[str], None] | None = None,
        on_thinking_chunk: Callable[[str], None] | None = None,
    ) -> str:
        model = str(payload.get("model", ""))
        disable_thinking = bool(payload.get("_disable_thinking"))
        stream_payload = {
            key: value
            for key, value in payload.items()
            if not key.startswith("_")
        }
        stream_payload["stream"] = True
        if disable_thinking:
            if self.supports_thinking(model):
                stream_payload["think"] = False
        else:
            think_setting = self._thinking_setting_for_model(model)
            if think_setting is not None:
                stream_payload["think"] = think_setting

        parts: list[str] = []
        thinking_started = False
        started_at = self._monotonic()
        try:
            for event in self.http.stream_json_lines("POST", "/chat", json_body=stream_payload):
                elapsed = self._monotonic() - started_at
                if elapsed > self.timeout_seconds:
                    raise OllamaResponseError(
                        f"Ollama {operation} stream exceeded {self.timeout_seconds:.0f}s without completing."
                    )
                if thinking_started and not parts and elapsed > self.thinking_response_timeout_seconds:
                    raise OllamaResponseError(
                        f"Ollama {operation} stream spent {self.thinking_response_timeout_seconds:.0f}s thinking without response content."
                    )
                if not isinstance(event, dict):
                    raise OllamaResponseError("Ollama returned a non-object stream event.")
                message = event.get("message")
                if not isinstance(message, dict):
                    if event.get("done") is True:
                        continue
                    raise OllamaResponseError("Ollama stream event did not contain message content.")

                thinking_chunk = message.get("thinking")
                if thinking_chunk is not None and not isinstance(thinking_chunk, str):
                    raise OllamaResponseError("Ollama stream chunk did not contain message.thinking.")
                if thinking_chunk:
                    if not thinking_started:
                        self._emit_thinking_event(kind="start", operation=operation, model=model)
                        thinking_started = True
                    self._emit_thinking_event(
                        kind="chunk",
                        operation=operation,
                        model=model,
                        chunk=thinking_chunk,
                    )
                    if on_thinking_chunk is not None:
                        on_thinking_chunk(thinking_chunk)

                chunk = message.get("content")
                if chunk is not None and not isinstance(chunk, str):
                    raise OllamaResponseError("Ollama stream chunk did not contain message.content.")
                if not chunk:
                    continue
                parts.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
        except HttpRequestError as exc:
            raise OllamaResponseError(f"Ollama {operation} stream failed: {exc}") from exc
        finally:
            if thinking_started:
                self._emit_thinking_event(kind="end", operation=operation, model=model)

        return "".join(parts).strip()


def _parse_required_reply_decision(content: str, *, fallback_reason: str) -> ModelDecision:
    try:
        return OllamaClient._parse_decision(content)
    except OllamaResponseError:
        payload_decision = _reply_decision_from_partial_payload(
            content,
            fallback_reason=fallback_reason,
        )
        if payload_decision is not None:
            return payload_decision
        reply_body = _plain_reply_body_from_unstructured_content(content)
        if reply_body is None:
            raise
        return ModelDecision(
            action="reply",
            reply_markdown=reply_body,
            reason=fallback_reason,
        )


def _normalize_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().casefold()


def _normalize_gif_id(value: Any) -> str | None | object:
    if value is None:
        return None
    if not isinstance(value, str):
        return _INVALID_GIF_ID
    normalized = value.strip().casefold()
    if normalized in {"", "null", "none", "no", "false"}:
        return None
    return normalized


def _coerce_candidate_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdecimal():
            return int(stripped)
    return None


def _coerce_confidence(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= confidence <= 1:
        return None
    return confidence


def _reply_decision_from_partial_payload(
    content: str,
    *,
    fallback_reason: str,
) -> ModelDecision | None:
    try:
        payload = _loads_json_object(content)
    except OllamaResponseError:
        return None
    if not isinstance(payload, dict):
        return None

    for key in ("reply_markdown", "raw", "body", "reply"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            gif_id = _normalize_gif_id(payload.get("gif_id"))
            return ModelDecision(
                action="reply",
                reply_markdown=value.strip(),
                reason=fallback_reason,
                gif_id=gif_id if gif_id is not _INVALID_GIF_ID else None,
            )
    return None


def _plain_reply_body_from_unstructured_content(content: str) -> str | None:
    body = content.strip()
    if not body:
        return None
    if _extract_embedded_json_object(body) is not None:
        return None
    if body.lstrip().startswith(("{", "[")):
        return None

    lines = body.splitlines()
    if len(lines) > 1:
        first = lines[0].strip().rstrip(":").casefold()
        if first in {"reply", "reply markdown", "forum reply", "post body"}:
            body = "\n".join(lines[1:]).strip()

    lowered = body.casefold()
    for prefix in ("reply_markdown:", "reply:", "forum reply:", "post body:"):
        if lowered.startswith(prefix):
            body = body[len(prefix) :].strip()
            break

    if not body:
        return None
    return body


def _resolve_selection_post_url(
    *,
    candidate_id: int | None,
    post_url: str | None,
    candidates: list[AutonomousCandidate] | None,
) -> str | None:
    if candidates:
        if candidate_id is not None:
            if 1 <= candidate_id <= len(candidates):
                return candidates[candidate_id - 1].post_url
            return None
        if post_url is not None:
            return _match_candidate_post_url(post_url, candidates)
        return None

    if post_url is not None:
        return extract_url_like(post_url) or post_url.strip()
    return None


def _match_candidate_post_url(
    selected_post_url: str,
    candidates: list[AutonomousCandidate],
) -> str | None:
    selected_canonical = canonical_post_url(selected_post_url)
    selected_key = topic_post_key_from_url(selected_post_url)
    for candidate in candidates:
        if selected_canonical and selected_canonical == canonical_post_url(candidate.post_url):
            return candidate.post_url
        if selected_key == (candidate.topic_id, candidate.post_number):
            return candidate.post_url
    return None


def _normalize_ollama_host(host: str) -> str:
    clean = host.rstrip("/")
    parts = urlsplit(clean)
    if parts.path.endswith("/api"):
        return clean
    return f"{clean}/api"


def _loads_json_object(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError as original_exc:
        extracted = _extract_embedded_json_object(content)
        if extracted is not None:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        raise OllamaResponseError(
            f"Ollama returned invalid JSON: {_preview_invalid_response(content)}"
        ) from original_exc


def _extract_embedded_json_object(content: str) -> str | None:
    decoder = json.JSONDecoder()
    text = content.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return text[index : index + end]
    return None


def _structured_retry_payload(
    payload: dict[str, Any],
    *,
    invalid_content: str,
    error: str,
) -> dict[str, Any]:
    retry_payload = dict(payload)
    messages = list(payload.get("messages", []))
    messages.append(
        {
            "role": "user",
            "content": "\n".join(
                [
                    STRUCTURED_RESPONSE_RETRY_PROMPT,
                    f"Parser error: {error}",
                    "Invalid response:",
                    _preview_invalid_response(invalid_content, limit=2000),
                ]
            ),
        }
    )
    retry_payload["messages"] = messages
    return retry_payload


def _structured_options(
    options: dict[str, Any] | None,
    *,
    num_predict_limit: int,
) -> dict[str, Any]:
    structured_options = dict(options or {})
    structured_options.setdefault("temperature", 0)
    configured_num_predict = _optional_int_option(structured_options.get("num_predict"))
    if (
        configured_num_predict is None
        or configured_num_predict <= 0
        or configured_num_predict > num_predict_limit
    ):
        structured_options["num_predict"] = num_predict_limit
    return structured_options


def _optional_int_option(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _structured_attempt_payload(
    payload: dict[str, Any],
    *,
    operation: str,
    is_retry: bool,
) -> dict[str, Any]:
    attempt_payload = dict(payload)
    num_predict_limit = (
        SELECTION_NUM_PREDICT_LIMIT
        if operation == "autonomous_selection"
        else REPLY_NUM_PREDICT_LIMIT
    )
    attempt_payload["options"] = _structured_options(
        attempt_payload.get("options")
        if isinstance(attempt_payload.get("options"), dict)
        else None,
        num_predict_limit=num_predict_limit,
    )
    if operation == "autonomous_selection" or is_retry:
        attempt_payload["_disable_thinking"] = True
    return attempt_payload


def _decision_retry_payload(
    payload: dict[str, Any],
    error: str,
    invalid_content: str | None,
) -> dict[str, Any]:
    retry_payload = dict(payload)
    messages = list(payload.get("messages", []))
    retry_lines = [
        DECISION_RETRY_PROMPT,
        f"Previous failure: {error}",
    ]
    if invalid_content:
        retry_lines.extend(
            [
                "Invalid response:",
                _preview_invalid_response(invalid_content, limit=1000),
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "\n".join(retry_lines),
        }
    )
    retry_payload["messages"] = messages
    retry_payload["options"] = _structured_options(
        retry_payload.get("options") if isinstance(retry_payload.get("options"), dict) else None,
        num_predict_limit=REPLY_NUM_PREDICT_LIMIT,
    )
    retry_payload["_disable_thinking"] = True
    return retry_payload


def _autonomous_reply_retry_payload(
    payload: dict[str, Any],
    error: str,
    invalid_content: str | None,
) -> dict[str, Any]:
    retry_payload = dict(payload)
    messages = list(payload.get("messages", []))
    retry_lines = [
        AUTONOMOUS_REPLY_RETRY_PROMPT,
        f"Previous failure: {error}",
    ]
    if invalid_content:
        retry_lines.extend(
            [
                "Invalid response:",
                _preview_invalid_response(invalid_content, limit=1000),
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "\n".join(retry_lines),
        }
    )
    retry_payload["messages"] = messages
    retry_payload["options"] = _structured_options(
        retry_payload.get("options") if isinstance(retry_payload.get("options"), dict) else None,
        num_predict_limit=REPLY_NUM_PREDICT_LIMIT,
    )
    retry_payload["_disable_thinking"] = True
    return retry_payload


def _preview_invalid_response(content: str, *, limit: int = 500) -> str:
    normalized = content.strip().replace("\r", "\\r").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}...<truncated>"


def _truncate_prompt_text(content: str, limit: int) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}...<truncated>"


def _build_context_prompt(
    identity: BotIdentity,
    context: TopicContext,
    available_gifs: list[GifOption] | None = None,
) -> str:
    messages = _build_context_messages(identity, context, available_gifs)
    return "\n\n".join(
        f"{message['role'].upper()} MESSAGE:\n{message['content']}"
        for message in messages
    )


def _build_context_messages(
    identity: BotIdentity,
    context: TopicContext,
    available_gifs: list[GifOption] | None = None,
) -> list[dict[str, str]]:
    target_section = _format_target_summary(context.target_post)
    messages = [
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"Bot username: {identity.username}",
                    f"Bot display name: {identity.name or ''}",
                    f"Trigger: {context.trigger}",
                    f"Actor username: {context.actor_username or ''}",
                    f"Topic title: {context.topic_title}",
                    f"Topic id: {context.topic_id}",
                    f"Topic archetype: {context.topic_archetype or ''}",
                    target_section,
                    "",
                    "The topic transcript follows as chat messages in chronological order.",
                    "Each message is one Discourse post. Assistant-role messages are the bot's own earlier posts in this topic.",
                    "Use the transcript to avoid acting unaware of earlier discussion or earlier bot replies.",
                ]
            ),
        }
    ]
    if context.recent_posts:
        messages.extend(_format_post_message(identity, post) for post in context.recent_posts)
    else:
        messages.append(
            {
                "role": "user",
                "content": "Topic transcript: (no posts available)",
            }
        )
    messages.append(
        {
            "role": "user",
            "content": _format_gif_options(available_gifs),
        }
    )
    return messages


def _build_optional_docs_context_messages(
    roblox_docs_context: str | None,
) -> list[dict[str, str]]:
    if not roblox_docs_context or not roblox_docs_context.strip():
        return []
    return [
        {
            "role": "user",
            "content": roblox_docs_context.strip(),
        }
    ]


def _build_context_prompt_with_instruction(
    identity: BotIdentity,
    context: TopicContext,
    instruction: str,
    available_gifs: list[GifOption] | None = None,
) -> str:
    base_prompt = _build_context_prompt(identity, context, available_gifs)
    return "\n".join(
        [
            base_prompt,
            "",
            instruction,
        ]
    )


def _format_target_summary(post: TopicPost | None) -> str:
    if post is None:
        return "Target post: (not available)"
    return (
        f"Target post number: {post.post_number}\n"
        f"Target post author: {post.username}"
    )


def _format_target_post(post: TopicPost | None) -> str:
    if post is None:
        return "Target post: (not available)"
    return (
        f"Target post number: {post.post_number}\n"
        f"Target post author: {post.username}\n"
        f"Target post text: {_post_text(post)}"
    )


def _format_post(post: TopicPost) -> str:
    text = _post_text(post)
    return (
        f"Post #{post.post_number} by {post.username}"
        f"{f' at {post.created_at}' if post.created_at else ''}:\n{text}"
    )


def _format_post_message(identity: BotIdentity, post: TopicPost) -> dict[str, str]:
    is_bot_post = _same_username(post.username, identity.username)
    author_label = "your earlier forum post" if is_bot_post else f"forum post by {post.username}"
    content = (
        f"Post #{post.post_number}, {author_label}"
        f"{f' at {post.created_at}' if post.created_at else ''}:\n"
        f"{_post_text(post)}"
    )
    return {
        "role": "assistant" if is_bot_post else "user",
        "content": content,
    }


def _post_text(post: TopicPost) -> str:
    return post.raw or strip_html(post.cooked) or ""


def _build_manual_request_prompt(
    identity: BotIdentity,
    context: TopicContext,
    user_request: str,
    available_gifs: list[GifOption] | None = None,
) -> str:
    return _build_context_prompt_with_instruction(
        identity,
        context,
        _build_manual_request_instruction(user_request),
        available_gifs,
    )


def _build_manual_request_instruction(user_request: str) -> str:
    gif_expectation = (
        "Operator GIF requirement: required when a fitting GIF is available."
        if _manual_request_mentions_gif(user_request)
        else "Operator GIF requirement: optional."
    )
    return "\n".join(
        [
            f"Operator request: {user_request}",
            gif_expectation,
            "Write the reply requested by the operator.",
        ]
    )


def _build_autonomous_reply_prompt(
    identity: BotIdentity,
    context: TopicContext,
    selection_reason: str,
    available_gifs: list[GifOption] | None = None,
) -> str:
    return _build_context_prompt_with_instruction(
        identity,
        context,
        _build_autonomous_reply_instruction(selection_reason),
        available_gifs,
    )


def _build_autonomous_reply_instruction(selection_reason: str) -> str:
    return "\n".join(
        [
            f"Selection reason: {selection_reason}",
            "Write the forum reply now. Keep the system prompt's persona and style intact.",
        ]
    )


def _manual_request_mentions_gif(user_request: str) -> bool:
    normalized = user_request.strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in (" gif", "gifs", "gif ", "gif,", "gif.", "gif!", "gif?", "gif-")) or normalized.startswith("gif")


def _same_username(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return left.strip().casefold() == right.strip().casefold()


def _build_autonomous_selection_prompt(
    identity: BotIdentity,
    candidates: list[AutonomousCandidate],
    min_confidence: float,
) -> str:
    candidate_sections = []
    for index, candidate in enumerate(candidates, start=1):
        context = candidate.context
        target_section = _format_selection_target_post(context.target_post)
        recent_posts = "\n\n".join(
            _format_selection_recent_post(post)
            for post in context.recent_posts[-AUTONOMOUS_SELECTION_RECENT_POST_LIMIT:]
        )
        candidate_sections.append(
            "\n".join(
                [
                    f"Candidate ID: {index}",
                    f"Post URL for reference only, do not return it: {candidate.post_url}",
                    f"Topic title: {context.topic_title}",
                    f"Topic id: {candidate.topic_id}",
                    f"Topic archetype: {context.topic_archetype or ''}",
                    f"Latest poster: {candidate.actor_username or ''}",
                    target_section,
                    "Recent conversation:",
                    recent_posts or "(no recent posts available)",
                ]
            )
        )
    return "\n\n".join(
        [
            f"Bot username: {identity.username}",
            f"Bot display name: {identity.name or ''}",
            f"Minimum confidence required to choose a post: {min_confidence:.2f}",
            "",
            *candidate_sections,
            "",
            "Select one post only if confidence meets or exceeds the minimum.",
            "The chosen post does not have to be a question; it can be a good opportunity for information, perspective, disagreement, or clarification.",
            "If selecting a post, return only its Candidate ID in candidate_id.",
            "Do not write the reply. The reply will be written later in a separate step.",
        ]
    )


def _format_selection_target_post(post: TopicPost | None) -> str:
    if post is None:
        return "Target post: (not available)"
    return (
        f"Target post number: {post.post_number}\n"
        f"Target post author: {post.username}\n"
        f"Target post text: {_truncate_prompt_text(_post_text(post), AUTONOMOUS_SELECTION_POST_TEXT_LIMIT)}"
    )


def _format_selection_recent_post(post: TopicPost) -> str:
    text = _truncate_prompt_text(_post_text(post), AUTONOMOUS_SELECTION_POST_TEXT_LIMIT)
    return (
        f"Post #{post.post_number} by {post.username}"
        f"{f' at {post.created_at}' if post.created_at else ''}:\n{text}"
    )


def _format_gif_options(available_gifs: list[GifOption] | None) -> str:
    if not available_gifs:
        return "Available optional GIFs: none."
    lines = ["Available optional GIFs:"]
    for option in available_gifs:
        lines.append(f"- {option.gif_id}: {option.description}")
    lines.append("Return gif_id as null when no GIF should be included.")
    return "\n".join(lines)


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


def _collect_model_aliases(model: str, details: dict[str, Any] | None) -> set[str]:
    aliases = {model.strip().lower()}
    if ":" in model:
        aliases.add(model.split(":", 1)[0].strip().lower())

    detail_block = details.get("details") if isinstance(details, dict) else None
    if isinstance(detail_block, dict):
        family = detail_block.get("family")
        if isinstance(family, str) and family.strip():
            aliases.add(family.strip().lower())
        families = detail_block.get("families")
        if isinstance(families, list):
            aliases.update(
                str(item).strip().lower()
                for item in families
                if isinstance(item, str) and item.strip()
            )
    return aliases


def _aliases_support_thinking(aliases: set[str]) -> bool:
    return any(
        alias == prefix or alias.startswith(f"{prefix}:") or alias.startswith(f"{prefix}-")
        for alias in aliases
        for prefix in KNOWN_THINKING_MODEL_PREFIXES
    )
