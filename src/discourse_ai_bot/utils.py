from __future__ import annotations

from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
import re
from typing import Iterable
from urllib.parse import urlsplit


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(unescape(value))
    return parser.get_text()


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def datetime_to_storage(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def take_last(items: Iterable[object], count: int) -> list[object]:
    values = list(items)
    if count <= 0:
        return []
    return values[-count:]


def parse_duration_seconds(value: str, *, field_name: str) -> float:
    raw = value.strip().lower()
    if not raw:
        raise ValueError(f"{field_name} is required.")

    match = re.fullmatch(
        r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)?",
        raw,
    )
    if match is None:
        raise ValueError(
            f"{field_name} must be a duration like '30s', '1m', or '1h'."
        )

    amount = float(match.group("amount"))
    unit = match.group("unit") or "s"
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        multiplier = 1.0
    elif unit in {"m", "min", "mins", "minute", "minutes"}:
        multiplier = 60.0
    else:
        multiplier = 3600.0
    return amount * multiplier


def extract_url_like(value: str) -> str | None:
    markdown_match = re.search(r"\]\((https?://[^)\s]+)\)", value)
    if markdown_match:
        return markdown_match.group(1).rstrip(".,;")
    raw_match = re.search(r"https?://[^\s<>)\]]+", value)
    if raw_match:
        return raw_match.group(0).rstrip(".,;")
    return None


def canonical_post_url(value: str | None) -> str | None:
    if value is None:
        return None
    url = extract_url_like(value) or value.strip()
    if not url:
        return None
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    path = "/" + "/".join(segment for segment in parts.path.split("/") if segment)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}"


def topic_post_key_from_url(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    url = extract_url_like(value) or value.strip()
    path = urlsplit(url).path or url
    segments = [segment for segment in path.split("/") if segment]
    try:
        topic_index = segments.index("t")
    except ValueError:
        return None
    numeric_segments: list[int] = []
    for segment in segments[topic_index + 1 :]:
        try:
            numeric_segments.append(int(segment))
        except ValueError:
            continue
    if len(numeric_segments) < 2:
        return None
    return numeric_segments[-2], numeric_segments[-1]
