from __future__ import annotations

from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
import re
from typing import Iterable


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
