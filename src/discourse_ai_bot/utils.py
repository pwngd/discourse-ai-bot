from __future__ import annotations

from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
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
