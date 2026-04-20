from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


class HttpRequestError(RuntimeError):
    """Raised when a remote HTTP endpoint fails."""


@dataclass(frozen=True)
class HttpError(HttpRequestError):
    status_code: int
    url: str
    body: str

    def __str__(self) -> str:
        return f"HTTP {self.status_code} for {self.url}: {self.body}"


class JsonHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        default_headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.default_headers = default_headers or {}
        self.timeout_seconds = timeout_seconds

    def request_json(
        self,
        method: str,
        path_or_url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        form_body: dict[str, Any] | Iterable[tuple[str, Any]] | None = None,
        multipart_body: dict[str, Any] | None = None,
        multipart_files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> Any:
        url = _resolve_url(self.base_url, path_or_url)
        request_headers = {
            "Accept": "application/json",
            **self.default_headers,
            **(headers or {}),
        }

        data: bytes | None = None
        body_count = sum(
            body is not None
            for body in (json_body, form_body, multipart_body if (multipart_body or multipart_files) else None)
        )
        if body_count > 1:
            raise ValueError("Only one of json_body, form_body, or multipart body may be provided.")
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        elif form_body is not None:
            if isinstance(form_body, dict):
                encoded = urlencode(form_body, doseq=True)
            else:
                encoded = urlencode(list(form_body), doseq=True)
            data = encoded.encode("utf-8")
            request_headers.setdefault(
                "Content-Type",
                "application/x-www-form-urlencoded; charset=UTF-8",
            )
        elif multipart_body is not None or multipart_files is not None:
            boundary, data = _encode_multipart_form_data(
                multipart_body or {},
                multipart_files or {},
            )
            request_headers.setdefault(
                "Content-Type",
                f"multipart/form-data; boundary={boundary}",
            )

        request = Request(url=url, data=data, headers=request_headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            raise HttpError(status_code=exc.code, url=url, body=body) from exc
        except URLError as exc:
            raise HttpRequestError(f"Unable to reach {url}: {exc.reason}") from exc

        if not raw_body.strip():
            return {}

        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HttpRequestError(f"Expected JSON from {url}, got: {raw_body}") from exc

    def stream_json_lines(
        self,
        method: str,
        path_or_url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        form_body: dict[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> Iterable[Any]:
        url = _resolve_url(self.base_url, path_or_url)
        request_headers = {
            "Accept": "application/json",
            **self.default_headers,
            **(headers or {}),
        }

        data: bytes | None = None
        if json_body is not None and form_body is not None:
            raise ValueError("Only one of json_body or form_body may be provided.")
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        elif form_body is not None:
            if isinstance(form_body, dict):
                encoded = urlencode(form_body, doseq=True)
            else:
                encoded = urlencode(list(form_body), doseq=True)
            data = encoded.encode("utf-8")
            request_headers.setdefault(
                "Content-Type",
                "application/x-www-form-urlencoded; charset=UTF-8",
            )

        request = Request(url=url, data=data, headers=request_headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise HttpRequestError(f"Expected JSON lines from {url}, got: {line}") from exc
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            raise HttpError(status_code=exc.code, url=url, body=body) from exc
        except URLError as exc:
            raise HttpRequestError(f"Unable to reach {url}: {exc.reason}") from exc


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/"


def _resolve_url(base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    return urljoin(base_url, path_or_url.lstrip("/"))


def _encode_multipart_form_data(
    fields: dict[str, Any],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[str, bytes]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, (filename, content, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(chunks)
