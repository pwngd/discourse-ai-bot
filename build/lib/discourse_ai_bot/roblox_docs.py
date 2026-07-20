from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml

from discourse_ai_bot.utils import strip_html


REFERENCE_KINDS = ("classes", "datatypes", "enums")
DEFAULT_GITHUB_API_BASE = "https://api.github.com/repos/Roblox/creator-docs"
DEFAULT_RAW_BASE = "https://raw.githubusercontent.com/Roblox/creator-docs"
DEFAULT_LOCAL_PATH = "vendor/creator-docs"
CREATOR_DOCS_BASE = "https://create.roblox.com/docs/reference/engine"
USER_AGENT = "discourse-ai-bot roblox-docs verifier"

CODING_HINT_RE = re.compile(
    r"(```|`[^`]+`|\b(?:script|scripts|scripting|code|coding|lua|luau|"
    r"api|reference|docs?|property|properties|method|function|event|"
    r"callback|enum|datatype|class|server|client|replicat|error|"
    r"Instance\.new|GetService|WaitForChild|FindFirstChild|require|"
    r"RemoteEvent|RemoteFunction|ModuleScript|LocalScript|Script)\b)",
    re.IGNORECASE,
)
ROBLOX_API_HINT_RE = re.compile(
    r"\b(?:Instance\.new|game:GetService|Enum\.[A-Z]\w+|"
    r"[A-Z][A-Za-z0-9_]+[.:][A-Za-z_][A-Za-z0-9_]*|"
    r"[A-Z][A-Za-z0-9_]*(?:Service|Params|Type|Info|Result)|"
    r"(?:Humanoid|BasePart|Part|Model|Workspace|Players|Player|TweenService|"
    r"RunService|UserInputService|ReplicatedStorage|ServerStorage|"
    r"ServerScriptService|StarterGui|StarterPlayer|CFrame|Vector3|UDim2|"
    r"RaycastParams|OverlapParams|RemoteEvent|RemoteFunction|BindableEvent|"
    r"DataStoreService|MessagingService|TeleportService|CollectionService))\b"
)
CLASS_MEMBER_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]+)[.:]([A-Za-z_][A-Za-z0-9_]*)\b")
INSTANCE_NEW_RE = re.compile(r"\bInstance\.new\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", re.IGNORECASE)
GET_SERVICE_RE = re.compile(r"\bGetService\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", re.IGNORECASE)
ENUM_RE = re.compile(r"\bEnum\.([A-Za-z_][A-Za-z0-9_]*)\b")
BACKTICK_RE = re.compile(r"`([^`\n]{2,80})`")
IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")

COMMON_NON_API_WORDS = {
    "Roblox",
    "Studio",
    "Discourse",
    "DevForum",
    "HTTP",
    "JSON",
    "URL",
    "API",
    "Lua",
    "Luau",
}


class RobloxDocsError(RuntimeError):
    """Raised when official Roblox docs cannot be queried or parsed."""


@dataclass(frozen=True)
class RobloxDocsMatch:
    kind: str
    name: str
    url: str
    summary: str
    type_name: str | None = None
    tags: tuple[str, ...] = ()
    deprecation_message: str | None = None
    members: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class RobloxDocsContext:
    matches: tuple[RobloxDocsMatch, ...]

    def format_for_prompt(self, *, max_chars: int) -> str:
        if not self.matches:
            return ""

        lines = [
            "Verified Roblox API docs context:",
            "Use this official Creator Docs context only for Roblox/Luau coding API claims. If it does not cover the claim, say so or avoid asserting it.",
        ]
        for match in self.matches:
            label = {"classes": "class", "datatypes": "datatype", "enums": "enum"}.get(
                match.kind,
                match.kind,
            )
            title_bits = [f"- {label}: {match.name}"]
            if match.type_name:
                title_bits.append(f"type={match.type_name}")
            if match.tags:
                title_bits.append(f"tags={', '.join(match.tags)}")
            lines.append(" ".join(title_bits))
            if match.summary:
                lines.append(f"  Summary: {match.summary}")
            if match.deprecation_message:
                lines.append(f"  Deprecated: {match.deprecation_message}")
            if match.members:
                lines.append("  Relevant members:")
                for member in match.members:
                    member_bits = [f"    - {member['name']}"]
                    if member.get("section"):
                        member_bits.append(f"({member['section']})")
                    if member.get("type"):
                        member_bits.append(f"type={member['type']}")
                    if member.get("thread_safety"):
                        member_bits.append(f"thread_safety={member['thread_safety']}")
                    if member.get("security"):
                        member_bits.append(f"security={member['security']}")
                    lines.append(" ".join(member_bits))
                    if member.get("summary"):
                        lines.append(f"      Summary: {member['summary']}")
                    if member.get("deprecation_message"):
                        lines.append(f"      Deprecated: {member['deprecation_message']}")
            lines.append(f"  Source: {match.url}")

        content = "\n".join(lines).strip()
        if len(content) <= max_chars:
            return content
        return content[: max(0, max_chars - 32)].rstrip() + "\n...<docs context truncated>"


class RobloxDocsClient:
    def __init__(
        self,
        *,
        ref: str = "main",
        timeout_seconds: float = 8.0,
        cache_ttl_seconds: float = 86400.0,
        max_terms: int = 6,
        max_results: int = 4,
        max_context_chars: int = 6000,
        source: str = "auto",
        local_path: str | Path | None = DEFAULT_LOCAL_PATH,
        github_api_base: str = DEFAULT_GITHUB_API_BASE,
        raw_base: str = DEFAULT_RAW_BASE,
        monotonic_fn: Callable[[], float] = time.monotonic,
        fetch_json: Callable[[str, float], Any] | None = None,
        fetch_text: Callable[[str, float], str] | None = None,
    ) -> None:
        self.ref = ref.strip() or "main"
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_terms = max_terms
        self.max_results = max_results
        self.max_context_chars = max_context_chars
        self.source = _normalize_source(source)
        self.local_path = _resolve_local_path(local_path)
        self.github_api_base = github_api_base.rstrip("/")
        self.raw_base = raw_base.rstrip("/")
        self._monotonic = monotonic_fn
        self._fetch_json = fetch_json or _fetch_json_url
        self._fetch_text = fetch_text or _fetch_text_url
        self._index_cache: dict[str, tuple[float, dict[str, str]]] = {}
        self._doc_cache: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}

    def context_for_text(self, text: str) -> RobloxDocsContext | None:
        if not is_likely_roblox_coding_question(text):
            return None

        query = extract_query_terms(text, max_terms=self.max_terms)
        if not query.class_terms and not query.member_terms and not query.enum_terms:
            return None

        matches = self.search(
            class_terms=query.class_terms,
            member_terms=query.member_terms,
            enum_terms=query.enum_terms,
        )
        if not matches:
            return None
        return RobloxDocsContext(tuple(matches[: self.max_results]))

    def search(
        self,
        *,
        class_terms: tuple[str, ...],
        member_terms: tuple[str, ...],
        enum_terms: tuple[str, ...],
    ) -> list[RobloxDocsMatch]:
        candidates: list[tuple[str, str]] = []
        class_matches = self._candidate_matches("classes", class_terms)
        enum_matches = self._candidate_matches(
            "enums",
            _dedupe_terms((*enum_terms, *class_terms)),
        )
        datatype_matches = self._candidate_matches("datatypes", class_terms)

        ordered_candidates = (
            class_matches[:1]
            + enum_matches
            + datatype_matches
            + class_matches[1:]
        )
        for candidate in ordered_candidates:
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= self.max_terms:
                break

        matches: list[RobloxDocsMatch] = []
        for kind, name in candidates:
            doc = self._load_reference_doc(kind, name)
            if not doc:
                continue
            matches.append(_match_from_doc(kind, doc, member_terms=member_terms))
            if len(matches) >= self.max_results:
                break
        return matches

    def _candidate_matches(
        self,
        kind: str,
        terms: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        index = self._reference_index(kind)
        matches: list[tuple[str, str]] = []
        for term in terms:
            canonical = index.get(term.casefold())
            if canonical is not None and (kind, canonical) not in matches:
                matches.append((kind, canonical))
        return matches

    def _reference_index(self, kind: str) -> dict[str, str]:
        now = self._monotonic()
        cache_key = self._cache_key(kind)
        cached = self._index_cache.get(cache_key)
        if cached is not None and now - cached[0] < self.cache_ttl_seconds:
            return cached[1]

        if self._can_read_local_kind(kind):
            index = self._local_reference_index(kind)
            self._index_cache[cache_key] = (now, index)
            return index

        url = (
            f"{self.github_api_base}/contents/content/en-us/reference/engine/"
            f"{kind}?ref={quote(self.ref)}"
        )
        try:
            payload = self._fetch_json(url, self.timeout_seconds)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RobloxDocsError(f"Unable to load Roblox docs index for {kind}: {exc}") from exc
        if not isinstance(payload, list):
            raise RobloxDocsError(f"Roblox docs index for {kind} was not a list.")

        index: dict[str, str] = {}
        for item in payload:
            if not isinstance(item, dict) or item.get("type") != "file":
                continue
            filename = item.get("name")
            if not isinstance(filename, str) or not filename.endswith(".yaml"):
                continue
            name = filename[:-5]
            index[name.casefold()] = name
        self._index_cache[cache_key] = (now, index)
        return index

    def _load_reference_doc(self, kind: str, name: str) -> dict[str, Any] | None:
        now = self._monotonic()
        cache_key = (self._cache_key(kind), name)
        cached = self._doc_cache.get(cache_key)
        if cached is not None and now - cached[0] < self.cache_ttl_seconds:
            return cached[1]

        if self._can_read_local_kind(kind):
            doc = self._load_local_reference_doc(kind, name)
            if doc is not None:
                self._doc_cache[cache_key] = (now, doc)
                return doc

        url = (
            f"{self.raw_base}/{quote(self.ref)}/content/en-us/reference/engine/"
            f"{kind}/{quote(name)}.yaml"
        )
        try:
            raw = self._fetch_text(url, self.timeout_seconds)
            parsed = yaml.safe_load(raw)
        except (HTTPError, URLError, TimeoutError, OSError, yaml.YAMLError) as exc:
            raise RobloxDocsError(f"Unable to load Roblox docs for {kind}/{name}: {exc}") from exc
        if parsed is not None and not isinstance(parsed, dict):
            raise RobloxDocsError(f"Roblox docs for {kind}/{name} did not parse to an object.")
        doc = parsed if isinstance(parsed, dict) else None
        self._doc_cache[cache_key] = (now, doc)
        return doc

    def _cache_key(self, kind: str) -> str:
        if self._can_read_local_kind(kind):
            assert self.local_path is not None
            return f"local:{self.local_path}:{kind}"
        return f"remote:{self.ref}:{kind}"

    def _can_read_local_kind(self, kind: str) -> bool:
        if self.source == "remote" or self.local_path is None:
            return False
        return self._local_reference_dir(kind).is_dir()

    def _local_reference_index(self, kind: str) -> dict[str, str]:
        reference_dir = self._local_reference_dir(kind)
        index: dict[str, str] = {}
        for path in reference_dir.glob("*.yaml"):
            if path.is_file():
                index[path.stem.casefold()] = path.stem
        return index

    def _load_local_reference_doc(self, kind: str, name: str) -> dict[str, Any] | None:
        path = self._local_reference_dir(kind) / f"{name}.yaml"
        if not path.is_file():
            return None
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RobloxDocsError(f"Unable to load local Roblox docs for {kind}/{name}: {exc}") from exc
        if parsed is not None and not isinstance(parsed, dict):
            raise RobloxDocsError(f"Local Roblox docs for {kind}/{name} did not parse to an object.")
        return parsed if isinstance(parsed, dict) else None

    def _local_reference_dir(self, kind: str) -> Path:
        assert self.local_path is not None
        return self.local_path / "content" / "en-us" / "reference" / "engine" / kind


@dataclass(frozen=True)
class _QueryTerms:
    class_terms: tuple[str, ...]
    member_terms: tuple[str, ...]
    enum_terms: tuple[str, ...]


def is_likely_roblox_coding_question(text: str) -> bool:
    if not text.strip():
        return False
    return CODING_HINT_RE.search(text) is not None and ROBLOX_API_HINT_RE.search(text) is not None


def extract_query_terms(text: str, *, max_terms: int) -> _QueryTerms:
    class_terms: list[str] = []
    member_terms: list[str] = []
    enum_terms: list[str] = []

    def add(target: list[str], value: str) -> None:
        clean = value.strip().strip("`'\"()[]{}.,:;")
        if not clean or clean in COMMON_NON_API_WORDS:
            return
        if clean not in target:
            target.append(clean)

    for class_name, member_name in CLASS_MEMBER_RE.findall(text):
        if class_name != "Enum":
            add(class_terms, class_name)
            add(member_terms, member_name)
    for class_name in INSTANCE_NEW_RE.findall(text):
        add(class_terms, class_name)
    for service_name in GET_SERVICE_RE.findall(text):
        add(class_terms, service_name)
    for enum_name in ENUM_RE.findall(text):
        add(enum_terms, enum_name)
    for inline in BACKTICK_RE.findall(text):
        for class_name, member_name in CLASS_MEMBER_RE.findall(inline):
            if class_name != "Enum":
                add(class_terms, class_name)
                add(member_terms, member_name)
        for identifier in IDENTIFIER_RE.findall(inline):
            add(class_terms, identifier)
    for identifier in IDENTIFIER_RE.findall(text):
        add(class_terms, identifier)
        if len(class_terms) >= max_terms:
            break

    return _QueryTerms(
        class_terms=tuple(class_terms[:max_terms]),
        member_terms=tuple(member_terms[:max_terms]),
        enum_terms=tuple(enum_terms[:max_terms]),
    )


def _dedupe_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        selected.append(term)
        seen.add(key)
    return tuple(selected)


def _normalize_source(source: str) -> str:
    normalized = source.strip().lower() or "auto"
    if normalized not in {"auto", "local", "remote"}:
        raise ValueError("Roblox docs source must be 'auto', 'local', or 'remote'.")
    return normalized


def _resolve_local_path(local_path: str | Path | None) -> Path | None:
    if local_path is None:
        return None
    path = Path(local_path)
    if not str(path).strip():
        return None
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _match_from_doc(
    kind: str,
    doc: dict[str, Any],
    *,
    member_terms: tuple[str, ...],
) -> RobloxDocsMatch:
    name = str(doc.get("name") or "")
    members = _select_members(doc, member_terms=member_terms)
    return RobloxDocsMatch(
        kind=kind,
        name=name,
        url=_creator_docs_url(kind, name),
        summary=_clean_text(doc.get("summary")),
        type_name=_optional_str(doc.get("type")),
        tags=_string_tuple(doc.get("tags")),
        deprecation_message=_clean_text(doc.get("deprecation_message")) or None,
        members=tuple(members),
    )


def _select_members(
    doc: dict[str, Any],
    *,
    member_terms: tuple[str, ...],
) -> list[dict[str, str]]:
    if not member_terms:
        return []

    normalized_terms = {term.casefold() for term in member_terms}
    selected: list[dict[str, str]] = []
    for section in ("properties", "methods", "events", "callbacks", "items"):
        entries = doc.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_name = _optional_str(entry.get("name"))
            if raw_name is None:
                continue
            short_name = _short_member_name(raw_name)
            if (
                raw_name.casefold() not in normalized_terms
                and short_name.casefold() not in normalized_terms
            ):
                continue
            selected.append(_format_member(section, entry, raw_name=raw_name))
            if len(selected) >= 8:
                return selected
    return selected


def _format_member(section: str, entry: dict[str, Any], *, raw_name: str) -> dict[str, str]:
    result = {
        "section": section,
        "name": raw_name,
        "summary": _clean_text(entry.get("summary")),
    }
    type_name = _optional_str(entry.get("type"))
    if type_name:
        result["type"] = type_name
    thread_safety = _optional_str(entry.get("thread_safety"))
    if thread_safety:
        result["thread_safety"] = thread_safety
    security = _format_security(entry.get("security"))
    if security:
        result["security"] = security
    deprecation_message = _clean_text(entry.get("deprecation_message"))
    if deprecation_message:
        result["deprecation_message"] = deprecation_message
    return result


def _creator_docs_url(kind: str, name: str) -> str:
    path_name = quote(name)
    if kind == "classes":
        return f"{CREATOR_DOCS_BASE}/classes/{path_name}"
    if kind == "datatypes":
        return f"{CREATOR_DOCS_BASE}/datatypes/{path_name}"
    if kind == "enums":
        return f"{CREATOR_DOCS_BASE}/enums/{path_name}"
    return f"{CREATOR_DOCS_BASE}/{kind}/{path_name}"


def _short_member_name(value: str) -> str:
    return re.split(r"[.:]", value)[-1]


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = strip_html(value)
    text = re.sub(r"`(?:Class|Datatype|Enum)\.([^`|]+)(?:\|([^`]+))?`", _link_label, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _link_label(match: re.Match[str]) -> str:
    return match.group(2) or match.group(1)


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _format_security(value: object) -> str:
    if isinstance(value, dict):
        parts = []
        read = _optional_str(value.get("read"))
        write = _optional_str(value.get("write"))
        if read:
            parts.append(f"read={read}")
        if write:
            parts.append(f"write={write}")
        return ", ".join(parts)
    if isinstance(value, str):
        return value.strip()
    return ""


def _fetch_json_url(url: str, timeout_seconds: float) -> Any:
    raw = _fetch_text_url(url, timeout_seconds)
    return json.loads(raw)


def _fetch_text_url(url: str, timeout_seconds: float) -> str:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")
