from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from discourse_ai_bot.roblox_docs import (
    RobloxDocsClient,
    extract_query_terms,
    is_likely_roblox_coding_question,
)


class RobloxDocsTests(unittest.TestCase):
    def test_non_coding_text_is_not_treated_as_docs_question(self) -> None:
        self.assertFalse(
            is_likely_roblox_coding_question("Roblox should improve the avatar marketplace.")
        )

    def test_coding_text_extracts_class_member_and_enum_terms(self) -> None:
        terms = extract_query_terms(
            'Does `Part.Shape` use Enum.PartType after Instance.new("Part")?',
            max_terms=6,
        )

        self.assertIn("Part", terms.class_terms)
        self.assertIn("Shape", terms.member_terms)
        self.assertIn("PartType", terms.enum_terms)

    def test_context_fetches_official_yaml_and_formats_relevant_members(self) -> None:
        json_calls: list[str] = []
        text_calls: list[str] = []

        def fetch_json(url: str, timeout_seconds: float) -> object:
            json_calls.append(url)
            if url.endswith("/classes?ref=main"):
                return [{"name": "Part.yaml", "type": "file"}]
            if url.endswith("/datatypes?ref=main"):
                return [{"name": "CFrame.yaml", "type": "file"}]
            if url.endswith("/enums?ref=main"):
                return [{"name": "PartType.yaml", "type": "file"}]
            return []

        def fetch_text(url: str, timeout_seconds: float) -> str:
            text_calls.append(url)
            if url.endswith("/classes/Part.yaml"):
                return """
name: Part
type: class
summary: A common type of BasePart.
tags: []
deprecation_message: ''
properties:
  - name: Part.Shape
    summary: Sets the overall shape of the object.
    type: PartType
    thread_safety: ReadSafe
    security:
      read: None
      write: None
methods: []
events: []
callbacks: []
"""
            if url.endswith("/enums/PartType.yaml"):
                return """
name: PartType
type: enum
summary: Controls the Part.Shape of an object.
tags: []
deprecation_message: ''
items:
  - name: Ball
    summary: A spherical shape.
    value: 0
"""
            raise AssertionError(f"unexpected URL {url}")

        client = RobloxDocsClient(source="remote", fetch_json=fetch_json, fetch_text=fetch_text)

        context = client.context_for_text(
            'In Luau, does `Part.Shape` use `Enum.PartType` for Instance.new("Part")?'
        )

        self.assertIsNotNone(context)
        assert context is not None
        formatted = context.format_for_prompt(max_chars=4000)
        self.assertIn("Verified Roblox API docs context", formatted)
        self.assertIn("Part.Shape", formatted)
        self.assertIn("type=PartType", formatted)
        self.assertIn("https://create.roblox.com/docs/reference/engine/classes/Part", formatted)
        self.assertEqual(len(text_calls), 2)

        context_again = client.context_for_text("Luau code question: can Part.Shape be set?")
        self.assertIsNotNone(context_again)
        self.assertEqual(len(text_calls), 2)
        self.assertGreaterEqual(len(json_calls), 3)

    def test_non_coding_text_does_not_fetch_indexes(self) -> None:
        def fetch_json(url: str, timeout_seconds: float) -> object:
            raise AssertionError("docs index should not be fetched")

        def fetch_text(url: str, timeout_seconds: float) -> str:
            raise AssertionError("docs file should not be fetched")

        client = RobloxDocsClient(source="remote", fetch_json=fetch_json, fetch_text=fetch_text)

        self.assertIsNone(client.context_for_text("Parties in Roblox can be fun."))

    def test_bare_enum_name_can_match_enum_reference(self) -> None:
        def fetch_json(url: str, timeout_seconds: float) -> object:
            if url.endswith("/classes?ref=main"):
                return []
            if url.endswith("/datatypes?ref=main"):
                return []
            if url.endswith("/enums?ref=main"):
                return [{"name": "PartType.yaml", "type": "file"}]
            return []

        def fetch_text(url: str, timeout_seconds: float) -> str:
            return """
name: PartType
type: enum
summary: Controls the Part.Shape of an object.
tags: []
deprecation_message: ''
items: []
"""

        client = RobloxDocsClient(fetch_json=fetch_json, fetch_text=fetch_text)

        context = client.context_for_text("Luau code question: what is PartType used for?")

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("enum: PartType", context.format_for_prompt(max_chars=1000))

    def test_reads_local_creator_docs_checkout_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_dir = root / "content" / "en-us" / "reference" / "engine"
            classes_dir = reference_dir / "classes"
            enums_dir = reference_dir / "enums"
            datatypes_dir = reference_dir / "datatypes"
            classes_dir.mkdir(parents=True)
            enums_dir.mkdir(parents=True)
            datatypes_dir.mkdir(parents=True)
            (classes_dir / "Part.yaml").write_text(
                """
name: Part
type: class
summary: Local class summary.
tags: []
deprecation_message: ''
properties:
  - name: Part.Shape
    summary: Local shape summary.
    type: PartType
methods: []
events: []
callbacks: []
""",
                encoding="utf-8",
            )

            def fetch_json(url: str, timeout_seconds: float) -> object:
                raise AssertionError("remote index should not be fetched")

            def fetch_text(url: str, timeout_seconds: float) -> str:
                raise AssertionError("remote file should not be fetched")

            client = RobloxDocsClient(
                source="auto",
                local_path=root,
                fetch_json=fetch_json,
                fetch_text=fetch_text,
            )

            context = client.context_for_text("Luau code question: is Part.Shape valid?")

            self.assertIsNotNone(context)
            assert context is not None
            formatted = context.format_for_prompt(max_chars=1000)
            self.assertIn("Local class summary", formatted)
            self.assertIn("Part.Shape", formatted)

    def test_falls_back_to_remote_when_local_checkout_is_missing(self) -> None:
        json_calls: list[str] = []

        def fetch_json(url: str, timeout_seconds: float) -> object:
            json_calls.append(url)
            if url.endswith("/classes?ref=main"):
                return [{"name": "Part.yaml", "type": "file"}]
            if url.endswith("/datatypes?ref=main") or url.endswith("/enums?ref=main"):
                return []
            return []

        def fetch_text(url: str, timeout_seconds: float) -> str:
            return """
name: Part
type: class
summary: Remote fallback summary.
tags: []
deprecation_message: ''
properties: []
methods: []
events: []
callbacks: []
"""

        client = RobloxDocsClient(
            source="auto",
            local_path="does-not-exist",
            fetch_json=fetch_json,
            fetch_text=fetch_text,
        )

        context = client.context_for_text("Luau code question: can I create a Part?")

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("Remote fallback summary", context.format_for_prompt(max_chars=1000))
        self.assertTrue(json_calls)


if __name__ == "__main__":
    unittest.main()
