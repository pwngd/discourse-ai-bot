from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discourse_ai_bot.classifier import NotificationClassifier
from discourse_ai_bot.models import Notification
from discourse_ai_bot.storage import BotStorage


class ClassifierTests(unittest.TestCase):
    def test_classifier_accepts_supported_trigger(self) -> None:
        classifier = NotificationClassifier(
            "bot",
            allowed_triggers=("mentioned", "replied", "private_message"),
            notification_types={1: "mentioned"},
        )
        notification = Notification(
            notification_id=10,
            notification_type=1,
            type_name=None,
            read=False,
            created_at="2026-01-01T00:00:00+00:00",
            topic_id=123,
            post_number=2,
            slug="topic",
            data={"username": "alice"},
        )
        classified = classifier.classify(notification)
        self.assertIsNotNone(classified)
        self.assertEqual(classified.trigger, "mentioned")

    def test_classifier_ignores_self_notifications(self) -> None:
        classifier = NotificationClassifier(
            "bot",
            allowed_triggers=("mentioned",),
            notification_types={1: "mentioned"},
        )
        notification = Notification(
            notification_id=11,
            notification_type=1,
            type_name=None,
            read=False,
            created_at=None,
            topic_id=123,
            post_number=2,
            slug=None,
            data={"username": "bot"},
        )
        self.assertIsNone(classifier.classify(notification))


class StorageTests(unittest.TestCase):
    def test_jobs_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            storage.enqueue_job(
                notification_id=100,
                topic_id=200,
                reply_to_post_number=3,
                raw="Hello",
                decision_reason="Useful reply",
                due_at="2026-01-01T00:00:00+00:00",
                created_at="2026-01-01T00:00:00+00:00",
                presence_channel="/discourse-presence/reply/200",
            )

            reopened = BotStorage(database_path)
            pending = reopened.get_pending_job(100)

        self.assertIsNotNone(pending)
        self.assertEqual(pending.topic_id, 200)
        self.assertEqual(pending.decision_reason, "Useful reply")

    def test_manual_commands_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bot.sqlite3"
            storage = BotStorage(database_path)
            command_id = storage.enqueue_manual_command(
                post_url="https://forum.example.com/t/topic/200/3",
                user_request="Tell them thanks.",
                created_at="2026-01-01T00:00:00+00:00",
            )

            reopened = BotStorage(database_path)
            commands = reopened.list_manual_commands()

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].command_id, command_id)
        self.assertEqual(commands[0].status, "queued")
