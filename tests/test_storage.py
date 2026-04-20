from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discourse_ai_bot.storage import BotStorage


class StorageMaintenanceTests(unittest.TestCase):
    def test_clear_queue_removes_only_pending_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            storage.enqueue_job(
                notification_id=1,
                topic_id=10,
                reply_to_post_number=2,
                raw="hello",
                decision_reason="reason",
                due_at="2026-01-01T00:00:00+00:00",
                created_at="2026-01-01T00:00:00+00:00",
                presence_channel=None,
            )
            queued_id = storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/1",
                user_request="reply",
                created_at="2026-01-01T00:00:00+00:00",
            )
            completed_id = storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/2",
                user_request="reply",
                created_at="2026-01-01T00:00:00+00:00",
            )
            storage.schedule_manual_command(
                queued_id,
                topic_id=10,
                reply_to_post_number=2,
                raw="queued",
                ollama_reason="because",
                due_at="2026-01-01T00:00:10+00:00",
                presence_channel=None,
            )
            storage.complete_manual_command(
                completed_id,
                response_post_id=99,
                completed_at="2026-01-01T00:00:20+00:00",
            )

            result = storage.clear_queue()
            remaining = storage.list_manual_commands()

        self.assertEqual(result["pending_replies_deleted"], 1)
        self.assertEqual(result["manual_commands_deleted"], 1)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].status, "completed")

    def test_reset_database_removes_all_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            storage.record_handled(
                1,
                action="skip",
                reason="done",
                handled_at="2026-01-01T00:00:00+00:00",
            )
            storage.enqueue_job(
                notification_id=2,
                topic_id=10,
                reply_to_post_number=2,
                raw="hello",
                decision_reason="reason",
                due_at="2026-01-01T00:00:00+00:00",
                created_at="2026-01-01T00:00:00+00:00",
                presence_channel=None,
            )
            storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/1",
                user_request="reply",
                created_at="2026-01-01T00:00:00+00:00",
            )

            result = storage.reset_database()

        self.assertEqual(result["handled_notifications_deleted"], 1)
        self.assertEqual(result["pending_replies_deleted"], 1)
        self.assertEqual(result["manual_commands_deleted"], 1)

    def test_stats_summary_reports_queue_and_handled_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = BotStorage(Path(temp_dir) / "bot.sqlite3")
            storage.record_handled(
                1,
                action="reply",
                reason="done",
                handled_at="2026-01-01T00:00:00+00:00",
            )
            storage.record_handled(
                2,
                action="skip",
                reason="skip",
                handled_at="2026-01-01T00:00:00+00:00",
            )
            storage.enqueue_job(
                notification_id=3,
                topic_id=10,
                reply_to_post_number=2,
                raw="hello",
                decision_reason="reason",
                due_at="2026-01-01T00:00:00+00:00",
                created_at="2026-01-01T00:00:00+00:00",
                presence_channel=None,
            )
            storage.reschedule_job(
                3,
                due_at="2026-01-01T00:10:00+00:00",
                attempts=1,
                last_error="bad target",
            )
            queued_id = storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/1",
                user_request="reply",
                created_at="2026-01-01T00:00:00+00:00",
            )
            scheduled_id = storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/2",
                user_request="reply",
                created_at="2026-01-01T00:00:00+00:00",
            )
            completed_id = storage.enqueue_manual_command(
                post_url="https://forum.example.com/p/3",
                user_request="reply",
                created_at="2026-01-01T00:00:00+00:00",
            )
            storage.schedule_manual_command(
                scheduled_id,
                topic_id=10,
                reply_to_post_number=2,
                raw="scheduled",
                ollama_reason="because",
                due_at="2026-01-01T00:00:10+00:00",
                presence_channel=None,
            )
            storage.reschedule_manual_command_generation(
                queued_id,
                available_at="2026-01-01T00:05:00+00:00",
                attempts=1,
                last_error="bad url",
            )
            storage.complete_manual_command(
                completed_id,
                response_post_id=99,
                completed_at="2026-01-01T00:00:20+00:00",
            )

            summary = storage.stats_summary()

        self.assertEqual(summary["handled_total"], 2)
        self.assertEqual(summary["handled_replied"], 1)
        self.assertEqual(summary["handled_skipped"], 1)
        self.assertEqual(summary["pending_replies"], 1)
        self.assertEqual(summary["pending_reply_errors"], 1)
        self.assertEqual(summary["manual_total"], 3)
        self.assertEqual(summary["manual_queued"], 1)
        self.assertEqual(summary["manual_scheduled"], 1)
        self.assertEqual(summary["manual_completed"], 1)
        self.assertEqual(summary["manual_errors"], 1)
