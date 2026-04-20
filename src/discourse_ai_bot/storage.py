from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from discourse_ai_bot.models import ManualCommand, PendingJob


class BotStorage:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self) -> sqlite3.Connection:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS handled_notifications (
                    notification_id INTEGER PRIMARY KEY,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    response_post_id INTEGER,
                    handled_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_replies (
                    notification_id INTEGER PRIMARY KEY,
                    topic_id INTEGER NOT NULL,
                    reply_to_post_number INTEGER,
                    raw TEXT NOT NULL,
                    gif_id TEXT,
                    decision_reason TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    presence_channel TEXT,
                    last_presence_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_pending_replies_due_at
                ON pending_replies(due_at);

                CREATE TABLE IF NOT EXISTS manual_commands (
                    command_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_url TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    topic_id INTEGER,
                    reply_to_post_number INTEGER,
                    raw TEXT,
                    gif_id TEXT,
                    ollama_reason TEXT,
                    due_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    presence_channel TEXT,
                    last_presence_at TEXT,
                    response_post_id INTEGER,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_manual_commands_status_available_at
                ON manual_commands(status, available_at);

                CREATE INDEX IF NOT EXISTS idx_manual_commands_status_due_at
                ON manual_commands(status, due_at);
                """
            )
            _ensure_column(connection, "pending_replies", "gif_id", "TEXT")
            _ensure_column(connection, "manual_commands", "gif_id", "TEXT")

    def is_handled(self, notification_id: int) -> bool:
        with self._session() as connection:
            row = connection.execute(
                "SELECT 1 FROM handled_notifications WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
        return row is not None

    def get_pending_job(self, notification_id: int) -> PendingJob | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT * FROM pending_replies WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
        return _row_to_pending_job(row) if row else None

    def list_pending_jobs(self) -> list[PendingJob]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT * FROM pending_replies ORDER BY due_at ASC, notification_id ASC"
            ).fetchall()
        return [_row_to_pending_job(row) for row in rows]

    def list_due_jobs(self, due_at: str) -> list[PendingJob]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM pending_replies
                WHERE due_at <= ?
                ORDER BY due_at ASC, notification_id ASC
                """,
                (due_at,),
            ).fetchall()
        return [_row_to_pending_job(row) for row in rows]

    def enqueue_job(
        self,
        *,
        notification_id: int,
        topic_id: int,
        reply_to_post_number: int | None,
        raw: str,
        decision_reason: str,
        due_at: str,
        created_at: str,
        presence_channel: str | None,
        gif_id: str | None = None,
    ) -> bool:
        with self._session() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO pending_replies (
                    notification_id,
                    topic_id,
                    reply_to_post_number,
                    raw,
                    gif_id,
                    decision_reason,
                    due_at,
                    attempts,
                    last_error,
                    created_at,
                    presence_channel,
                    last_presence_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, NULL)
                """,
                (
                    notification_id,
                    topic_id,
                    reply_to_post_number,
                    raw,
                    gif_id,
                    decision_reason,
                    due_at,
                    created_at,
                    presence_channel,
                ),
            )
            rowcount = cursor.rowcount
        return rowcount > 0

    def update_job_presence(self, notification_id: int, last_presence_at: str) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE pending_replies
                SET last_presence_at = ?
                WHERE notification_id = ?
                """,
                (last_presence_at, notification_id),
            )

    def reschedule_job(
        self,
        notification_id: int,
        *,
        due_at: str,
        attempts: int,
        last_error: str,
    ) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE pending_replies
                SET due_at = ?, attempts = ?, last_error = ?, last_presence_at = NULL
                WHERE notification_id = ?
                """,
                (due_at, attempts, last_error, notification_id),
            )

    def record_handled(
        self,
        notification_id: int,
        *,
        action: str,
        reason: str,
        handled_at: str,
        response_post_id: int | None = None,
    ) -> None:
        with self._session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO handled_notifications (
                    notification_id,
                    action,
                    reason,
                    response_post_id,
                    handled_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (notification_id, action, reason, response_post_id, handled_at),
            )
            connection.execute(
                "DELETE FROM pending_replies WHERE notification_id = ?",
                (notification_id,),
            )

    def enqueue_manual_command(
        self,
        *,
        post_url: str,
        user_request: str,
        created_at: str,
    ) -> int:
        with self._session() as connection:
            cursor = connection.execute(
                """
                INSERT INTO manual_commands (
                    post_url,
                    user_request,
                    status,
                    created_at,
                    available_at,
                    topic_id,
                    reply_to_post_number,
                    raw,
                    ollama_reason,
                    due_at,
                    attempts,
                    last_error,
                    presence_channel,
                    last_presence_at,
                    response_post_id,
                    completed_at
                ) VALUES (?, ?, 'queued', ?, ?, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, NULL, NULL)
                """,
                (post_url, user_request, created_at, created_at),
            )
            command_id = int(cursor.lastrowid)
        return command_id

    def list_ready_manual_commands(self, available_at: str) -> list[ManualCommand]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM manual_commands
                WHERE status = 'queued' AND available_at <= ?
                ORDER BY available_at ASC, command_id ASC
                """,
                (available_at,),
            ).fetchall()
        return [_row_to_manual_command(row) for row in rows]

    def list_scheduled_manual_commands(self) -> list[ManualCommand]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM manual_commands
                WHERE status = 'scheduled'
                ORDER BY due_at ASC, command_id ASC
                """
            ).fetchall()
        return [_row_to_manual_command(row) for row in rows]

    def list_due_manual_commands(self, due_at: str) -> list[ManualCommand]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM manual_commands
                WHERE status = 'scheduled' AND due_at <= ?
                ORDER BY due_at ASC, command_id ASC
                """,
                (due_at,),
            ).fetchall()
        return [_row_to_manual_command(row) for row in rows]

    def schedule_manual_command(
        self,
        command_id: int,
        *,
        topic_id: int,
        reply_to_post_number: int | None,
        raw: str,
        ollama_reason: str,
        due_at: str,
        presence_channel: str | None,
        gif_id: str | None = None,
    ) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE manual_commands
                SET status = 'scheduled',
                    topic_id = ?,
                    reply_to_post_number = ?,
                    raw = ?,
                    gif_id = ?,
                    ollama_reason = ?,
                    due_at = ?,
                    attempts = 0,
                    last_error = NULL,
                    presence_channel = ?,
                    last_presence_at = NULL
                WHERE command_id = ?
                """,
                (
                    topic_id,
                    reply_to_post_number,
                    raw,
                    gif_id,
                    ollama_reason,
                    due_at,
                    presence_channel,
                    command_id,
                ),
            )

    def reschedule_manual_command_generation(
        self,
        command_id: int,
        *,
        available_at: str,
        attempts: int,
        last_error: str,
    ) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE manual_commands
                SET available_at = ?, attempts = ?, last_error = ?
                WHERE command_id = ?
                """,
                (available_at, attempts, last_error, command_id),
            )

    def update_manual_command_presence(self, command_id: int, last_presence_at: str) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE manual_commands
                SET last_presence_at = ?
                WHERE command_id = ?
                """,
                (last_presence_at, command_id),
            )

    def reschedule_manual_command_send(
        self,
        command_id: int,
        *,
        due_at: str,
        attempts: int,
        last_error: str,
    ) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE manual_commands
                SET due_at = ?, attempts = ?, last_error = ?, last_presence_at = NULL
                WHERE command_id = ?
                """,
                (due_at, attempts, last_error, command_id),
            )

    def complete_manual_command(
        self,
        command_id: int,
        *,
        response_post_id: int | None,
        completed_at: str,
    ) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE manual_commands
                SET status = 'completed',
                    response_post_id = ?,
                    completed_at = ?,
                    last_error = NULL,
                    presence_channel = NULL,
                    last_presence_at = NULL
                WHERE command_id = ?
                """,
                (response_post_id, completed_at, command_id),
            )

    def list_manual_commands(self) -> list[ManualCommand]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM manual_commands
                ORDER BY created_at ASC, command_id ASC
                """
            ).fetchall()
        return [_row_to_manual_command(row) for row in rows]

    def clear_queue(self) -> dict[str, int]:
        with self._session() as connection:
            pending_replies_deleted = connection.execute(
                "DELETE FROM pending_replies"
            ).rowcount
            manual_commands_deleted = connection.execute(
                """
                DELETE FROM manual_commands
                WHERE status IN ('queued', 'scheduled')
                """
            ).rowcount
        return {
            "pending_replies_deleted": int(pending_replies_deleted),
            "manual_commands_deleted": int(manual_commands_deleted),
        }

    def reset_database(self) -> dict[str, int]:
        with self._session() as connection:
            handled_notifications_deleted = connection.execute(
                "DELETE FROM handled_notifications"
            ).rowcount
            pending_replies_deleted = connection.execute(
                "DELETE FROM pending_replies"
            ).rowcount
            manual_commands_deleted = connection.execute(
                "DELETE FROM manual_commands"
            ).rowcount
        return {
            "handled_notifications_deleted": int(handled_notifications_deleted),
            "pending_replies_deleted": int(pending_replies_deleted),
            "manual_commands_deleted": int(manual_commands_deleted),
        }

    def stats_summary(self) -> dict[str, int]:
        with self._session() as connection:
            handled_total = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM handled_notifications",
            )
            handled_replied = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM handled_notifications WHERE action = 'reply'",
            )
            handled_skipped = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM handled_notifications WHERE action = 'skip'",
            )
            pending_replies = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM pending_replies",
            )
            pending_reply_errors = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM pending_replies WHERE last_error IS NOT NULL",
            )
            manual_total = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM manual_commands",
            )
            manual_queued = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM manual_commands WHERE status = 'queued'",
            )
            manual_scheduled = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM manual_commands WHERE status = 'scheduled'",
            )
            manual_completed = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM manual_commands WHERE status = 'completed'",
            )
            manual_errors = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM manual_commands WHERE last_error IS NOT NULL",
            )
        return {
            "handled_total": handled_total,
            "handled_replied": handled_replied,
            "handled_skipped": handled_skipped,
            "pending_replies": pending_replies,
            "pending_reply_errors": pending_reply_errors,
            "manual_total": manual_total,
            "manual_queued": manual_queued,
            "manual_scheduled": manual_scheduled,
            "manual_completed": manual_completed,
            "manual_errors": manual_errors,
        }


def _row_to_pending_job(row: sqlite3.Row) -> PendingJob:
    return PendingJob(
        notification_id=int(row["notification_id"]),
        topic_id=int(row["topic_id"]),
        reply_to_post_number=row["reply_to_post_number"],
        raw=str(row["raw"]),
        gif_id=row["gif_id"],
        decision_reason=str(row["decision_reason"]),
        due_at=str(row["due_at"]),
        attempts=int(row["attempts"]),
        last_error=row["last_error"],
        created_at=str(row["created_at"]),
        presence_channel=row["presence_channel"],
        last_presence_at=row["last_presence_at"],
    )


def _row_to_manual_command(row: sqlite3.Row) -> ManualCommand:
    return ManualCommand(
        command_id=int(row["command_id"]),
        post_url=str(row["post_url"]),
        user_request=str(row["user_request"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        available_at=str(row["available_at"]),
        topic_id=row["topic_id"],
        reply_to_post_number=row["reply_to_post_number"],
        raw=row["raw"],
        gif_id=row["gif_id"],
        ollama_reason=row["ollama_reason"],
        due_at=row["due_at"],
        attempts=int(row["attempts"]),
        last_error=row["last_error"],
        presence_channel=row["presence_channel"],
        last_presence_at=row["last_presence_at"],
        response_post_id=row["response_post_id"],
        completed_at=row["completed_at"],
    )


def _scalar_int(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in columns:
        return
    connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
    )
