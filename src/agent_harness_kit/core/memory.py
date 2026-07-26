"""Persistence for HarnessContext across runs and process restarts."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from agent_harness_kit.core.context import HarnessContext


class MemoryProvider(ABC):
    """Loads and saves a HarnessContext keyed by session_id.

    ``load`` never raises for an unknown session — it returns a fresh, empty
    ``HarnessContext`` for that id instead, so callers (chiefly
    ``HarnessLoop.run_supervised``) can always treat the return value as
    "the context to work with", creating it implicitly on first use.
    """

    @abstractmethod
    def load(self, session_id: str) -> HarnessContext:
        """Return the stored context for ``session_id``, or a new empty one."""
        raise NotImplementedError

    @abstractmethod
    def save(self, session_id: str, context: HarnessContext) -> None:
        """Persist ``context`` under ``session_id``, overwriting any prior value."""
        raise NotImplementedError


class InMemoryProvider(MemoryProvider):
    """Process-local, dict-backed storage. Useful for tests and single-process runs.

    Nothing here survives past the life of the Python process — use
    ``SQLiteMemoryProvider`` if you need the run to survive a restart.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def load(self, session_id: str) -> HarnessContext:
        data = self._store.get(session_id)
        if data is None:
            return HarnessContext(session_id=session_id)
        return HarnessContext.from_dict(data)

    def save(self, session_id: str, context: HarnessContext) -> None:
        self._store[session_id] = context.to_dict()


class SQLiteMemoryProvider(MemoryProvider):
    """Durable storage backed by a SQLite file, one row per session_id.

    Uses only the stdlib ``sqlite3`` and ``json`` modules — the whole
    ``HarnessContext`` is serialized to a JSON blob and stored in a single
    ``TEXT`` column, keyed by ``session_id`` (primary key, so a save is an
    upsert).
    """

    def __init__(self, db_path: str = "agent_harness.db") -> None:
        self.db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_context (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def load(self, session_id: str) -> HarnessContext:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT data FROM harness_context WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return HarnessContext(session_id=session_id)
        return HarnessContext.from_dict(json.loads(row[0]))

    def save(self, session_id: str, context: HarnessContext) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO harness_context (session_id, data)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET data = excluded.data
                """,
                (session_id, json.dumps(context.to_dict())),
            )
            conn.commit()
        finally:
            conn.close()
