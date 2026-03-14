"""SQLite storage backend — the default, zero-config backend."""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Backend
from ..types import Memory, MemoryType


class SQLiteBackend(Backend):
    """Stores memories in a local SQLite database.

    Args:
        db_path: Filesystem path for the ``.db`` file, or ``":memory:"``
                 for a purely in-memory database (default).
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                type        INTEGER NOT NULL,
                content     TEXT    NOT NULL,
                vector      BLOB,
                timestamp   REAL    NOT NULL,
                score       REAL    DEFAULT 0.0,
                metadata    TEXT    DEFAULT '{}'
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memories (id, type, content, vector, timestamp, score, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                memory.id,
                memory.type.value,
                memory.content,
                memory.vector.tobytes() if memory.vector is not None else None,
                memory.timestamp,
                memory.score,
                json.dumps(memory.metadata),
            ),
        )
        self._conn.commit()

    def load(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT id, type, content, vector, timestamp, score, metadata FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def search(self, query: str, limit: int = 10) -> List[Memory]:
        rows = self._conn.execute(
            "SELECT id, type, content, vector, timestamp, score, metadata "
            "FROM memories WHERE content LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def all(self) -> List[Memory]:
        rows = self._conn.execute(
            "SELECT id, type, content, vector, timestamp, score, metadata FROM memories"
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def delete(self, memory_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def clear(self) -> None:
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_memory(row: tuple) -> Memory:
        mid, mtype, content, vec_bytes, ts, score, meta_json = row
        vector = np.frombuffer(vec_bytes, dtype=np.float32).copy() if vec_bytes else np.zeros(384, dtype=np.float32)
        return Memory(
            id=mid,
            type=MemoryType(mtype),
            content=content,
            vector=vector,
            timestamp=ts,
            score=score,
            metadata=json.loads(meta_json) if meta_json else {},
        )

    def close(self) -> None:
        self._conn.close()
