"""
Lightweight SQLite conversation memory so chats persist across app restarts
(a real differentiator vs. a stateless chatbot wrapper).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from . import config


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.SQLITE_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def ensure_session(session_id: str, title: str = "New chat"):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, title, created_at) VALUES (?, ?, ?)",
            (session_id, title, datetime.utcnow().isoformat()),
        )
    conn.close()


def save_message(session_id: str, role: str, content: str):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat()),
        )
    conn.close()


def load_history(session_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]


def list_sessions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{"session_id": s, "title": t, "created_at": c} for s, t, c in rows]


def rename_session(session_id: str, title: str):
    conn = _connect()
    with conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id)
        )
    conn.close()


def delete_session(session_id: str):
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.close()
