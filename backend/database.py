"""
SQLite database for user interaction history and personalization.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "recommendation_history.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database schema."""
    try:
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    extracted_preferences TEXT,
                    recommended_product_ids TEXT,
                    session_id TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_timestamp 
                ON recommendation_history(timestamp)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


def save_recommendation(
    query: str,
    preferences: Dict[str, Any],
    product_ids: List[str],
    session_id: Optional[str] = None,
) -> int:
    """Save a recommendation event to history."""
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recommendation_history 
                (timestamp, query, extracted_preferences, recommended_product_ids, session_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    query,
                    json.dumps(preferences),
                    json.dumps(product_ids),
                    session_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to save recommendation: {e}")
        return -1


def get_recent_history(limit: int = 5, session_id: Optional[str] = None) -> List[Dict]:
    """Get recent recommendation history."""
    try:
        with get_connection() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT * FROM recommendation_history 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM recommendation_history 
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []


def get_recent_preferences(session_id: Optional[str] = None, limit: int = 10) -> List[str]:
    """Extract recent search queries for personalization context."""
    try:
        history = get_recent_history(limit=limit, session_id=session_id)
        return [h["query"] for h in history if h.get("query")]
    except Exception as e:
        logger.error(f"Failed to get recent preferences: {e}")
        return []


def clear_old_history(days: int = 30) -> None:
    """Clear history older than specified days."""
    try:
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM recommendation_history 
                WHERE timestamp < datetime('now', ?)
                """,
                (f"-{days} days",),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to clear old history: {e}")
