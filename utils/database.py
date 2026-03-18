# ============================================================
#  database.py — SQLite History & Statistics
#  Author : Salar Ahmed | FYP 2025-2026
# ============================================================

import sqlite3, os, datetime
from typing import List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/history.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT,
                source      TEXT,
                url         TEXT,
                published   TEXT,
                label       TEXT,
                confidence  REAL,
                bert_vote   TEXT,
                nb_vote     TEXT,
                rf_vote     TEXT,
                model_used  TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


def save_history(entry: Dict):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO predictions
               (title, source, url, published, label, confidence,
                bert_vote, nb_vote, rf_vote, model_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("title",      "")[:300],
                entry.get("source",     "manual"),
                entry.get("url",        ""),
                entry.get("published",  datetime.datetime.utcnow().isoformat()),
                entry.get("label",      "UNKNOWN"),
                entry.get("confidence", 0.0),
                entry.get("bert_vote",  "N/A"),
                entry.get("nb_vote",    "N/A"),
                entry.get("rf_vote",    "N/A"),
                entry.get("model_used", "unknown"),
            ),
        )
        conn.commit()


def get_history(limit: int = 50) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> Dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        fake  = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='FAKE'").fetchone()[0]
        true  = conn.execute("SELECT COUNT(*) FROM predictions WHERE label='TRUE'").fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM predictions"
        ).fetchone()[0] or 0.0
        by_source = conn.execute(
            "SELECT source, COUNT(*) as cnt FROM predictions GROUP BY source"
        ).fetchall()
        recent = conn.execute(
            "SELECT DATE(created_at) as day, COUNT(*) as cnt "
            "FROM predictions GROUP BY day ORDER BY day DESC LIMIT 7"
        ).fetchall()

    return {
        "total"       : total,
        "fake_count"  : fake,
        "true_count"  : true,
        "avg_confidence": round(avg_conf, 4),
        "fake_ratio"  : round(fake / total, 4) if total else 0,
        "by_source"   : [dict(r) for r in by_source],
        "daily_counts": [dict(r) for r in recent],
    }
