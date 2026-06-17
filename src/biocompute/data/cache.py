"""SQLite-based API result cache for biocompute.

Caches external API responses (PubMed, Semantic Scholar, HPA, String DB,
OpenTargets, ClinicalTrials) so repeated queries for the same gene+disease
across generations and re-runs skip the network call.

LLM calls are NOT cached — they should always be fresh.
"""

import json
import os
import sqlite3
import time

_CACHE_DB = os.path.expanduser("~/.biocompute/api_cache.db")
_CACHE_TTL = 86400 * 7  # 7 days


class ApiCache:
    def __init__(self, db_path: str = _CACHE_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at REAL
            )
        """)
        self.conn.commit()

    def get(self, key: str) -> dict | None:
        """Return cached value or None if missing/expired.

        Always returns None when BIOCOMPUTE_NO_CACHE env var is set.
        """
        if os.environ.get("BIOCOMPUTE_NO_CACHE"):
            return None
        row = self.conn.execute(
            "SELECT value, created_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value, created_at = row
        if time.time() - created_at > _CACHE_TTL:
            self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self.conn.commit()
            return None
        return json.loads(value)

    def set(self, key: str, value: dict) -> None:
        """Store a value in the cache (upsert)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# Module-level singleton
_cache: ApiCache | None = None


def get_cache() -> ApiCache:
    """Return the module-level ApiCache singleton."""
    global _cache
    if _cache is None:
        _cache = ApiCache(db_path=_CACHE_DB)
    return _cache


def reset_cache() -> None:
    """Close and discard the module-level singleton (for testing)."""
    global _cache
    if _cache is not None:
        _cache.close()
        _cache = None


def cache_key(api: str, gene: str, disease: str = "") -> str:
    """Build a deterministic cache key from API name, gene, and disease."""
    return f"{api}:{gene}:{disease}".lower()
