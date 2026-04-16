"""Tests for the SQLite-based API result cache."""

import os
import tempfile
import time

import pytest

from biocompute.data.cache import ApiCache, cache_key


@pytest.fixture
def tmp_cache(tmp_path):
    """Create a cache using a temporary directory."""
    db_path = os.path.join(str(tmp_path), "test_cache.db")
    cache = ApiCache(db_path=db_path)
    yield cache
    cache.close()


class TestCacheKey:
    def test_basic_key(self):
        assert cache_key("pubmed", "CXCR4", "cancer") == "pubmed:cxcr4:cancer"

    def test_key_lowercased(self):
        assert cache_key("PubMed", "VEGF", "Heart Failure") == "pubmed:vegf:heart failure"

    def test_key_without_disease(self):
        assert cache_key("hpa", "TP53") == "hpa:tp53:"

    def test_key_with_empty_disease(self):
        assert cache_key("string", "BRCA1", "") == "string:brca1:"


class TestApiCache:
    def test_set_and_get(self, tmp_cache):
        data = {"pmid_count": 42, "pmids": ["123", "456"]}
        tmp_cache.set("pubmed:cxcr4:cancer", data)
        result = tmp_cache.get("pubmed:cxcr4:cancer")
        assert result == data

    def test_get_missing_key(self, tmp_cache):
        assert tmp_cache.get("nonexistent:key:here") is None

    def test_overwrite_existing(self, tmp_cache):
        tmp_cache.set("key:a:b", {"v": 1})
        tmp_cache.set("key:a:b", {"v": 2})
        assert tmp_cache.get("key:a:b") == {"v": 2}

    def test_ttl_expiry(self, tmp_cache, monkeypatch):
        data = {"count": 10}
        tmp_cache.set("key:gene:disease", data)

        # Confirm it's there now
        assert tmp_cache.get("key:gene:disease") == data

        # Fast-forward time past TTL (7 days + 1 second)
        future = time.time() + 86400 * 7 + 1
        monkeypatch.setattr(time, "time", lambda: future)

        assert tmp_cache.get("key:gene:disease") is None

    def test_ttl_not_expired(self, tmp_cache, monkeypatch):
        data = {"count": 10}
        tmp_cache.set("key:gene:disease", data)

        # Fast-forward time to just before TTL
        future = time.time() + 86400 * 7 - 10
        monkeypatch.setattr(time, "time", lambda: future)

        assert tmp_cache.get("key:gene:disease") == data

    def test_no_cache_env_var(self, tmp_cache, monkeypatch):
        data = {"count": 5}
        tmp_cache.set("key:gene:d", data)

        monkeypatch.setenv("BIOCOMPUTE_NO_CACHE", "1")
        assert tmp_cache.get("key:gene:d") is None

        monkeypatch.delenv("BIOCOMPUTE_NO_CACHE")
        assert tmp_cache.get("key:gene:d") == data

    def test_complex_json_roundtrip(self, tmp_cache):
        data = {
            "gene": "CXCR4",
            "tissues": [{"tissue": "brain", "level": "high"}],
            "nested": {"a": [1, 2, 3], "b": None},
            "score": 0.95,
        }
        tmp_cache.set("complex:key:", data)
        assert tmp_cache.get("complex:key:") == data
