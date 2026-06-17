"""Shared pytest fixtures for the biocompute test suite."""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_api_cache(tmp_path, monkeypatch):
    """Point the API cache at a per-test temp directory and reset the singleton
    between tests so cached API results never leak across tests."""
    from biocompute.data import cache as _cache_mod

    # Reset any leftover singleton from a previous test
    _cache_mod.reset_cache()

    # Override the default DB path to a per-test temp file
    monkeypatch.setattr(_cache_mod, "_CACHE_DB", str(tmp_path / "test_cache.db"))

    yield

    _cache_mod.reset_cache()
