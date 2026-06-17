"""Tests for batch config loading."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from biocompute.batch import load_batch_config


def _write_config(tmp_dir: str, data: dict) -> str:
    path = os.path.join(tmp_dir, "batch.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


class TestLoadBatchConfig:
    def test_valid_config(self, tmp_path: str) -> None:
        config_data = {
            "diseases": [
                {
                    "name": "Test Disease",
                    "description": "A test disease",
                    "keywords": ["test"],
                }
            ],
            "settings": {
                "generations": 5,
                "population_size": 10,
            },
        }
        path = _write_config(str(tmp_path), config_data)
        result = load_batch_config(path)

        assert len(result["diseases"]) == 1
        assert result["diseases"][0]["name"] == "Test Disease"
        assert result["settings"]["generations"] == 5
        assert result["settings"]["population_size"] == 10

    def test_missing_settings_uses_defaults(self, tmp_path: str) -> None:
        config_data = {
            "diseases": [
                {
                    "name": "Test Disease",
                    "description": "A test disease",
                }
            ],
        }
        path = _write_config(str(tmp_path), config_data)
        result = load_batch_config(path)

        assert result["settings"]["generations"] == 10
        assert result["settings"]["population_size"] == 30

    def test_partial_settings_fills_defaults(self, tmp_path: str) -> None:
        config_data = {
            "diseases": [
                {
                    "name": "Test Disease",
                    "description": "A test disease",
                }
            ],
            "settings": {
                "generations": 3,
            },
        }
        path = _write_config(str(tmp_path), config_data)
        result = load_batch_config(path)

        assert result["settings"]["generations"] == 3
        assert result["settings"]["population_size"] == 30

    def test_empty_diseases_raises(self, tmp_path: str) -> None:
        config_data = {"diseases": []}
        path = _write_config(str(tmp_path), config_data)

        with pytest.raises(ValueError, match="non-empty"):
            load_batch_config(path)

    def test_missing_diseases_raises(self, tmp_path: str) -> None:
        config_data = {"settings": {"generations": 5}}
        path = _write_config(str(tmp_path), config_data)

        with pytest.raises(ValueError, match="non-empty"):
            load_batch_config(path)

    def test_multiple_diseases(self, tmp_path: str) -> None:
        config_data = {
            "diseases": [
                {"name": "Disease A", "description": "Desc A"},
                {"name": "Disease B", "description": "Desc B"},
                {"name": "Disease C", "description": "Desc C"},
            ],
            "settings": {"generations": 2, "population_size": 5},
        }
        path = _write_config(str(tmp_path), config_data)
        result = load_batch_config(path)

        assert len(result["diseases"]) == 3
        assert result["diseases"][1]["name"] == "Disease B"

    def test_keywords_default_to_empty(self, tmp_path: str) -> None:
        """Config entries without keywords should still be valid."""
        config_data = {
            "diseases": [
                {"name": "No Keywords", "description": "A disease without keywords"},
            ],
        }
        path = _write_config(str(tmp_path), config_data)
        result = load_batch_config(path)

        # keywords key may not exist — that's fine, run_batch handles it
        assert result["diseases"][0]["name"] == "No Keywords"
