# pyright: reportMissingImports=false

"""Tests for live calibration — all API calls are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from biocompute.calibration.ground_truth import CALIBRATION_SET, CalibrationEntry
from biocompute.calibration.live import (
    LiveCalibrationResult,
    _entry_to_hypothesis,
    _entry_to_query,
    evaluate_calibration_live,
)
from biocompute.models import FitnessScores, Weights


def _make_mock_raw_data(
    gene: str,
    disease: str,
    *,
    pmid_count: int = 10,
    citations: int = 50,
    interaction_count: int = 5,
) -> dict[str, object]:
    """Build realistic raw_data dict matching what collect_bio_data returns."""
    return {
        "literature": {
            "gene": gene,
            "disease": disease,
            "pmid_count": pmid_count,
            "pmids": [f"PMID{i}" for i in range(min(pmid_count, 5))],
            "total_citations": citations,
            "influential_citations": citations // 5,
            "source": "pubmed+semantic_scholar",
        },
        "expression": {
            "gene": gene,
            "tissues": [
                {"tissue": "brain", "level": "high"},
                {"tissue": "liver", "level": "medium"},
            ],
            "source": "hpa",
        },
        "pathway": {
            "gene": gene,
            "interactions": [{"partner": f"P{i}", "score": 0.8} for i in range(interaction_count)],
            "interaction_count": interaction_count,
            "source": "string",
        },
        "druggability": {
            "gene": gene,
            "tractability": ["small_molecule"],
            "known_drugs_count": 2,
            "safety_liabilities": [],
            "source": "opentargets",
        },
        "safety": {
            "gene": gene,
            "tractability": ["small_molecule"],
            "known_drugs_count": 2,
            "safety_liabilities": [],
            "source": "opentargets",
        },
        "ip": {
            "source": "opentargets_heuristic",
            "freedom_estimate": 0.6,
        },
    }


@pytest.fixture()
def mock_collect():
    """Patch _collect_single_hypothesis to return predetermined raw data."""

    async def _fake_collect(client, hypothesis, query):
        return _make_mock_raw_data(hypothesis.target_gene, query.name)

    with patch(
        "biocompute.calibration.live._collect_single_entry",
    ) as mock_entry:
        # We patch at a higher level — _collect_single_entry — to avoid
        # needing httpx. Each call returns a tuple (key, scores, error).
        # Instead, patch the lower-level async function used inside it.
        mock_entry.side_effect = None  # clear
        yield _fake_collect


def test_entry_to_hypothesis():
    entry = CalibrationEntry("VEGF", "mAb", "systemic", "cancer", "SUCCESS")
    h = _entry_to_hypothesis(entry)
    assert h.target_gene == "VEGF"
    assert h.modality == "mAb"
    assert h.delivery == "systemic"
    assert h.duration == "chronic"
    assert h.tissue_context == ""


def test_entry_to_query():
    entry = CalibrationEntry("VEGF", "mAb", "systemic", "cancer", "SUCCESS")
    q = _entry_to_query(entry)
    assert q.name == "cancer"
    assert q.description == "cancer"
    assert "VEGF" in q.keywords
    assert "cancer" in q.keywords


def test_evaluate_calibration_live_processes_all_entries():
    """Verify live calibration processes all 12 entries with mocked API data."""

    # Mock _collect_single_entry to return predetermined scores
    async def _fake_collect(entry):
        from biocompute.calibration.tune import _calibration_key

        key = _calibration_key(entry)
        # Give SUCCESS entries higher scores than FAIL entries
        if entry.outcome == "SUCCESS":
            scores = FitnessScores(
                literature_strength=0.8,
                expression_specificity=0.7,
                pathway_centrality=0.7,
                druggability=0.8,
                safety_profile=0.7,
                ip_freedom=0.5,
            )
            return (key, scores, 0.72, "")
        else:
            scores = FitnessScores(
                literature_strength=0.3,
                expression_specificity=0.2,
                pathway_centrality=0.3,
                druggability=0.3,
                safety_profile=0.2,
                ip_freedom=0.2,
            )
            return (key, scores, 0.25, "")

    with patch(
        "biocompute.calibration.live._collect_single_entry",
        side_effect=_fake_collect,
    ):
        result = evaluate_calibration_live(CALIBRATION_SET)

    # All 12 entries should be scored
    assert len(result["scores_map"]) == len(CALIBRATION_SET)
    assert len(result["skipped"]) == 0

    # Evaluation should have separation data
    evaluation = result["evaluation"]
    assert "separation_score" in evaluation
    assert "success_mean" in evaluation
    assert "fail_mean" in evaluation
    assert len(evaluation["success_scores"]) > 0
    assert len(evaluation["fail_scores"]) > 0

    # With our mock data, successes should score higher than failures
    assert evaluation["success_mean"] > evaluation["fail_mean"]
    assert evaluation["separation_score"] > 0.5

    # Tuned weights should be present
    assert result["tuned_weights"] is not None
    assert result["tuned_evaluation"] is not None

    # Elapsed time should be recorded
    assert result["elapsed_seconds"] >= 0


def test_evaluate_calibration_live_handles_failures_gracefully():
    """When an entry fails, it should be skipped and others still processed."""

    call_count = 0

    async def _partial_fail(entry):
        nonlocal call_count
        from biocompute.calibration.tune import _calibration_key

        call_count += 1
        key = _calibration_key(entry)

        # Fail on the second entry
        if call_count == 2:
            return (key, None, 0.0, "Simulated API failure")

        scores = FitnessScores(
            literature_strength=0.5,
            expression_specificity=0.5,
            pathway_centrality=0.5,
            druggability=0.5,
            safety_profile=0.5,
            ip_freedom=0.5,
        )
        return (key, scores, 0.5, "")

    with patch(
        "biocompute.calibration.live._collect_single_entry",
        side_effect=_partial_fail,
    ):
        result = evaluate_calibration_live(CALIBRATION_SET)

    # One entry should be skipped
    assert len(result["skipped"]) == 1
    assert len(result["scores_map"]) == len(CALIBRATION_SET) - 1

    # Should still produce valid evaluation
    assert result["evaluation"]["separation_score"] >= 0.0


def test_evaluate_calibration_live_returns_correct_structure():
    """Verify the returned dict has all required keys."""

    async def _fixed_scores(entry):
        from biocompute.calibration.tune import _calibration_key

        key = _calibration_key(entry)
        scores = FitnessScores(
            literature_strength=0.5,
            expression_specificity=0.5,
            pathway_centrality=0.5,
            druggability=0.5,
            safety_profile=0.5,
            ip_freedom=0.5,
        )
        return (key, scores, 0.5, "")

    with patch(
        "biocompute.calibration.live._collect_single_entry",
        side_effect=_fixed_scores,
    ):
        result = evaluate_calibration_live(CALIBRATION_SET)

    # Check all required keys
    assert "scores_map" in result
    assert "evaluation" in result
    assert "tuned_weights" in result
    assert "tuned_evaluation" in result
    assert "skipped" in result
    assert "elapsed_seconds" in result

    # scores_map keys should be (str, str) tuples
    for key in result["scores_map"]:
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)

    # Each value should be a FitnessScores instance
    for scores in result["scores_map"].values():
        assert isinstance(scores, FitnessScores)


def test_evaluate_calibration_live_with_custom_weights():
    """Verify custom weights are passed through to evaluation."""

    async def _fixed_scores(entry):
        from biocompute.calibration.tune import _calibration_key

        key = _calibration_key(entry)
        scores = FitnessScores(
            literature_strength=0.8,
            expression_specificity=0.6,
            pathway_centrality=0.5,
            druggability=0.7,
            safety_profile=0.6,
            ip_freedom=0.4,
        )
        return (key, scores, 0.5, "")

    custom_weights = Weights(
        literature_strength=0.30,
        expression_specificity=0.25,
        pathway_centrality=0.15,
        druggability=0.15,
        safety_profile=0.10,
        ip_freedom=0.05,
    )

    with patch(
        "biocompute.calibration.live._collect_single_entry",
        side_effect=_fixed_scores,
    ):
        result = evaluate_calibration_live(CALIBRATION_SET, weights=custom_weights)

    # Should still produce valid results
    assert len(result["scores_map"]) == len(CALIBRATION_SET)
    assert result["evaluation"]["separation_score"] >= 0.0


def test_evaluate_calibration_live_progress_callback():
    """Verify progress callback is invoked for each entry."""
    progress_calls: list[tuple[int, int]] = []

    def on_progress(i: int, total: int, entry: object) -> None:
        progress_calls.append((i, total))

    async def _fixed_scores(entry):
        from biocompute.calibration.tune import _calibration_key

        key = _calibration_key(entry)
        scores = FitnessScores(
            literature_strength=0.5,
            expression_specificity=0.5,
            pathway_centrality=0.5,
            druggability=0.5,
            safety_profile=0.5,
            ip_freedom=0.5,
        )
        return (key, scores, 0.5, "")

    with patch(
        "biocompute.calibration.live._collect_single_entry",
        side_effect=_fixed_scores,
    ):
        evaluate_calibration_live(
            CALIBRATION_SET,
            on_progress=on_progress,
        )

    # Progress should be called once per entry
    assert len(progress_calls) == len(CALIBRATION_SET)
    # Each call should report correct total
    for i, (idx, total) in enumerate(progress_calls):
        assert idx == i
        assert total == len(CALIBRATION_SET)
