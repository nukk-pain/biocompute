# pyright: reportMissingImports=false

"""Tests for clinical trial outcome scoring and API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from biocompute.fitness.clinical import score_clinical


# --- score_clinical tests ---


def test_score_clinical_no_trials_returns_neutral():
    """No trial data should return 1.0 (neutral)."""
    data = {
        "completed_count": 0,
        "failed_count": 0,
        "phase3_failures": 0,
        "failure_ratio": 0.0,
        "failed_trial_names": [],
        "source": "clinicaltrials_gov",
    }
    score, evidence = score_clinical(data, gene="CXCL12")
    assert score == 1.0
    assert evidence == []


def test_score_clinical_phase3_failures_penalize():
    """Phase 2/3 failures should reduce score."""
    data = {
        "completed_count": 5,
        "failed_count": 3,
        "phase3_failures": 2,
        "failure_ratio": 0.375,
        "failed_trial_names": ["tanezumab Phase 3", "drug X Phase 2"],
        "source": "clinicaltrials_gov",
    }
    score, evidence = score_clinical(data, gene="NGF")
    # 2 phase3 failures: penalty = min(2 * 0.15, 0.5) = 0.3
    # score = 1.0 - 0.3 = 0.7
    assert score == pytest.approx(0.7, abs=0.01)
    assert len(evidence) == 1
    assert "phase2/3_failures=2" in evidence[0].summary


def test_score_clinical_high_failure_ratio_extra_penalty():
    """High failure ratio with many trials triggers additional penalty."""
    data = {
        "completed_count": 1,
        "failed_count": 5,
        "phase3_failures": 1,
        "failure_ratio": 0.833,
        "failed_trial_names": ["trial1", "trial2", "trial3"],
        "source": "clinicaltrials_gov",
    }
    score, evidence = score_clinical(data, gene="APP")
    # phase3 penalty: min(1*0.15, 0.5) = 0.15
    # failure_ratio 0.833 > 0.5 and total 6 > 3: additional -0.2
    # score = 1.0 - 0.15 - 0.2 = 0.65
    assert score == pytest.approx(0.65, abs=0.01)
    assert len(evidence) == 1


def test_score_clinical_all_completed_bonus():
    """All completed, no failures should give a bonus."""
    data = {
        "completed_count": 5,
        "failed_count": 0,
        "phase3_failures": 0,
        "failure_ratio": 0.0,
        "failed_trial_names": [],
        "source": "clinicaltrials_gov",
    }
    score, evidence = score_clinical(data, gene="TNF")
    # score = 1.0 + 0.1 bonus = 1.0 (capped)
    assert score == 1.0
    assert len(evidence) == 1
    assert "completed=5" in evidence[0].summary


def test_score_clinical_max_phase3_penalty_capped():
    """Phase 3 failure penalty should be capped at 0.5."""
    data = {
        "completed_count": 2,
        "failed_count": 10,
        "phase3_failures": 5,
        "failure_ratio": 0.833,
        "failed_trial_names": ["a", "b", "c", "d", "e"],
        "source": "clinicaltrials_gov",
    }
    score, evidence = score_clinical(data, gene="TEST")
    # phase3 penalty: min(5*0.15, 0.5) = 0.5
    # failure_ratio 0.833 > 0.5 and total 12 > 3: additional -0.2
    # score = 1.0 - 0.5 - 0.2 = 0.3
    assert score == pytest.approx(0.3, abs=0.01)


def test_score_clinical_clamps_to_zero():
    """Score should never go below 0.0."""
    data = {
        "completed_count": 0,
        "failed_count": 20,
        "phase3_failures": 10,
        "failure_ratio": 1.0,
        "failed_trial_names": [],
        "source": "clinicaltrials_gov",
    }
    score, evidence = score_clinical(data, gene="BAD")
    # phase3 penalty: 0.5 (capped), failure_ratio penalty: 0.2
    # score = 1.0 - 0.5 - 0.2 = 0.3 (not negative because cap works)
    assert score >= 0.0
    assert score <= 1.0


def test_score_clinical_handles_missing_fields():
    """Should handle empty/missing fields gracefully."""
    data = {"source": "clinicaltrials_gov"}
    score, evidence = score_clinical(data)
    assert score == 1.0
    assert evidence == []


def test_score_clinical_handles_wrong_types():
    """Should handle wrong types gracefully."""
    data = {
        "completed_count": "not_a_number",
        "failed_count": None,
        "phase3_failures": [],
        "failure_ratio": "high",
        "source": "clinicaltrials_gov",
    }
    score, evidence = score_clinical(data)
    assert score == 1.0
    assert evidence == []


# --- API client tests ---


@pytest.mark.asyncio
async def test_get_clinical_outcome_with_failures():
    """Test API client returns correct structure with failed studies."""
    from biocompute.data.clinical_trials import get_clinical_outcome

    mock_failed_response = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "briefTitle": "Tanezumab Phase 3 OA Trial"
                    },
                    "designModule": {
                        "phases": ["PHASE3"],
                    },
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "briefTitle": "NGF Inhibitor Phase 1 Trial"
                    },
                    "designModule": {
                        "phases": ["PHASE1"],
                    },
                }
            },
        ]
    }
    mock_completed_response = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "briefTitle": "NGF Completed Study"
                    },
                    "designModule": {
                        "phases": ["PHASE2"],
                    },
                }
            },
        ]
    }

    import httpx

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # First call = failed studies, second call = completed studies
    mock_response_failed = MagicMock()
    mock_response_failed.raise_for_status = lambda: None
    mock_response_failed.json.return_value = mock_failed_response

    mock_response_completed = MagicMock()
    mock_response_completed.raise_for_status = lambda: None
    mock_response_completed.json.return_value = mock_completed_response

    mock_client.get.side_effect = [mock_response_failed, mock_response_completed]

    result = await get_clinical_outcome(mock_client, "NGF", "osteoarthritis")

    assert result["gene"] == "NGF"
    assert result["disease"] == "osteoarthritis"
    assert result["completed_count"] == 1
    assert result["failed_count"] == 2
    assert result["phase3_failures"] == 1
    assert result["source"] == "clinicaltrials_gov"
    assert len(result["failed_trial_names"]) == 2


@pytest.mark.asyncio
async def test_get_clinical_outcome_no_studies():
    """Test API client with no matching studies."""
    from biocompute.data.clinical_trials import get_clinical_outcome

    import httpx

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    mock_response = MagicMock()
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {"studies": []}

    mock_client.get.return_value = mock_response

    result = await get_clinical_outcome(mock_client, "CXCL12", "CLL")

    assert result["completed_count"] == 0
    assert result["failed_count"] == 0
    assert result["phase3_failures"] == 0
    assert result["failure_ratio"] == 0.0


@pytest.mark.asyncio
async def test_get_clinical_outcome_api_failure():
    """Test API client degrades gracefully on network errors."""
    from biocompute.data.clinical_trials import get_clinical_outcome

    import httpx

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = httpx.HTTPError("Connection refused")

    result = await get_clinical_outcome(mock_client, "TEST", "disease")

    # Should return safe defaults (all zeros)
    assert result["completed_count"] == 0
    assert result["failed_count"] == 0
    assert result["phase3_failures"] == 0
    assert result["failure_ratio"] == 0.0


# --- Integration: evaluator applies clinical penalty ---


def test_evaluator_applies_clinical_penalty():
    """Verify evaluator applies clinical penalty to fitness when clinical_score < 0.7."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="NGF",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="joint",
    )
    query = DiseaseQuery(name="osteoarthritis", description="OA pain")

    raw_data = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "joint", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 2},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 3,
            "failed_count": 5,
            "phase3_failures": 2,
            "failure_ratio": 0.625,
            "failed_trial_names": ["tanezumab Phase 3"],
            "source": "clinicaltrials_gov",
        },
    }

    # With clinical penalty
    scored_with = evaluate_all_dimensions(hypothesis, query, raw_data, Weights())

    # Without clinical data (neutral)
    raw_data_no_clinical = {**raw_data, "clinical": {}}
    scored_without = evaluate_all_dimensions(hypothesis, query, raw_data_no_clinical, Weights())

    # Clinical penalty should reduce fitness
    assert scored_with.fitness < scored_without.fitness


def test_evaluator_no_penalty_when_clinical_score_high():
    """No penalty when clinical score >= 0.7."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="TNF",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="joint",
    )
    query = DiseaseQuery(name="rheumatoid arthritis", description="RA")

    raw_data = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "joint", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 2},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 10,
            "failed_count": 0,
            "phase3_failures": 0,
            "failure_ratio": 0.0,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
    }

    # With clean clinical data
    scored_with = evaluate_all_dimensions(hypothesis, query, raw_data, Weights())

    # Without clinical data (neutral — score = 1.0 from empty section)
    raw_data_no_clinical = {**raw_data, "clinical": {}}
    scored_without = evaluate_all_dimensions(hypothesis, query, raw_data_no_clinical, Weights())

    # Both should be equal since clinical_score >= 0.7 means no penalty
    assert scored_with.fitness == scored_without.fitness
