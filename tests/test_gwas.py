# pyright: reportMissingImports=false

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from biocompute.data.gwas import get_gwas_evidence
from biocompute.fitness.evaluator import evaluate_all_dimensions
from biocompute.fitness.gwas import has_strong_gwas_signal, score_gwas
from biocompute.models import DiseaseQuery, TherapeuticHypothesis


@pytest.mark.asyncio
async def test_get_gwas_evidence_parses_opentargets_genetics_rows() -> None:
    search_response = httpx.Response(
        200,
        request=httpx.Request(
            "POST", "https://api.platform.opentargets.org/api/v4/graphql"
        ),
        json={
            "data": {
                "search": {
                    "hits": [{"id": "EFO_0003767", "name": "hypercholesterolemia"}]
                }
            }
        },
    )
    evidence_response = httpx.Response(
        200,
        request=httpx.Request(
            "POST", "https://api.platform.opentargets.org/api/v4/graphql"
        ),
        json={
            "data": {
                "disease": {
                    "id": "EFO_0003767",
                    "associatedTargets": {
                        "rows": [
                            {
                                "target": {"approvedSymbol": "PCSK9"},
                                "score": 0.92,
                                "evidences": {
                                    "rows": [
                                        {
                                            "score": 0.81,
                                            "datasourceId": "ot_genetics_portal",
                                        },
                                        {
                                            "score": 0.73,
                                            "datasourceId": "ot_genetics_portal",
                                        },
                                        {
                                            "score": 0.2,
                                            "datasourceId": "eva",
                                        },
                                    ]
                                },
                            }
                        ]
                    },
                }
            }
        },
    )

    mock_post = AsyncMock(side_effect=[search_response, evidence_response])
    with patch("biocompute.data.gwas.httpx.AsyncClient.post", mock_post):
        async with httpx.AsyncClient() as client:
            result = await get_gwas_evidence(client, "PCSK9", "hypercholesterolemia")

    assert result["gene"] == "PCSK9"
    assert result["disease"] == "hypercholesterolemia"
    assert result["disease_id"] == "EFO_0003767"
    assert result["hit_count"] == 2
    assert result["max_score"] == 0.81
    assert result["scores"] == [0.81, 0.73]
    assert result["source"] == "opentargets_gwas"


@pytest.mark.asyncio
async def test_get_gwas_evidence_returns_neutral_payload_on_failure() -> None:
    with patch(
        "biocompute.data.gwas.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("network down"),
    ):
        async with httpx.AsyncClient() as client:
            result = await get_gwas_evidence(client, "PCSK9", "hypercholesterolemia")

    assert result["gene"] == "PCSK9"
    assert result["disease"] == "hypercholesterolemia"
    assert result["hit_count"] == 0
    assert result["scores"] == []
    assert result["max_score"] == 0.0
    assert result["source"] == "opentargets_gwas"
    assert result["error"] == "API call failed"


def test_score_gwas_uses_hit_count_and_score_thresholds() -> None:
    score, evidence = score_gwas(
        {
            "gene": "PCSK9",
            "disease": "hypercholesterolemia",
            "hit_count": 5,
            "scores": [0.42, 0.63, 0.66, 0.58, 0.61],
            "max_score": 0.66,
            "source": "opentargets_gwas",
        }
    )

    assert score == 0.66
    assert len(evidence) == 1
    assert evidence[0].source_type == "opentargets_gwas"
    assert "hits=5" in evidence[0].summary


def test_has_strong_gwas_signal_requires_high_score_or_many_hits() -> None:
    assert has_strong_gwas_signal({"hit_count": 2, "max_score": 0.8}) is True
    assert has_strong_gwas_signal({"hit_count": 11, "max_score": 0.35}) is True
    assert has_strong_gwas_signal({"hit_count": 3, "max_score": 0.61}) is False
    assert has_strong_gwas_signal({"hit_count": 0, "max_score": 0.0}) is False


def test_evaluator_applies_post_hoc_gwas_boost_for_strong_signal() -> None:
    hypothesis = TherapeuticHypothesis(
        target_gene="PCSK9",
        modality="mAb",
        delivery="systemic IV",
        duration="chronic",
        tissue_context="liver",
    )
    query = DiseaseQuery(name="Hypercholesterolemia", description="High LDL")
    base_raw_data = {
        "literature": {
            "pmid_count": 20,
            "total_citations": 200,
            "influential_citations": 10,
        },
        "expression": {"tissues": [{"Tissue": "liver", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        "druggability": {
            "tractability": [{"modality": "AB", "value": True}],
            "known_drugs_count": 1,
        },
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 1,
            "failed_count": 0,
            "phase3_failures": 0,
            "failure_ratio": 0.0,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
    }

    without_gwas = evaluate_all_dimensions(hypothesis, query, base_raw_data)
    with_gwas = evaluate_all_dimensions(
        hypothesis,
        query,
        {
            **base_raw_data,
            "gwas": {
                "gene": "PCSK9",
                "disease": "Hypercholesterolemia",
                "hit_count": 2,
                "scores": [0.82, 0.77],
                "max_score": 0.82,
                "source": "opentargets_gwas",
            },
        },
    )

    assert with_gwas.fitness == pytest.approx(without_gwas.fitness * 1.1)


def test_evaluator_keeps_no_gwas_evidence_neutral() -> None:
    hypothesis = TherapeuticHypothesis(
        target_gene="CXCL12",
        modality="mAb",
        delivery="local injection",
        duration="single-dose",
        tissue_context="skeletal muscle",
    )
    query = DiseaseQuery(name="MPS", description="Myofascial pain")
    raw_data = {
        "literature": {
            "pmid_count": 10,
            "total_citations": 100,
            "influential_citations": 5,
        },
        "expression": {"tissues": [{"Tissue": "skeletal muscle", "Level": "High"}]},
        "pathway": {"interaction_count": 8, "interactions": [{"score": 0.75}] * 8},
        "druggability": {"tractability": [], "known_drugs_count": 0},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.8},
        "clinical": {
            "completed_count": 0,
            "failed_count": 0,
            "phase3_failures": 0,
            "failure_ratio": 0.0,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
    }

    without_gwas = evaluate_all_dimensions(hypothesis, query, raw_data)
    with_empty_gwas = evaluate_all_dimensions(
        hypothesis,
        query,
        {
            **raw_data,
            "gwas": {
                "gene": "CXCL12",
                "disease": "MPS",
                "hit_count": 0,
                "scores": [],
                "max_score": 0.0,
                "source": "opentargets_gwas",
            },
        },
    )

    assert with_empty_gwas.fitness == pytest.approx(without_gwas.fitness)


def test_evaluator_gwas_boost_does_not_override_other_penalties() -> None:
    hypothesis = TherapeuticHypothesis(
        target_gene="APP",
        modality="ASO",
        delivery="intrathecal",
        duration="chronic",
        tissue_context="brain",
    )
    query = DiseaseQuery(name="Alzheimer disease", description="Neurodegeneration")
    raw_data = {
        "literature": {
            "pmid_count": 30,
            "total_citations": 400,
            "influential_citations": 30,
        },
        "expression": {"tissues": [{"Tissue": "brain", "Level": "High"}]},
        "pathway": {"interaction_count": 12, "interactions": [{"score": 0.9}] * 12},
        "druggability": {
            "tractability": [{"modality": "ASO", "value": True}],
            "known_drugs_count": 0,
        },
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.7},
        "clinical": {
            "completed_count": 0,
            "failed_count": 4,
            "phase3_failures": 3,
            "failure_ratio": 1.0,
            "failed_trial_names": ["Trial A", "Trial B"],
            "source": "clinicaltrials_gov",
        },
        "gwas": {
            "gene": "APP",
            "disease": "Alzheimer disease",
            "hit_count": 12,
            "scores": [0.44, 0.39, 0.41],
            "max_score": 0.44,
            "source": "opentargets_gwas",
        },
    }

    scored = evaluate_all_dimensions(hypothesis, query, raw_data)

    assert scored.fitness < 0.5
