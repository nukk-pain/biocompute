# pyright: reportMissingImports=false

"""Tests for LLM-based clinical feasibility assessment."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from biocompute.fitness.llm_clinical import (
    _normalize,
    _safe_defaults,
    _validate_drug_claims,
    assess_clinical_feasibility,
)


# --- assess_clinical_feasibility tests ---


def test_assess_returns_approved_drug_info():
    """LLM returns valid JSON with approved drug data."""
    mock_response = {
        "has_approved_drug": True,
        "approved_drugs": ["bevacizumab"],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.9,
        "rationale": "Bevacizumab is FDA-approved for multiple cancer types.",
    }

    with patch(
        "biocompute.data.llm.query_llm_json",
        return_value=mock_response,
    ):
        result = assess_clinical_feasibility("VEGF", "cancer")

    assert result["has_approved_drug"] is True
    assert "bevacizumab" in result["approved_drugs"]
    assert result["feasibility_score"] == pytest.approx(0.9)
    assert result["has_phase3_failure"] is False


def test_assess_returns_phase3_failure_info():
    """LLM returns valid JSON with Phase 3 failure data."""
    mock_response = {
        "has_approved_drug": False,
        "approved_drugs": [],
        "has_phase3_failure": True,
        "failed_drugs": ["aducanumab", "solanezumab"],
        "feasibility_score": 0.2,
        "rationale": "Multiple anti-amyloid antibodies have shown limited efficacy.",
    }

    with patch(
        "biocompute.data.llm.query_llm_json",
        return_value=mock_response,
    ):
        result = assess_clinical_feasibility("APP", "alzheimer")

    assert result["has_approved_drug"] is False
    assert result["has_phase3_failure"] is True
    assert len(result["failed_drugs"]) == 2
    assert result["feasibility_score"] == pytest.approx(0.2)


def test_assess_returns_defaults_on_llm_exception():
    """Should return safe defaults when query_llm_json raises."""
    with patch(
        "biocompute.data.llm.query_llm_json",
        side_effect=RuntimeError("Claude CLI failed"),
    ):
        result = assess_clinical_feasibility("TEST", "disease")

    assert result["has_approved_drug"] is False
    assert result["feasibility_score"] == pytest.approx(0.5)
    assert result["rationale"] == "LLM assessment unavailable"


def test_assess_returns_defaults_on_parse_failure():
    """Should return safe defaults when LLM returns non-dict (None)."""
    with patch(
        "biocompute.data.llm.query_llm_json",
        return_value=None,
    ):
        result = assess_clinical_feasibility("TEST", "disease")

    assert result["has_approved_drug"] is False
    assert result["feasibility_score"] == pytest.approx(0.5)


def test_assess_returns_defaults_on_list_response():
    """Should return safe defaults when LLM returns a list instead of dict."""
    with patch(
        "biocompute.data.llm.query_llm_json",
        return_value=["not", "a", "dict"],
    ):
        result = assess_clinical_feasibility("TEST", "disease")

    assert result["has_approved_drug"] is False
    assert result["feasibility_score"] == pytest.approx(0.5)


# --- _normalize tests ---


def test_normalize_fills_missing_keys():
    """Normalize should fill missing keys with defaults."""
    result = _normalize({"has_approved_drug": True})

    assert result["has_approved_drug"] is True
    assert result["approved_drugs"] == []
    assert result["has_phase3_failure"] is False
    assert result["failed_drugs"] == []
    assert result["feasibility_score"] == pytest.approx(0.5)
    assert result["rationale"] == ""


def test_normalize_clamps_feasibility_score():
    """Feasibility score should be clamped to [0.0, 1.0]."""
    result = _normalize({"feasibility_score": 1.5})
    assert result["feasibility_score"] == pytest.approx(1.0)

    result = _normalize({"feasibility_score": -0.3})
    assert result["feasibility_score"] == pytest.approx(0.0)


def test_normalize_handles_wrong_types():
    """Should handle incorrect types gracefully."""
    result = _normalize({
        "has_approved_drug": "yes",  # truthy string -> True
        "approved_drugs": "not_a_list",
        "feasibility_score": "high",
    })
    assert result["has_approved_drug"] is True
    assert result["approved_drugs"] == []
    assert result["feasibility_score"] == pytest.approx(0.5)


# --- _safe_defaults tests ---


def test_safe_defaults_structure():
    """Safe defaults should have all expected keys."""
    defaults = _safe_defaults()
    assert defaults["has_approved_drug"] is False
    assert defaults["approved_drugs"] == []
    assert defaults["has_phase3_failure"] is False
    assert defaults["failed_drugs"] == []
    assert defaults["feasibility_score"] == pytest.approx(0.5)
    assert isinstance(defaults["rationale"], str)


# --- _validate_drug_claims tests ---


def test_validate_known_drug_verified():
    """Known target + matching drug should be verified."""
    result = {
        "has_approved_drug": True,
        "approved_drugs": ["Bevacizumab"],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.9,
        "rationale": "FDA-approved",
    }
    validated = _validate_drug_claims(result, "VEGF")
    assert validated["drug_verification"] == "verified"
    assert "bevacizumab" in validated["verified_drugs"]


def test_validate_hallucinated_drug_unverified():
    """Known target + non-matching drug should be flagged as unverified."""
    result = {
        "has_approved_drug": True,
        "approved_drugs": ["fakemab"],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.9,
        "rationale": "Fakemab is FDA-approved (hallucinated).",
    }
    validated = _validate_drug_claims(result, "VEGF")
    assert validated["drug_verification"] == "unverified"
    assert validated["verified_drugs"] == []


def test_validate_unknown_gene_no_reference():
    """Unknown gene (not in KNOWN_DRUGS) should return no_reference."""
    result = {
        "has_approved_drug": True,
        "approved_drugs": ["somemab"],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.8,
        "rationale": "Some rationale.",
    }
    validated = _validate_drug_claims(result, "UNKNOWNGENE123")
    assert validated["drug_verification"] == "no_reference"


def test_validate_empty_approved_drugs_no_reference():
    """Known gene but empty approved_drugs list should return no_reference."""
    result = {
        "has_approved_drug": False,
        "approved_drugs": [],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.5,
        "rationale": "No drugs.",
    }
    validated = _validate_drug_claims(result, "VEGF")
    assert validated["drug_verification"] == "no_reference"


def test_validate_case_insensitive_matching():
    """Drug matching should be case-insensitive."""
    result = {
        "has_approved_drug": True,
        "approved_drugs": ["ADALIMUMAB"],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.9,
        "rationale": "Adalimumab is approved.",
    }
    validated = _validate_drug_claims(result, "tnf")
    assert validated["drug_verification"] == "verified"
    assert "adalimumab" in validated["verified_drugs"]


# --- Evaluator integration tests ---


def test_evaluator_llm_feasibility_boost_on_approved_drug():
    """Approved drug should boost fitness when clinical_score >= 0.7 (no significant failures)."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="VEGF",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="tumor",
    )
    query = DiseaseQuery(name="cancer", description="solid tumor")

    # Clinical data with NO significant failures (clinical_score >= 0.7)
    raw_data_with_llm = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "tumor", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 2},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 5,
            "failed_count": 1,
            "phase3_failures": 0,
            "failure_ratio": 0.17,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
        "llm_clinical": {
            "has_approved_drug": True,
            "approved_drugs": ["bevacizumab"],
            "has_phase3_failure": False,
            "failed_drugs": [],
            "feasibility_score": 0.9,
            "rationale": "FDA-approved",
        },
    }

    # Without LLM data
    raw_data_without_llm = {**raw_data_with_llm}
    del raw_data_without_llm["llm_clinical"]

    scored_with = evaluate_all_dimensions(hypothesis, query, raw_data_with_llm, Weights())
    scored_without = evaluate_all_dimensions(hypothesis, query, raw_data_without_llm, Weights())

    # LLM boost should raise fitness when clinical_score >= 0.7
    assert scored_with.fitness > scored_without.fitness


def test_clinical_penalty_cannot_be_undone_by_llm():
    """Clinical penalty stays even if LLM says has_approved_drug=True."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="VEGF",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="tumor",
    )
    query = DiseaseQuery(name="cancer", description="solid tumor")

    # Clinical data with real Phase 3 failures -> clinical_score < 0.7
    raw_data_with_llm = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "tumor", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 2},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 5,
            "failed_count": 5,
            "phase3_failures": 3,
            "failure_ratio": 0.5,
            "failed_trial_names": ["trial1", "trial2", "trial3"],
            "source": "clinicaltrials_gov",
        },
        # LLM says approved drug — but clinical trials show real failures
        "llm_clinical": {
            "has_approved_drug": True,
            "approved_drugs": ["bevacizumab"],
            "has_phase3_failure": False,
            "failed_drugs": [],
            "feasibility_score": 0.9,
            "rationale": "FDA-approved",
        },
    }

    # Without LLM data (clinical penalty applies, nothing else)
    raw_data_without_llm = {**raw_data_with_llm}
    del raw_data_without_llm["llm_clinical"]

    scored_with = evaluate_all_dimensions(hypothesis, query, raw_data_with_llm, Weights())
    scored_without = evaluate_all_dimensions(hypothesis, query, raw_data_without_llm, Weights())

    # Clinical penalty must NOT be undone by LLM boost — fitness should be equal
    assert scored_with.fitness == pytest.approx(scored_without.fitness)


def test_evaluator_llm_feasibility_penalty_on_phase3_failure():
    """Phase 3 failure with low feasibility should reduce fitness."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="APP",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="brain",
    )
    query = DiseaseQuery(name="alzheimer", description="neurodegeneration")

    raw_data_with_penalty = {
        "literature": {"pmid_count": 20, "total_citations": 100, "influential_citations": 5},
        "expression": {"tissues": [{"Tissue": "brain", "Level": "Medium"}]},
        "pathway": {"interaction_count": 5, "interactions": [{"score": 0.6}] * 5},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 1},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.5},
        "clinical": {
            "completed_count": 3,
            "failed_count": 0,
            "phase3_failures": 0,
            "failure_ratio": 0.0,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
        "llm_clinical": {
            "has_approved_drug": False,
            "approved_drugs": [],
            "has_phase3_failure": True,
            "failed_drugs": ["solanezumab"],
            "feasibility_score": 0.2,
            "rationale": "Multiple anti-amyloid failures",
        },
    }

    raw_data_without_llm = {**raw_data_with_penalty}
    del raw_data_without_llm["llm_clinical"]

    scored_with = evaluate_all_dimensions(hypothesis, query, raw_data_with_penalty, Weights())
    scored_without = evaluate_all_dimensions(hypothesis, query, raw_data_without_llm, Weights())

    # LLM penalty (feasibility 0.2 < 0.3) should reduce fitness
    assert scored_with.fitness < scored_without.fitness


def test_evaluator_no_effect_when_llm_clinical_absent():
    """No llm_clinical key in raw_data should not change fitness."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="TNF",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="joint",
    )
    query = DiseaseQuery(name="RA", description="rheumatoid arthritis")

    raw_data = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "joint", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 2},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {},
    }

    # With empty llm_clinical
    raw_data_empty = {**raw_data, "llm_clinical": {}}
    scored_empty = evaluate_all_dimensions(hypothesis, query, raw_data_empty, Weights())

    # Without llm_clinical key at all
    scored_absent = evaluate_all_dimensions(hypothesis, query, raw_data, Weights())

    assert scored_empty.fitness == scored_absent.fitness


def test_evaluator_llm_feasibility_no_penalty_when_feasibility_above_threshold():
    """Phase 3 failure with feasibility >= 0.3 should NOT trigger soft penalty."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="TEST",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="tissue",
    )
    query = DiseaseQuery(name="disease", description="test")

    raw_data = {
        "literature": {"pmid_count": 20, "total_citations": 100, "influential_citations": 5},
        "expression": {"tissues": [{"Tissue": "tissue", "Level": "Medium"}]},
        "pathway": {"interaction_count": 5, "interactions": [{"score": 0.6}] * 5},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 1},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.5},
        "clinical": {},
    }

    # Phase 3 failure but feasibility_score=0.5 (above 0.3 threshold)
    raw_data_llm = {
        **raw_data,
        "llm_clinical": {
            "has_approved_drug": False,
            "has_phase3_failure": True,
            "feasibility_score": 0.5,
        },
    }

    scored_with = evaluate_all_dimensions(hypothesis, query, raw_data_llm, Weights())
    scored_without = evaluate_all_dimensions(hypothesis, query, raw_data, Weights())

    # No penalty since feasibility >= 0.3
    assert scored_with.fitness == scored_without.fitness


# --- Cross-validation: LLM hallucination detection ---


def test_hallucination_detected_llm_claims_drug_but_opentargets_zero():
    """LLM claims has_approved_drug=True but OpenTargets known_drugs_count=0.

    The boost should NOT be applied — treated as hallucination.
    Fitness should equal the no-LLM baseline (with clinical penalty if applicable).
    """
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="FAKEGENE",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="tumor",
    )
    query = DiseaseQuery(name="cancer", description="solid tumor")

    raw_data_hallucinated = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "tumor", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        # known_drugs_count=0 — OpenTargets says NO approved drugs
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 0},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 5,
            "failed_count": 5,
            "phase3_failures": 3,
            "failure_ratio": 0.5,
            "failed_trial_names": ["trial1", "trial2", "trial3"],
            "source": "clinicaltrials_gov",
        },
        # LLM claims approved drug exists — hallucination
        "llm_clinical": {
            "has_approved_drug": True,
            "approved_drugs": ["fantasimab"],
            "has_phase3_failure": False,
            "failed_drugs": [],
            "feasibility_score": 0.9,
            "rationale": "Fantasimab is FDA-approved (hallucinated).",
        },
    }

    # Same data but without the hallucinated LLM claim
    raw_data_no_llm = {**raw_data_hallucinated}
    del raw_data_no_llm["llm_clinical"]

    scored_hallucinated = evaluate_all_dimensions(
        hypothesis, query, raw_data_hallucinated, Weights()
    )
    scored_no_llm = evaluate_all_dimensions(
        hypothesis, query, raw_data_no_llm, Weights()
    )

    # Hallucination detected: boost reverted, fitness should equal no-LLM baseline
    assert scored_hallucinated.fitness == pytest.approx(scored_no_llm.fitness)


def test_consistent_llm_and_opentargets():
    """LLM claims has_approved_drug=True and OpenTargets confirms known_drugs=3.

    The boost SHOULD be applied when clinical_score >= 0.7 — no hallucination.
    """
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="VEGF",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="tumor",
    )
    query = DiseaseQuery(name="cancer", description="solid tumor")

    # Clinical data with NO significant failures (clinical_score >= 0.7)
    raw_data_consistent = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "tumor", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        # known_drugs_count=3 — OpenTargets CONFIRMS approved drugs
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 3},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 5,
            "failed_count": 1,
            "phase3_failures": 0,
            "failure_ratio": 0.17,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
        # LLM claims approved drug — consistent with OpenTargets
        "llm_clinical": {
            "has_approved_drug": True,
            "approved_drugs": ["bevacizumab"],
            "has_phase3_failure": False,
            "failed_drugs": [],
            "feasibility_score": 0.9,
            "rationale": "Bevacizumab is FDA-approved.",
        },
    }

    # Same data but without LLM
    raw_data_no_llm = {**raw_data_consistent}
    del raw_data_no_llm["llm_clinical"]

    scored_consistent = evaluate_all_dimensions(
        hypothesis, query, raw_data_consistent, Weights()
    )
    scored_no_llm = evaluate_all_dimensions(
        hypothesis, query, raw_data_no_llm, Weights()
    )

    # Consistent: boost IS applied, fitness should be higher than no-LLM baseline
    assert scored_consistent.fitness > scored_no_llm.fitness


def test_evaluator_unverified_drug_blocks_boost():
    """LLM claims approved drug for known gene but drug name doesn't match — boost blocked."""
    from biocompute.fitness.evaluator import evaluate_all_dimensions
    from biocompute.models import DiseaseQuery, TherapeuticHypothesis, Weights

    hypothesis = TherapeuticHypothesis(
        target_gene="VEGF",
        modality="mAb",
        delivery="systemic",
        duration="chronic",
        tissue_context="tumor",
    )
    query = DiseaseQuery(name="cancer", description="solid tumor")

    raw_data_unverified = {
        "literature": {"pmid_count": 30, "total_citations": 200, "influential_citations": 10},
        "expression": {"tissues": [{"Tissue": "tumor", "Level": "High"}]},
        "pathway": {"interaction_count": 10, "interactions": [{"score": 0.8}] * 10},
        "druggability": {"tractability": [{"modality": "AB", "value": True}], "known_drugs_count": 3},
        "safety": {"safety_liabilities": []},
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.6},
        "clinical": {
            "completed_count": 5,
            "failed_count": 1,
            "phase3_failures": 0,
            "failure_ratio": 0.17,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
        # LLM claims a hallucinated drug for a known gene
        "llm_clinical": {
            "has_approved_drug": True,
            "approved_drugs": ["hallucinomab"],
            "has_phase3_failure": False,
            "failed_drugs": [],
            "feasibility_score": 0.9,
            "rationale": "Hallucinomab is approved (hallucinated).",
            "drug_verification": "unverified",
            "verified_drugs": [],
        },
    }

    raw_data_no_llm = {**raw_data_unverified}
    del raw_data_no_llm["llm_clinical"]

    scored_unverified = evaluate_all_dimensions(
        hypothesis, query, raw_data_unverified, Weights()
    )
    scored_no_llm = evaluate_all_dimensions(
        hypothesis, query, raw_data_no_llm, Weights()
    )

    # Unverified drug should NOT get the boost — fitness equals no-LLM baseline
    assert scored_unverified.fitness == pytest.approx(scored_no_llm.fitness)
