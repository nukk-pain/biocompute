# pyright: reportMissingImports=false

from biocompute.fitness.druggability import score_druggability
from biocompute.fitness.evaluator import evaluate_all_dimensions
from biocompute.fitness.expression import merge_expression_tissues
from biocompute.fitness.expression import score_expression
from biocompute.fitness.ip import score_ip
from biocompute.fitness.literature import score_literature
from biocompute.fitness.pathway import score_pathway
from biocompute.fitness.safety import classify_delivery, score_safety
from biocompute.models import DiseaseQuery, FitnessScores, TherapeuticHypothesis


def test_score_literature_high_evidence():
    data = {"pmid_count": 50, "total_citations": 500, "influential_citations": 20}
    score, evidence = score_literature(data)
    assert 0.7 <= score <= 1.0
    assert len(evidence) >= 1


def test_score_literature_no_evidence():
    data = {"pmid_count": 0, "total_citations": 0, "influential_citations": 0}
    score, evidence = score_literature(data)
    assert score == 0.0
    assert evidence == []


def test_score_expression_high_specificity():
    data = {
        "tissues": [
            {"Tissue": "skeletal muscle", "Level": "High"},
            {"Tissue": "liver", "Level": "Not detected"},
            {"Tissue": "brain", "Level": "Low"},
        ]
    }
    score, evidence = score_expression(data, target_tissue="skeletal muscle")
    assert 0.5 <= score <= 1.0
    assert len(evidence) == 1


def test_merge_expression_tissues_prefers_higher_level_from_hpa_or_gtex() -> None:
    merged = merge_expression_tissues(
        [
            {"Tissue": "skeletal muscle", "Level": "Low"},
            {"Tissue": "liver", "Level": "Medium"},
        ],
        [
            {"Tissue": "skeletal muscle", "Level": "High"},
            {"Tissue": "brain cortex", "Level": "Low"},
        ],
    )

    assert merged == [
        {"Tissue": "skeletal muscle", "Level": "High"},
        {"Tissue": "liver", "Level": "Medium"},
        {"Tissue": "brain cortex", "Level": "Low"},
    ]


def test_merge_expression_tissues_normalizes_live_gtex_tissue_names() -> None:
    merged = merge_expression_tissues(
        [{"Tissue": "skeletal muscle", "Level": "Low"}],
        [{"Tissue": "SkeletalMuscle", "Level": "High"}],
    )

    assert merged == [{"Tissue": "skeletal muscle", "Level": "High"}]


def test_score_expression_merges_gtex_tissues_when_present() -> None:
    data = {
        "tissues": [{"Tissue": "skeletal muscle", "Level": "Low"}],
        "gtex_tissues": [{"Tissue": "skeletal muscle", "Level": "High"}],
    }

    score, _ = score_expression(data, target_tissue="skeletal muscle")

    assert score >= 0.6


def test_score_expression_matches_live_gtex_multiword_tissue_names() -> None:
    data = {
        "tissues": [{"Tissue": "BrainCortex", "Level": "High"}],
        "source": "gtex",
    }

    score, evidence = score_expression(data, target_tissue="migraine")

    assert score >= 0.6
    assert evidence[0].source_type == "gtex"


def test_score_pathway_high_centrality():
    data = {"interaction_count": 30, "interactions": [{"score": 0.9}] * 30}
    score, evidence = score_pathway(data)
    assert 0.5 <= score <= 1.0
    assert len(evidence) == 1


def test_score_druggability_with_tractability():
    data = {
        "tractability": [{"modality": "AB", "value": True}],
        "known_drugs_count": 3,
    }
    score, evidence = score_druggability(data)
    assert score > 0.5
    assert len(evidence) == 1


def test_score_safety_local_single():
    assert classify_delivery("local injection") == "local"
    assert classify_delivery("systemic IV") == "systemic"
    assert classify_delivery("topical LNP") == "local"


def test_score_safety_systemic_high_risk():
    data = {
        "safety_liabilities": [
            {"event": "cardiotoxicity"},
            {"event": "hepatotoxicity"},
        ]
    }
    score, evidence = score_safety(data, delivery="systemic IV", duration="chronic")
    assert score < 0.5
    assert len(evidence) == 1


def test_score_safety_local_same_gene_lower_risk():
    data = {
        "safety_liabilities": [
            {"event": "cardiotoxicity"},
            {"event": "hepatotoxicity"},
        ]
    }
    score_sys, _ = score_safety(data, delivery="systemic IV", duration="chronic")
    score_loc, _ = score_safety(
        data, delivery="local injection", duration="single-dose"
    )
    assert score_loc > score_sys


def test_score_ip_llm_fallback():
    data = {"source": "llm_estimate", "freedom_estimate": 0.7}
    score, evidence = score_ip(data)
    # No known_drugs_count -> defaults to 0 -> novel target bonus +0.1
    assert abs(score - 0.8) < 1e-9
    assert evidence[0].source_type == "llm_estimate"


def test_score_ip_novel_target_bonus():
    """Zero known drugs = novel target, gets +0.1 bonus."""
    data = {
        "source": "opentargets_heuristic",
        "freedom_estimate": 0.6,
        "known_drugs_count": 0,
    }
    score, _ = score_ip(data)
    assert abs(score - 0.7) < 1e-9  # 0.6 + 0.1


def test_score_ip_few_drugs_penalty():
    """1-5 known drugs = moderate competition, gets -0.1 penalty."""
    data = {
        "source": "opentargets_heuristic",
        "freedom_estimate": 0.6,
        "known_drugs_count": 3,
    }
    score, _ = score_ip(data)
    assert abs(score - 0.5) < 1e-9  # 0.6 - 0.1


def test_score_ip_moderate_drugs_penalty():
    """6-10 known drugs = significant competition, gets -0.2 penalty."""
    data = {
        "source": "opentargets_heuristic",
        "freedom_estimate": 0.6,
        "known_drugs_count": 8,
    }
    score, _ = score_ip(data)
    assert abs(score - 0.4) < 1e-9  # 0.6 - 0.2


def test_score_ip_crowded_field_penalty():
    """11+ known drugs = crowded field, gets -0.3 penalty."""
    data = {
        "source": "opentargets_heuristic",
        "freedom_estimate": 0.6,
        "known_drugs_count": 15,
    }
    score, _ = score_ip(data)
    assert abs(score - 0.3) < 1e-9  # 0.6 - 0.3


def test_score_ip_clamped_to_zero():
    """Score never goes below 0.0."""
    data = {
        "source": "opentargets_heuristic",
        "freedom_estimate": 0.1,
        "known_drugs_count": 15,
    }
    score, _ = score_ip(data)
    assert score == 0.0  # 0.1 - 0.3 clamped to 0.0


def test_score_ip_clamped_to_one():
    """Score never exceeds 1.0."""
    data = {
        "source": "opentargets_heuristic",
        "freedom_estimate": 0.95,
        "known_drugs_count": 0,
    }
    score, _ = score_ip(data)
    assert score == 1.0  # 0.95 + 0.1 clamped to 1.0


def test_score_ip_differentiation():
    """More drugs = lower IP freedom score."""
    data_novel = {"freedom_estimate": 0.6, "known_drugs_count": 0}
    data_crowded = {"freedom_estimate": 0.6, "known_drugs_count": 15}
    score_novel, _ = score_ip(data_novel)
    score_crowded, _ = score_ip(data_crowded)
    assert score_novel > score_crowded


def test_evaluate_all_dimensions_applies_safety_veto_and_collects_evidence():
    hypothesis = TherapeuticHypothesis(
        target_gene="CXCL12",
        modality="mAb",
        delivery="systemic IV",
        duration="chronic",
        tissue_context="skeletal muscle",
    )
    query = DiseaseQuery(name="MPS", description="Myofascial pain")
    raw_data = {
        "literature": {
            "pmid_count": 50,
            "total_citations": 500,
            "influential_citations": 20,
        },
        "expression": {
            "tissues": [
                {"Tissue": "skeletal muscle", "Level": "High"},
                {"Tissue": "liver", "Level": "Not detected"},
            ]
        },
        "pathway": {
            "interaction_count": 30,
            "interactions": [{"score": 0.95}] * 30,
        },
        "druggability": {
            "tractability": [],
            "known_drugs_count": 0,
        },
        "safety": {
            "safety_liabilities": [
                {"event": "cardiotoxicity"},
                {"event": "hepatotoxicity"},
                {"event": "neurotoxicity"},
                {"event": "renal toxicity"},
                {"event": "myelosuppression"},
            ]
        },
        "ip": {"source": "llm_estimate", "freedom_estimate": 0.7},
    }

    scored = evaluate_all_dimensions(hypothesis, query, raw_data)

    assert isinstance(scored.scores, FitnessScores)
    assert scored.scores.safety_profile == 0.0
    assert scored.scores.druggability == 0.0
    assert scored.fitness == 0.0  # safety veto fires when druggability <= 0.5
    assert len(scored.evidence) == 6
