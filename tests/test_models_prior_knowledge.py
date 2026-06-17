from biocompute.models import (
    EvidenceMaturity,
    PriorKnowledge,
    ScoredHypothesis,
    TherapeuticHypothesis,
    FitnessScores,
)


def test_evidence_maturity_enum():
    assert EvidenceMaturity.L0_HYPOTHESIS.value == 0
    assert EvidenceMaturity.L1_ASSOCIATION.value == 1
    assert EvidenceMaturity.L2_IN_VITRO.value == 2
    assert EvidenceMaturity.L3_IN_VIVO.value == 3
    assert EvidenceMaturity.L4_CLINICAL.value == 4
    assert EvidenceMaturity.L5_CLINICAL_FAIL.value == 5


def test_prior_knowledge_dataclass():
    pk = PriorKnowledge(
        gene="TEST",
        disease="Test Disease",
        maturity=EvidenceMaturity.L2_IN_VITRO,
        known_facts=["fact1"],
        attempted_approaches=["approach1"],
        gaps=["gap1"],
        key_papers=["paper1"],
        summary="Test summary",
    )
    assert pk.maturity == EvidenceMaturity.L2_IN_VITRO
    assert pk.gene == "TEST"
    assert pk.disease == "Test Disease"


def test_scored_hypothesis_with_prior_knowledge():
    hypothesis = TherapeuticHypothesis(
        target_gene="TEST",
        modality="small molecule",
        delivery="oral",
        duration="chronic",
        tissue_context="liver",
    )
    scores = FitnessScores()
    pk = PriorKnowledge(
        gene="TEST",
        disease="Test Disease",
        maturity=EvidenceMaturity.L3_IN_VIVO,
        known_facts=["fact1"],
        attempted_approaches=["approach1"],
        gaps=["gap1"],
        key_papers=["paper1"],
        summary="Test summary",
    )

    # Test with prior knowledge
    sh = ScoredHypothesis(
        hypothesis=hypothesis, fitness=0.5, scores=scores, prior_knowledge=pk
    )
    assert sh.prior_knowledge == pk

    # Test without prior knowledge (backward compatibility)
    sh_no_pk = ScoredHypothesis(hypothesis=hypothesis, fitness=0.5, scores=scores)
    assert sh_no_pk.prior_knowledge is None


def test_import_smoke():
    from biocompute.models import EvidenceMaturity, PriorKnowledge

    assert EvidenceMaturity.L2_IN_VITRO.value == 2
    pk = PriorKnowledge(
        gene="TEST",
        disease="Test Disease",
        maturity=EvidenceMaturity.L1_ASSOCIATION,
        known_facts=[],
        attempted_approaches=[],
        gaps=[],
        key_papers=[],
        summary="Test",
    )
    assert pk.maturity == EvidenceMaturity.L1_ASSOCIATION
