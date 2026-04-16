from biocompute.models import (
    DiseaseQuery,
    DiscoveryResult,
    TherapeuticHypothesis,
    FitnessScores,
    RunMetadata,
    Weights,
    compute_fitness,
    Evidence,
)


def test_disease_query_defaults():
    q = DiseaseQuery(name="MPS", description="Myofascial pain")
    assert q.keywords == []
    assert q.known_targets == []
    assert q.known_failures == []


def test_hypothesis_has_unique_id():
    h1 = TherapeuticHypothesis("NGF", "mAb", "systemic", "chronic", "joint")
    h2 = TherapeuticHypothesis("NGF", "mAb", "systemic", "chronic", "joint")
    assert h1.id != h2.id


def test_hypothesis_defaults():
    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    assert h.parent_id is None
    assert h.mutation_type == "seed"
    assert h.generation == 0


def test_fitness_scores_dimensions():
    s = FitnessScores(
        literature_strength=0.8,
        expression_specificity=0.6,
        pathway_centrality=0.5,
        druggability=0.7,
        safety_profile=0.9,
        ip_freedom=0.4,
    )
    dims = s.dimensions()
    assert dims["literature_strength"] == 0.8
    assert len(dims) == 6


def test_compute_fitness_weighted_sum():
    scores = FitnessScores(
        literature_strength=1.0,
        expression_specificity=1.0,
        pathway_centrality=1.0,
        druggability=1.0,
        safety_profile=1.0,
        ip_freedom=1.0,
    )
    weights = Weights()
    result = compute_fitness(scores, weights)
    assert abs(result - 1.0) < 1e-9


def test_compute_fitness_safety_veto():
    # Safety veto fires when safety < threshold AND no approved drugs (druggability <= 0.5)
    scores = FitnessScores(
        literature_strength=1.0,
        expression_specificity=1.0,
        pathway_centrality=1.0,
        druggability=0.0,
        safety_profile=0.1,
        ip_freedom=1.0,
    )
    weights = Weights(safety_threshold=0.2)
    result = compute_fitness(scores, weights)
    assert result == 0.1 * 0.1


def test_compute_fitness_safety_veto_bypassed_by_druggability():
    # Safety veto does NOT fire when druggability > 0.5 AND no severe class effect
    # VEGF has penalty 0.15 (< 0.2 threshold) → bypass allowed
    scores = FitnessScores(
        literature_strength=1.0,
        expression_specificity=1.0,
        pathway_centrality=1.0,
        druggability=1.0,
        safety_profile=0.1,
        ip_freedom=1.0,
    )
    weights = Weights(safety_threshold=0.2)
    result = compute_fitness(scores, weights, gene="VEGF")
    assert result > 0.5  # not vetoed, full weighted sum applies


def test_compute_fitness_safety_veto_bypassed_without_gene():
    # Without gene info, no class effect lookup → bypass still works for high druggability
    scores = FitnessScores(
        literature_strength=1.0,
        expression_specificity=1.0,
        pathway_centrality=1.0,
        druggability=1.0,
        safety_profile=0.1,
        ip_freedom=1.0,
    )
    weights = Weights(safety_threshold=0.2)
    result = compute_fitness(scores, weights)
    assert result > 0.5  # no gene = no class effect = bypass allowed


def test_compute_fitness_safety_veto_not_bypassed_for_severe_class_effect():
    # NGF has CLASS_EFFECT penalty 0.3 (>= 0.2) → severe → veto stands
    # even though druggability > 0.5
    scores = FitnessScores(
        literature_strength=1.0,
        expression_specificity=1.0,
        pathway_centrality=1.0,
        druggability=1.0,
        safety_profile=0.1,
        ip_freedom=1.0,
    )
    weights = Weights(safety_threshold=0.2)
    result = compute_fitness(scores, weights, gene="NGF")
    assert result == 0.1 * 0.1  # vetoed despite high druggability


def test_compute_fitness_safety_veto_bypassed_for_mild_class_effect():
    # TNF has CLASS_EFFECT penalty 0.1 (< 0.2) → mild → bypass allowed
    # Adalimumab is an approved success
    scores = FitnessScores(
        literature_strength=1.0,
        expression_specificity=1.0,
        pathway_centrality=1.0,
        druggability=1.0,
        safety_profile=0.1,
        ip_freedom=1.0,
    )
    weights = Weights(safety_threshold=0.2)
    result = compute_fitness(scores, weights, gene="TNF")
    assert result > 0.5  # not vetoed, Adalimumab approved


def test_compute_fitness_safety_at_threshold_passes():
    scores = FitnessScores(
        literature_strength=0.5,
        expression_specificity=0.5,
        pathway_centrality=0.5,
        druggability=0.5,
        safety_profile=0.2,
        ip_freedom=0.5,
    )
    weights = Weights(safety_threshold=0.2)
    result = compute_fitness(scores, weights)
    assert result > 0.1


def test_evidence_creation():
    e = Evidence(
        source_type="pubmed",
        source_id="PMID:12345",
        summary="CXCL12 elevated in scar tissue",
        relevance_score=0.8,
    )
    assert e.source_type == "pubmed"


def test_discovery_result_db_path_defaults_to_none():
    result = DiscoveryResult(
        query=DiseaseQuery(name="MPS", description="Myofascial pain"),
        candidates=[],
        metadata=RunMetadata(),
    )
    assert result.db_path is None


def test_discovery_result_db_path_stores_value():
    result = DiscoveryResult(
        query=DiseaseQuery(name="MPS", description="Myofascial pain"),
        candidates=[],
        metadata=RunMetadata(),
        db_path="/tmp/test.db",
    )
    assert result.db_path == "/tmp/test.db"
