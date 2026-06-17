import os
import sqlite3
import tempfile
from typing import cast

from biocompute.archive.store import ArchiveStore
from biocompute.models import (
    Evidence,
    EvidenceMaturity,
    FitnessScores,
    PriorKnowledge,
    StrategyPriorArt,
    TherapeuticHypothesis,
)


def make_store():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    return ArchiveStore(db_path), db_path


def test_create_store_initializes_tables():
    store, path = make_store()
    assert os.path.exists(path)
    store.close()


def test_save_and_load_hypothesis():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    scores = FitnessScores(
        literature_strength=0.8,
        expression_specificity=0.6,
        pathway_centrality=0.5,
        druggability=0.7,
        safety_profile=0.9,
        ip_freedom=0.4,
    )
    store.save_hypothesis(hypothesis, scores, fitness_total=0.65)

    loaded = store.get_hypothesis(hypothesis.id)
    assert loaded is not None
    assert loaded["target_gene"] == "CXCL12"
    assert loaded["fitness_total"] == 0.65
    store.close()


def test_save_evidence():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("NGF", "mAb", "systemic", "chronic", "joint")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.5)

    evidence_item = Evidence("pubmed", "PMID:12345", "NGF elevated in OA", 0.7)
    store.save_evidence(hypothesis.id, evidence_item)

    evidence = store.get_evidence(hypothesis.id)
    assert len(evidence) == 1
    assert evidence[0]["source_id"] == "PMID:12345"
    store.close()


def test_get_scores_reloads_saved_dimension_values():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    scores = FitnessScores(
        literature_strength=0.8,
        expression_specificity=0.6,
        pathway_centrality=0.5,
        druggability=0.7,
        safety_profile=0.9,
        ip_freedom=0.4,
    )
    store.save_hypothesis(hypothesis, scores, fitness_total=0.65)

    loaded_scores = store.get_scores(hypothesis.id)

    assert loaded_scores is not None
    assert loaded_scores.literature_strength == 0.8
    assert loaded_scores.expression_specificity == 0.6
    assert loaded_scores.ip_freedom == 0.4
    store.close()


def test_save_critique():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("NGF", "mAb", "systemic", "chronic", "joint")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.5)

    store.save_critique(hypothesis.id, "RPOA risk from systemic NGF blockade", "sonnet")

    critiques = store.get_critiques(hypothesis.id)
    assert len(critiques) == 1
    critique_text = critiques[0]["critique_text"]
    assert isinstance(critique_text, str)
    assert "RPOA" in critique_text
    store.close()


def test_get_top_hypotheses():
    store, _ = make_store()
    for gene, fitness in [("A", 0.9), ("B", 0.3), ("C", 0.7), ("D", 0.5)]:
        hypothesis = TherapeuticHypothesis(
            gene, "mAb", "systemic", "chronic", "systemic"
        )
        store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=fitness)

    top = store.get_top_hypotheses(n=2)
    assert len(top) == 2
    assert top[0]["target_gene"] == "A"
    assert top[1]["target_gene"] == "C"
    store.close()


def test_get_generation_hypotheses():
    store, _ = make_store()
    first = TherapeuticHypothesis(
        "A", "mAb", "systemic", "chronic", "systemic", generation=0
    )
    second = TherapeuticHypothesis("B", "VHH", "local", "single", "scar", generation=1)
    store.save_hypothesis(first, FitnessScores(), fitness_total=0.5)
    store.save_hypothesis(second, FitnessScores(), fitness_total=0.6)

    generation_zero = store.get_generation(0)
    assert len(generation_zero) == 1
    assert generation_zero[0]["target_gene"] == "A"

    generation_one = store.get_generation(1)
    assert len(generation_one) == 1
    assert generation_one[0]["target_gene"] == "B"
    store.close()


def test_save_hypothesis_with_source_and_raw_data():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    scores = FitnessScores(
        literature_strength=0.8,
        expression_specificity=0.6,
        pathway_centrality=0.5,
        druggability=0.7,
        safety_profile=0.9,
        ip_freedom=0.4,
    )
    dimension_sources = {
        "literature_strength": "pubmed+semantic_scholar",
        "expression_specificity": "hpa",
        "pathway_centrality": "string",
        "druggability": "opentargets",
        "safety_profile": "opentargets",
        "ip_freedom": "opentargets_heuristic",
    }
    dimension_raw_data = {
        "literature_strength": {"pmid_count": 12, "total_citations": 44},
        "druggability": {"known_drugs_count": 2},
    }
    store.save_hypothesis(
        hypothesis,
        scores,
        fitness_total=0.65,
        dimension_sources=dimension_sources,
        dimension_raw_data=dimension_raw_data,
    )

    metadata = store.get_scores_with_metadata(hypothesis.id)
    assert len(metadata) == 6

    lit_row = next(r for r in metadata if r["dimension"] == "literature_strength")
    assert lit_row["source"] == "pubmed+semantic_scholar"
    assert lit_row["raw_data"] == {"pmid_count": 12, "total_citations": 44}
    assert lit_row["score"] == 0.8

    drug_row = next(r for r in metadata if r["dimension"] == "druggability")
    assert drug_row["source"] == "opentargets"
    assert drug_row["raw_data"] == {"known_drugs_count": 2}

    path_row = next(r for r in metadata if r["dimension"] == "pathway_centrality")
    assert path_row["source"] == "string"
    assert path_row["raw_data"] is None

    loaded_scores = store.get_scores(hypothesis.id)
    assert loaded_scores is not None
    assert loaded_scores.literature_strength == 0.8
    store.close()


def test_save_and_load_prior_knowledge_round_trip():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("SMAD3", "ASO", "local", "repeat", "scar")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.72)
    prior_knowledge = PriorKnowledge(
        gene="SMAD3",
        disease="Hypertrophic scarring",
        maturity=EvidenceMaturity.L2_IN_VITRO,
        known_facts=["SMAD3 amplifies fibrotic TGF-beta signaling."],
        attempted_approaches=[
            "Preclinical antisense knockdown reduced fibrosis markers."
        ],
        gaps=["No scar-focused human interventional data identified."],
        key_papers=["PMID:123456", "PMID:789012"],
        summary="The pathway is validated preclinically, but translation into scar therapy remains early.",
    )

    store.save_prior_knowledge(hypothesis.id, prior_knowledge)

    loaded = store.get_prior_knowledge(hypothesis.id)

    assert loaded == prior_knowledge
    store.close()


def test_get_prior_knowledge_returns_none_for_missing_row_or_old_schema():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("APP", "ASO", "cns", "chronic", "brain")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.41)

    assert store.get_prior_knowledge(hypothesis.id) is None

    _ = store.conn.execute("DROP TABLE prior_knowledge")
    store.conn.commit()

    assert store.get_prior_knowledge(hypothesis.id) is None
    store.close()


def test_get_prior_knowledge_returns_none_for_legacy_row_missing_newer_columns():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("APP", "ASO", "cns", "chronic", "brain")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.41)

    _ = store.conn.execute("DROP TABLE prior_knowledge")
    _ = store.conn.execute(
        """CREATE TABLE prior_knowledge (
               hypothesis_id TEXT PRIMARY KEY,
               gene TEXT,
               disease TEXT,
               maturity INTEGER,
               known_facts TEXT,
               gaps TEXT
           )"""
    )
    _ = store.conn.execute(
        "INSERT INTO prior_knowledge (hypothesis_id, gene, disease, maturity, known_facts, gaps) VALUES (?, ?, ?, ?, ?, ?)",
        (
            hypothesis.id,
            "APP",
            "Alzheimer disease",
            int(EvidenceMaturity.L1_ASSOCIATION),
            '["Amyloid biology is implicated in disease."]',
            '["Translation remains difficult."]',
        ),
    )
    store.conn.commit()

    assert store.get_prior_knowledge(hypothesis.id) is None
    store.close()


def test_prior_knowledge_preserves_maturity_and_list_fields():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("APP", "ASO", "cns", "chronic", "brain")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.38)
    prior_knowledge = PriorKnowledge(
        gene="APP",
        disease="Alzheimer disease",
        maturity=EvidenceMaturity.L5_CLINICAL_FAIL,
        known_facts=[
            "APP processing is central to amyloid biology.",
            "Human genetics support target relevance.",
        ],
        attempted_approaches=["Multiple beta-secretase programs failed clinically."],
        gaps=["Safer modality selection is still unresolved."],
        key_papers=["PMID:111111", "PMID:222222"],
        summary="Clinical attempts show strong biological validation but repeated translational failure.",
    )

    store.save_prior_knowledge(hypothesis.id, prior_knowledge)

    row = cast(
        sqlite3.Row | None,
        store.conn.execute(
            "SELECT maturity, known_facts, attempted_approaches, gaps, key_papers FROM prior_knowledge WHERE hypothesis_id = ?",
            (hypothesis.id,),
        ).fetchone(),
    )

    assert row is not None
    assert row["maturity"] == int(EvidenceMaturity.L5_CLINICAL_FAIL)
    assert row["known_facts"] == (
        '["APP processing is central to amyloid biology.",'
        '"Human genetics support target relevance."]'
    )
    assert row["attempted_approaches"] == (
        '["Multiple beta-secretase programs failed clinically."]'
    )
    assert row["gaps"] == '["Safer modality selection is still unresolved."]'
    assert row["key_papers"] == '["PMID:111111","PMID:222222"]'

    loaded = store.get_prior_knowledge(hypothesis.id)

    assert loaded is not None
    assert loaded.maturity is EvidenceMaturity.L5_CLINICAL_FAIL
    assert loaded.known_facts == prior_knowledge.known_facts
    assert loaded.attempted_approaches == prior_knowledge.attempted_approaches
    assert loaded.gaps == prior_knowledge.gaps
    assert loaded.key_papers == prior_knowledge.key_papers


def test_save_and_load_strategy_prior_art_round_trip():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("SMAD3", "mRNA-LNP", "local", "repeat", "scar")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.72)
    strategy_prior_art = StrategyPriorArt(
        strategy="SMAD7 overexpression",
        disease_class="fibrosis",
        prior_studies=["AAV5-Smad7 in corneal fibrosis"],
        modality_status={"AAV": "in vivo confirmed", "mRNA-LNP": "not attempted"},
        our_differentiation=["repeat dosing flexibility"],
        key_papers=["PMID:28339457"],
        summary="Prior art exists for AAV while mRNA-LNP remains untested.",
    )

    store.save_strategy_prior_art(hypothesis.id, strategy_prior_art)

    loaded = store.get_strategy_prior_art(hypothesis.id)

    assert loaded == strategy_prior_art
    store.close()


def test_get_strategy_prior_art_returns_none_for_missing_row_or_old_schema():
    store, _ = make_store()
    hypothesis = TherapeuticHypothesis("APP", "ASO", "cns", "chronic", "brain")
    store.save_hypothesis(hypothesis, FitnessScores(), fitness_total=0.41)

    assert store.get_strategy_prior_art(hypothesis.id) is None

    _ = store.conn.execute("DROP TABLE strategy_prior_art")
    store.conn.commit()

    assert store.get_strategy_prior_art(hypothesis.id) is None
    store.close()
    store.close()
