# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import os
import tempfile
from collections.abc import Callable
from collections.abc import Generator
from typing import cast
from unittest.mock import AsyncMock, patch

import biocompute.engine as engine_module
import pytest
from biocompute.engine import EngineConfig, EvolutionEngine, collect_bio_data
from biocompute.data.cache import reset_cache
from biocompute.archive.store import ArchiveStore
from biocompute.models import (
    DiseaseQuery,
    EvidenceMaturity,
    FitnessScores,
    PRIOR_KNOWLEDGE_TOP_N,
    PriorKnowledge,
    ScoredHypothesis,
    StrategyPriorArt,
    TherapeuticHypothesis,
)


def make_hypothesis(gene: str, gen: int = 0) -> TherapeuticHypothesis:
    return TherapeuticHypothesis(
        gene,
        "mAb",
        "systemic",
        "chronic",
        "systemic",
        generation=gen,
    )


def make_prior_knowledge(gene: str, disease: str, summary: str) -> PriorKnowledge:
    return PriorKnowledge(
        gene=gene,
        disease=disease,
        maturity=EvidenceMaturity.L1_ASSOCIATION,
        known_facts=[f"{gene} has prior literature"],
        attempted_approaches=[],
        gaps=[],
        key_papers=[],
        summary=summary,
    )


def make_dimension_raw_data(pmids: list[str]) -> dict[str, object]:
    return {
        "literature": {
            "gene": "GENE",
            "disease": "Disease",
            "pmid_count": len(pmids),
            "pmids": pmids,
            "total_citations": 0,
            "influential_citations": 0,
            "negative_count": 0,
            "source": "pubmed+semantic_scholar",
        },
        "expression": {"tissues": [], "source": "hpa+gtex"},
        "pathway": {
            "interaction_count": 0,
            "interactions": [],
            "source": "string",
        },
        "druggability": {
            "tractability": [],
            "known_drugs_count": 0,
            "source": "opentargets",
        },
        "safety": {"safety_liabilities": [], "source": "opentargets"},
        "ip": {"freedom_estimate": 0.0, "source": "opentargets_heuristic"},
        "clinical": {
            "completed_count": 0,
            "failed_count": 0,
            "phase3_failures": 0,
            "failure_ratio": 0.0,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
        },
        "gwas": {
            "scores": [],
            "hit_count": 0,
            "max_score": 0.0,
            "source": "opentargets_gwas",
        },
        "llm_clinical": {
            "feasibility_score": 0.0,
            "has_approved_drug": False,
        },
    }


async def fake_prior_knowledge_fetch(
    _client: object,
    pmids: list[str],
) -> list[dict[str, str]]:
    return [
        {
            "pmid": pmids[0],
            "title": f"Paper {pmids[0]}",
            "abstract": "Prior target evidence.",
            "year": "2025",
        }
    ]


def fake_prior_knowledge_assessment(
    gene: str,
    disease: str,
    _abstracts: list[dict[str, str]],
) -> PriorKnowledge:
    return make_prior_knowledge(gene, disease, f"{gene} prior knowledge attached")


def make_strategy_prior_art(gene: str) -> StrategyPriorArt:
    return StrategyPriorArt(
        strategy=f"{gene} modulation",
        disease_class="fibrosis",
        prior_studies=[f"{gene} prior study"],
        modality_status={"mAb": "supported", "mRNA-LNP": "not attempted"},
        our_differentiation=["localized delivery"],
        key_papers=[f"PMID:{gene}"],
        summary=f"{gene} strategy prior art attached",
    )


def _batch_collector() -> Callable[
    [list[TherapeuticHypothesis], DiseaseQuery, object], list[dict[str, object]]
]:
    return cast(
        Callable[
            [list[TherapeuticHypothesis], DiseaseQuery, object], list[dict[str, object]]
        ],
        getattr(engine_module, "_collect_all_bio_data_batch"),
    )


@pytest.fixture(autouse=True)
def clear_api_cache() -> Generator[None, None, None]:
    reset_cache()
    yield
    reset_cache()


def test_engine_config_defaults() -> None:
    config = EngineConfig()

    assert config.max_generations == 10
    assert config.population_size == 30
    assert config.top_n == 10
    assert config.diverse_n == 5
    assert config.critique_top_k == 5


def test_engine_should_stop_max_generations() -> None:
    config = EngineConfig(max_generations=3)
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    engine = EvolutionEngine(config, db_path)

    engine.generation = 3

    assert engine.should_stop() is True


def test_engine_should_stop_convergence() -> None:
    config = EngineConfig(max_generations=100)
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    engine = EvolutionEngine(config, db_path)

    engine.generation = 5
    engine.best_scores_history = [0.8, 0.8, 0.8]

    assert engine.should_stop() is True


def test_engine_should_not_stop_improving() -> None:
    config = EngineConfig(max_generations=100)
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    engine = EvolutionEngine(config, db_path)

    engine.generation = 5
    engine.best_scores_history = [0.5, 0.6, 0.8]

    assert engine.should_stop() is False


def test_collect_bio_data_normalizes_existing_client_shapes() -> None:
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.data.pubmed.search_and_count",
            new=AsyncMock(
                return_value={
                    "pmid_count": 12,
                    "pmids": ["1", "2"],
                    "source": "pubmed",
                }
            ),
        ),
        patch(
            "biocompute.data.semantic_scholar.get_citation_count",
            new=AsyncMock(
                return_value={
                    "total_citations": 44,
                    "influential_citations": 5,
                    "source": "semantic_scholar",
                }
            ),
        ),
        patch(
            "biocompute.data.hpa.get_tissue_expression",
            new=AsyncMock(
                return_value={"tissues": [{"Tissue": "muscle", "Level": "High"}]}
            ),
        ),
        patch(
            "biocompute.data.gtex.get_tissue_expression",
            new=AsyncMock(
                return_value={"tissues": [{"Tissue": "muscle", "Level": "Medium"}]}
            ),
        ),
        patch(
            "biocompute.data.string_db.get_interaction_partners",
            new=AsyncMock(
                return_value={"interaction_count": 7, "interactions": [{"score": 0.9}]}
            ),
        ),
        patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl",
            new=AsyncMock(return_value="ENSG00000107562"),
        ),
        patch(
            "biocompute.data.opentargets.get_target_info",
            new=AsyncMock(
                return_value={
                    "tractability": [{"modality": "AB", "value": True}],
                    "known_drugs_count": 2,
                    "safety_liabilities": [{"event": "hepatotoxicity"}],
                    "source": "opentargets",
                }
            ),
        ),
        patch(
            "biocompute.data.clinical_trials.get_clinical_outcome",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "completed_count": 0,
                    "failed_count": 0,
                    "phase3_failures": 0,
                    "failure_ratio": 0.0,
                    "failed_trial_names": [],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.data.gwas.get_gwas_evidence",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "disease_id": "EFO_0001234",
                    "scores": [0.71, 0.64],
                    "hit_count": 2,
                    "max_score": 0.71,
                    "source": "opentargets_gwas",
                }
            ),
        ),
    ):
        raw_data = collect_bio_data(hypothesis, query)

    literature = raw_data["literature"]
    expression = raw_data["expression"]
    pathway = raw_data["pathway"]
    druggability = raw_data["druggability"]
    safety = raw_data["safety"]
    ip = raw_data["ip"]
    clinical = raw_data["clinical"]
    gwas = raw_data["gwas"]

    assert isinstance(literature, dict)
    assert isinstance(expression, dict)
    assert isinstance(pathway, dict)
    assert isinstance(druggability, dict)
    assert isinstance(safety, dict)
    assert isinstance(ip, dict)
    assert isinstance(clinical, dict)
    assert isinstance(gwas, dict)

    tissues = expression["tissues"]
    liabilities = safety["safety_liabilities"]
    assert isinstance(tissues, list)
    assert isinstance(liabilities, list)
    assert isinstance(tissues[0], dict)
    assert isinstance(liabilities[0], dict)

    assert literature["pmid_count"] == 12
    assert literature["influential_citations"] == 5
    assert tissues[0]["Tissue"] == "muscle"
    assert tissues[0]["Level"] == "High"
    assert pathway["interaction_count"] == 7
    assert druggability["known_drugs_count"] == 2
    assert liabilities[0]["event"] == "hepatotoxicity"
    assert ip["source"] == "opentargets_heuristic"
    assert clinical["source"] == "clinicaltrials_gov"
    assert gwas["hit_count"] == 2
    assert gwas["max_score"] == 0.71


def test_collect_bio_data_merges_live_gtex_tissue_ids_with_hpa_names() -> None:
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.data.pubmed.search_and_count",
            new=AsyncMock(
                return_value={"pmid_count": 1, "pmids": [], "source": "pubmed"}
            ),
        ),
        patch(
            "biocompute.data.semantic_scholar.get_citation_count",
            new=AsyncMock(
                return_value={"total_citations": 1, "influential_citations": 0}
            ),
        ),
        patch(
            "biocompute.data.pubmed.search_negative_evidence",
            new=AsyncMock(
                return_value={"negative_count": 0, "source": "pubmed_negative"}
            ),
        ),
        patch(
            "biocompute.data.hpa.get_tissue_expression",
            new=AsyncMock(
                return_value={
                    "tissues": [{"Tissue": "skeletal muscle", "Level": "Low"}]
                }
            ),
        ),
        patch(
            "biocompute.data.gtex.get_tissue_expression",
            new=AsyncMock(
                return_value={
                    "tissues": [{"Tissue": "SkeletalMuscle", "Level": "High"}]
                }
            ),
        ),
        patch(
            "biocompute.data.string_db.get_interaction_partners",
            new=AsyncMock(return_value={"interaction_count": 0, "interactions": []}),
        ),
        patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl",
            new=AsyncMock(return_value="ENSG00000107562"),
        ),
        patch(
            "biocompute.data.opentargets.get_target_info",
            new=AsyncMock(
                return_value={
                    "tractability": [],
                    "known_drugs_count": 0,
                    "safety_liabilities": [],
                }
            ),
        ),
        patch(
            "biocompute.data.clinical_trials.get_clinical_outcome",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "completed_count": 0,
                    "failed_count": 0,
                    "phase3_failures": 0,
                    "failure_ratio": 0.0,
                    "failed_trial_names": [],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.data.gwas.get_gwas_evidence",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "disease_id": None,
                    "scores": [],
                    "hit_count": 0,
                    "max_score": 0.0,
                    "source": "opentargets_gwas",
                }
            ),
        ),
    ):
        raw_data = collect_bio_data(hypothesis, query)

    expression = raw_data["expression"]
    assert isinstance(expression, dict)
    assert expression["tissues"] == [{"Tissue": "skeletal muscle", "Level": "High"}]


def test_collect_bio_data_degrades_gracefully_on_partial_failure() -> None:
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.data.pubmed.search_and_count",
            new=AsyncMock(side_effect=RuntimeError("pubmed down")),
        ),
        patch(
            "biocompute.data.semantic_scholar.get_citation_count",
            new=AsyncMock(
                return_value={"total_citations": 21, "influential_citations": 3}
            ),
        ),
        patch(
            "biocompute.data.hpa.get_tissue_expression",
            new=AsyncMock(return_value={"tissues": []}),
        ),
        patch(
            "biocompute.data.gtex.get_tissue_expression",
            new=AsyncMock(side_effect=RuntimeError("gtex down")),
        ),
        patch(
            "biocompute.data.string_db.get_interaction_partners",
            new=AsyncMock(return_value={"interaction_count": 0, "interactions": []}),
        ),
        patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "biocompute.data.opentargets.get_target_info",
            new=AsyncMock(
                return_value={
                    "tractability": [],
                    "known_drugs_count": 0,
                    "safety_liabilities": [],
                }
            ),
        ),
        patch(
            "biocompute.data.clinical_trials.get_clinical_outcome",
            new=AsyncMock(side_effect=RuntimeError("clinical_trials down")),
        ),
        patch(
            "biocompute.data.gwas.get_gwas_evidence",
            new=AsyncMock(side_effect=RuntimeError("gwas down")),
        ),
    ):
        raw_data = collect_bio_data(hypothesis, query)

    literature = raw_data["literature"]
    expression = raw_data["expression"]
    druggability = raw_data["druggability"]
    clinical = raw_data["clinical"]
    gwas = raw_data["gwas"]
    assert isinstance(literature, dict)
    assert isinstance(expression, dict)
    assert isinstance(druggability, dict)
    assert isinstance(clinical, dict)
    assert isinstance(gwas, dict)

    assert literature["pmid_count"] == 0
    assert literature["total_citations"] == 21
    assert expression["tissues"] == []
    assert expression["source"] == "hpa+gtex"
    assert druggability["known_drugs_count"] == 0
    assert clinical["failed_count"] == 0
    assert clinical["source"] == "clinicaltrials_gov"
    assert gwas["hit_count"] == 0
    assert gwas["source"] == "opentargets_gwas"

    # Verify error keys are present in failed sections
    assert "error" in literature
    assert "pubmed:" in literature["error"]
    assert "RuntimeError" in literature["error"]
    assert "error" in clinical
    assert "clinicaltrials:" in clinical["error"]
    assert "error" in gwas
    assert "gwas:" in gwas["error"]

    # Verify api_errors aggregated at top level
    api_errors = raw_data.get("api_errors")
    assert isinstance(api_errors, list)
    assert len(api_errors) == 4
    error_sources = [e.split(":")[0] for e in api_errors]
    assert "pubmed" in error_sources
    assert "clinicaltrials" in error_sources
    assert "gtex" in error_sources
    assert "gwas" in error_sources


def test_collect_bio_data_no_api_errors_when_all_succeed() -> None:
    """When all APIs succeed, api_errors key should not be present."""
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.data.pubmed.search_and_count",
            new=AsyncMock(
                return_value={
                    "pmid_count": 5,
                    "pmids": ["1"],
                    "source": "pubmed",
                }
            ),
        ),
        patch(
            "biocompute.data.semantic_scholar.get_citation_count",
            new=AsyncMock(
                return_value={
                    "total_citations": 10,
                    "influential_citations": 2,
                    "source": "semantic_scholar",
                }
            ),
        ),
        patch(
            "biocompute.data.pubmed.search_negative_evidence",
            new=AsyncMock(
                return_value={
                    "negative_count": 0,
                    "source": "pubmed_negative",
                }
            ),
        ),
        patch(
            "biocompute.data.hpa.get_tissue_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "hpa"}),
        ),
        patch(
            "biocompute.data.gtex.get_tissue_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "gtex"}),
        ),
        patch(
            "biocompute.data.string_db.get_interaction_partners",
            new=AsyncMock(
                return_value={
                    "interaction_count": 3,
                    "interactions": [],
                    "source": "string",
                }
            ),
        ),
        patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl",
            new=AsyncMock(return_value="ENSG00000107562"),
        ),
        patch(
            "biocompute.data.opentargets.get_target_info",
            new=AsyncMock(
                return_value={
                    "tractability": [],
                    "known_drugs_count": 0,
                    "safety_liabilities": [],
                    "source": "opentargets",
                }
            ),
        ),
        patch(
            "biocompute.data.clinical_trials.get_clinical_outcome",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "completed_count": 0,
                    "failed_count": 0,
                    "phase3_failures": 0,
                    "failure_ratio": 0.0,
                    "failed_trial_names": [],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.data.gwas.get_gwas_evidence",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "disease_id": "EFO_0001234",
                    "scores": [0.4],
                    "hit_count": 1,
                    "max_score": 0.4,
                    "source": "opentargets_gwas",
                }
            ),
        ),
    ):
        raw_data = collect_bio_data(hypothesis, query)

    assert "api_errors" not in raw_data


def test_collect_bio_data_all_apis_fail_collects_all_errors() -> None:
    """When all APIs fail, api_errors should contain errors from every source."""
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.data.pubmed.search_and_count",
            new=AsyncMock(side_effect=RuntimeError("pubmed down")),
        ),
        patch(
            "biocompute.data.semantic_scholar.get_citation_count",
            new=AsyncMock(side_effect=TimeoutError("s2 timeout")),
        ),
        patch(
            "biocompute.data.pubmed.search_negative_evidence",
            new=AsyncMock(side_effect=ConnectionError("neg down")),
        ),
        patch(
            "biocompute.data.hpa.get_tissue_expression",
            new=AsyncMock(side_effect=RuntimeError("hpa down")),
        ),
        patch(
            "biocompute.data.gtex.get_tissue_expression",
            new=AsyncMock(side_effect=RuntimeError("gtex down")),
        ),
        patch(
            "biocompute.data.string_db.get_interaction_partners",
            new=AsyncMock(side_effect=RuntimeError("string down")),
        ),
        patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl",
            new=AsyncMock(side_effect=RuntimeError("resolve down")),
        ),
        patch(
            "biocompute.data.opentargets.get_target_info",
            new=AsyncMock(side_effect=RuntimeError("opentargets down")),
        ),
        patch(
            "biocompute.data.clinical_trials.get_clinical_outcome",
            new=AsyncMock(side_effect=RuntimeError("clinical down")),
        ),
        patch(
            "biocompute.data.gwas.get_gwas_evidence",
            new=AsyncMock(side_effect=RuntimeError("gwas down")),
        ),
    ):
        raw_data = collect_bio_data(hypothesis, query)

    api_errors = raw_data.get("api_errors")
    assert isinstance(api_errors, list)
    # literature (has pubmed + s2 + negative errors merged), expression, pathway,
    # druggability/safety (shared target dict), clinical, gwas = at least 6 sections with errors
    assert len(api_errors) >= 6

    # Verify each error string contains the exception type
    for err in api_errors:
        assert isinstance(err, str)
        assert "Error" in err or "error" in err.lower()

    # Verify specific API names appear
    all_errors_joined = " ".join(api_errors)
    assert "pubmed" in all_errors_joined
    assert "hpa" in all_errors_joined
    assert "gtex" in all_errors_joined
    assert "string_db" in all_errors_joined
    assert "opentargets" in all_errors_joined
    assert "clinicaltrials" in all_errors_joined
    assert "gwas" in all_errors_joined


def test_build_final_candidates_reloads_archived_scores() -> None:
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    engine = EvolutionEngine(EngineConfig(), db_path)
    hypothesis = make_hypothesis("CXCL12")
    scores = FitnessScores(
        literature_strength=0.8,
        expression_specificity=0.6,
        pathway_centrality=0.5,
        druggability=0.7,
        safety_profile=0.9,
        ip_freedom=0.4,
    )

    try:
        engine.store.save_hypothesis(hypothesis, scores, fitness_total=0.65)
        candidates = engine._build_final_candidates()  # pyright: ignore[reportPrivateUsage]
    finally:
        engine.store.close()

    assert len(candidates) == 1
    assert candidates[0].scores.literature_strength == 0.8
    assert candidates[0].scores.expression_specificity == 0.6
    assert candidates[0].scores.ip_freedom == 0.4


def test_collect_all_bio_data_batch_includes_gwas_section() -> None:
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.engine._collect_literature_data",
            new=AsyncMock(
                return_value={"pmid_count": 5, "source": "pubmed+semantic_scholar"}
            ),
        ),
        patch(
            "biocompute.engine._safe_get_tissue_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "hpa"}),
        ),
        patch(
            "biocompute.engine._safe_get_gtex_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "gtex"}),
        ),
        patch(
            "biocompute.engine._safe_get_interaction_partners",
            new=AsyncMock(
                return_value={
                    "interaction_count": 0,
                    "interactions": [],
                    "source": "string",
                }
            ),
        ),
        patch(
            "biocompute.engine._collect_target_data",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "tractability": [],
                    "known_drugs_count": 0,
                    "safety_liabilities": [],
                    "source": "opentargets",
                }
            ),
        ),
        patch(
            "biocompute.engine._safe_get_clinical_outcome",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "completed_count": 0,
                    "failed_count": 0,
                    "phase3_failures": 0,
                    "failure_ratio": 0.0,
                    "failed_trial_names": [],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.engine._safe_get_gwas_evidence",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "disease_id": "EFO_0001234",
                    "scores": [0.72],
                    "hit_count": 1,
                    "max_score": 0.72,
                    "source": "opentargets_gwas",
                }
            ),
        ),
        patch(
            "biocompute.engine._safe_get_llm_feasibility",
            new=AsyncMock(
                return_value={"feasibility_score": 0.5, "has_approved_drug": False}
            ),
        ),
    ):
        batch = _batch_collector()([hypothesis], query, None)

    assert len(batch) == 1
    raw_data = batch[0]
    gwas = cast(dict[str, object], raw_data["gwas"])
    assert gwas["hit_count"] == 1
    assert gwas["max_score"] == 0.72


def test_collect_all_bio_data_batch_keeps_gwas_failure_graceful() -> None:
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.engine._collect_literature_data",
            new=AsyncMock(
                return_value={"pmid_count": 5, "source": "pubmed+semantic_scholar"}
            ),
        ),
        patch(
            "biocompute.engine._safe_get_tissue_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "hpa"}),
        ),
        patch(
            "biocompute.engine._safe_get_gtex_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "gtex"}),
        ),
        patch(
            "biocompute.engine._safe_get_interaction_partners",
            new=AsyncMock(
                return_value={
                    "interaction_count": 0,
                    "interactions": [],
                    "source": "string",
                }
            ),
        ),
        patch(
            "biocompute.engine._collect_target_data",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "tractability": [],
                    "known_drugs_count": 0,
                    "safety_liabilities": [],
                    "source": "opentargets",
                }
            ),
        ),
        patch(
            "biocompute.engine._safe_get_clinical_outcome",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "completed_count": 0,
                    "failed_count": 0,
                    "phase3_failures": 0,
                    "failure_ratio": 0.0,
                    "failed_trial_names": [],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.engine._safe_get_gwas_evidence",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "disease_id": None,
                    "scores": [],
                    "hit_count": 0,
                    "max_score": 0.0,
                    "source": "opentargets_gwas",
                    "error": "gwas: RuntimeError: gwas down",
                }
            ),
        ),
        patch(
            "biocompute.engine._safe_get_llm_feasibility",
            new=AsyncMock(
                return_value={"feasibility_score": 0.5, "has_approved_drug": False}
            ),
        ),
    ):
        batch = _batch_collector()([hypothesis], query, None)

    assert len(batch) == 1
    raw_data = batch[0]
    gwas = cast(dict[str, object], raw_data["gwas"])
    assert gwas["hit_count"] == 0
    assert gwas["source"] == "opentargets_gwas"
    api_errors = raw_data.get("api_errors")
    assert isinstance(api_errors, list)
    assert "gwas: RuntimeError: gwas down" in api_errors


def test_collect_bio_data_includes_llm_clinical_like_batch_path() -> None:
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    llm_payload = {
        "feasibility_score": 0.8,
        "has_approved_drug": True,
        "drug_verification": "verified",
    }

    with (
        patch(
            "biocompute.data.pubmed.search_and_count",
            new=AsyncMock(
                return_value={"pmid_count": 5, "pmids": ["1"], "source": "pubmed"}
            ),
        ),
        patch(
            "biocompute.data.semantic_scholar.get_citation_count",
            new=AsyncMock(
                return_value={
                    "total_citations": 10,
                    "influential_citations": 2,
                    "source": "semantic_scholar",
                }
            ),
        ),
        patch(
            "biocompute.data.pubmed.search_negative_evidence",
            new=AsyncMock(
                return_value={"negative_count": 0, "source": "pubmed_negative"}
            ),
        ),
        patch(
            "biocompute.data.hpa.get_tissue_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "hpa"}),
        ),
        patch(
            "biocompute.data.gtex.get_tissue_expression",
            new=AsyncMock(return_value={"tissues": [], "source": "gtex"}),
        ),
        patch(
            "biocompute.data.string_db.get_interaction_partners",
            new=AsyncMock(
                return_value={
                    "interaction_count": 0,
                    "interactions": [],
                    "source": "string",
                }
            ),
        ),
        patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl",
            new=AsyncMock(return_value="ENSG00000107562"),
        ),
        patch(
            "biocompute.data.opentargets.get_target_info",
            new=AsyncMock(
                return_value={
                    "tractability": [],
                    "known_drugs_count": 1,
                    "safety_liabilities": [],
                    "source": "opentargets",
                }
            ),
        ),
        patch(
            "biocompute.data.clinical_trials.get_clinical_outcome",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "completed_count": 0,
                    "failed_count": 0,
                    "phase3_failures": 0,
                    "failure_ratio": 0.0,
                    "failed_trial_names": [],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.data.gwas.get_gwas_evidence",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "disease_id": "EFO_0001234",
                    "scores": [0.4],
                    "hit_count": 1,
                    "max_score": 0.4,
                    "source": "opentargets_gwas",
                }
            ),
        ),
        patch(
            "biocompute.engine._safe_get_llm_feasibility",
            new=AsyncMock(return_value=llm_payload),
        ),
    ):
        single_raw_data = collect_bio_data(hypothesis, query)
        batch_raw_data = _batch_collector()([hypothesis], query, None)[0]

    assert single_raw_data["llm_clinical"] == llm_payload
    assert batch_raw_data["llm_clinical"] == llm_payload


def test_run_enriches_only_top_ranked_candidates_with_prior_knowledge() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain")
    hypotheses = [make_hypothesis(f"GENE{i}") for i in range(PRIOR_KNOWLEDGE_TOP_N + 2)]
    fitness_by_gene = {
        hypothesis.target_gene: float(len(hypotheses) - index)
        for index, hypothesis in enumerate(hypotheses)
    }

    def collect_data(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
    ) -> dict[str, object]:
        gene_suffix = hypothesis.target_gene.removeprefix("GENE")
        return make_dimension_raw_data([gene_suffix or "0"])

    def fake_evaluate(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
        _raw_data: dict[str, object],
        _weights: object,
    ) -> ScoredHypothesis:
        return ScoredHypothesis(
            hypothesis=hypothesis,
            fitness=fitness_by_gene[hypothesis.target_gene],
            scores=FitnessScores(
                literature_strength=fitness_by_gene[hypothesis.target_gene]
            ),
        )

    fetch_abstracts = AsyncMock(side_effect=fake_prior_knowledge_fetch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        engine = EvolutionEngine(
            EngineConfig(
                max_generations=0,
                population_size=len(hypotheses),
                use_db_seed=False,
            ),
            db_path,
        )

        with (
            patch(
                "biocompute.engine.generate_seed_population", return_value=hypotheses
            ),
            patch(
                "biocompute.engine.evaluate_all_dimensions", side_effect=fake_evaluate
            ),
            patch(
                "biocompute.data.pubmed_abstracts.fetch_abstracts",
                new=fetch_abstracts,
            ),
            patch(
                "biocompute.fitness.prior_knowledge.assess_prior_knowledge",
                side_effect=fake_prior_knowledge_assessment,
            ) as assess_prior_knowledge,
            patch(
                "biocompute.fitness.strategy_prior_art._generate_strategy_queries",
                return_value=["strategy query"],
            ),
            patch(
                "biocompute.fitness.strategy_prior_art._search_strategy_abstracts",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = engine.run(query, collect_data_fn=collect_data)

    assert len(result.candidates) == len(hypotheses)
    enriched_genes = [
        candidate.hypothesis.target_gene
        for candidate in result.candidates
        if candidate.prior_knowledge is not None
    ]
    assert enriched_genes == [
        hypothesis.target_gene for hypothesis in hypotheses[:PRIOR_KNOWLEDGE_TOP_N]
    ]
    assert all(
        candidate.prior_knowledge is not None
        for candidate in result.candidates[:PRIOR_KNOWLEDGE_TOP_N]
    )
    assert all(
        candidate.prior_knowledge is None
        for candidate in result.candidates[PRIOR_KNOWLEDGE_TOP_N:]
    )
    assert fetch_abstracts.await_count == PRIOR_KNOWLEDGE_TOP_N
    assert assess_prior_knowledge.call_count == PRIOR_KNOWLEDGE_TOP_N


def test_run_enriches_top_unique_genes_when_leading_candidates_repeat_gene() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain")
    hypotheses = [
        make_hypothesis("DUP"),
        make_hypothesis("DUP"),
        make_hypothesis("GENE1"),
        make_hypothesis("GENE2"),
        make_hypothesis("GENE3"),
        make_hypothesis("GENE4"),
        make_hypothesis("GENE5"),
    ]
    fitness_by_id = {
        hypothesis.id: float(len(hypotheses) - index)
        for index, hypothesis in enumerate(hypotheses)
    }

    def collect_data(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
    ) -> dict[str, object]:
        return make_dimension_raw_data([hypothesis.id])

    def fake_evaluate(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
        _raw_data: dict[str, object],
        _weights: object,
    ) -> ScoredHypothesis:
        fitness = fitness_by_id[hypothesis.id]
        return ScoredHypothesis(
            hypothesis=hypothesis,
            fitness=fitness,
            scores=FitnessScores(literature_strength=fitness),
        )

    fetch_abstracts = AsyncMock(side_effect=fake_prior_knowledge_fetch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        engine = EvolutionEngine(
            EngineConfig(
                max_generations=0,
                population_size=len(hypotheses),
                use_db_seed=False,
            ),
            db_path,
        )

        with (
            patch(
                "biocompute.engine.generate_seed_population", return_value=hypotheses
            ),
            patch(
                "biocompute.engine.evaluate_all_dimensions", side_effect=fake_evaluate
            ),
            patch(
                "biocompute.data.pubmed_abstracts.fetch_abstracts",
                new=fetch_abstracts,
            ),
            patch(
                "biocompute.fitness.prior_knowledge.assess_prior_knowledge",
                side_effect=fake_prior_knowledge_assessment,
            ) as assess_prior_knowledge,
            patch(
                "biocompute.fitness.strategy_prior_art._generate_strategy_queries",
                return_value=["strategy query"],
            ),
            patch(
                "biocompute.fitness.strategy_prior_art._search_strategy_abstracts",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = engine.run(query, collect_data_fn=collect_data)

    enriched_genes = [
        candidate.hypothesis.target_gene
        for candidate in result.candidates
        if candidate.prior_knowledge is not None
    ]

    assert enriched_genes == ["DUP", "GENE1", "GENE2", "GENE3", "GENE4"]
    assert result.candidates[0].prior_knowledge is not None
    assert result.candidates[1].prior_knowledge is None
    assert result.candidates[5].prior_knowledge is not None
    assert result.candidates[6].prior_knowledge is None
    assert fetch_abstracts.await_count == PRIOR_KNOWLEDGE_TOP_N
    assert assess_prior_knowledge.call_count == PRIOR_KNOWLEDGE_TOP_N


def test_run_keeps_discovery_result_when_prior_knowledge_pubmed_or_llm_fail() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain")
    hypotheses = [make_hypothesis("GENE1"), make_hypothesis("GENE2")]
    fitness_by_gene = {"GENE1": 2.0, "GENE2": 1.0}

    def collect_data(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
    ) -> dict[str, object]:
        gene_suffix = hypothesis.target_gene.removeprefix("GENE")
        return make_dimension_raw_data([gene_suffix])

    def fake_evaluate(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
        _raw_data: dict[str, object],
        _weights: object,
    ) -> ScoredHypothesis:
        return ScoredHypothesis(
            hypothesis=hypothesis,
            fitness=fitness_by_gene[hypothesis.target_gene],
            scores=FitnessScores(
                literature_strength=fitness_by_gene[hypothesis.target_gene]
            ),
        )

    async def fake_fetch_abstracts(
        _client: object, pmids: list[str]
    ) -> list[dict[str, str]]:
        if pmids == ["1"]:
            return [
                {
                    "pmid": "1",
                    "title": "Paper 1",
                    "abstract": "Evidence body.",
                    "year": "2025",
                }
            ]
        return []

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        engine = EvolutionEngine(
            EngineConfig(
                max_generations=0,
                population_size=len(hypotheses),
                use_db_seed=False,
            ),
            db_path,
        )

        with (
            patch(
                "biocompute.engine.generate_seed_population", return_value=hypotheses
            ),
            patch(
                "biocompute.engine.evaluate_all_dimensions", side_effect=fake_evaluate
            ),
            patch(
                "biocompute.data.pubmed_abstracts.fetch_abstracts",
                new=AsyncMock(side_effect=fake_fetch_abstracts),
            ),
            patch(
                "biocompute.data.llm.query_llm_json",
                side_effect=RuntimeError("llm down"),
            ),
        ):
            result = engine.run(query, collect_data_fn=collect_data)

    assert result.db_path == db_path
    assert [candidate.hypothesis.target_gene for candidate in result.candidates] == [
        "GENE1",
        "GENE2",
    ]
    assert result.metadata.total_hypotheses == 2
    assert result.candidates[0].prior_knowledge is not None
    assert result.candidates[1].prior_knowledge is not None
    assert (
        result.candidates[0].prior_knowledge.summary
        == "Prior knowledge assessment unavailable."
    )
    assert (
        result.candidates[1].prior_knowledge.summary
        == "No usable PubMed abstracts available for prior-knowledge assessment."
    )
    assert result.candidates[0].api_errors == []
    assert result.candidates[1].api_errors == []


def test_run_persists_prior_knowledge_for_verification_reuse() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain")
    hypotheses = [make_hypothesis("GENE1")]

    def collect_data(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
    ) -> dict[str, object]:
        gene_suffix = hypothesis.target_gene.removeprefix("GENE")
        return make_dimension_raw_data([gene_suffix or "0"])

    def fake_evaluate(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
        _raw_data: dict[str, object],
        _weights: object,
    ) -> ScoredHypothesis:
        return ScoredHypothesis(
            hypothesis=hypothesis,
            fitness=2.0,
            scores=FitnessScores(literature_strength=2.0),
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        engine = EvolutionEngine(
            EngineConfig(
                max_generations=0,
                population_size=len(hypotheses),
                use_db_seed=False,
            ),
            db_path,
        )

        with (
            patch(
                "biocompute.engine.generate_seed_population", return_value=hypotheses
            ),
            patch(
                "biocompute.engine.evaluate_all_dimensions", side_effect=fake_evaluate
            ),
            patch(
                "biocompute.data.pubmed_abstracts.fetch_abstracts",
                new=AsyncMock(side_effect=fake_prior_knowledge_fetch),
            ),
            patch(
                "biocompute.fitness.prior_knowledge.assess_prior_knowledge",
                side_effect=fake_prior_knowledge_assessment,
            ),
        ):
            result = engine.run(query, collect_data_fn=collect_data)

        store = ArchiveStore(db_path)
        try:
            persisted = store.get_prior_knowledge(result.candidates[0].hypothesis.id)
        finally:
            store.close()

    assert persisted is not None
    assert persisted.gene == "GENE1"
    assert persisted.summary == "GENE1 prior knowledge attached"


def test_run_enriches_strategy_prior_art_for_top_three_candidates_only() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain")
    hypotheses = [make_hypothesis(f"GENE{index}") for index in range(1, 5)]
    strategy_calls: list[tuple[str, str, str, list[dict[str, str]]]] = []

    def collect_data(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
    ) -> dict[str, object]:
        gene_suffix = hypothesis.target_gene.removeprefix("GENE")
        return make_dimension_raw_data([gene_suffix or "0"])

    def fake_evaluate(
        hypothesis: TherapeuticHypothesis,
        _query: DiseaseQuery,
        _raw_data: dict[str, object],
        _weights: object,
    ) -> ScoredHypothesis:
        fitness = 10.0 - float(hypothesis.target_gene.removeprefix("GENE"))
        return ScoredHypothesis(
            hypothesis=hypothesis,
            fitness=fitness,
            scores=FitnessScores(literature_strength=fitness),
        )

    async def fake_fetch_abstracts(
        _client: object,
        pmids: list[str],
        max_abstracts: int = 10,
    ) -> list[dict[str, str]]:
        return [
            {
                "pmid": pmids[0],
                "title": f"Paper {pmids[0]}",
                "abstract": f"Abstract {pmids[0]}",
                "year": "2025",
            }
        ][:max_abstracts]

    async def fake_search_strategy_abstracts(
        _client: object,
        _queries: list[str],
    ) -> list[dict[str, str]]:
        return [
            {
                "pmid": "28339457",
                "title": "Strategy paper",
                "abstract": "Abstract 28339457",
                "year": "2017",
            }
        ]

    def fake_strategy_assessment(
        gene: str,
        disease: str,
        modality: str,
        abstracts: list[dict[str, str]],
    ) -> dict[str, object]:
        strategy_calls.append((gene, disease, modality, abstracts))
        prior_art = make_strategy_prior_art(gene)
        return {
            "prior_studies": prior_art.prior_studies,
            "modality_status": prior_art.modality_status,
            "our_differentiation": prior_art.our_differentiation,
            "summary": prior_art.summary,
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        engine = EvolutionEngine(
            EngineConfig(
                max_generations=0,
                population_size=len(hypotheses),
                use_db_seed=False,
            ),
            db_path,
        )

        with (
            patch(
                "biocompute.engine.generate_seed_population", return_value=hypotheses
            ),
            patch(
                "biocompute.engine.evaluate_all_dimensions", side_effect=fake_evaluate
            ),
            patch(
                "biocompute.data.pubmed_abstracts.fetch_abstracts",
                new=AsyncMock(side_effect=fake_fetch_abstracts),
            ),
            patch(
                "biocompute.fitness.prior_knowledge.assess_prior_knowledge",
                side_effect=fake_prior_knowledge_assessment,
            ),
            patch(
                "biocompute.fitness.strategy_prior_art._generate_strategy_queries",
                return_value=["SMAD7 overexpression fibrosis"],
            ),
            patch(
                "biocompute.fitness.strategy_prior_art._search_strategy_abstracts",
                new=AsyncMock(side_effect=fake_search_strategy_abstracts),
            ),
            patch(
                "biocompute.fitness.strategy_prior_art.assess_strategy_prior_art",
                side_effect=fake_strategy_assessment,
            ),
        ):
            result = engine.run(query, collect_data_fn=collect_data)

        store = ArchiveStore(db_path)
        try:
            persisted_top = [
                store.get_strategy_prior_art(candidate.hypothesis.id)
                for candidate in result.candidates[:3]
            ]
            persisted_fourth = store.get_strategy_prior_art(
                result.candidates[3].hypothesis.id
            )
        finally:
            store.close()

    assert [call[0] for call in strategy_calls] == ["GENE1", "GENE2", "GENE3"]
    assert all(call[1] == "MPS" for call in strategy_calls)
    assert all(call[2] == "mAb" for call in strategy_calls)
    assert all(call[3][0]["abstract"].startswith("Abstract") for call in strategy_calls)
    assert [
        candidate.strategy_prior_art is not None for candidate in result.candidates
    ] == [
        True,
        True,
        True,
        False,
    ]
    assert all(prior_art is not None for prior_art in persisted_top)
    assert persisted_fourth is None


def test_collect_bio_data_handles_non_finite_semantic_scholar_citation_counts() -> None:
    hypothesis = make_hypothesis("CXCL12")
    query = DiseaseQuery("MPS", "Myofascial pain")

    with (
        patch(
            "biocompute.data.pubmed.search_and_count",
            new=AsyncMock(
                return_value={"pmid_count": 2, "pmids": ["1", "2"], "source": "pubmed"}
            ),
        ),
        patch(
            "biocompute.data.semantic_scholar.search_papers",
            new=AsyncMock(
                return_value=[
                    {
                        "paperId": "s1",
                        "title": "Paper 1",
                        "citationCount": float("inf"),
                        "influentialCitationCount": float("nan"),
                    },
                    {
                        "paperId": "s2",
                        "title": "Paper 2",
                        "citationCount": float("nan"),
                        "influentialCitationCount": float("-inf"),
                    },
                ]
            ),
        ),
        patch(
            "biocompute.data.pubmed.search_negative_evidence",
            new=AsyncMock(
                return_value={"negative_count": 0, "source": "pubmed_negative"}
            ),
        ),
        patch(
            "biocompute.data.hpa.get_tissue_expression",
            new=AsyncMock(return_value={"tissues": []}),
        ),
        patch(
            "biocompute.data.gtex.get_tissue_expression",
            new=AsyncMock(return_value={"tissues": []}),
        ),
        patch(
            "biocompute.data.string_db.get_interaction_partners",
            new=AsyncMock(return_value={"interaction_count": 0, "interactions": []}),
        ),
        patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl",
            new=AsyncMock(return_value="ENSG00000107562"),
        ),
        patch(
            "biocompute.data.opentargets.get_target_info",
            new=AsyncMock(
                return_value={
                    "tractability": [],
                    "known_drugs_count": 0,
                    "safety_liabilities": [],
                    "source": "opentargets",
                }
            ),
        ),
        patch(
            "biocompute.data.clinical_trials.get_clinical_outcome",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "completed_count": 0,
                    "failed_count": 0,
                    "phase3_failures": 0,
                    "failure_ratio": 0.0,
                    "failed_trial_names": [],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.data.gwas.get_gwas_evidence",
            new=AsyncMock(
                return_value={
                    "gene": "CXCL12",
                    "disease": "MPS",
                    "disease_id": "EFO_0001234",
                    "scores": [0.4],
                    "hit_count": 1,
                    "max_score": 0.4,
                    "source": "opentargets_gwas",
                }
            ),
        ),
    ):
        raw_data = collect_bio_data(hypothesis, query)

    literature = raw_data["literature"]
    assert isinstance(literature, dict)
    assert literature["pmid_count"] == 2
    assert literature["total_citations"] == 0
    assert literature["influential_citations"] == 0


def test_engine_caps_evaluated_children_to_population_size() -> None:
    # Setup
    config = EngineConfig(
        population_size=5,
        top_n=2,
        diverse_n=1,
        critique_top_k=0,
        use_db_seed=False,
    )
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    engine = EvolutionEngine(config, db_path)

    # Mock survivors
    survivors = [make_hypothesis(f"GENE{i}") for i in range(3)]
    engine.population = [
        ScoredHypothesis(hypothesis=h, fitness=1.0, scores=FitnessScores())
        for h in survivors
    ]

    # Mock select_survivors to return our survivors
    with patch("biocompute.engine.select_survivors", return_value=engine.population):
        # Mock mutate_hypothesis to return 10 children
        children = [make_hypothesis(f"CHILD{i}") for i in range(10)]
        with patch(
            "biocompute.engine.mutate_hypothesis", return_value=children
        ) as mock_mutate:
            # Mock _collect_all_bio_data_batch to track evaluations
            evaluated_hypotheses = []

            def fake_collect_batch(hypotheses, query, collect_data_fn):
                evaluated_hypotheses.extend(hypotheses)
                return [make_dimension_raw_data([]) for _ in hypotheses]

            with patch(
                "biocompute.engine._collect_all_bio_data_batch",
                side_effect=fake_collect_batch,
            ):
                with (
                    patch.object(
                        EvolutionEngine,
                        "_enrich_prior_knowledge",
                        return_value=None,
                    ),
                    patch.object(
                        EvolutionEngine,
                        "_enrich_strategy_prior_art",
                        return_value=None,
                    ),
                ):
                    # Run one generation
                    engine.generation = 0
                    query = DiseaseQuery("MPS", "Myofascial pain")

                    # Mock seed population to return survivors
                    with patch(
                        "biocompute.engine.generate_seed_population",
                        return_value=survivors,
                    ):
                        # Mock should_stop to run once
                        with patch.object(
                            engine, "should_stop", side_effect=[False, True]
                        ):
                            _ = engine.run(query)

                    # Assertions
                    assert mock_mutate.call_count == 3

                    # Filter evaluated_hypotheses to only include children
                    evaluated_children = [
                        h
                        for h in evaluated_hypotheses
                        if h.target_gene.startswith("CHILD")
                    ]

                    assert len(evaluated_children) <= config.population_size
                    assert evaluated_children != children[:5]
