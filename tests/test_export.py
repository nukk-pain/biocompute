import json
import os
import tempfile
from unittest.mock import patch

from biocompute.archive.export import (
    _EVIDENCE_SOURCE_MAP,
    _gene_id_cache,
    _lookup_ncbi_gene_id,
    _map_evidence_source,
    export_for_neuroregen,
    export_for_neuroregen_pipeline,
)
from biocompute.archive.store import ArchiveStore
from biocompute.models import (
    Evidence,
    EvidenceMaturity,
    FitnessScores,
    PriorKnowledge,
    TherapeuticHypothesis,
)


def _make_run_dir(
    disease: str = "Test Disease", description: str = "A test description"
):
    """Create a temporary run directory with config.json and run.db."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "run.db")
    config_path = os.path.join(tmp, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"disease": disease, "description": description}, f)
    return tmp, db_path


def test_export_basic_structure():
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h1 = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    scores1 = FitnessScores(
        literature_strength=0.9,
        expression_specificity=0.7,
        pathway_centrality=0.8,
        druggability=0.6,
        safety_profile=0.95,
        ip_freedom=0.8,
    )
    store.save_hypothesis(h1, scores1, fitness_total=0.82)

    h2 = TherapeuticHypothesis("NGF", "mAb", "systemic", "chronic", "joint")
    scores2 = FitnessScores(
        literature_strength=0.7,
        expression_specificity=0.5,
        pathway_centrality=0.6,
        druggability=0.5,
        safety_profile=0.1,
        ip_freedom=0.3,
    )
    store.save_hypothesis(h2, scores2, fitness_total=0.45)
    store.close()

    result = export_for_neuroregen(db_path, top_n=5)

    assert result["source"] == "biocompute"
    assert result["version"] == "0.1.0"
    assert result["disease"] == "Test Disease"
    assert result["description"] == "A test description"
    assert len(result["candidates"]) == 2


def test_export_candidate_fields():
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    scores = FitnessScores(
        literature_strength=0.9,
        expression_specificity=0.7,
        pathway_centrality=0.8,
        druggability=0.6,
        safety_profile=0.95,
        ip_freedom=0.8,
    )
    store.save_hypothesis(h, scores, fitness_total=0.82)
    store.save_evidence(
        h.id, Evidence("pubmed", "PMID:111", "CXCL12 in fibrosis", 0.85)
    )
    store.save_evidence(
        h.id, Evidence("semantic_scholar", "S2:222", "Chemokine axis", 0.7)
    )
    store.close()

    result = export_for_neuroregen(db_path, top_n=5)
    candidate = result["candidates"][0]

    assert candidate["gene"]["symbol"] == "CXCL12"
    assert candidate["score"] == 0.82
    assert candidate["modality"] == "VHH"
    assert candidate["delivery"] == "local"
    assert isinstance(candidate["evidence"], list)
    assert len(candidate["evidence"]) == 2
    assert isinstance(candidate["pathway"], list)
    assert isinstance(candidate["fitness_breakdown"], dict)

    breakdown = candidate["fitness_breakdown"]
    assert breakdown["literature_strength"] == 0.9
    assert breakdown["expression_specificity"] == 0.7
    assert breakdown["pathway_centrality"] == 0.8
    assert breakdown["druggability"] == 0.6
    assert breakdown["safety_profile"] == 0.95
    assert breakdown["ip_freedom"] == 0.8


def test_export_evidence_format():
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("VEGF", "mAb", "systemic", "chronic", "tumor")
    store.save_hypothesis(h, FitnessScores(), fitness_total=0.5)
    store.save_evidence(
        h.id, Evidence("pubmed", "PMID:999", "VEGF role in angiogenesis", 0.9)
    )
    store.close()

    result = export_for_neuroregen(db_path, top_n=5)
    ev = result["candidates"][0]["evidence"][0]

    assert ev["source"] == "Literature"
    assert ev["description"] == "VEGF role in angiogenesis"
    assert ev["confidence"] == 0.9


def test_export_respects_top_n():
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    for i, gene in enumerate(["A", "B", "C", "D", "E"]):
        h = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "systemic")
        store.save_hypothesis(h, FitnessScores(), fitness_total=0.9 - i * 0.1)
    store.close()

    result = export_for_neuroregen(db_path, top_n=3)
    assert len(result["candidates"]) == 3
    # Should be ordered by fitness_total DESC
    assert result["candidates"][0]["gene"]["symbol"] == "A"
    assert result["candidates"][1]["gene"]["symbol"] == "B"
    assert result["candidates"][2]["gene"]["symbol"] == "C"


def test_export_pathway_from_raw_data():
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    scores = FitnessScores(pathway_centrality=0.8)
    raw_data = {
        "pathway_centrality": {
            "interactions": [
                {"partner": "CXCR4", "score": 0.95},
                {"partner": "ACKR3", "score": 0.8},
            ]
        }
    }
    store.save_hypothesis(h, scores, fitness_total=0.6, dimension_raw_data=raw_data)
    store.close()

    result = export_for_neuroregen(db_path, top_n=5)
    candidate = result["candidates"][0]

    assert len(candidate["pathway"]) == 2
    assert candidate["pathway"][0]["partner"] == "CXCR4"


def test_export_prior_knowledge():
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    store.save_hypothesis(h, FitnessScores(), fitness_total=0.8)

    pk = PriorKnowledge(
        gene="CXCL12",
        disease="Test Disease",
        maturity=EvidenceMaturity.L2_IN_VITRO,
        known_facts=["Fact 1", "Fact 2"],
        attempted_approaches=["Approach 1"],
        gaps=["Gap 1"],
        key_papers=["Paper 1"],
        summary="A summary",
    )
    store.save_prior_knowledge(h.id, pk)
    store.close()

    result = export_for_neuroregen(db_path, top_n=5)
    candidate = result["candidates"][0]

    assert "prior_knowledge" in candidate
    pk_data = candidate["prior_knowledge"]
    assert pk_data["maturity"] == "L2_IN_VITRO"
    assert pk_data["summary"] == "A summary"
    assert pk_data["known_facts"] == ["Fact 1", "Fact 2"]
    assert pk_data["attempted_approaches"] == ["Approach 1"]
    assert pk_data["gaps"] == ["Gap 1"]
    assert pk_data["key_papers"] == ["Paper 1"]


# --- Tests for export_for_neuroregen_pipeline ---


def test_pipeline_export_returns_bare_array():
    """Pipeline export returns a list, not a dict wrapper."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    store.save_hypothesis(h, FitnessScores(), fitness_total=0.711)
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=6387):
        result = export_for_neuroregen_pipeline(db_path, top_n=5)

    assert isinstance(result, list)
    assert len(result) == 1


def test_pipeline_export_gene_object_format():
    """Pipeline export gene field has symbol, ncbi_id (from NCBI lookup), and uniprot_id."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    store.save_hypothesis(h, FitnessScores(), fitness_total=0.711)
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=6387):
        result = export_for_neuroregen_pipeline(db_path, top_n=5)

    gene = result[0]["gene"]

    assert gene["symbol"] == "CXCL12"
    assert gene["ncbi_id"] == 6387
    assert gene["uniprot_id"] is None


def test_pipeline_export_evidence_source_mapping():
    """Pipeline export maps evidence source types to neuroregen names."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    store.save_hypothesis(h, FitnessScores(), fitness_total=0.711)
    store.save_evidence(
        h.id,
        Evidence(
            "pubmed+semantic_scholar",
            "PMID:111",
            "50 publications, 407 citations",
            0.955,
        ),
    )
    store.save_evidence(
        h.id, Evidence("hpa", "HPA:222", "High expression in tissue", 0.8)
    )
    store.save_evidence(h.id, Evidence("string", "STR:333", "Network centrality", 0.7))
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=6387):
        result = export_for_neuroregen_pipeline(db_path, top_n=5)
    evidence = result[0]["evidence"]

    assert len(evidence) == 3
    assert evidence[0]["source"] == "Literature"
    assert evidence[0]["description"] == "50 publications, 407 citations"
    assert evidence[0]["confidence"] == 0.955
    assert evidence[1]["source"] == "Expression"
    assert evidence[2]["source"] == "Pathway"


def test_pipeline_export_pathway_extracts_partner_names():
    """Pipeline export extracts partner names as strings from pathway raw_data."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("CXCL12", "VHH", "local", "single", "scar")
    scores = FitnessScores(pathway_centrality=0.8)
    raw_data = {
        "pathway_centrality": {
            "interactions": [
                {"partner": "CXCR4", "score": 0.95},
                {"partner": "ACKR3", "score": 0.8},
            ]
        }
    }
    store.save_hypothesis(h, scores, fitness_total=0.711, dimension_raw_data=raw_data)
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=6387):
        result = export_for_neuroregen_pipeline(db_path, top_n=5)
    pathway = result[0]["pathway"]

    assert pathway == ["CXCR4", "ACKR3"]


def test_pipeline_export_deduplicates_by_gene():
    """Pipeline export returns unique genes — keeps best-scoring hypothesis per gene."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    # Insert 3 hypotheses for CXCR4 with different scores + 1 different gene
    for i, (gene, score) in enumerate(
        [
            ("CXCR4", 0.95),
            ("CXCR4", 0.88),
            ("CXCR4", 0.80),
            ("VEGF", 0.70),
        ]
    ):
        h = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "systemic")
        store.save_hypothesis(h, FitnessScores(), fitness_total=score)
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=None):
        result = export_for_neuroregen_pipeline(db_path, top_n=10)

    gene_symbols = [c["gene"]["symbol"] for c in result]
    assert gene_symbols == ["CXCR4", "VEGF"]
    # Best CXCR4 score should be kept
    assert result[0]["score"] == 0.95


def test_export_deduplicates_by_gene():
    """Non-pipeline export also returns unique genes."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    for gene, score in [("CXCR4", 0.95), ("CXCR4", 0.88), ("VEGF", 0.70)]:
        h = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "systemic")
        store.save_hypothesis(h, FitnessScores(), fitness_total=score)
    store.close()

    result = export_for_neuroregen(db_path, top_n=10)
    gene_symbols = [c["gene"]["symbol"] for c in result["candidates"]]
    assert gene_symbols == ["CXCR4", "VEGF"]
    assert result["candidates"][0]["score"] == 0.95


def test_pipeline_export_respects_top_n():
    """Pipeline export limits output to top_n candidates."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    for i, gene in enumerate(["A", "B", "C", "D"]):
        h = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "systemic")
        store.save_hypothesis(h, FitnessScores(), fitness_total=0.9 - i * 0.1)
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=None):
        result = export_for_neuroregen_pipeline(db_path, top_n=2)
    assert len(result) == 2
    assert result[0]["gene"]["symbol"] == "A"
    assert result[1]["gene"]["symbol"] == "B"


def test_pipeline_export_top_n_counts_unique_genes_when_duplicates_lead() -> None:
    """Pipeline export should return top_n unique genes, not top_n raw rows."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    for gene, score in [
        ("CXCR4", 0.95),
        ("CXCR4", 0.94),
        ("VEGF", 0.93),
        ("STAT3", 0.92),
    ]:
        h = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "systemic")
        store.save_hypothesis(h, FitnessScores(), fitness_total=score)
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=None):
        result = export_for_neuroregen_pipeline(db_path, top_n=2)

    assert [candidate["gene"]["symbol"] for candidate in result] == ["CXCR4", "VEGF"]


def test_export_top_n_counts_unique_genes_when_duplicates_lead() -> None:
    """Wrapped export should return top_n unique genes, not top_n raw rows."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    for gene, score in [
        ("CXCR4", 0.95),
        ("CXCR4", 0.94),
        ("VEGF", 0.93),
        ("STAT3", 0.92),
    ]:
        h = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "systemic")
        store.save_hypothesis(h, FitnessScores(), fitness_total=score)
    store.close()

    result = export_for_neuroregen(db_path, top_n=2)

    assert [candidate["gene"]["symbol"] for candidate in result["candidates"]] == [
        "CXCR4",
        "VEGF",
    ]


def test_evidence_source_map_coverage():
    """Source map covers all expected biocompute source types."""
    assert _map_evidence_source("pubmed") == "Literature"
    assert _map_evidence_source("semantic_scholar") == "Literature"
    assert _map_evidence_source("pubmed+semantic_scholar") == "Literature"
    assert _map_evidence_source("hpa") == "Expression"
    assert _map_evidence_source("gtex") == "Expression"
    assert _map_evidence_source("hpa+gtex") == "Expression"
    assert _map_evidence_source("string") == "Pathway"
    assert _map_evidence_source("opentargets") == "Literature"
    assert _map_evidence_source("opentargets_heuristic") == "Literature"
    assert _map_evidence_source("clinicaltrials_gov") == "Literature"
    assert _map_evidence_source("opentargets_gwas") == "Literature"
    assert _map_evidence_source("class_effect") == "Literature"
    # Unknown sources pass through unchanged
    assert _map_evidence_source("unknown_source") == "unknown_source"


def test_pipeline_export_no_modality_or_delivery():
    """Pipeline export does not include modality, delivery, or fitness_breakdown."""
    run_dir, db_path = _make_run_dir()
    store = ArchiveStore(db_path)

    h = TherapeuticHypothesis("STAT3", "mAb", "systemic", "chronic", "joint")
    store.save_hypothesis(h, FitnessScores(), fitness_total=0.6)
    store.close()

    with patch("biocompute.archive.export._lookup_ncbi_gene_id", return_value=None):
        result = export_for_neuroregen_pipeline(db_path, top_n=5)
    candidate = result[0]

    assert "modality" not in candidate
    assert "delivery" not in candidate
    assert "fitness_breakdown" not in candidate
    # Only gene, score, evidence, pathway
    assert set(candidate.keys()) == {"gene", "score", "evidence", "pathway"}


# --- Tests for _lookup_ncbi_gene_id ---


def test_lookup_ncbi_gene_id_success():
    """NCBI lookup returns gene ID from esearch response."""
    _gene_id_cache.clear()
    mock_response = {
        "esearchresult": {"idlist": ["6387"], "count": "1"},
    }
    with patch("biocompute.archive.export.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        result = _lookup_ncbi_gene_id("CXCL12")
    assert result == 6387


def test_lookup_ncbi_gene_id_not_found():
    """NCBI lookup returns None when no IDs are found."""
    _gene_id_cache.clear()
    mock_response = {
        "esearchresult": {"idlist": [], "count": "0"},
    }
    with patch("biocompute.archive.export.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        result = _lookup_ncbi_gene_id("NONEXISTENT_GENE")
    assert result is None


def test_lookup_ncbi_gene_id_network_error():
    """NCBI lookup returns None on network failure."""
    _gene_id_cache.clear()
    with patch("biocompute.archive.export.httpx.get", side_effect=Exception("timeout")):
        result = _lookup_ncbi_gene_id("CXCL12")
    assert result is None


def test_lookup_ncbi_gene_id_caches_results():
    """Repeated lookups for the same gene use the cache, not HTTP."""
    _gene_id_cache.clear()
    mock_response = {
        "esearchresult": {"idlist": ["6387"], "count": "1"},
    }
    with patch("biocompute.archive.export.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        first = _lookup_ncbi_gene_id("CXCL12")
        second = _lookup_ncbi_gene_id("CXCL12")
    assert first == 6387
    assert second == 6387
    # httpx.get should only be called once — second call uses cache
    mock_get.assert_called_once()
