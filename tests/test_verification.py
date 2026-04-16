import json
import os
import tempfile
from collections.abc import Mapping
from unittest.mock import AsyncMock, patch

import pytest

from biocompute.archive.store import ArchiveStore
from biocompute.models import (
    EvidenceMaturity,
    FitnessScores,
    PriorKnowledge,
    TherapeuticHypothesis,
)
from biocompute.verification.literature import (
    PaperSummary,
    TargetVerification,
    _verify_targets_async,
    classify_evidence_strength,
    verify_targets,
)
from biocompute.verification.report import generate_verification_report


def _clinical_status(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "completed_count": 0,
        "failed_count": 0,
        "phase3_failures": 0,
        "failure_ratio": 0.0,
        "failed_trial_names": [],
        "source": "clinicaltrials_gov",
        "status_summary": "No matching ClinicalTrials.gov studies found.",
    }
    base.update(overrides)
    return base


def _llm_status(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "has_approved_drug": False,
        "approved_drugs": [],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.5,
        "rationale": "LLM assessment unavailable",
        "drug_verification": "no_reference",
        "verified_drugs": [],
        "feasibility_label": "Moderate",
    }
    base.update(overrides)
    return base


def _pathway_note(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "pathway": "TGF-β/SMAD3",
        "summary": "Direct SMAD3 programs are sparse, but upstream TGF-β programs provide indirect clinical context.",
        "examples": [
            "P144 peptide (TGF-β1 inhibitor): Phase 2 skin-fibrosis program completed without visible follow-up development.",
            "Fresolimumab (anti-TGF-β mAb): early clinical testing reached Phase 1 but programs were discontinued.",
        ],
        "interpretation": "Pathway translation is possible, but systemic blockade has tractability limits.",
        "source_note": "Curated from docs/findings/scar-platform-targets.md",
    }
    base.update(overrides)
    return base


# --- helpers ---


def _make_run_dir(disease: str = "Test Disease"):
    """Create a temporary run directory with config.json, run.db, and some hypotheses."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "run.db")
    config_path = os.path.join(tmp, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"disease": disease, "description": "A test description"}, f)
    return tmp, db_path


def _seed_hypotheses(db_path: str, genes: list[str]) -> None:
    store = ArchiveStore(db_path)
    for i, gene in enumerate(genes):
        h = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "tissue")
        scores = FitnessScores(literature_strength=0.8 - i * 0.1)
        store.save_hypothesis(h, scores, fitness_total=0.9 - i * 0.1)
    store.close()


# --- dataclass creation tests ---


def test_paper_summary_creation():
    ps = PaperSummary(
        pmid="12345678",
        title="Test paper title",
        year=2024,
        citation_count=42,
        relevance="BRCA1 as target for breast cancer",
    )
    assert ps.pmid == "12345678"
    assert ps.title == "Test paper title"
    assert ps.year == 2024
    assert ps.citation_count == 42
    assert ps.relevance == "BRCA1 as target for breast cancer"


def test_target_verification_creation():
    tv = TargetVerification(
        gene="BRCA1",
        disease="Breast Cancer",
        fitness=0.85,
        pubmed_count=25,
        top_papers=[],
        evidence_strength="Strong",
        summary="BRCA1 has strong evidence as a therapeutic target for Breast Cancer (25 papers, 500 citations)",
    )
    assert tv.gene == "BRCA1"
    assert tv.disease == "Breast Cancer"
    assert tv.fitness == 0.85
    assert tv.pubmed_count == 25
    assert tv.evidence_strength == "Strong"


def test_target_verification_defaults():
    tv = TargetVerification(gene="X", disease="Y", fitness=0.5, pubmed_count=0)
    assert tv.top_papers == []
    assert tv.evidence_strength == "No evidence"
    assert tv.summary == ""
    assert tv.clinical_status == {}
    assert tv.llm_feasibility == {}
    assert tv.pathway_trial_note == {}
    assert tv.prior_knowledge is None


# --- evidence strength classification ---


def test_classify_strong():
    assert classify_evidence_strength(20, 100) == "Strong"
    assert classify_evidence_strength(50, 500) == "Strong"


def test_classify_moderate_by_papers():
    assert classify_evidence_strength(5, 10) == "Moderate"
    assert classify_evidence_strength(19, 50) == "Moderate"


def test_classify_moderate_by_citations():
    assert classify_evidence_strength(3, 20) == "Moderate"
    assert classify_evidence_strength(2, 100) == "Moderate"


def test_classify_weak():
    assert classify_evidence_strength(1, 0) == "Weak"
    assert classify_evidence_strength(4, 19) == "Weak"


def test_classify_no_evidence():
    assert classify_evidence_strength(0, 0) == "No evidence"
    assert classify_evidence_strength(0, 50) == "No evidence"


def test_build_summary_handles_no_evidence_wording() -> None:
    from biocompute.verification.literature import _build_summary

    summary = _build_summary("GENE0", "Disease X", "No evidence", 0, 0)

    assert "no direct literature evidence" in summary
    assert "no evidence evidence" not in summary


def test_classify_boundary_strong_needs_both():
    # 20 papers but only 99 citations -> Moderate (needs both conditions for Strong)
    assert classify_evidence_strength(20, 99) == "Moderate"
    # 19 papers with 100 citations -> Moderate
    assert classify_evidence_strength(19, 100) == "Moderate"


# --- report generation ---


def test_report_empty():
    report = generate_verification_report([])
    assert "No targets to verify" in report


def test_report_with_mock_data():
    papers = [
        PaperSummary("111", "Paper A", 2023, 50, "Gene1 in Disease"),
        PaperSummary("222", "Paper B", 2022, 30, "Gene1 mechanism"),
        PaperSummary("333", "Paper C", 2021, 10, "Gene1 pathway"),
    ]
    verifications = [
        TargetVerification(
            gene="GENE1",
            disease="Disease A",
            fitness=0.85,
            pubmed_count=25,
            top_papers=papers,
            evidence_strength="Strong",
            summary="GENE1 has strong evidence as a therapeutic target for Disease A (25 papers, 500 citations)",
            clinical_status=_clinical_status(
                completed_count=2,
                failed_count=1,
                phase3_failures=1,
                failed_trial_names=["Trial A"],
                status_summary="Failure signal present (1 stopped trials, including 1 phase 2/3 failures).",
            ),
            llm_feasibility=_llm_status(
                feasibility_score=0.25,
                feasibility_label="Low",
                has_phase3_failure=True,
                failed_drugs=["Drug X"],
                rationale="Prior late-stage failures reduce confidence.",
            ),
            pathway_trial_note=_pathway_note(),
        ),
        TargetVerification(
            gene="GENE2",
            disease="Disease A",
            fitness=0.70,
            pubmed_count=2,
            top_papers=[],
            evidence_strength="Weak",
            summary="GENE2 has weak evidence as a therapeutic target for Disease A (2 papers, 5 citations)",
            clinical_status=_clinical_status(),
            llm_feasibility=_llm_status(),
        ),
    ]

    report = generate_verification_report(verifications)

    # Summary table present
    assert "| Gene | Fitness | Papers | Evidence |" in report
    assert "GENE1" in report
    assert "GENE2" in report
    assert "0.850" in report
    assert "0.700" in report
    assert "Strong" in report
    assert "Weak" in report

    # Overall assessment
    assert "1 strong" in report
    assert "1 weak" in report

    # Paper details
    assert "PMID:111" in report
    assert "Paper A" in report
    assert "No papers found" in report  # GENE2 has no papers
    assert "Clinical Trial Status" in report
    assert "Failure signal present" in report
    assert "LLM Clinical Feasibility" in report
    assert "Drug X" in report
    assert "Pathway-Level Trial Note" in report
    assert "TGF-β/SMAD3" in report
    assert "P144 peptide" in report


def test_report_shows_only_top_3_papers():
    papers = [
        PaperSummary(str(i), f"Paper {i}", 2020 + i, i * 10, "relevance")
        for i in range(5)
    ]
    v = TargetVerification(
        gene="X",
        disease="Y",
        fitness=0.5,
        pubmed_count=5,
        top_papers=papers,
        evidence_strength="Moderate",
        summary="summary",
        clinical_status=_clinical_status(),
        llm_feasibility=_llm_status(),
    )
    report = generate_verification_report([v])
    # Only papers 0, 1, 2 shown (top 3)
    assert "PMID:0" in report
    assert "PMID:1" in report
    assert "PMID:2" in report
    assert "PMID:3" not in report
    assert "PMID:4" not in report


def test_report_includes_prior_knowledge_framing():
    prior_knowledge = PriorKnowledge(
        gene="GENE1",
        disease="Disease A",
        maturity=EvidenceMaturity.L4_CLINICAL,
        known_facts=["Human genetic association has been replicated."],
        attempted_approaches=["A monoclonal antibody reached early clinical testing."],
        gaps=["No validated responder biomarker is established."],
        key_papers=["PMID:123"],
        summary="Biology is validated, but durable therapeutic translation remains uncertain.",
    )
    verification = TargetVerification(
        gene="GENE1",
        disease="Disease A",
        fitness=0.85,
        pubmed_count=25,
        top_papers=[],
        evidence_strength="Strong",
        summary="GENE1 has strong evidence as a therapeutic target for Disease A (25 papers, 500 citations)",
        clinical_status=_clinical_status(),
        llm_feasibility=_llm_status(),
        prior_knowledge=prior_knowledge,
    )

    report = generate_verification_report([verification])

    assert "Prior Knowledge Framing" in report
    assert "Evidence maturity: L4 clinical evidence" in report
    assert "Known facts:" in report
    assert "Human genetic association has been replicated." in report
    assert "Attempted approaches:" in report
    assert "A monoclonal antibody reached early clinical testing." in report
    assert "Remaining gaps:" in report
    assert "No validated responder biomarker is established." in report
    assert "durable therapeutic translation remains uncertain" in report


def test_report_prior_knowledge_fallback_is_conservative():
    verification = TargetVerification(
        gene="GENE2",
        disease="Disease A",
        fitness=0.70,
        pubmed_count=1,
        top_papers=[],
        evidence_strength="Weak",
        summary="GENE2 has weak evidence as a therapeutic target for Disease A (1 papers, 5 citations)",
        clinical_status=_clinical_status(),
        llm_feasibility=_llm_status(),
    )

    report = generate_verification_report([verification])

    assert "Prior Knowledge Framing" in report
    assert "Stored prior knowledge was not available for this hypothesis." in report
    assert "without inferring evidence maturity" in report


# --- verify_targets with mocked API calls ---


@pytest.mark.asyncio
async def test_verify_targets_with_mocked_apis():
    run_dir, db_path = _make_run_dir("Breast Cancer")
    _seed_hypotheses(db_path, ["BRCA1", "TP53"])

    mock_pmids = ["11111", "22222", "33333", "44444", "55555"]

    mock_esummary_response = {
        "result": {
            "11111": {"title": "BRCA1 in breast cancer therapy", "pubdate": "2023 Jan"},
            "22222": {"title": "Targeting BRCA1 pathway", "pubdate": "2022 Mar"},
            "33333": {"title": "BRCA1 DNA repair mechanism", "pubdate": "2021"},
            "44444": {"title": "Novel BRCA1 inhibitors", "pubdate": "2020 Jun"},
            "55555": {"title": "BRCA1 clinical trial results", "pubdate": "2019"},
        }
    }

    mock_s2_papers = [
        {
            "paperId": "s1",
            "title": "BRCA1 in breast cancer therapy",
            "citationCount": 45,
            "influentialCitationCount": 5,
            "year": 2023,
        },
        {
            "paperId": "s2",
            "title": "Other paper",
            "citationCount": 30,
            "influentialCitationCount": 2,
            "year": 2022,
        },
    ]

    async def mock_search_pubmed(client, query, max_results=20):
        return mock_pmids

    async def mock_search_papers(client, query, limit=10):
        return mock_s2_papers

    async def mock_fetch_titles(client, pmids):
        result = {}
        for pmid in pmids:
            entry = mock_esummary_response["result"].get(pmid, {})
            title = entry.get("title", "")
            pubdate = entry.get("pubdate", "")
            year = int(pubdate[:4]) if pubdate and len(pubdate) >= 4 else 0
            if title:
                result[pmid] = (title, year)
        return result

    with (
        patch("biocompute.verification.literature.search_pubmed", mock_search_pubmed),
        patch("biocompute.verification.literature.search_papers", mock_search_papers),
        patch(
            "biocompute.verification.literature._fetch_paper_titles", mock_fetch_titles
        ),
        patch(
            "biocompute.verification.literature.get_clinical_outcome",
            AsyncMock(
                return_value={
                    "completed_count": 1,
                    "failed_count": 2,
                    "phase3_failures": 1,
                    "failure_ratio": 0.67,
                    "failed_trial_names": ["Trial Alpha"],
                    "source": "clinicaltrials_gov",
                }
            ),
        ),
        patch(
            "biocompute.verification.literature.assess_clinical_feasibility",
            return_value={
                "has_approved_drug": False,
                "approved_drugs": [],
                "has_phase3_failure": True,
                "failed_drugs": ["Drug Beta"],
                "feasibility_score": 0.2,
                "rationale": "Late-stage failure signal present.",
                "drug_verification": "no_reference",
                "verified_drugs": [],
            },
        ),
    ):
        results = await _verify_targets_async(db_path, top_n=2)

    assert len(results) == 2
    # First target (highest fitness)
    assert results[0].gene == "BRCA1"
    assert results[0].disease == "Breast Cancer"
    assert results[0].pubmed_count == 5
    assert results[0].evidence_strength == "Moderate"  # 5 papers, 75 citations
    assert len(results[0].top_papers) == 5
    assert results[0].clinical_status["failed_count"] == 2
    assert results[0].clinical_status["phase3_failures"] == 1
    clinical_status = results[0].clinical_status
    assert isinstance(clinical_status, Mapping)
    assert "Failure signal present" in str(clinical_status["status_summary"])
    assert results[0].llm_feasibility["feasibility_score"] == pytest.approx(0.2)
    assert results[0].llm_feasibility["feasibility_label"] == "Low"
    assert results[0].llm_feasibility["failed_drugs"] == ["Drug Beta"]
    assert results[0].pathway_trial_note == {}

    # Second target
    assert results[1].gene == "TP53"


@pytest.mark.asyncio
async def test_verify_targets_reuses_stored_prior_knowledge() -> None:
    run_dir, db_path = _make_run_dir("Fibrosis")
    store = ArchiveStore(db_path)
    hypothesis = TherapeuticHypothesis(
        "SMAD3", "mAb", "systemic", "chronic", "fibrotic tissue"
    )
    store.save_hypothesis(
        hypothesis, FitnessScores(literature_strength=0.8), fitness_total=0.91
    )
    prior_knowledge = PriorKnowledge(
        gene="SMAD3",
        disease="Fibrosis",
        maturity=EvidenceMaturity.L3_IN_VIVO,
        known_facts=["Fibrotic signaling is elevated in animal models."],
        attempted_approaches=["Upstream TGF-β blockade has been tested."],
        gaps=["Targeted local delivery remains underdefined."],
        key_papers=["PMID:999"],
        summary="Preclinical biology is established, but therapeutic translation needs tighter modality control.",
    )
    store.save_prior_knowledge(hypothesis.id, prior_knowledge)
    store.close()

    with (
        patch(
            "biocompute.verification.literature.search_pubmed",
            AsyncMock(return_value=[]),
        ),
        patch(
            "biocompute.verification.literature.search_papers",
            AsyncMock(return_value=[]),
        ),
        patch(
            "biocompute.verification.literature.get_clinical_outcome",
            AsyncMock(return_value=_clinical_status()),
        ),
        patch(
            "biocompute.verification.literature.assess_clinical_feasibility",
            return_value=_llm_status(),
        ),
    ):
        results = await _verify_targets_async(db_path, top_n=1)

    assert len(results) == 1
    assert results[0].gene == "SMAD3"
    assert results[0].prior_knowledge == prior_knowledge


@pytest.mark.asyncio
async def test_verify_targets_adds_smad3_pathway_trial_note():
    run_dir, db_path = _make_run_dir("Hypertrophic Scarring")
    _seed_hypotheses(db_path, ["SMAD3"])

    with (
        patch(
            "biocompute.verification.literature.search_pubmed",
            AsyncMock(return_value=[]),
        ),
        patch(
            "biocompute.verification.literature.search_papers",
            AsyncMock(return_value=[]),
        ),
        patch(
            "biocompute.verification.literature.get_clinical_outcome",
            AsyncMock(return_value=_clinical_status()),
        ),
        patch(
            "biocompute.verification.literature.assess_clinical_feasibility",
            return_value=_llm_status(),
        ),
    ):
        results = await _verify_targets_async(db_path, top_n=1)

    assert len(results) == 1
    assert results[0].gene == "SMAD3"
    assert results[0].pathway_trial_note["pathway"] == "TGF-β/SMAD3"
    assert "P144 peptide" in str(results[0].pathway_trial_note["examples"])


@pytest.mark.asyncio
async def test_verify_targets_empty_db():
    run_dir, db_path = _make_run_dir()
    # Create DB schema but don't add any hypotheses
    store = ArchiveStore(db_path)
    store.close()

    with (
        patch(
            "biocompute.verification.literature.search_pubmed",
            AsyncMock(return_value=[]),
        ),
        patch(
            "biocompute.verification.literature.search_papers",
            AsyncMock(return_value=[]),
        ),
    ):
        results = await _verify_targets_async(db_path, top_n=5)

    assert results == []


@pytest.mark.asyncio
async def test_verify_targets_api_failure_graceful():
    """verify_targets handles API failures gracefully, returning 0 papers."""
    run_dir, db_path = _make_run_dir("Cancer")
    _seed_hypotheses(db_path, ["BRCA1"])

    async def failing_search(client, query, max_results=20):
        raise Exception("503 Service Unavailable")

    async def failing_s2(client, query, limit=10):
        raise Exception("429 Too Many Requests")

    with (
        patch("biocompute.verification.literature.search_pubmed", failing_search),
        patch("biocompute.verification.literature.search_papers", failing_s2),
        patch(
            "biocompute.verification.literature.get_clinical_outcome",
            AsyncMock(side_effect=Exception("503 Service Unavailable")),
        ),
        patch(
            "biocompute.verification.literature.assess_clinical_feasibility",
            side_effect=RuntimeError("Claude CLI failed"),
        ),
    ):
        results = await _verify_targets_async(db_path, top_n=1)

    assert len(results) == 1
    assert results[0].pubmed_count == 0
    assert results[0].evidence_strength == "No evidence"
    assert results[0].top_papers == []
    assert results[0].clinical_status["status_summary"] == (
        "No matching ClinicalTrials.gov studies found."
    )
    assert results[0].llm_feasibility["rationale"] == "LLM assessment unavailable"
    assert results[0].pathway_trial_note == {}


@pytest.mark.asyncio
async def test_verify_targets_handles_malformed_semantic_scholar_citation_count():
    run_dir, db_path = _make_run_dir("Breast Cancer")
    _seed_hypotheses(db_path, ["BRCA1"])

    async def mock_fetch_titles(client, pmids):
        return {"11111": ("BRCA1 in breast cancer therapy", 2023)}

    with (
        patch(
            "biocompute.verification.literature.search_pubmed",
            AsyncMock(return_value=["11111"]),
        ),
        patch(
            "biocompute.verification.literature.search_papers",
            AsyncMock(
                return_value=[
                    {
                        "paperId": "s1",
                        "title": "BRCA1 in breast cancer therapy",
                        "citationCount": None,
                    },
                    {
                        "paperId": "s2",
                        "title": "Other paper",
                        "citationCount": "unknown",
                    },
                    {
                        "paperId": "s3",
                        "title": "Infinite citations paper",
                        "citationCount": float("inf"),
                    },
                    {
                        "paperId": "s4",
                        "title": "NaN citations paper",
                        "citationCount": float("nan"),
                    },
                ]
            ),
        ),
        patch(
            "biocompute.verification.literature._fetch_paper_titles", mock_fetch_titles
        ),
        patch(
            "biocompute.verification.literature.get_clinical_outcome",
            AsyncMock(return_value=_clinical_status()),
        ),
        patch(
            "biocompute.verification.literature.assess_clinical_feasibility",
            return_value=_llm_status(),
        ),
    ):
        results = await _verify_targets_async(db_path, top_n=1)

    assert len(results) == 1
    assert results[0].gene == "BRCA1"
    assert results[0].evidence_strength == "Weak"
    assert results[0].summary.endswith("(1 papers, 0 citations)")
    assert len(results[0].top_papers) == 1
    assert results[0].top_papers[0].citation_count == 0


@pytest.mark.asyncio
async def test_verify_targets_deduplicates_by_gene() -> None:
    run_dir, db_path = _make_run_dir("Breast Cancer")
    store = ArchiveStore(db_path)

    first = TherapeuticHypothesis("BRCA1", "mAb", "systemic", "chronic", "systemic")
    second = TherapeuticHypothesis("BRCA1", "siRNA", "local", "acute", "tumor")
    third = TherapeuticHypothesis("TP53", "mAb", "systemic", "chronic", "systemic")
    store.save_hypothesis(
        first, FitnessScores(literature_strength=0.8), fitness_total=0.95
    )
    store.save_hypothesis(
        second, FitnessScores(literature_strength=0.7), fitness_total=0.90
    )
    store.save_hypothesis(
        third, FitnessScores(literature_strength=0.6), fitness_total=0.85
    )
    store.close()

    with (
        patch(
            "biocompute.verification.literature.search_pubmed",
            AsyncMock(return_value=[]),
        ),
        patch(
            "biocompute.verification.literature.search_papers",
            AsyncMock(return_value=[]),
        ),
        patch(
            "biocompute.verification.literature.get_clinical_outcome",
            AsyncMock(return_value=_clinical_status()),
        ),
        patch(
            "biocompute.verification.literature.assess_clinical_feasibility",
            return_value=_llm_status(),
        ),
    ):
        results = await _verify_targets_async(db_path, top_n=2)

    assert [result.gene for result in results] == ["BRCA1", "TP53"]
    assert all(result.pathway_trial_note == {} for result in results)


# --- CLI test ---


def test_verify_help():
    from click.testing import CliRunner
    from biocompute.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["verify", "--help"])
    assert result.exit_code == 0
    assert "Verify top targets against literature evidence" in result.output
    assert "--top" in result.output
    assert "--output" in result.output


def test_verify_missing_db():
    from click.testing import CliRunner
    from biocompute.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["verify", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "not found" in result.output
