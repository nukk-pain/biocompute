# pyright: reportMissingImports=false

from datetime import datetime

from biocompute.archive.report import generate_report
from biocompute.models import (
    DiseaseQuery,
    DiscoveryResult,
    Evidence,
    EvidenceMaturity,
    FitnessScores,
    PriorKnowledge,
    RunMetadata,
    ScoredHypothesis,
    StrategyPriorArt,
    TherapeuticHypothesis,
)


def test_generate_report_contains_key_sections() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain from scar tissue")
    hypothesis = TherapeuticHypothesis(
        "CXCL12",
        "VHH",
        "local injection",
        "single-dose",
        "scar tissue",
    )
    scored = ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=0.75,
        scores=FitnessScores(
            literature_strength=0.8,
            expression_specificity=0.6,
            pathway_centrality=0.5,
            druggability=0.7,
            safety_profile=0.9,
            ip_freedom=0.4,
        ),
        evidence=[Evidence("pubmed", "PMID:12345", "CXCL12 in scar tissue", 0.8)],
        critiques=["No animal model data"],
    )
    metadata = RunMetadata(
        started_at=datetime(2026, 4, 10, 12, 0),
        finished_at=datetime(2026, 4, 10, 12, 30),
        generations_run=5,
        total_hypotheses=150,
    )
    result = DiscoveryResult(query=query, candidates=[scored], metadata=metadata)

    report = generate_report(result)

    assert "MPS" in report
    assert "CXCL12" in report
    assert "0.75" in report or "0.750" in report
    assert "Evidence" in report or "evidence" in report.lower()


def test_generate_report_includes_prior_knowledge() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain from scar tissue")
    hypothesis = TherapeuticHypothesis(
        "CXCL12",
        "VHH",
        "local injection",
        "single-dose",
        "scar tissue",
    )
    pk = PriorKnowledge(
        gene="CXCL12",
        disease="MPS",
        maturity=EvidenceMaturity.L2_IN_VITRO,
        known_facts=["Fact 1"],
        attempted_approaches=["Approach 1"],
        gaps=["Gap 1"],
        key_papers=["Paper 1"],
        summary="A summary",
    )
    scored = ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=0.75,
        scores=FitnessScores(),
        prior_knowledge=pk,
    )
    metadata = RunMetadata()
    result = DiscoveryResult(query=query, candidates=[scored], metadata=metadata)

    report = generate_report(result)

    assert "Prior Knowledge" in report
    assert "L2_IN_VITRO" in report
    assert "A summary" in report
    assert "Fact 1" in report
    assert "Approach 1" in report
    assert "Gap 1" in report


def test_generate_report_includes_unavailable_prior_knowledge() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain from scar tissue")
    hypothesis = TherapeuticHypothesis(
        "CXCL12",
        "VHH",
        "local injection",
        "single-dose",
        "scar tissue",
    )
    scored = ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=0.75,
        scores=FitnessScores(),
        prior_knowledge=None,
    )
    metadata = RunMetadata()
    result = DiscoveryResult(query=query, candidates=[scored], metadata=metadata)

    report = generate_report(result)

    assert "Prior Knowledge" in report
    assert "Unavailable" in report


def test_generate_report_includes_strategy_prior_art_when_available() -> None:
    query = DiseaseQuery("MPS", "Myofascial pain from scar tissue")
    hypothesis = TherapeuticHypothesis(
        "SMAD3",
        "mRNA-LNP",
        "local injection",
        "single-dose",
        "scar tissue",
    )
    scored = ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=0.81,
        scores=FitnessScores(),
        strategy_prior_art=StrategyPriorArt(
            strategy="SMAD7 overexpression",
            disease_class="fibrosis",
            prior_studies=["AAV5-Smad7 corneal fibrosis"],
            modality_status={"AAV": "in vivo confirmed", "mRNA-LNP": "not attempted"},
            our_differentiation=["transient expression", "repeat dosing"],
            key_papers=["PMID:28339457"],
            summary="AAV prior art exists while mRNA-LNP remains open.",
        ),
    )
    metadata = RunMetadata()
    result = DiscoveryResult(query=query, candidates=[scored], metadata=metadata)

    report = generate_report(result)

    assert "Strategy Prior Art" in report
    assert "SMAD7 overexpression" in report
    assert "fibrosis" in report
    assert "AAV5-Smad7 corneal fibrosis" in report
    assert "AAV: in vivo confirmed" in report
    assert "mRNA-LNP: not attempted" in report
    assert "transient expression" in report
    assert "PMID:28339457" in report
    assert "AAV prior art exists while mRNA-LNP remains open." in report
