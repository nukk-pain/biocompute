"""Automatic literature verification for discovery results."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from biocompute.archive.store import ArchiveStore
from biocompute.data.clinical_trials import get_clinical_outcome
from biocompute.data.pubmed import search_pubmed, _build_disease_query, EUTILS_BASE
from biocompute.data.semantic_scholar import search_papers
from biocompute.fitness.llm_clinical import assess_clinical_feasibility
from biocompute.models import PriorKnowledge


@dataclass
class PaperSummary:
    pmid: str
    title: str
    year: int
    citation_count: int
    relevance: str  # one-line summary of relevance to disease-target pair


@dataclass
class TargetVerification:
    gene: str
    disease: str
    fitness: float
    pubmed_count: int
    top_papers: list[PaperSummary] = field(default_factory=list)
    evidence_strength: str = (
        "No evidence"  # "Strong", "Moderate", "Weak", "No evidence"
    )
    summary: str = ""
    clinical_status: Mapping[str, object] = field(default_factory=dict)
    llm_feasibility: Mapping[str, object] = field(default_factory=dict)
    pathway_trial_note: Mapping[str, object] = field(default_factory=dict)
    prior_knowledge: PriorKnowledge | None = None


def _default_clinical_status() -> dict[str, object]:
    return {
        "completed_count": 0,
        "failed_count": 0,
        "phase3_failures": 0,
        "failure_ratio": 0.0,
        "failed_trial_names": [],
        "source": "clinicaltrials_gov",
        "status_summary": "No matching ClinicalTrials.gov studies found.",
    }


def _summarize_clinical_status(clinical_data: dict[str, object]) -> str:
    completed_count = clinical_data.get("completed_count", 0)
    completed_count = completed_count if isinstance(completed_count, int) else 0

    failed_count = clinical_data.get("failed_count", 0)
    failed_count = failed_count if isinstance(failed_count, int) else 0

    phase3_failures = clinical_data.get("phase3_failures", 0)
    phase3_failures = phase3_failures if isinstance(phase3_failures, int) else 0

    if completed_count == 0 and failed_count == 0:
        return "No matching ClinicalTrials.gov studies found."
    if failed_count == 0:
        return f"Completed-trial signal only ({completed_count} completed, no stopped trials found)."
    if phase3_failures > 0:
        return f"Failure signal present ({failed_count} stopped trials, including {phase3_failures} phase 2/3 failures)."
    return f"Mixed clinical signal ({completed_count} completed, {failed_count} stopped trials)."


def _enrich_clinical_status(clinical_data: dict[str, object]) -> dict[str, object]:
    enriched = _default_clinical_status()
    enriched.update(clinical_data)
    failed_trial_names = enriched.get("failed_trial_names", [])
    if not isinstance(failed_trial_names, list):
        enriched["failed_trial_names"] = []
    enriched["status_summary"] = _summarize_clinical_status(enriched)
    return enriched


def _default_llm_feasibility() -> dict[str, object]:
    return {
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


def _pathway_trial_note(gene: str) -> dict[str, object]:
    if gene.upper() != "SMAD3":
        return {}

    return {
        "pathway": "TGF-β/SMAD3",
        "summary": (
            "Direct SMAD3 clinical programs remain sparse, but upstream TGF-β pathway drugs "
            "show a mixed translational record that is relevant to scar/fibrosis strategy."
        ),
        "examples": [
            "P144 peptide (TGF-β1 inhibitor): Phase 2 skin-fibrosis program completed without visible follow-up development.",
            "Fresolimumab (anti-TGF-β mAb): early clinical testing reached Phase 1 but programs were discontinued.",
            "Galunisertib (ALK5 / TGF-βRI inhibitor): advanced to Phase 2 in oncology, with hepatotoxicity concerns limiting enthusiasm.",
        ],
        "interpretation": (
            "The pathway appears clinically actionable, but systemic TGF-β blockade has shown tractability and safety limits; "
            "that supports exploring local or modality-constrained SMAD3 strategies rather than assuming broad pathway inhibition is sufficient."
        ),
        "source_note": "Curated from docs/findings/scar-platform-targets.md",
    }


def _feasibility_label(score: object) -> str:
    if not isinstance(score, (int, float)):
        return "Moderate"
    value = float(score)
    if value >= 0.7:
        return "High"
    if value <= 0.3:
        return "Low"
    return "Moderate"


def _enrich_llm_feasibility(payload: dict[str, object]) -> dict[str, object]:
    enriched = _default_llm_feasibility()
    enriched.update(payload)

    approved_drugs = enriched.get("approved_drugs", [])
    if not isinstance(approved_drugs, list):
        enriched["approved_drugs"] = []

    failed_drugs = enriched.get("failed_drugs", [])
    if not isinstance(failed_drugs, list):
        enriched["failed_drugs"] = []

    verified_drugs = enriched.get("verified_drugs", [])
    if not isinstance(verified_drugs, list):
        enriched["verified_drugs"] = []

    enriched["feasibility_label"] = _feasibility_label(
        enriched.get("feasibility_score")
    )
    return enriched


def classify_evidence_strength(paper_count: int, total_citations: int) -> str:
    """Classify evidence strength based on paper count and citation totals.

    Rules:
        20+ papers with 100+ total citations -> "Strong"
        5-19 papers or 20+ total citations -> "Moderate"
        1-4 papers -> "Weak"
        0 papers -> "No evidence"
    """
    if paper_count == 0:
        return "No evidence"
    if paper_count >= 20 and total_citations >= 100:
        return "Strong"
    if paper_count >= 5 or total_citations >= 20:
        return "Moderate"
    return "Weak"


def _build_summary(
    gene: str, disease: str, strength: str, count: int, citations: int
) -> str:
    if strength == "No evidence":
        return (
            f"{gene} currently has no direct literature evidence as a therapeutic target "
            f"for {disease} ({count} papers, {citations} citations)"
        )
    return (
        f"{gene} has {strength.lower()} evidence as a therapeutic target "
        f"for {disease} ({count} papers, {citations} citations)"
    )


def _normalize_citation_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        return int(value)
    return 0


async def _fetch_paper_titles(
    client: httpx.AsyncClient,
    pmids: list[str],
) -> dict[str, tuple[str, int]]:
    """Fetch titles and publication years for a list of PMIDs via PubMed efetch.

    Returns:
        Dict mapping pmid -> (title, year). Missing entries are omitted.
    """
    if not pmids:
        return {}
    try:
        response = await client.get(
            f"{EUTILS_BASE}/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
            },
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}

    result: dict[str, tuple[str, int]] = {}
    uid_data = data.get("result", {})
    for pmid in pmids:
        entry = uid_data.get(pmid, {})
        title = entry.get("title", "")
        # pubdate format is typically "YYYY Mon DD" or "YYYY"
        pubdate = entry.get("pubdate", "")
        year = 0
        if pubdate and len(pubdate) >= 4:
            try:
                year = int(pubdate[:4])
            except ValueError:
                year = 0
        if title:
            result[pmid] = (title, year)
    return result


async def _verify_single_target(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
    fitness: float,
    prior_knowledge: PriorKnowledge | None = None,
) -> TargetVerification:
    """Verify a single target against PubMed and Semantic Scholar."""
    # Step 1: Search PubMed (use keyword splitting for rare disease names)
    disease_clause = _build_disease_query(disease)
    query = f'"{gene}" AND {disease_clause} AND therapeutic'
    try:
        pmids = await search_pubmed(client, query, max_results=50)
        if not pmids:
            fallback = f'"{gene}" AND {disease_clause}'
            pmids = await search_pubmed(client, fallback, max_results=50)
    except Exception:
        pmids = []

    pubmed_count = len(pmids)

    # Step 2: Fetch paper titles for top PMIDs
    top_pmids = pmids[:10]
    titles_map = await _fetch_paper_titles(client, top_pmids)

    # Step 3: Search Semantic Scholar for citation counts
    s2_query = f"{gene} {disease} therapeutic target"
    try:
        s2_papers = await search_papers(client, s2_query, limit=20)
    except Exception:
        s2_papers = []

    total_citations = sum(
        _normalize_citation_count(p.get("citationCount")) for p in s2_papers
    )

    # Build citation lookup from S2 results (by title similarity — best effort)
    s2_by_title: dict[str, int] = {}
    for p in s2_papers:
        t = p.get("title", "").lower().strip()
        if t:
            s2_by_title[t] = _normalize_citation_count(p.get("citationCount"))

    # Step 4: Build PaperSummary list
    papers: list[PaperSummary] = []
    for pmid in top_pmids:
        if pmid not in titles_map:
            continue
        title, year = titles_map[pmid]
        # Try to match S2 citation count by title
        cite_count = s2_by_title.get(title.lower().strip(), 0)
        papers.append(
            PaperSummary(
                pmid=pmid,
                title=title,
                year=year,
                citation_count=cite_count,
                relevance=f"{gene} as therapeutic target for {disease}",
            )
        )

    # Step 5: Classify and summarize
    strength = classify_evidence_strength(pubmed_count, total_citations)
    summary = _build_summary(gene, disease, strength, pubmed_count, total_citations)

    # Step 6: Clinical trial signal
    try:
        clinical_status = await get_clinical_outcome(client, gene, disease)
    except Exception:
        clinical_status = _default_clinical_status()
    clinical_status = _enrich_clinical_status(clinical_status)

    # Step 7: LLM clinical feasibility
    try:
        llm_feasibility = await asyncio.to_thread(
            assess_clinical_feasibility, gene, disease
        )
    except Exception:
        llm_feasibility = _default_llm_feasibility()
    if not isinstance(llm_feasibility, dict):
        llm_feasibility = _default_llm_feasibility()
    llm_feasibility = _enrich_llm_feasibility(llm_feasibility)

    pathway_trial_note = _pathway_trial_note(gene)

    return TargetVerification(
        gene=gene,
        disease=disease,
        fitness=fitness,
        pubmed_count=pubmed_count,
        top_papers=papers,
        evidence_strength=strength,
        summary=summary,
        clinical_status=clinical_status,
        llm_feasibility=llm_feasibility,
        pathway_trial_note=pathway_trial_note,
        prior_knowledge=prior_knowledge,
    )


async def _verify_targets_async(
    db_path: str,
    top_n: int = 5,
) -> list[TargetVerification]:
    """Async implementation of verify_targets."""
    # Load top-N unique genes by best fitness so verification breadth matches
    # export/pipeline breadth even when multiple hypotheses share the same gene.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT * FROM hypotheses ORDER BY fitness_total DESC"
    ).fetchall()
    conn.close()

    seen_genes: set[str] = set()
    rows: list[sqlite3.Row] = []
    for row in all_rows:
        gene = row["target_gene"]
        if gene in seen_genes:
            continue
        rows.append(row)
        seen_genes.add(gene)
        if len(rows) >= top_n:
            break

    if not rows:
        return []

    # Load disease name from config.json
    run_dir = os.path.dirname(db_path)
    config_path = os.path.join(run_dir, "config.json")
    disease = "unknown"
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        disease = config.get("disease", "unknown")

    store = ArchiveStore(db_path)
    try:
        # Process targets sequentially (S2 rate limit)
        verifications: list[TargetVerification] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for row in rows:
                hypothesis_id = row["id"]
                gene = row["target_gene"]
                fitness = row["fitness_total"]
                prior_knowledge = None
                if isinstance(hypothesis_id, str):
                    prior_knowledge = store.get_prior_knowledge(hypothesis_id)
                verification = await _verify_single_target(
                    client,
                    gene,
                    disease,
                    fitness,
                    prior_knowledge=prior_knowledge,
                )
                verifications.append(verification)
    finally:
        store.close()

    return verifications


def verify_targets(db_path: str, top_n: int = 5) -> list[TargetVerification]:
    """Fetch and verify top discovery targets against published literature.

    Args:
        db_path: Path to the SQLite run.db file.
        top_n: Number of top hypotheses to verify.

    Returns:
        List of TargetVerification results, one per target.
    """
    return asyncio.run(_verify_targets_async(db_path, top_n))
