"""Export biocompute results to neuroregen Phase B compatible JSON."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import httpx

from biocompute.archive.store import ArchiveStore

_gene_id_cache: dict[str, int | None] = {}


def _select_top_unique_hypotheses(
    conn: sqlite3.Connection,
    top_n: int,
) -> list[sqlite3.Row]:
    """Return the best-scoring hypothesis per gene, capped at top_n unique genes."""
    rows = conn.execute(
        "SELECT * FROM hypotheses ORDER BY fitness_total DESC"
    ).fetchall()

    seen_genes: set[str] = set()
    hypotheses: list[sqlite3.Row] = []
    for row in rows:
        gene = row["target_gene"]
        if gene in seen_genes:
            continue
        hypotheses.append(row)
        seen_genes.add(gene)
        if len(hypotheses) >= top_n:
            break

    return hypotheses


def _lookup_ncbi_gene_id(gene_symbol: str) -> int | None:
    """Look up NCBI Gene ID for a human gene symbol via NCBI esearch.

    Results are cached in-process to avoid duplicate requests for the
    same gene across candidates.
    """
    if gene_symbol in _gene_id_cache:
        return _gene_id_cache[gene_symbol]
    try:
        resp = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "gene",
                "term": f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
                "retmode": "json",
            },
            timeout=10,
        )
        data = resp.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        result = int(ids[0]) if ids else None
    except Exception:
        result = None
    _gene_id_cache[gene_symbol] = result
    return result


def export_for_neuroregen(db_path: str, top_n: int = 5) -> dict[str, Any]:
    """Export top candidates from a biocompute run as neuroregen-compatible JSON.

    Args:
        db_path: Path to the SQLite run.db file.
        top_n: Number of top candidates to include (by fitness_total DESC).

    Returns:
        Dictionary with source, version, disease, description, and candidates.
    """
    run_dir = os.path.dirname(db_path)
    config_path = os.path.join(run_dir, "config.json")

    # Load disease metadata from config.json
    disease_name = "unknown"
    description = ""
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        disease_name = config.get("disease", "unknown")
        description = config.get("description", "")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    store = ArchiveStore(db_path)
    hypotheses = _select_top_unique_hypotheses(conn, top_n)

    candidates: list[dict[str, Any]] = []
    for hyp in hypotheses:
        hypothesis_id = hyp["id"]

        # Load scores (6 dimensions)
        score_rows = conn.execute(
            "SELECT dimension, score, raw_data FROM scores WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchall()

        fitness_breakdown: dict[str, float] = {}
        pathway_interactions: list[dict[str, Any]] = []
        for row in score_rows:
            fitness_breakdown[row["dimension"]] = row["score"]

            # Extract pathway interactions from raw_data if available
            if row["dimension"] == "pathway_centrality" and row["raw_data"]:
                try:
                    raw = json.loads(row["raw_data"])
                    if isinstance(raw, dict) and "interactions" in raw:
                        pathway_interactions = raw["interactions"]
                except (json.JSONDecodeError, TypeError):
                    pass

        # Load evidence
        evidence_rows = conn.execute(
            "SELECT source_type, summary, relevance_score FROM evidence WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchall()

        evidence_list = [
            {
                "source": _map_evidence_source(ev["source_type"]),
                "description": ev["summary"],
                "confidence": ev["relevance_score"],
            }
            for ev in evidence_rows
            if ev["source_type"] != "api_error"
        ]

        # Load prior knowledge
        pk = store.get_prior_knowledge(hypothesis_id)
        prior_knowledge_data = None
        if pk:
            prior_knowledge_data = {
                "maturity": pk.maturity.name,
                "summary": pk.summary,
                "known_facts": pk.known_facts,
                "attempted_approaches": pk.attempted_approaches,
                "gaps": pk.gaps,
                "key_papers": pk.key_papers,
            }

        candidate: dict[str, Any] = {
            "gene": {"symbol": hyp["target_gene"]},
            "score": hyp["fitness_total"],
            "modality": hyp["modality"],
            "delivery": hyp["delivery"],
            "evidence": evidence_list,
            "pathway": pathway_interactions,
            "fitness_breakdown": fitness_breakdown,
            "prior_knowledge": prior_knowledge_data,
        }
        candidates.append(candidate)

    conn.close()
    store.close()

    return {
        "source": "biocompute",
        "version": "0.1.0",
        "disease": disease_name,
        "description": description,
        "candidates": candidates,
    }


# Mapping from biocompute evidence source_type to neuroregen pipeline source names
_EVIDENCE_SOURCE_MAP = {
    "pubmed": "Literature",
    "semantic_scholar": "Literature",
    "pubmed+semantic_scholar": "Literature",
    "hpa": "Expression",
    "gtex": "Expression",
    "hpa+gtex": "Expression",
    "string": "Pathway",
    "opentargets": "Literature",
    "opentargets_heuristic": "Literature",
    "clinicaltrials_gov": "Literature",
    "opentargets_gwas": "Literature",
    "class_effect": "Literature",
}


def _map_evidence_source(source_type: str) -> str:
    """Map a biocompute evidence source_type to a neuroregen pipeline source name."""
    return _EVIDENCE_SOURCE_MAP.get(source_type, source_type)


def export_for_neuroregen_pipeline(
    db_path: str, top_n: int = 5
) -> list[dict[str, Any]]:
    """Export as a bare array matching neuroregen's Vec<TargetCandidate> schema.

    Unlike export_for_neuroregen(), this returns a bare JSON array (not wrapped
    in {"candidates": [...]}) with simplified fields for direct pipeline consumption.

    Args:
        db_path: Path to the SQLite run.db file.
        top_n: Number of top candidates to include (by fitness_total DESC).

    Returns:
        List of dicts, each with gene, score, evidence, and pathway fields.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    hypotheses = _select_top_unique_hypotheses(conn, top_n)

    candidates: list[dict[str, Any]] = []
    for hyp in hypotheses:
        hypothesis_id = hyp["id"]

        # Extract pathway interaction partner names from raw_data
        pathway_names: list[str] = []
        score_rows = conn.execute(
            "SELECT dimension, raw_data FROM scores WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchall()
        for row in score_rows:
            if row["dimension"] == "pathway_centrality" and row["raw_data"]:
                try:
                    raw = json.loads(row["raw_data"])
                    if isinstance(raw, dict) and "interactions" in raw:
                        for interaction in raw["interactions"]:
                            if (
                                isinstance(interaction, dict)
                                and "partner" in interaction
                            ):
                                pathway_names.append(interaction["partner"])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Load evidence with source mapping
        evidence_rows = conn.execute(
            "SELECT source_type, summary, relevance_score FROM evidence WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchall()

        evidence_list = [
            {
                "source": _map_evidence_source(ev["source_type"]),
                "description": ev["summary"],
                "confidence": ev["relevance_score"],
            }
            for ev in evidence_rows
            if ev["source_type"] != "api_error"
        ]

        gene_symbol = hyp["target_gene"]
        candidate: dict[str, Any] = {
            "gene": {
                "symbol": gene_symbol,
                "ncbi_id": _lookup_ncbi_gene_id(gene_symbol),
                "uniprot_id": None,
            },
            "score": hyp["fitness_total"],
            "evidence": evidence_list,
            "pathway": pathway_names,
        }
        candidates.append(candidate)

    conn.close()
    return candidates
