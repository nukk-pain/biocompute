# pyright: reportMissingImports=false

from __future__ import annotations

from collections.abc import Mapping

from biocompute.fitness.clinical import score_clinical
from biocompute.fitness.druggability import score_druggability
from biocompute.fitness.expression import score_expression
from biocompute.fitness.gwas import has_strong_gwas_signal, score_gwas
from biocompute.fitness.ip import score_ip
from biocompute.fitness.literature import score_literature
from biocompute.fitness.pathway import score_pathway
from biocompute.fitness.safety import score_safety
from biocompute.models import (
    DiseaseQuery,
    Evidence,
    FitnessScores,
    ScoredHypothesis,
    TherapeuticHypothesis,
    Weights,
    compute_fitness,
)


def evaluate_all_dimensions(
    hypothesis: TherapeuticHypothesis,
    query: DiseaseQuery,
    raw_data: Mapping[str, object],
    weights: Weights | None = None,
) -> ScoredHypothesis:
    if weights is None:
        weights = Weights()

    del query
    all_evidence: list[Evidence] = []

    literature_score, literature_evidence = score_literature(
        _section(raw_data, "literature")
    )
    all_evidence.extend(literature_evidence)

    expression_score, expression_evidence = score_expression(
        _section(raw_data, "expression"),
        target_tissue=hypothesis.tissue_context,
    )
    all_evidence.extend(expression_evidence)

    pathway_score, pathway_evidence = score_pathway(_section(raw_data, "pathway"))
    all_evidence.extend(pathway_evidence)

    druggability_score, druggability_evidence = score_druggability(
        _section(raw_data, "druggability")
    )
    all_evidence.extend(druggability_evidence)

    safety_score, safety_evidence = score_safety(
        _section(raw_data, "safety"),
        delivery=hypothesis.delivery,
        duration=hypothesis.duration,
        gene=hypothesis.target_gene,
    )
    all_evidence.extend(safety_evidence)

    ip_score, ip_evidence = score_ip(_section(raw_data, "ip"))
    all_evidence.extend(ip_evidence)

    scores = FitnessScores(
        literature_strength=literature_score,
        expression_specificity=expression_score,
        pathway_centrality=pathway_score,
        druggability=druggability_score,
        safety_profile=safety_score,
        ip_freedom=ip_score,
    )
    fitness = compute_fitness(scores, weights, gene=hypothesis.target_gene)

    # Post-hoc clinical trial penalty (not a 7th dimension — applied like safety veto)
    clinical_score, clinical_evidence = score_clinical(
        _section(raw_data, "clinical"),
        gene=hypothesis.target_gene,
    )
    all_evidence.extend(clinical_evidence)

    clinical_penalty_applied = False
    if clinical_score < 0.7:
        fitness *= clinical_score
        clinical_penalty_applied = True

    gwas_data = _section(raw_data, "gwas")
    _, gwas_evidence = score_gwas(gwas_data)
    all_evidence.extend(gwas_evidence)
    if has_strong_gwas_signal(gwas_data):
        fitness *= 1.1

    # LLM clinical feasibility — only boosts when NO clinical failure signal
    llm_feasibility = _get_llm_feasibility(raw_data)
    if llm_feasibility is not None and not clinical_penalty_applied:
        if llm_feasibility.get("has_approved_drug"):
            # Additional check: are the claimed drugs verified?
            verification = llm_feasibility.get("drug_verification", "no_reference")
            if verification == "unverified":
                # LLM claims drugs that don't match known database — don't trust
                pass  # skip boost
            else:
                # Verify against OpenTargets
                druggability_data = _section(raw_data, "druggability")
                known_drugs = druggability_data.get("known_drugs_count", 0)
                known_drugs = known_drugs if isinstance(known_drugs, int) else 0
                if known_drugs > 0:
                    # Both LLM and API agree — apply boost
                    fitness *= 1.05
        elif llm_feasibility.get("has_phase3_failure") and not llm_feasibility.get(
            "has_approved_drug"
        ):
            # Phase 3 failed, no approval -> additional penalty
            feasibility = llm_feasibility.get("feasibility_score", 0.5)
            if isinstance(feasibility, (int, float)) and feasibility < 0.3:
                fitness *= feasibility + 0.2  # soft penalty
    elif llm_feasibility is not None and clinical_penalty_applied:
        # LLM says approved but clinical trials show failures — trust clinical data
        # Apply additional LLM-confirmed failure penalty if applicable
        if llm_feasibility.get("has_phase3_failure") and not llm_feasibility.get(
            "has_approved_drug"
        ):
            feasibility = llm_feasibility.get("feasibility_score", 0.5)
            if isinstance(feasibility, (int, float)) and feasibility < 0.3:
                fitness *= feasibility + 0.2

    # Record API errors as evidence so they appear in reports/exports
    api_errors = raw_data.get("api_errors")
    if isinstance(api_errors, list):
        for error_msg in api_errors:
            if isinstance(error_msg, str):
                all_evidence.append(Evidence("api_error", "error", error_msg, 0.0))

    return ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=fitness,
        scores=scores,
        evidence=all_evidence,
    )


def _get_llm_feasibility(raw_data: Mapping[str, object]) -> Mapping[str, object] | None:
    """Extract LLM clinical feasibility data from raw_data, if present."""
    value = raw_data.get("llm_clinical")
    if isinstance(value, dict) and value:
        return value
    return None


def _section(raw_data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw_data.get(key, {})
    return value if isinstance(value, Mapping) else {}
