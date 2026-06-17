"""Markdown report generation for literature verification results."""

from __future__ import annotations

from biocompute.models import EvidenceMaturity, PriorKnowledge
from biocompute.verification.literature import TargetVerification


def _format_prior_knowledge_list(items: list[str]) -> list[str]:
    return items if items else ["No stored items recorded."]


def _maturity_label(maturity: EvidenceMaturity) -> str:
    labels = {
        EvidenceMaturity.L0_HYPOTHESIS: "L0 hypothesis-stage signal",
        EvidenceMaturity.L1_ASSOCIATION: "L1 disease association",
        EvidenceMaturity.L2_IN_VITRO: "L2 in vitro support",
        EvidenceMaturity.L3_IN_VIVO: "L3 in vivo support",
        EvidenceMaturity.L4_CLINICAL: "L4 clinical evidence",
        EvidenceMaturity.L5_CLINICAL_FAIL: "L5 clinical failure signal",
    }
    return labels.get(maturity, "Unspecified evidence maturity")


def _append_prior_knowledge_section(
    lines: list[str], prior_knowledge: PriorKnowledge | None
) -> None:
    lines.append("**Prior Knowledge Framing:**")
    lines.append("")

    if prior_knowledge is None:
        lines.append(
            "- Stored prior knowledge was not available for this hypothesis. This verification therefore reports current literature and translational checks only, without inferring evidence maturity, prior attempted approaches, or remaining gaps beyond the retrieved sources."
        )
        lines.append("")
        return

    lines.append(f"- Evidence maturity: {_maturity_label(prior_knowledge.maturity)}")
    lines.append(f"- Framing summary: {prior_knowledge.summary}")
    lines.append("- Known facts:")
    for item in _format_prior_knowledge_list(prior_knowledge.known_facts):
        lines.append(f"  - {item}")
    lines.append("- Attempted approaches:")
    for item in _format_prior_knowledge_list(prior_knowledge.attempted_approaches):
        lines.append(f"  - {item}")
    lines.append("- Remaining gaps:")
    for item in _format_prior_knowledge_list(prior_knowledge.gaps):
        lines.append(f"  - {item}")
    lines.append("")


def generate_verification_report(verifications: list[TargetVerification]) -> str:
    """Generate a markdown report from verification results.

    Args:
        verifications: List of TargetVerification results.

    Returns:
        Markdown-formatted verification report string.
    """
    if not verifications:
        return "# Literature Verification Report\n\nNo targets to verify.\n"

    lines: list[str] = [
        "# Literature Verification Report",
        "",
        "## Summary",
        "",
        "| Gene | Fitness | Papers | Evidence |",
        "|------|---------|--------|----------|",
    ]

    total_citations = 0
    for v in verifications:
        target_citations = sum(p.citation_count for p in v.top_papers)
        total_citations += target_citations
        lines.append(
            f"| {v.gene} | {v.fitness:.3f} | {v.pubmed_count} | {v.evidence_strength} |"
        )

    lines.append("")

    # Overall assessment
    strengths = [v.evidence_strength for v in verifications]
    strong_count = strengths.count("Strong")
    moderate_count = strengths.count("Moderate")
    weak_count = strengths.count("Weak")
    no_ev_count = strengths.count("No evidence")

    lines.append("## Overall Assessment")
    lines.append("")
    lines.append(
        f"Of {len(verifications)} targets verified: "
        f"{strong_count} strong, {moderate_count} moderate, "
        f"{weak_count} weak, {no_ev_count} no evidence."
    )
    lines.append("")

    # Per-target details
    lines.append("## Target Details")
    lines.append("")

    for v in verifications:
        lines.append(f"### {v.gene}")
        lines.append("")
        lines.append(v.summary)
        lines.append("")

        _append_prior_knowledge_section(lines, v.prior_knowledge)

        top_3 = v.top_papers[:3]
        if top_3:
            lines.append("**Key Papers:**")
            lines.append("")
            for p in top_3:
                year_str = str(p.year) if p.year else "n/a"
                lines.append(f"- PMID:{p.pmid} ({year_str}) — {p.title}")
            lines.append("")
        else:
            lines.append("No papers found in PubMed for this target.")
            lines.append("")

        clinical = v.clinical_status
        clinical_summary = clinical.get(
            "status_summary", "No matching ClinicalTrials.gov studies found."
        )
        completed_count = clinical.get("completed_count", 0)
        failed_count = clinical.get("failed_count", 0)
        phase3_failures = clinical.get("phase3_failures", 0)
        failed_trial_names = clinical.get("failed_trial_names", [])

        lines.append("**Clinical Trial Status:**")
        lines.append("")
        lines.append(f"- Summary: {clinical_summary}")
        lines.append(f"- Completed studies: {completed_count}")
        lines.append(f"- Stopped studies: {failed_count}")
        lines.append(f"- Phase 2/3 failures: {phase3_failures}")
        if isinstance(failed_trial_names, list) and failed_trial_names:
            failed_titles = ", ".join(str(name) for name in failed_trial_names[:3])
            lines.append(f"- Example stopped trials: {failed_titles}")
        lines.append("")

        llm = v.llm_feasibility
        feasibility_score = llm.get("feasibility_score", 0.5)
        feasibility_label = llm.get("feasibility_label", "Moderate")
        rationale = llm.get("rationale", "LLM assessment unavailable")
        approved_drugs = llm.get("approved_drugs", [])
        failed_drugs = llm.get("failed_drugs", [])
        drug_verification = llm.get("drug_verification", "no_reference")

        approved_text = (
            ", ".join(str(drug) for drug in approved_drugs)
            if isinstance(approved_drugs, list) and approved_drugs
            else "None reported"
        )
        failed_text = (
            ", ".join(str(drug) for drug in failed_drugs)
            if isinstance(failed_drugs, list) and failed_drugs
            else "None reported"
        )

        score_text = (
            f"{float(feasibility_score):.3f} ({feasibility_label})"
            if isinstance(feasibility_score, (int, float))
            else f"0.500 ({feasibility_label})"
        )

        lines.append("**LLM Clinical Feasibility:**")
        lines.append("")
        lines.append(f"- Feasibility score: {score_text}")
        lines.append(f"- Approved drugs: {approved_text}")
        lines.append(f"- Phase 2/3 failed drugs: {failed_text}")
        lines.append(f"- Drug verification: {drug_verification}")
        lines.append(f"- Rationale: {rationale}")
        lines.append("")

        pathway_note = v.pathway_trial_note
        if pathway_note:
            lines.append("**Pathway-Level Trial Note:**")
            lines.append("")
            lines.append(f"- Pathway: {pathway_note.get('pathway', 'n/a')}")
            lines.append(
                f"- Summary: {pathway_note.get('summary', 'No pathway-level note available.')}"
            )
            examples = pathway_note.get("examples", [])
            if isinstance(examples, list):
                for example in examples:
                    lines.append(f"- {example}")
            lines.append(
                f"- Interpretation: {pathway_note.get('interpretation', 'n/a')}"
            )
            lines.append(f"- Source note: {pathway_note.get('source_note', 'n/a')}")
            lines.append("")

    return "\n".join(lines)
