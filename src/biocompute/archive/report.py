from __future__ import annotations

from biocompute.models import DiscoveryResult, deduplicate_by_gene


def generate_report(result: DiscoveryResult, dedup: bool = True) -> str:
    query = result.query
    metadata = result.metadata

    lines = [
        "# BioCompute Discovery Report",
        "",
        "## Query",
        f"- **Disease:** {query.name}",
        f"- **Description:** {query.description}",
    ]
    if query.keywords:
        lines.append(f"- **Keywords:** {', '.join(query.keywords)}")
    lines.append("")

    lines.append("## Run Summary")
    lines.append(f"- **Generations:** {metadata.generations_run}")
    lines.append(f"- **Total hypotheses explored:** {metadata.total_hypotheses}")
    if metadata.started_at and metadata.finished_at:
        duration = metadata.finished_at - metadata.started_at
        lines.append(f"- **Duration:** {duration}")
    lines.append("")

    lines.append("## Top Candidates")
    lines.append("")
    display_candidates = (
        deduplicate_by_gene(result.candidates) if dedup else result.candidates
    )
    for index, scored in enumerate(display_candidates[:20], start=1):
        hypothesis = scored.hypothesis
        lines.append(
            f"### #{index}: {hypothesis.target_gene} (fitness: {scored.fitness:.3f})"
        )
        lines.append(f"- **Modality:** {hypothesis.modality}")
        lines.append(f"- **Delivery:** {hypothesis.delivery}")
        lines.append(f"- **Duration:** {hypothesis.duration}")
        lines.append(f"- **Tissue:** {hypothesis.tissue_context}")
        if hypothesis.parent_id:
            lines.append(
                f"- **Derived from:** {hypothesis.parent_id} ({hypothesis.mutation_type})"
            )
        lines.append(f"- **Generation:** {hypothesis.generation}")

        score_values = scored.scores.dimensions()
        if score_values:
            formatted_scores = ", ".join(
                f"{name}={value:.2f}" for name, value in score_values.items()
            )
            lines.append(f"- **Scores:** {formatted_scores}")

        if scored.evidence:
            lines.append("- **Evidence:**")
            for evidence in scored.evidence[:5]:
                lines.append(
                    f"  - [{evidence.source_type}] {evidence.source_id}: {evidence.summary}"
                )

        if scored.critiques:
            lines.append("- **Critiques:**")
            for critique in scored.critiques[:3]:
                lines.append(f"  - {critique}")

        if scored.prior_knowledge:
            pk = scored.prior_knowledge
            lines.append("- **Prior Knowledge:**")
            lines.append(f"  - **Maturity:** {pk.maturity.name}")
            lines.append(f"  - **Summary:** {pk.summary}")
            if pk.known_facts:
                lines.append(f"  - **Known Facts:** {', '.join(pk.known_facts)}")
            if pk.attempted_approaches:
                lines.append(
                    f"  - **Attempted Approaches:** {', '.join(pk.attempted_approaches)}"
                )
            if pk.gaps:
                lines.append(f"  - **Gaps:** {', '.join(pk.gaps)}")
        else:
            lines.append("- **Prior Knowledge:** *Unavailable*")

        if scored.strategy_prior_art:
            strategy_prior_art = scored.strategy_prior_art
            lines.append("- **Strategy Prior Art:**")
            lines.append(f"  - **Strategy:** {strategy_prior_art.strategy}")
            lines.append(f"  - **Disease Class:** {strategy_prior_art.disease_class}")
            lines.append(f"  - **Summary:** {strategy_prior_art.summary}")
            if strategy_prior_art.prior_studies:
                lines.append(
                    f"  - **Prior Studies:** {', '.join(strategy_prior_art.prior_studies)}"
                )
            if strategy_prior_art.modality_status:
                modality_entries = ", ".join(
                    f"{modality}: {status}"
                    for modality, status in strategy_prior_art.modality_status.items()
                )
                lines.append(f"  - **Modality Status:** {modality_entries}")
            if strategy_prior_art.our_differentiation:
                lines.append(
                    "  - **Our Differentiation:** "
                    + ", ".join(strategy_prior_art.our_differentiation)
                )
            if strategy_prior_art.key_papers:
                lines.append(
                    f"  - **Key Papers:** {', '.join(strategy_prior_art.key_papers)}"
                )
        lines.append("")

    return "\n".join(lines)
