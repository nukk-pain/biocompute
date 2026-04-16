from __future__ import annotations

from biocompute.models import ScoredHypothesis


def select_survivors(
    population: list[ScoredHypothesis],
    top_n: int = 10,
    diverse_n: int = 5,
    max_per_gene: int = 2,
) -> list[ScoredHypothesis]:
    if len(population) <= top_n + diverse_n:
        return list(population)

    by_score = sorted(population, key=lambda scored: scored.fitness, reverse=True)

    # Phase 1: Gene-aware top selection (max_per_gene per gene)
    top: list[ScoredHypothesis] = []
    gene_counts: dict[str, int] = {}
    overflow: list[ScoredHypothesis] = []

    for scored in by_score:
        gene = scored.hypothesis.target_gene
        count = gene_counts.get(gene, 0)
        if count < max_per_gene and len(top) < top_n:
            top.append(scored)
            gene_counts[gene] = count + 1
        else:
            overflow.append(scored)

    # Fill remaining top slots from overflow if top_n not reached
    for scored in overflow:
        if len(top) >= top_n:
            break
        top.append(scored)

    # Phase 2: Diverse selection — prefer genes NOT in top
    top_ids = {id(s) for s in top}
    remaining = [s for s in by_score if id(s) not in top_ids]
    top_genes = {s.hypothesis.target_gene for s in top}

    # Prioritize unseen genes for diversity
    unseen = [s for s in remaining if s.hypothesis.target_gene not in top_genes]
    seen = [s for s in remaining if s.hypothesis.target_gene in top_genes]

    diverse_pool = unseen + seen
    diverse_count = min(diverse_n, len(diverse_pool))
    diverse = diverse_pool[:diverse_count]  # already sorted by fitness

    return top + diverse
