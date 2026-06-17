from __future__ import annotations

from biocompute.data.llm import parse_json_from_response, query_llm  # pyright: ignore[reportMissingImports]
from biocompute.models import DiseaseQuery, ScoredHypothesis, TherapeuticHypothesis

MUTATE_SYSTEM_PROMPT = """You are a drug target discovery expert generating new therapeutic 
hypotheses by modifying existing ones. You use three mutation strategies:

1. PATHWAY_NEIGHBOR: Same signaling pathway, different gene (upstream or downstream)
2. MODALITY_SWITCH: Same gene, different therapeutic modality or delivery route
3. LATERAL_JUMP: Different pathway entirely, but addressing the same disease mechanism

Return ONLY a JSON object:
{
  "mutations": [
    {
      "target_gene": "GENE",
      "modality": "...",
      "delivery": "...",
      "duration": "...",
      "tissue_context": "...",
      "mutation_type": "pathway_neighbor|modality_switch|lateral_jump",
      "rationale": "why this mutation addresses parent's weaknesses"
    }
  ]
}"""


def build_mutate_prompt(parent: ScoredHypothesis, query: DiseaseQuery) -> str:
    hypothesis = parent.hypothesis
    weakest_dimensions = sorted(
        parent.scores.dimensions().items(), key=lambda item: item[1]
    )[:2]

    parts = [
        f"Disease: {query.name} — {query.description}",
        "\nParent hypothesis:",
        f"  Gene: {hypothesis.target_gene}, Modality: {hypothesis.modality}",
        (
            "  Delivery: "
            f"{hypothesis.delivery}, Duration: {hypothesis.duration}, Tissue: {hypothesis.tissue_context}"
        ),
        f"  Fitness: {parent.fitness:.2f}",
        (
            "  Weakest dimensions: "
            + ", ".join(f"{name}={score:.2f}" for name, score in weakest_dimensions)
        ),
    ]
    if parent.critiques:
        parts.append(f"  Known risks: {'; '.join(parent.critiques[:3])}")

    parts.append(
        "\nGenerate 3 mutated hypotheses (one PATHWAY_NEIGHBOR, one MODALITY_SWITCH, one LATERAL_JUMP)."
    )
    parts.append("Each should address the parent's weaknesses or known risks.")

    return "\n".join(parts)


def parse_mutation_response(
    response: str,
    parent_id: str,
    generation: int,
) -> list[TherapeuticHypothesis]:
    data = parse_json_from_response(response)
    if data is None:
        return []

    raw_mutations: object = []
    if isinstance(data, dict):
        raw_mutations = data.get("mutations", [])
    elif isinstance(data, list):
        raw_mutations = data

    if not isinstance(raw_mutations, list):
        return []

    children: list[TherapeuticHypothesis] = []
    for item in raw_mutations:
        if not isinstance(item, dict):
            continue

        target_gene = item.get("target_gene", "")
        if not isinstance(target_gene, str):
            continue
        gene = target_gene.strip()
        if not gene:
            continue

        modality = item.get("modality", "unknown")
        delivery = item.get("delivery", "unknown")
        duration = item.get("duration", "unknown")
        tissue_context = item.get("tissue_context", "unknown")
        mutation_type = item.get("mutation_type", "unknown")

        children.append(
            TherapeuticHypothesis(
                target_gene=gene,
                modality=modality if isinstance(modality, str) else "unknown",
                delivery=delivery if isinstance(delivery, str) else "unknown",
                duration=duration if isinstance(duration, str) else "unknown",
                tissue_context=(
                    tissue_context if isinstance(tissue_context, str) else "unknown"
                ),
                parent_id=parent_id,
                generation=generation,
                mutation_type=mutation_type
                if isinstance(mutation_type, str)
                else "unknown",
            )
        )

    return children


def mutate_hypothesis(
    parent: ScoredHypothesis,
    query: DiseaseQuery,
    generation: int,
    model: str = "haiku",
) -> list[TherapeuticHypothesis]:
    prompt = build_mutate_prompt(parent, query)
    response = query_llm(prompt, model=model, system_prompt=MUTATE_SYSTEM_PROMPT)
    return parse_mutation_response(
        response,
        parent_id=parent.hypothesis.id,
        generation=generation,
    )
