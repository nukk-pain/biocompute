from __future__ import annotations

from biocompute.data.llm import parse_json_from_response, query_llm  # pyright: ignore[reportMissingImports]
from biocompute.models import DiseaseQuery, TherapeuticHypothesis

SEED_SYSTEM_PROMPT = """You are a drug target discovery expert. Given a disease description, 
propose therapeutic target hypotheses. Think broadly — include well-known targets AND 
unconventional ones from different pathways. For each hypothesis, specify the target gene, 
therapeutic modality, delivery route, treatment duration, and target tissue.

Return ONLY a JSON object with this exact structure:
{
  "hypotheses": [
    {
      "target_gene": "GENE_SYMBOL",
      "modality": "mAb|VHH|siRNA|mRNA-decoy|small molecule|antisense",
      "delivery": "systemic IV|local injection|topical LNP|oral|intrathecal",
      "duration": "chronic|acute|single-dose",
      "tissue_context": "specific tissue name",
      "rationale": "brief scientific rationale"
    }
  ]
}"""


def build_seed_prompt(query: DiseaseQuery, n: int = 30) -> str:
    parts = [
        f"Disease: {query.name}",
        f"Description: {query.description}",
    ]
    if query.keywords:
        parts.append(f"Keywords: {', '.join(query.keywords)}")
    if query.known_targets:
        parts.append(
            "Known targets (include but go beyond these): "
            f"{', '.join(query.known_targets)}"
        )
    if query.known_failures:
        parts.append(
            f"Failed approaches (learn from these): {', '.join(query.known_failures)}"
        )

    parts.append(f"\nPropose exactly {n} diverse therapeutic target hypotheses.")
    parts.append("Include targets from at least 5 different biological pathways.")
    parts.append(
        "Consider both conventional and unconventional modalities/delivery routes."
    )

    return "\n".join(parts)


def parse_seed_response(response: str) -> list[TherapeuticHypothesis]:
    data = parse_json_from_response(response)
    if data is None:
        return []

    raw_hypotheses: object = []
    if isinstance(data, dict):
        raw_hypotheses = data.get("hypotheses", [])
    elif isinstance(data, list):
        raw_hypotheses = data

    if not isinstance(raw_hypotheses, list):
        return []

    results: list[TherapeuticHypothesis] = []
    for item in raw_hypotheses:
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

        results.append(
            TherapeuticHypothesis(
                target_gene=gene,
                modality=modality if isinstance(modality, str) else "unknown",
                delivery=delivery if isinstance(delivery, str) else "unknown",
                duration=duration if isinstance(duration, str) else "unknown",
                tissue_context=(
                    tissue_context if isinstance(tissue_context, str) else "unknown"
                ),
                mutation_type="seed",
                generation=0,
            )
        )
    return results


def generate_seed_population(
    query: DiseaseQuery,
    n: int = 30,
    model: str = "haiku",
) -> list[TherapeuticHypothesis]:
    prompt = build_seed_prompt(query, n)
    response = query_llm(prompt, model=model, system_prompt=SEED_SYSTEM_PROMPT)
    return parse_seed_response(response)[:n]
