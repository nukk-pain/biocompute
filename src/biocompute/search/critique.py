from __future__ import annotations

import json

from biocompute.data.llm import parse_json_from_response, query_llm  # pyright: ignore[reportMissingImports]
from biocompute.models import DiseaseQuery, ScoredHypothesis

CRITIQUE_SYSTEM_PROMPT = """You are a skeptical drug development reviewer. Given a therapeutic 
target hypothesis, identify specific reasons why it could FAIL. Be concrete — cite biological 
mechanisms, clinical precedents, and practical risks. Do not be encouraging.

Return ONLY a JSON object:
{
  "critiques": [
    "Specific failure reason 1",
    "Specific failure reason 2",
    "Specific failure reason 3"
  ]
}"""


def build_critique_prompt(scored: ScoredHypothesis, query: DiseaseQuery) -> str:
    hypothesis = scored.hypothesis
    dimension_scores = {
        name: round(score, 2) for name, score in scored.scores.dimensions().items()
    }

    return "\n".join(
        [
            f"Disease: {query.name} — {query.description}",
            "\nHypothesis under review:",
            f"  Gene: {hypothesis.target_gene}, Modality: {hypothesis.modality}",
            (
                "  Delivery: "
                f"{hypothesis.delivery}, Duration: {hypothesis.duration}, Tissue: {hypothesis.tissue_context}"
            ),
            f"  Fitness: {scored.fitness:.2f}",
            f"  Dimension scores: {json.dumps(dimension_scores)}",
            "\nAs a skeptical critic, list 3-5 specific reasons this hypothesis could fail.",
        ]
    )


def parse_critique_response(response: str) -> list[str]:
    data = parse_json_from_response(response)
    if data is not None:
        if isinstance(data, dict):
            critiques = data.get("critiques", [])
            if isinstance(critiques, list):
                return [item for item in critiques if isinstance(item, str)]
            return []
        if isinstance(data, list):
            return [item for item in data if isinstance(item, str)]

    lines = [line.strip().lstrip("0123456789.-) ") for line in response.split("\n")]
    return [line for line in lines if len(line) > 0]


def critique_hypothesis(
    scored: ScoredHypothesis,
    query: DiseaseQuery,
    model: str = "sonnet",
) -> list[str]:
    prompt = build_critique_prompt(scored, query)
    response = query_llm(prompt, model=model, system_prompt=CRITIQUE_SYSTEM_PROMPT)
    return parse_critique_response(response)
