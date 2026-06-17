from __future__ import annotations

from collections.abc import Mapping
import math

from biocompute.models import Evidence


def score_pathway(data: Mapping[str, object]) -> tuple[float, list[Evidence]]:
    interaction_count = data.get("interaction_count", 0)
    count = interaction_count if isinstance(interaction_count, int) else 0
    if count <= 0:
        return 0.0, [Evidence("string", "none", "No interactions found", 0.0)]

    base_score = min(math.log1p(count) / math.log1p(200), 1.0)
    interactions_value = data.get("interactions", [])
    interactions = interactions_value if isinstance(interactions_value, list) else []
    high_confidence = sum(
        1
        for interaction in interactions
        if isinstance(interaction, dict)
        and isinstance(interaction.get("score"), int | float)
        and float(interaction.get("score", 0.0)) > 0.7
    )

    quality_factor = high_confidence / count if count > 0 else 0
    score = base_score * (0.5 + 0.5 * quality_factor)

    evidence = [
        Evidence(
            "string",
            f"interactions:{count}",
            f"{count} interaction partners ({high_confidence} high-confidence)",
            score,
        )
    ]
    return score, evidence
