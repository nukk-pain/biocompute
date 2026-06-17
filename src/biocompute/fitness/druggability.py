from __future__ import annotations

from collections.abc import Mapping

from biocompute.models import Evidence


def score_druggability(data: Mapping[str, object]) -> tuple[float, list[Evidence]]:
    tractability_value = data.get("tractability", [])
    tractability = tractability_value if isinstance(tractability_value, list) else []
    known_drugs_value = data.get("known_drugs_count", 0)
    known_drugs = known_drugs_value if isinstance(known_drugs_value, int) else 0

    score = 0.0
    tractable_modalities = [
        str(item.get("modality"))
        for item in tractability
        if isinstance(item, dict) and item.get("value")
    ]
    if tractable_modalities:
        score += 0.4

    if known_drugs > 0:
        score += min(known_drugs / 5, 1.0) * 0.4

    if "AB" in tractable_modalities:
        score += 0.2

    score = min(score, 1.0)
    evidence = [
        Evidence(
            str(data.get("source", "opentargets")),
            f"drugs:{known_drugs}",
            (
                f"Tractable modalities: {tractable_modalities}, "
                f"{known_drugs} known drugs"
            ),
            score,
        )
    ]
    return score, evidence
