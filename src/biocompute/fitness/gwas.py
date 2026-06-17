from __future__ import annotations

from collections.abc import Mapping

from biocompute.models import Evidence


def score_gwas(data: Mapping[str, object]) -> tuple[float, list[Evidence]]:
    hit_count = _as_int(data.get("hit_count"))
    max_score = _as_float(data.get("max_score"))
    if max_score == 0.0:
        max_score = max(_scores(data), default=0.0)

    if hit_count == 0 and max_score == 0.0:
        return 0.0, []

    score = max(max_score, _hit_count_score(hit_count))
    evidence = [
        Evidence(
            source_type=str(data.get("source", "opentargets_gwas")),
            source_id=f"gwas:{data.get('gene', 'unknown')}:{data.get('disease', 'unknown')}",
            summary=f"hits={hit_count}; max_score={max_score:.2f}; boost_candidate={score >= 0.7}",
            relevance_score=score,
        )
    ]
    return score, evidence


def has_strong_gwas_signal(data: Mapping[str, object]) -> bool:
    score, _ = score_gwas(data)
    return score >= 0.7


def _hit_count_score(hit_count: int) -> float:
    if hit_count >= 10:
        return 0.9
    if hit_count >= 4:
        return 0.6
    if hit_count >= 1:
        return 0.3
    return 0.0


def _scores(data: Mapping[str, object]) -> list[float]:
    payload = data.get("scores")
    if not isinstance(payload, list):
        return []

    scores: list[float] = []
    for value in payload:
        if isinstance(value, int | float):
            scores.append(float(value))
    return scores


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
