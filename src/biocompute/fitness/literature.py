from __future__ import annotations

from collections.abc import Mapping
import math

from biocompute.models import Evidence


def score_literature(data: Mapping[str, object]) -> tuple[float, list[Evidence]]:
    pmid_count = data.get("pmid_count", 0)
    total_citations = data.get("total_citations", 0)
    influential = data.get("influential_citations", 0)

    total_citations_value = total_citations if isinstance(total_citations, int) else 0
    influential_value = influential if isinstance(influential, int) else 0
    pmid_count_value = pmid_count if isinstance(pmid_count, int) else 0

    if pmid_count_value <= 0 and total_citations_value <= 0:
        return 0.0, []

    pub_score = min(math.log1p(pmid_count_value) / math.log1p(50), 1.0)
    cite_bonus = (
        min(influential_value / 20, 1.0) * 0.3 if influential_value > 0 else 0.0
    )
    if pmid_count_value <= 0 and total_citations_value > 0:
        pub_score = min(math.log1p(total_citations_value) / math.log1p(500), 0.5)
    score = min(pub_score * 0.7 + cite_bonus, 1.0)

    # Negative evidence penalty
    negative_count = data.get("negative_count", 0)
    negative_count_value = int(negative_count) if isinstance(negative_count, int | float) else 0
    negative_penalty = 0.0
    if negative_count_value > 0 and pmid_count_value > 0:
        negative_ratio = negative_count_value / pmid_count_value
        if negative_ratio > 0.3:
            negative_penalty = 0.15
        elif negative_ratio > 0.1:
            negative_penalty = 0.05
    score = max(score - negative_penalty, 0.0)

    negative_suffix = ""
    if negative_count_value > 0:
        negative_suffix = f", {negative_count_value} negative (penalty -{negative_penalty:.2f})"

    evidence = [
        Evidence(
            source_type=str(data.get("source", "pubmed")),
            source_id=f"count:{pmid_count}",
            summary=(
                f"{pmid_count} publications, {total_citations_value} citations "
                f"({influential_value} influential){negative_suffix}"
            ),
            relevance_score=score,
        )
    ]
    return score, evidence
