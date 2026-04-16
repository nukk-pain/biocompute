from __future__ import annotations

from collections.abc import Mapping

from biocompute.models import Evidence


def score_ip(data: Mapping[str, object]) -> tuple[float, list[Evidence]]:
    freedom_estimate = data.get("freedom_estimate", 0.5)
    base_score = (
        float(freedom_estimate) if isinstance(freedom_estimate, int | float) else 0.5
    )
    source = str(data.get("source", "llm_estimate"))

    # Adjust based on competitive landscape (known drugs = existing patents)
    known_drugs = data.get("known_drugs_count", 0)
    known_drugs = known_drugs if isinstance(known_drugs, int) else 0

    if known_drugs > 10:
        # Crowded field — many existing drugs, many patents
        ip_adjustment = -0.3
    elif known_drugs > 5:
        ip_adjustment = -0.2
    elif known_drugs > 0:
        ip_adjustment = -0.1
    else:
        # Novel target — potentially wide IP freedom
        ip_adjustment = 0.1

    score = max(0.0, min(base_score + ip_adjustment, 1.0))

    evidence = [
        Evidence(
            source,
            "ip_estimate",
            f"IP freedom estimate: {score:.2f} (base={base_score:.2f}, drugs={known_drugs}, adj={ip_adjustment:+.1f})",
            score,
        )
    ]
    return score, evidence
