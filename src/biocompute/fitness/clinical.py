"""Clinical trial outcome scoring.

Penalizes targets with failed clinical trials (especially Phase 2/3
failures) and gives a small bonus to targets with only successful
completions.  Applied as a post-hoc penalty on total fitness rather
than as a 7th fitness dimension.
"""

from __future__ import annotations

from collections.abc import Mapping

from biocompute.models import Evidence


def score_clinical(
    data: Mapping[str, object],
    gene: str = "",
) -> tuple[float, list[Evidence]]:
    """Score clinical trial outcomes for a gene+disease pair.

    Returns (score, evidence) where score is 0.0-1.0.
    - 1.0 = no trials found (neutral) or all completed successfully
    - Lower scores indicate clinical failure signals

    Scoring logic:
    - Base score = 1.0
    - Phase 2/3 failures: -0.15 each (capped at -0.5)
    - High failure ratio (>0.5 with >3 total trials): additional -0.2
    - All completed, no failures: +0.1 bonus (capped at 1.0)
    """
    completed_count = data.get("completed_count", 0)
    completed_count = completed_count if isinstance(completed_count, int) else 0

    failed_count = data.get("failed_count", 0)
    failed_count = failed_count if isinstance(failed_count, int) else 0

    phase3_failures = data.get("phase3_failures", 0)
    phase3_failures = phase3_failures if isinstance(phase3_failures, int) else 0

    failure_ratio = data.get("failure_ratio", 0.0)
    failure_ratio = (
        float(failure_ratio) if isinstance(failure_ratio, int | float) else 0.0
    )

    failed_names = data.get("failed_trial_names", [])
    failed_names = failed_names if isinstance(failed_names, list) else []

    total_trials = completed_count + failed_count

    # No clinical trial data — neutral score
    if total_trials == 0:
        return 1.0, []

    score = 1.0

    # Phase 2/3 failure penalty: -0.15 per failure, capped at -0.5
    if phase3_failures > 0:
        penalty = min(phase3_failures * 0.15, 0.5)
        score -= penalty

    # High failure ratio penalty
    if failure_ratio > 0.5 and total_trials > 3:
        score -= 0.2

    # Bonus for clean track record
    if completed_count > 0 and failed_count == 0:
        score += 0.1

    # Clamp to [0.0, 1.0]
    score = max(0.0, min(score, 1.0))

    # Build evidence
    summary_parts = [
        f"completed={completed_count}",
        f"failed={failed_count}",
        f"phase2/3_failures={phase3_failures}",
        f"ratio={failure_ratio:.2f}",
    ]
    if failed_names:
        names_str = ", ".join(str(n)[:50] for n in failed_names[:3])
        summary_parts.append(f"trials=[{names_str}]")

    evidence = [
        Evidence(
            source_type=str(data.get("source", "clinicaltrials_gov")),
            source_id=f"clinical:{gene}" if gene else "clinical:unknown",
            summary="; ".join(summary_parts),
            relevance_score=score,
        )
    ]

    return score, evidence
