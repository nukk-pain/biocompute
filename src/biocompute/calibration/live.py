# pyright: reportMissingImports=false

"""Live calibration using real bio API data.

Evaluates each CalibrationEntry against actual bio APIs (PubMed, Semantic
Scholar, HPA, String DB, OpenTargets) and computes real FitnessScores.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict

from biocompute.calibration.ground_truth import CalibrationEntry
from biocompute.calibration.tune import (
    CalibrationEvaluation,
    _calibration_key,
    evaluate_calibration,
    tune_weights,
)
from biocompute.fitness.evaluator import evaluate_all_dimensions
from biocompute.models import (
    DiseaseQuery,
    FitnessScores,
    TherapeuticHypothesis,
    Weights,
)


class LiveCalibrationResult(TypedDict):
    scores_map: dict[tuple[str, str], FitnessScores]
    fitness_map: dict[tuple[str, str], float]  # actual fitness with clinical penalty
    evaluation: CalibrationEvaluation
    tuned_weights: Weights
    tuned_evaluation: CalibrationEvaluation
    skipped: list[tuple[str, str, str]]  # (target_gene, disease, error_msg)
    elapsed_seconds: float


def _entry_to_hypothesis(entry: CalibrationEntry) -> TherapeuticHypothesis:
    """Build a minimal TherapeuticHypothesis from a CalibrationEntry."""
    return TherapeuticHypothesis(
        target_gene=entry.target_gene,
        modality=entry.modality,
        delivery=entry.delivery,
        duration="chronic",
        tissue_context="",
    )


def _entry_to_query(entry: CalibrationEntry) -> DiseaseQuery:
    """Build a DiseaseQuery from a CalibrationEntry's disease field."""
    return DiseaseQuery(
        name=entry.disease,
        description=entry.disease,
        keywords=[entry.target_gene, entry.disease],
    )


async def _collect_single_entry(
    entry: CalibrationEntry,
) -> tuple[tuple[str, str], FitnessScores | None, str]:
    """Collect bio data and score a single calibration entry.

    Returns (key, scores_or_none, error_message).
    Runs inside an existing event loop but creates its own httpx client
    to respect rate limits via sequential processing in the caller.
    """
    # Lazy import to avoid startup cost
    import httpx as _httpx

    key = _calibration_key(entry)
    hypothesis = _entry_to_hypothesis(entry)
    query = _entry_to_query(entry)

    try:
        async with _httpx.AsyncClient() as client:
            from biocompute.engine import _collect_single_hypothesis

            raw_data = await _collect_single_hypothesis(client, hypothesis, query)

        scored = evaluate_all_dimensions(hypothesis, query, raw_data)
        return (key, scored.scores, scored.fitness, "")
    except Exception as exc:
        return (key, None, 0.0, str(exc))


async def _collect_all_entries_sequentially(
    entries: list[CalibrationEntry],
    on_progress: Any | None = None,
) -> tuple[dict[tuple[str, str], FitnessScores], list[tuple[str, str, str]]]:
    """Process entries one at a time to respect Semantic Scholar rate limits.

    Args:
        entries: Calibration entries to evaluate.
        on_progress: Optional callable(index, total, entry) for progress reporting.

    Returns:
        (scores_map, skipped_list)
    """
    scores_map: dict[tuple[str, str], FitnessScores] = {}
    fitness_map: dict[tuple[str, str], float] = {}
    skipped: list[tuple[str, str, str]] = []

    for i, entry in enumerate(entries):
        if on_progress is not None:
            on_progress(i, len(entries), entry)

        key, scores, fitness, error = await _collect_single_entry(entry)

        if scores is not None:
            scores_map[key] = scores
            fitness_map[key] = fitness
        else:
            skipped.append((entry.target_gene, entry.disease, error))

        # Brief pause between entries to respect API rate limits
        if i < len(entries) - 1:
            await asyncio.sleep(0.5)

    return scores_map, fitness_map, skipped


def evaluate_calibration_live(
    entries: list[CalibrationEntry],
    weights: Weights | None = None,
    on_progress: Any | None = None,
) -> LiveCalibrationResult:
    """Run live calibration against real bio APIs.

    For each CalibrationEntry, calls actual bio APIs to collect data,
    then evaluates fitness scores and computes separation metrics.

    Args:
        entries: Calibration entries to evaluate.
        weights: Starting weights (defaults to Weights()).
        on_progress: Optional callable(index, total, entry) for progress.

    Returns:
        LiveCalibrationResult with scores, evaluation, tuned weights,
        skipped entries, and elapsed time.
    """
    if weights is None:
        weights = Weights()

    start_time = time.monotonic()

    scores_map, fitness_map, skipped = asyncio.run(
        _collect_all_entries_sequentially(entries, on_progress=on_progress)
    )

    # Evaluate using actual fitness values (includes clinical penalty)
    # Override evaluate_calibration to use pre-computed fitness instead of recomputing
    success_scores: list[float] = []
    fail_scores: list[float] = []
    for entry in entries:
        key = _calibration_key(entry)
        if key not in fitness_map:
            continue
        fitness = fitness_map[key]
        if entry.outcome == "SUCCESS":
            success_scores.append(fitness)
        else:
            fail_scores.append(fitness)

    correct_pairs = sum(
        1 for s in success_scores for f in fail_scores if s > f
    )
    total_pairs = len(success_scores) * len(fail_scores)

    evaluation: CalibrationEvaluation = {
        "separation_score": correct_pairs / total_pairs if total_pairs else 0.0,
        "success_mean": sum(success_scores) / len(success_scores) if success_scores else 0.0,
        "fail_mean": sum(fail_scores) / len(fail_scores) if fail_scores else 0.0,
        "success_scores": success_scores,
        "fail_scores": fail_scores,
    }

    # Tune weights (still uses raw scores — tuning optimizes the 6-dim weights)
    tuned_weights = tune_weights(entries, scores_map, steps=100)
    tuned_evaluation = evaluate_calibration(entries, scores_map, tuned_weights)

    elapsed = time.monotonic() - start_time

    return {
        "scores_map": scores_map,
        "fitness_map": fitness_map,
        "evaluation": evaluation,
        "tuned_weights": tuned_weights,
        "tuned_evaluation": tuned_evaluation,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
    }
