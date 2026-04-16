# pyright: reportMissingImports=false

from __future__ import annotations

import random
from typing import TypedDict

from biocompute.calibration.ground_truth import CalibrationEntry
from biocompute.models import FitnessScores, Weights, compute_fitness


class CalibrationEvaluation(TypedDict):
    separation_score: float
    success_mean: float
    fail_mean: float
    success_scores: list[float]
    fail_scores: list[float]


def _calibration_key(entry: CalibrationEntry) -> tuple[str, str]:
    """Build a unique lookup key from target_gene and disease context."""
    return (entry.target_gene, entry.disease)


def evaluate_calibration(
    entries: list[CalibrationEntry],
    scores_map: dict[str, FitnessScores] | dict[tuple[str, str], FitnessScores],
    weights: Weights,
) -> CalibrationEvaluation:
    success_scores: list[float] = []
    fail_scores: list[float] = []

    for entry in entries:
        key = _calibration_key(entry)
        scores = scores_map.get(key) or scores_map.get(entry.target_gene)  # type: ignore[call-overload]
        if scores is None:
            continue

        fitness = compute_fitness(scores, weights, gene=entry.target_gene)
        if entry.outcome == "SUCCESS":
            success_scores.append(fitness)
        else:
            fail_scores.append(fitness)

    if not success_scores or not fail_scores:
        return {
            "separation_score": 0.0,
            "success_mean": 0.0,
            "fail_mean": 0.0,
            "success_scores": success_scores,
            "fail_scores": fail_scores,
        }

    correct_pairs = 0
    total_pairs = 0
    for success_score in success_scores:
        for fail_score in fail_scores:
            total_pairs += 1
            if success_score > fail_score:
                correct_pairs += 1

    separation_score = correct_pairs / total_pairs if total_pairs else 0.0

    return {
        "separation_score": separation_score,
        "success_mean": sum(success_scores) / len(success_scores),
        "fail_mean": sum(fail_scores) / len(fail_scores),
        "success_scores": success_scores,
        "fail_scores": fail_scores,
    }


def tune_weights(
    entries: list[CalibrationEntry],
    scores_map: dict[str, FitnessScores] | dict[tuple[str, str], FitnessScores],
    steps: int = 50,
) -> Weights:
    best_weights = Weights()
    best_separation = evaluate_calibration(entries, scores_map, best_weights)[
        "separation_score"
    ]

    dimension_names = best_weights.dimensions()
    rng = random.Random(0)

    for _ in range(steps):
        raw_dimensions = {
            "literature_strength": rng.uniform(0.05, 0.4),
            "expression_specificity": rng.uniform(0.05, 0.3),
            "pathway_centrality": rng.uniform(0.05, 0.3),
            "druggability": rng.uniform(0.05, 0.3),
            "safety_profile": rng.uniform(0.05, 0.3),
            "ip_freedom": rng.uniform(0.05, 0.2),
        }
        total_weight = sum(raw_dimensions[name] for name in dimension_names)
        normalized_dimensions = {
            name: raw_dimensions[name] / total_weight for name in dimension_names
        }

        candidate = Weights(
            **normalized_dimensions,
            safety_threshold=rng.uniform(0.1, 0.3),
        )
        candidate_separation = evaluate_calibration(entries, scores_map, candidate)[
            "separation_score"
        ]
        if candidate_separation > best_separation:
            best_weights = candidate
            best_separation = candidate_separation

    return best_weights
