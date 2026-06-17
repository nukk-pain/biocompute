# pyright: reportMissingImports=false

from biocompute.calibration.ground_truth import CALIBRATION_SET, CalibrationEntry
from biocompute.calibration.tune import evaluate_calibration, tune_weights
from biocompute.models import FitnessScores, Weights


def test_calibration_set_has_successes_and_failures():
    successes = [entry for entry in CALIBRATION_SET if entry.outcome == "SUCCESS"]
    failures = [entry for entry in CALIBRATION_SET if entry.outcome == "FAIL"]
    assert len(successes) >= 30
    assert len(failures) >= 15


def test_evaluate_calibration_perfect_separation():
    entries = [
        CalibrationEntry("A", "mAb", "systemic", "disease", "SUCCESS"),
        CalibrationEntry("B", "mAb", "systemic", "disease", "FAIL"),
    ]
    scores_map = {
        "A": FitnessScores(
            literature_strength=0.9,
            safety_profile=0.9,
            druggability=0.8,
            expression_specificity=0.7,
            pathway_centrality=0.6,
            ip_freedom=0.5,
        ),
        "B": FitnessScores(
            literature_strength=0.2,
            safety_profile=0.3,
            druggability=0.1,
            expression_specificity=0.1,
            pathway_centrality=0.1,
            ip_freedom=0.1,
        ),
    }

    result = evaluate_calibration(entries, scores_map, Weights())

    assert result["separation_score"] == 1.0


def test_evaluate_calibration_no_separation():
    entries = [
        CalibrationEntry("A", "mAb", "systemic", "disease", "SUCCESS"),
        CalibrationEntry("B", "mAb", "systemic", "disease", "FAIL"),
    ]
    scores_map = {
        "A": FitnessScores(
            literature_strength=0.5,
            safety_profile=0.5,
            druggability=0.5,
            expression_specificity=0.5,
            pathway_centrality=0.5,
            ip_freedom=0.5,
        ),
        "B": FitnessScores(
            literature_strength=0.5,
            safety_profile=0.5,
            druggability=0.5,
            expression_specificity=0.5,
            pathway_centrality=0.5,
            ip_freedom=0.5,
        ),
    }

    result = evaluate_calibration(entries, scores_map, Weights())

    assert result["separation_score"] == 0.0


def test_evaluate_calibration_disambiguates_same_gene_different_disease():
    """VEGF appears as SUCCESS (cancer) and FAIL (heart failure). Tuple keys
    ensure they are scored separately (M4 fix)."""
    entries = [
        CalibrationEntry("VEGF", "mAb", "systemic", "cancer", "SUCCESS"),
        CalibrationEntry("VEGF", "mAb", "systemic", "heart failure", "FAIL"),
    ]
    scores_map = {
        ("VEGF", "cancer"): FitnessScores(
            literature_strength=0.9,
            safety_profile=0.9,
            druggability=0.8,
            expression_specificity=0.7,
            pathway_centrality=0.6,
            ip_freedom=0.5,
        ),
        ("VEGF", "heart failure"): FitnessScores(
            literature_strength=0.2,
            safety_profile=0.1,
            druggability=0.2,
            expression_specificity=0.1,
            pathway_centrality=0.1,
            ip_freedom=0.1,
        ),
    }

    result = evaluate_calibration(entries, scores_map, Weights())

    assert result["separation_score"] == 1.0
    assert len(result["success_scores"]) == 1
    assert len(result["fail_scores"]) == 1
    assert result["success_mean"] > result["fail_mean"]


def test_tune_weights_returns_normalized_dimensions():
    entries = [
        CalibrationEntry("A", "mAb", "systemic", "disease", "SUCCESS"),
        CalibrationEntry("B", "mAb", "systemic", "disease", "FAIL"),
    ]
    scores_map = {
        "A": FitnessScores(
            literature_strength=0.9,
            safety_profile=0.9,
            druggability=0.8,
            expression_specificity=0.7,
            pathway_centrality=0.6,
            ip_freedom=0.5,
        ),
        "B": FitnessScores(
            literature_strength=0.2,
            safety_profile=0.3,
            druggability=0.1,
            expression_specificity=0.1,
            pathway_centrality=0.1,
            ip_freedom=0.1,
        ),
    }

    weights = tune_weights(entries, scores_map, steps=5)
    total = sum(getattr(weights, dimension) for dimension in weights.dimensions())

    assert abs(total - 1.0) < 1e-9
