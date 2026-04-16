from dataclasses import asdict
from typing import cast

from biocompute.models import (
    FitnessScores,
    ScoredHypothesis,
    StrategyPriorArt,
    TherapeuticHypothesis,
)


def test_strategy_prior_art_dataclass_creation():
    strategy = StrategyPriorArt(
        strategy="SMAD7 overexpression",
        disease_class="fibrosis",
        prior_studies=["AAV5-Smad7 in corneal fibrosis"],
        modality_status={"AAV": "in vivo confirmed", "mRNA-LNP": "not attempted"},
        our_differentiation=["transient expression", "repeat dosing flexibility"],
        key_papers=["PMID:28339457"],
        summary="AAV delivery has prior validation, while mRNA-LNP remains open.",
    )

    assert strategy.strategy == "SMAD7 overexpression"
    assert strategy.disease_class == "fibrosis"
    assert strategy.modality_status["AAV"] == "in vivo confirmed"


def test_scored_hypothesis_strategy_prior_art_asdict_round_trip():
    hypothesis = TherapeuticHypothesis(
        target_gene="SMAD3",
        modality="mRNA",
        delivery="LNP",
        duration="transient",
        tissue_context="scar",
    )
    strategy = StrategyPriorArt(
        strategy="SMAD7 overexpression",
        disease_class="fibrosis",
        prior_studies=["AAV5-Smad7 in corneal fibrosis"],
        modality_status={"AAV": "in vivo confirmed", "mRNA-LNP": "not attempted"},
        our_differentiation=["scar-targeted LNP", "non-viral redosing"],
        key_papers=["PMID:28339457"],
        summary="Prior art exists for AAV, but not for mRNA-LNP in scar tissue.",
    )

    scored = ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=0.77,
        scores=FitnessScores(druggability=0.4),
        strategy_prior_art=strategy,
    )

    payload = asdict(scored)
    hypothesis_payload = cast(dict[str, object], payload["hypothesis"])
    scores_payload = cast(dict[str, float], payload["scores"])
    critiques_payload = cast(list[str], payload["critiques"])
    api_errors_payload = cast(list[str], payload["api_errors"])
    strategy_payload = cast(dict[str, object], payload["strategy_prior_art"])

    assert payload["strategy_prior_art"] == {
        "strategy": "SMAD7 overexpression",
        "disease_class": "fibrosis",
        "prior_studies": ["AAV5-Smad7 in corneal fibrosis"],
        "modality_status": {"AAV": "in vivo confirmed", "mRNA-LNP": "not attempted"},
        "our_differentiation": ["scar-targeted LNP", "non-viral redosing"],
        "key_papers": ["PMID:28339457"],
        "summary": "Prior art exists for AAV, but not for mRNA-LNP in scar tissue.",
    }

    round_tripped = ScoredHypothesis(
        hypothesis=TherapeuticHypothesis(
            target_gene=cast(str, hypothesis_payload["target_gene"]),
            modality=cast(str, hypothesis_payload["modality"]),
            delivery=cast(str, hypothesis_payload["delivery"]),
            duration=cast(str, hypothesis_payload["duration"]),
            tissue_context=cast(str, hypothesis_payload["tissue_context"]),
            id=cast(str, hypothesis_payload["id"]),
            parent_id=cast(str | None, hypothesis_payload["parent_id"]),
            mutation_type=cast(str, hypothesis_payload["mutation_type"]),
            generation=cast(int, hypothesis_payload["generation"]),
        ),
        fitness=cast(float, payload["fitness"]),
        scores=FitnessScores(**scores_payload),
        evidence=[],
        critiques=critiques_payload,
        api_errors=api_errors_payload,
        strategy_prior_art=StrategyPriorArt(
            strategy=cast(str, strategy_payload["strategy"]),
            disease_class=cast(str, strategy_payload["disease_class"]),
            prior_studies=cast(list[str], strategy_payload["prior_studies"]),
            modality_status=cast(dict[str, str], strategy_payload["modality_status"]),
            our_differentiation=cast(
                list[str], strategy_payload["our_differentiation"]
            ),
            key_papers=cast(list[str], strategy_payload["key_papers"]),
            summary=cast(str, strategy_payload["summary"]),
        ),
    )

    assert round_tripped == scored


def test_scored_hypothesis_strategy_prior_art_defaults_to_none():
    scored = ScoredHypothesis(
        hypothesis=TherapeuticHypothesis(
            target_gene="SMAD3",
            modality="mRNA",
            delivery="LNP",
            duration="transient",
            tissue_context="scar",
        ),
        fitness=0.55,
        scores=FitnessScores(),
    )

    assert scored.strategy_prior_art is None
