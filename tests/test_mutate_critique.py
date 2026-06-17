# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAny=false

import json

from biocompute.models import (
    DiseaseQuery,
    FitnessScores,
    ScoredHypothesis,
    TherapeuticHypothesis,
)
from biocompute.search.critique import (
    build_critique_prompt,
    parse_critique_response,
)
from biocompute.search.mutate import build_mutate_prompt, parse_mutation_response


def make_parent() -> ScoredHypothesis:
    hypothesis = TherapeuticHypothesis(
        "CXCL12",
        "VHH",
        "local injection",
        "single-dose",
        "scar tissue",
        id="parent001",
        generation=0,
    )
    return ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=0.72,
        scores=FitnessScores(
            literature_strength=0.8,
            expression_specificity=0.6,
            pathway_centrality=0.5,
            druggability=0.7,
            safety_profile=0.9,
            ip_freedom=0.4,
        ),
        critiques=["IP freedom is low — CXCR4 antagonists are heavily patented"],
    )


def test_build_mutate_prompt_includes_parent_info() -> None:
    parent = make_parent()
    query = DiseaseQuery("MPS", "Chronic myofascial pain")

    prompt = build_mutate_prompt(parent, query)

    assert "CXCL12" in prompt
    assert "VHH" in prompt
    assert "IP freedom is low" in prompt
    assert "PATHWAY_NEIGHBOR" in prompt
    assert "MODALITY_SWITCH" in prompt
    assert "LATERAL_JUMP" in prompt


def test_parse_mutation_response() -> None:
    response = json.dumps(
        {
            "mutations": [
                {
                    "target_gene": "CXCR4",
                    "modality": "siRNA",
                    "delivery": "local injection",
                    "duration": "single-dose",
                    "tissue_context": "scar tissue",
                    "mutation_type": "pathway_neighbor",
                    "rationale": "Target the receptor instead of the ligand",
                },
                {
                    "target_gene": "CXCL12",
                    "modality": "small molecule",
                    "delivery": "topical LNP",
                    "duration": "acute",
                    "tissue_context": "scar tissue",
                    "mutation_type": "modality_switch",
                    "rationale": "Avoid VHH IP issues with small molecule",
                },
            ]
        }
    )

    children = parse_mutation_response(response, parent_id="parent001", generation=1)

    assert len(children) == 2
    assert children[0].target_gene == "CXCR4"
    assert children[0].parent_id == "parent001"
    assert children[0].generation == 1
    assert children[0].mutation_type == "pathway_neighbor"


def test_parse_mutation_response_bad_json() -> None:
    children = parse_mutation_response("not json", parent_id="x", generation=1)
    assert children == []


def test_build_critique_prompt_includes_hypothesis() -> None:
    parent = make_parent()
    query = DiseaseQuery("MPS", "Chronic myofascial pain")

    prompt = build_critique_prompt(parent, query)

    assert "CXCL12" in prompt
    assert "VHH" in prompt
    assert "skeptical" in prompt.lower() or "critic" in prompt.lower()


def test_parse_critique_response() -> None:
    response = json.dumps(
        {
            "critiques": [
                "CXCL12 is involved in stem cell homing — blocking it may impair tissue repair",
                "VHH delivery to deep fascial tissue is technically challenging",
                "No animal model has demonstrated this mechanism in fascial scar",
            ]
        }
    )

    critiques = parse_critique_response(response)

    assert len(critiques) == 3
    assert "stem cell" in critiques[0]


def test_parse_critique_response_plain_text() -> None:
    critiques = parse_critique_response("1. Risk one\n2. Risk two\n3. Risk three")
    assert len(critiques) >= 1
