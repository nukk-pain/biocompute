# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAny=false

import json
from unittest.mock import patch

from biocompute.models import (
    DiseaseQuery,
    FitnessScores,
    ScoredHypothesis,
    TherapeuticHypothesis,
)
from biocompute.search.seed import (
    build_seed_prompt,
    generate_seed_population,
    parse_seed_response,
)
from biocompute.search.select import select_survivors


def test_build_seed_prompt_includes_disease():
    query = DiseaseQuery(
        name="MPS",
        description="Chronic pain from nerve hyperinnervation",
        keywords=["scar", "fascia"],
    )
    prompt = build_seed_prompt(query, n=10)
    assert "MPS" in prompt
    assert "hyperinnervation" in prompt
    assert "10" in prompt


def test_parse_seed_response_valid_json():
    response = json.dumps(
        {
            "hypotheses": [
                {
                    "target_gene": "CXCL12",
                    "modality": "VHH",
                    "delivery": "local injection",
                    "duration": "single-dose",
                    "tissue_context": "scar tissue",
                    "rationale": "CXCL12 drives nociceptor sprouting",
                },
                {
                    "target_gene": "NGF",
                    "modality": "mAb",
                    "delivery": "systemic IV",
                    "duration": "chronic",
                    "tissue_context": "joint",
                    "rationale": "NGF promotes pain signaling",
                },
            ]
        }
    )
    hypotheses = parse_seed_response(response)
    assert len(hypotheses) == 2
    assert hypotheses[0].target_gene == "CXCL12"
    assert hypotheses[1].target_gene == "NGF"
    assert all(h.mutation_type == "seed" for h in hypotheses)
    assert all(h.generation == 0 for h in hypotheses)


def test_parse_seed_response_bad_json_returns_empty():
    hypotheses = parse_seed_response("This is not JSON at all")
    assert hypotheses == []


def test_generate_seed_population_caps_llm_output_to_requested_n():
    query = DiseaseQuery(
        name="Hypertrophic Scarring",
        description="Pathologic skin fibrosis after wound healing",
    )
    response = json.dumps(
        {
            "hypotheses": [
                {
                    "target_gene": "TGFB1",
                    "modality": "mRNA-decoy",
                    "delivery": "topical LNP",
                    "duration": "single-dose",
                    "tissue_context": "skin/scar tissue",
                    "rationale": "TGF-beta signaling driver",
                },
                {
                    "target_gene": "ACTA2",
                    "modality": "siRNA",
                    "delivery": "local injection",
                    "duration": "single-dose",
                    "tissue_context": "scar tissue",
                    "rationale": "myofibroblast marker",
                },
                {
                    "target_gene": "CCN2",
                    "modality": "mAb",
                    "delivery": "systemic IV",
                    "duration": "chronic",
                    "tissue_context": "skin",
                    "rationale": "fibrotic signaling mediator",
                },
            ]
        }
    )

    with patch("biocompute.search.seed.query_llm", return_value=response):
        hypotheses = generate_seed_population(query, n=1)

    assert len(hypotheses) == 1
    assert hypotheses[0].target_gene == "TGFB1"


def make_scored(gene: str, fitness: float) -> ScoredHypothesis:
    hypothesis = TherapeuticHypothesis(gene, "mAb", "systemic", "chronic", "systemic")
    return ScoredHypothesis(
        hypothesis=hypothesis,
        fitness=fitness,
        scores=FitnessScores(),
    )


def test_select_survivors_keeps_top_n():
    population = [make_scored(f"G{i}", i * 0.1) for i in range(10)]
    survivors = select_survivors(population, top_n=3, diverse_n=0)
    assert len(survivors) == 3
    assert survivors[0].hypothesis.target_gene == "G9"
    assert survivors[1].hypothesis.target_gene == "G8"
    assert survivors[2].hypothesis.target_gene == "G7"


def test_select_survivors_adds_diversity():
    population = [make_scored(f"G{i}", i * 0.1) for i in range(10)]
    survivors = select_survivors(population, top_n=3, diverse_n=2)
    assert len(survivors) == 5
    top_genes = {survivor.hypothesis.target_gene for survivor in survivors[:3]}
    assert "G9" in top_genes


def test_select_survivors_limits_per_gene():
    # 5 CXCR4 (high fitness), 3 CXCL12 (medium), 2 NGF (low)
    population = (
        [make_scored("CXCR4", 0.9 + i * 0.01) for i in range(5)]
        + [make_scored("CXCL12", 0.6 + i * 0.01) for i in range(3)]
        + [make_scored("NGF", 0.3 + i * 0.01) for i in range(2)]
    )
    survivors = select_survivors(population, top_n=6, diverse_n=2, max_per_gene=2)
    top = survivors[:6]
    top_genes = [s.hypothesis.target_gene for s in top]
    assert top_genes.count("CXCR4") <= 2
    assert top_genes.count("CXCL12") >= 1
    assert top_genes.count("NGF") >= 1


def test_select_survivors_diverse_prefers_unseen_genes():
    # 4 CXCR4 (high), 2 CXCL12 (medium), 4 NGF (low)
    population = (
        [make_scored("CXCR4", 0.9 + i * 0.01) for i in range(4)]
        + [make_scored("CXCL12", 0.6 + i * 0.01) for i in range(2)]
        + [make_scored("NGF", 0.3 + i * 0.01) for i in range(4)]
    )
    # top_n=4 with max_per_gene=2 → 2 CXCR4 + 2 CXCL12 in top
    # diverse should prefer NGF (unseen in top) over more CXCR4/CXCL12
    survivors = select_survivors(population, top_n=4, diverse_n=2, max_per_gene=2)
    diverse = survivors[4:]
    diverse_genes = {s.hypothesis.target_gene for s in diverse}
    assert "NGF" in diverse_genes


def test_select_survivors_small_population():
    population = [make_scored("A", 0.9), make_scored("B", 0.3)]
    survivors = select_survivors(population, top_n=5, diverse_n=3)
    assert len(survivors) == 2
