from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest

from biocompute.fitness.prior_knowledge import assess_prior_knowledge
from biocompute.models import EvidenceMaturity


SAMPLE_ABSTRACTS = [
    {
        "pmid": "12345",
        "title": "Targeting SMAD3 in fibrotic scar tissue",
        "year": "2024",
        "abstract": "SMAD3 inhibition reduced fibroblast activation in vitro and improved collagen remodeling.",
    },
    {
        "pmid": "67890",
        "title": "Clinical experience with PCSK9 inhibition",
        "year": "2023",
        "abstract": "PCSK9 inhibitors have entered human clinical use with robust LDL lowering in patients.",
    },
]


def test_prior_knowledge_llm_prompt_includes_json_only_maturity_rules():
    mock_response: dict[str, object] = {
        "maturity": "L2",
        "known_facts": [],
        "attempted_approaches": [],
        "gaps": [],
        "key_papers": [],
        "summary": "stub",
    }

    with patch(
        "biocompute.data.llm.query_llm_json", return_value=mock_response
    ) as mock_query:
        _ = assess_prior_knowledge("SMAD3", "hypertrophic scar", SAMPLE_ABSTRACTS[:1])

    call_args = mock_query.call_args
    assert call_args is not None
    prompt = cast(str, call_args.args[0])

    assert "Return ONLY a JSON object" in prompt
    assert '"maturity": "L0|L1|L2|L3|L4|L5"' in prompt
    assert "L0: No usable papers or hypothesis-only evidence." in prompt
    assert (
        "L5: Clinical trial failed or human therapeutic effort clearly failed."
        in prompt
    )
    assert "Gene: SMAD3" in prompt
    assert "Disease: hypertrophic scar" in prompt
    assert mock_query.call_args.kwargs["model"] == "sonnet"


def test_prior_knowledge_llm_returns_normalized_prior_knowledge_from_valid_dict():
    mock_response = {
        "gene": "WRONGGENE",
        "disease": "Wrong Disease",
        "maturity": "L4",
        "known_facts": ["SMAD3 signaling is elevated in scar fibroblasts."],
        "attempted_approaches": [
            "Small-molecule inhibition reduced profibrotic signaling."
        ],
        "gaps": ["No approved scar-specific therapy has emerged."],
        "key_papers": ["PMID:12345"],
        "summary": "SMAD3 is biologically validated and remains therapeutically underexploited.",
    }

    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_prior_knowledge(
            "SMAD3", "hypertrophic scar", SAMPLE_ABSTRACTS[:1]
        )

    assert result.gene == "SMAD3"
    assert result.disease == "hypertrophic scar"
    assert result.maturity == EvidenceMaturity.L4_CLINICAL
    assert result.known_facts == ["SMAD3 signaling is elevated in scar fibroblasts."]
    assert result.attempted_approaches == [
        "Small-molecule inhibition reduced profibrotic signaling."
    ]
    assert result.gaps == ["No approved scar-specific therapy has emerged."]
    assert result.key_papers == ["PMID:12345"]
    assert (
        result.summary
        == "SMAD3 is biologically validated and remains therapeutically underexploited."
    )


def test_prior_knowledge_llm_returns_defaults_on_exception():
    with patch(
        "biocompute.data.llm.query_llm_json",
        side_effect=RuntimeError("Claude CLI failed"),
    ):
        result = assess_prior_knowledge(
            "APP", "alzheimer disease", SAMPLE_ABSTRACTS[:1]
        )

    assert result.gene == "APP"
    assert result.disease == "alzheimer disease"
    assert result.maturity == EvidenceMaturity.L0_HYPOTHESIS
    assert result.known_facts == []
    assert result.attempted_approaches == []
    assert result.gaps == []
    assert result.key_papers == []
    assert result.summary == "Prior knowledge assessment unavailable."


def test_prior_knowledge_llm_returns_defaults_on_non_dict_response():
    with patch("biocompute.data.llm.query_llm_json", return_value=None):
        result = assess_prior_knowledge(
            "APP", "alzheimer disease", SAMPLE_ABSTRACTS[:1]
        )

    assert result.maturity == EvidenceMaturity.L0_HYPOTHESIS
    assert result.summary == "Prior knowledge assessment unavailable."
    assert result.known_facts == []
    assert result.attempted_approaches == []
    assert result.gaps == []
    assert result.key_papers == []


def test_prior_knowledge_llm_returns_defaults_on_empty_abstracts():
    with patch("biocompute.data.llm.query_llm_json") as mock_query:
        result = assess_prior_knowledge(
            "NOVEL1",
            "rare disease",
            [{"pmid": "1", "title": "No abstract", "year": "2024", "abstract": "   "}],
        )

    mock_query.assert_not_called()
    assert result.gene == "NOVEL1"
    assert result.disease == "rare disease"
    assert result.maturity == EvidenceMaturity.L0_HYPOTHESIS
    assert (
        result.summary
        == "No usable PubMed abstracts available for prior-knowledge assessment."
    )


def test_prior_knowledge_llm_handles_contradictory_evidence():
    mock_response = {
        "maturity": "L4 / L2 mixed evidence",
        "known_facts": ["Cell studies show pathway suppression."],
        "attempted_approaches": "not-a-list",
        "gaps": ["Clinical observations are inconsistent across cohorts."],
        "key_papers": ["PMID:67890"],
        "summary": "Evidence is contradictory: cell data are supportive, but human findings are mixed.",
    }

    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_prior_knowledge("SMAD3", "hypertrophic scar", SAMPLE_ABSTRACTS)

    assert result.maturity == EvidenceMaturity.L4_CLINICAL
    assert result.attempted_approaches == []
    assert result.gaps == ["Clinical observations are inconsistent across cohorts."]
    assert "contradictory" in result.summary.lower()


def test_prior_knowledge_llm_normalizes_wrong_field_types_safely():
    mock_response = {
        "maturity": "unsupported",
        "known_facts": "not-a-list",
        "attempted_approaches": None,
        "gaps": {"bad": "shape"},
        "key_papers": True,
        "summary": 123,
    }

    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_prior_knowledge("GENE1", "disease x", SAMPLE_ABSTRACTS[:1])

    assert result.maturity == EvidenceMaturity.L0_HYPOTHESIS
    assert result.known_facts == []
    assert result.attempted_approaches == []
    assert result.gaps == []
    assert result.key_papers == []
    assert result.summary == "123"


@pytest.mark.parametrize(
    ("maturity_value", "expected"),
    [
        ("L0", EvidenceMaturity.L0_HYPOTHESIS),
        ("L2", EvidenceMaturity.L2_IN_VITRO),
        ("L4", EvidenceMaturity.L4_CLINICAL),
        ("L5", EvidenceMaturity.L5_CLINICAL_FAIL),
    ],
)
def test_prior_knowledge_maturity_map_explicit_levels(
    maturity_value: str, expected: EvidenceMaturity
):
    mock_response: dict[str, object] = {
        "maturity": maturity_value,
        "known_facts": [],
        "attempted_approaches": [],
        "gaps": [],
        "key_papers": [],
        "summary": "mapped",
    }
    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_prior_knowledge("GENE1", "disease x", SAMPLE_ABSTRACTS[:1])

    assert result.maturity == expected


def test_prior_knowledge_smad3_hypertrophic_scar_l4():
    mock_response = {
        "maturity": "L4",
        "known_facts": ["SMAD3 is elevated in hypertrophic scar tissue."],
        "attempted_approaches": ["Small-molecule inhibition."],
        "gaps": ["No approved therapy."],
        "key_papers": ["PMID:12345"],
        "summary": "SMAD3 is a validated target.",
    }
    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_prior_knowledge(
            "SMAD3", "hypertrophic scar", SAMPLE_ABSTRACTS[:1]
        )
    assert result.maturity == EvidenceMaturity.L4_CLINICAL


def test_prior_knowledge_app_alzheimer_l5():
    mock_response = {
        "maturity": "L5",
        "known_facts": ["APP mutations linked to Alzheimer's."],
        "attempted_approaches": ["Anti-amyloid antibodies failed in Phase 3."],
        "gaps": ["Need new targets."],
        "key_papers": ["PMID:99999"],
        "summary": "Clinical trials failed.",
    }
    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_prior_knowledge(
            "APP", "alzheimer disease", SAMPLE_ABSTRACTS[:1]
        )
    assert result.maturity == EvidenceMaturity.L5_CLINICAL_FAIL


def test_prior_knowledge_pcsk9_hyperlipidemia_clinical_maturity():
    mock_response = {
        "maturity": "CLINICAL",
        "known_facts": ["PCSK9 inhibition lowers LDL."],
        "attempted_approaches": ["Monoclonal antibodies."],
        "gaps": ["Cost."],
        "key_papers": ["PMID:67890"],
        "summary": "PCSK9 is a validated clinical target.",
    }
    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_prior_knowledge(
            "PCSK9", "hyperlipidemia", SAMPLE_ABSTRACTS[1:2]
        )
    assert result.maturity == EvidenceMaturity.L4_CLINICAL


def test_prior_knowledge_no_papers_l0_fallback():
    result = assess_prior_knowledge("GENE1", "disease x", [])
    assert result.maturity == EvidenceMaturity.L0_HYPOTHESIS
    assert "No usable PubMed abstracts" in result.summary
