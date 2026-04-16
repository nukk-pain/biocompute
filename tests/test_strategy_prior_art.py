# pyright: reportMissingImports=false

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from biocompute.fitness.strategy_prior_art import (
    _generate_strategy_queries,
    _search_strategy_abstracts,
    assess_strategy_prior_art,
)


@patch(
    "biocompute.data.llm.query_llm_json",
    return_value=[
        "SMAD7 overexpression fibrosis",
        "SMAD7 gene therapy scar",
        "SMAD3 siRNA hypertrophic scar",
        "TGF-beta blockade fibrosis",
        "extra query should be dropped",
        "sixth query also dropped",
    ],
)
def test_generate_strategy_queries_caps_at_five_and_keeps_smad7_patterns(mock_query):
    result = _generate_strategy_queries("SMAD3", "hypertrophic scar", "mRNA-LNP")

    assert result == [
        "SMAD7 overexpression fibrosis",
        "SMAD7 gene therapy scar",
        "SMAD3 siRNA hypertrophic scar",
        "TGF-beta blockade fibrosis",
        "extra query should be dropped",
    ]

    prompt = mock_query.call_args.args[0]
    assert "up to 5 PubMed-style search queries" in prompt
    assert "Gene: SMAD3" in prompt
    assert "Disease: hypertrophic scar" in prompt
    assert "Modality of interest: mRNA-LNP" in prompt
    assert mock_query.call_args.kwargs["model"] == "haiku"


@pytest.mark.asyncio
async def test_search_strategy_abstracts_deduplicates_pmids_and_collects_abstracts():
    with (
        patch(
            "biocompute.fitness.strategy_prior_art.search_pubmed",
            new=AsyncMock(side_effect=[["111", "222"], ["222", "333"]]),
        ) as mock_search,
        patch(
            "biocompute.fitness.strategy_prior_art.fetch_abstracts",
            new=AsyncMock(
                side_effect=[
                    [
                        {
                            "pmid": "111",
                            "title": "AAV5-Smad7 improves corneal fibrosis",
                            "abstract": "AAV5-Smad7 reduced fibrotic signaling in vivo.",
                            "year": "2017",
                            "authors": ["Gupta A"],
                        },
                        {
                            "pmid": "222",
                            "title": "Duplicate paper",
                            "abstract": "Shared PMID across search queries.",
                            "year": "2016",
                            "authors": ["Lee B"],
                        },
                    ],
                    [
                        {
                            "pmid": "222",
                            "title": "Duplicate paper",
                            "abstract": "Shared PMID across search queries.",
                            "year": "2016",
                            "authors": ["Lee B"],
                        },
                        {
                            "pmid": "333",
                            "title": "Smad7 skin transgenic fibrosis study",
                            "abstract": "Smad7 overexpression limited dermal fibrosis.",
                            "year": "2011",
                            "authors": ["Kim C"],
                        },
                    ],
                ]
            ),
        ) as mock_fetch,
    ):
        async with httpx.AsyncClient() as client:
            result = await _search_strategy_abstracts(
                client,
                ["SMAD7 overexpression fibrosis", "SMAD7 gene therapy scar"],
            )

    assert [record["pmid"] for record in result] == ["111", "222", "333"]
    assert result[0]["title"] == "AAV5-Smad7 improves corneal fibrosis"
    assert "AAV5-Smad7" in result[0]["abstract"]
    assert mock_search.await_count == 2
    assert mock_fetch.await_count == 2


def test_assess_strategy_prior_art_returns_safe_defaults_on_llm_failure():
    abstracts = [
        {
            "pmid": "28339457",
            "title": "AAV5-Smad7 gene therapy for corneal fibrosis",
            "year": "2017",
            "abstract": "AAV5-Smad7 reduced fibrosis markers and improved wound healing in vivo.",
        }
    ]

    with patch(
        "biocompute.data.llm.query_llm_json",
        side_effect=RuntimeError("Claude CLI failed"),
    ):
        result = assess_strategy_prior_art(
            "SMAD3", "hypertrophic scar", "mRNA-LNP", abstracts
        )

    assert result["prior_studies"] == []
    assert result["modality_status"] == {}
    assert result["our_differentiation"] == []
    assert result["summary"] == "Strategy prior-art assessment unavailable."


def test_assess_strategy_prior_art_extracts_aav5_smad7_prior_art():
    abstracts = [
        {
            "pmid": "28339457",
            "title": "AAV5-Smad7 gene therapy for corneal fibrosis",
            "year": "2017",
            "abstract": "AAV5-Smad7 reduced fibrosis markers and improved wound healing in vivo.",
        }
    ]

    mock_response = {
        "prior_studies": [
            "PMID:28339457 AAV5-Smad7 showed in vivo anti-fibrotic activity."
        ],
        "modality_status": {
            "AAV": "in vivo confirmed for Smad7 overexpression anti-fibrotic strategy",
            "mRNA-LNP": "not described in the provided abstracts",
        },
        "our_differentiation": [
            "mRNA-LNP could test the same anti-fibrotic strategy without persistent viral expression."
        ],
        "summary": "The anti-fibrotic Smad7-overexpression strategy has AAV prior art but no mRNA-LNP evidence in the supplied abstracts.",
    }

    with patch("biocompute.data.llm.query_llm_json", return_value=mock_response):
        result = assess_strategy_prior_art(
            "SMAD3", "hypertrophic scar", "mRNA-LNP", abstracts
        )

    assert result["prior_studies"] == [
        "PMID:28339457 AAV5-Smad7 showed in vivo anti-fibrotic activity."
    ]
    assert (
        result["modality_status"]["AAV"]
        == "in vivo confirmed for Smad7 overexpression anti-fibrotic strategy"
    )
    assert (
        result["modality_status"]["mRNA-LNP"]
        == "not described in the provided abstracts"
    )
    assert result["our_differentiation"] == [
        "mRNA-LNP could test the same anti-fibrotic strategy without persistent viral expression."
    ]
