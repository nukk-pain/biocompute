# pyright: reportMissingImports=false

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from biocompute.data.pubmed import search_negative_evidence
from biocompute.fitness.literature import score_literature


# ---------------------------------------------------------------------------
# search_negative_evidence tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client_with_negative_hits():
    """Return a mock httpx.AsyncClient that returns 5 PMIDs for negative search."""
    pmids = ["111", "222", "333", "444", "555"]
    response = MagicMock()
    response.json.return_value = {"esearchresult": {"idlist": pmids}}
    response.raise_for_status = MagicMock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = response
    return client


@pytest.fixture
def mock_client_with_no_hits():
    """Return a mock httpx.AsyncClient that returns 0 PMIDs."""
    response = MagicMock()
    response.json.return_value = {"esearchresult": {"idlist": []}}
    response.raise_for_status = MagicMock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = response
    return client


@pytest.mark.asyncio
async def test_search_negative_evidence_returns_count(mock_client_with_negative_hits):
    result = await search_negative_evidence(
        mock_client_with_negative_hits, "NGF", "osteoarthritis"
    )
    assert result["negative_count"] == 5
    assert result["source"] == "pubmed_negative"
    assert result["gene"] == "NGF"
    assert result["disease"] == "osteoarthritis"
    assert len(result["negative_pmids"]) == 5


@pytest.mark.asyncio
async def test_search_negative_evidence_no_hits(mock_client_with_no_hits):
    result = await search_negative_evidence(
        mock_client_with_no_hits, "BRCA1", "breast cancer"
    )
    assert result["negative_count"] == 0
    assert result["negative_pmids"] == []


@pytest.mark.asyncio
async def test_search_negative_evidence_query_includes_failure_terms(
    mock_client_with_negative_hits,
):
    await search_negative_evidence(
        mock_client_with_negative_hits, "NGF", "osteoarthritis"
    )
    call_args = mock_client_with_negative_hits.get.call_args
    query_param = call_args.kwargs.get("params", call_args[1].get("params", {}))
    term = query_param["term"]
    assert "failed" in term
    assert "discontinued" in term
    assert "toxicity" in term


# ---------------------------------------------------------------------------
# score_literature with negative evidence tests
# ---------------------------------------------------------------------------


def test_literature_score_no_negative_unchanged():
    """Without negative_count, score should be the same as before."""
    data = {"pmid_count": 30, "total_citations": 200, "influential_citations": 10}
    score, evidence = score_literature(data)
    assert score > 0.0
    # No negative suffix in summary
    assert "negative" not in evidence[0].summary


def test_literature_score_low_negative_ratio_no_penalty():
    """negative_ratio <= 0.1 should not penalize."""
    data = {
        "pmid_count": 30,
        "total_citations": 200,
        "influential_citations": 10,
        "negative_count": 2,  # 2/30 = 0.067 < 0.1
    }
    score_with_neg, _ = score_literature(data)

    data_no_neg = {
        "pmid_count": 30,
        "total_citations": 200,
        "influential_citations": 10,
    }
    score_without_neg, _ = score_literature(data_no_neg)

    assert score_with_neg == score_without_neg


def test_literature_score_moderate_negative_ratio_small_penalty():
    """negative_ratio > 0.1 and <= 0.3 should apply -0.05 penalty."""
    data = {
        "pmid_count": 30,
        "total_citations": 200,
        "influential_citations": 10,
        "negative_count": 5,  # 5/30 = 0.167 > 0.1
    }
    score_with_neg, evidence = score_literature(data)

    data_no_neg = {
        "pmid_count": 30,
        "total_citations": 200,
        "influential_citations": 10,
    }
    score_without_neg, _ = score_literature(data_no_neg)

    assert score_with_neg == pytest.approx(score_without_neg - 0.05, abs=1e-6)
    assert "negative" in evidence[0].summary
    assert "penalty -0.05" in evidence[0].summary


def test_literature_score_high_negative_ratio_heavy_penalty():
    """negative_ratio > 0.3 should apply -0.15 penalty."""
    data = {
        "pmid_count": 10,
        "total_citations": 100,
        "influential_citations": 5,
        "negative_count": 5,  # 5/10 = 0.5 > 0.3
    }
    score_with_neg, evidence = score_literature(data)

    data_no_neg = {
        "pmid_count": 10,
        "total_citations": 100,
        "influential_citations": 5,
    }
    score_without_neg, _ = score_literature(data_no_neg)

    assert score_with_neg == pytest.approx(score_without_neg - 0.15, abs=1e-6)
    assert "penalty -0.15" in evidence[0].summary


def test_literature_score_negative_penalty_does_not_go_below_zero():
    """Score should never go below 0.0 even with heavy penalty."""
    data = {
        "pmid_count": 1,
        "total_citations": 0,
        "influential_citations": 0,
        "negative_count": 1,  # 1/1 = 1.0 > 0.3 -> penalty 0.15
    }
    score, _ = score_literature(data)
    assert score >= 0.0


def test_literature_score_negative_with_zero_pmids_no_penalty():
    """If pmid_count is 0, negative_count should not cause division by zero."""
    data = {
        "pmid_count": 0,
        "total_citations": 50,
        "influential_citations": 0,
        "negative_count": 3,
    }
    score, _ = score_literature(data)
    # Should still produce a score from citation fallback, no crash
    assert score >= 0.0
