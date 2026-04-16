# pyright: reportMissingImports=false

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from biocompute.data.pubmed import _retry_get, _throttled_get, search_pubmed


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def _make_response(status_code: int, json_data: dict | None = None) -> httpx.Response:
    kwargs: dict = {
        "status_code": status_code,
        "request": httpx.Request("GET", ESEARCH_URL),
    }
    if json_data is not None:
        kwargs["json"] = json_data
    return httpx.Response(**kwargs)


@pytest.mark.asyncio
async def test_retry_get_succeeds_after_429():
    """429 then 200 — retry succeeds on the second attempt."""
    r429 = _make_response(429)
    r200 = _make_response(200, {"esearchresult": {"idlist": ["111"]}})
    mock_get = AsyncMock(side_effect=[r429, r200])

    with patch("biocompute.data.pubmed.httpx.AsyncClient.get", mock_get):
        async with httpx.AsyncClient() as client:
            resp = await _retry_get(client, ESEARCH_URL, {"db": "pubmed"})

    assert resp.status_code == 200
    assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_retry_get_raises_after_exhausted_retries():
    """Persistent 429 exhausts retries and raises HTTPStatusError."""
    r429 = _make_response(429)
    mock_get = AsyncMock(return_value=r429)

    with patch("biocompute.data.pubmed.httpx.AsyncClient.get", mock_get):
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError, match="retry exhausted"):
                await _retry_get(client, ESEARCH_URL, {"db": "pubmed"})

    assert mock_get.call_count == 3


@pytest.mark.asyncio
async def test_retry_get_retries_on_timeout():
    """TimeoutException triggers retry, then succeeds."""
    r200 = _make_response(200, {"esearchresult": {"idlist": []}})
    mock_get = AsyncMock(side_effect=[httpx.TimeoutException("timeout"), r200])

    with patch("biocompute.data.pubmed.httpx.AsyncClient.get", mock_get):
        async with httpx.AsyncClient() as client:
            resp = await _retry_get(client, ESEARCH_URL, {"db": "pubmed"})

    assert resp.status_code == 200
    assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_throttled_get_calls_retry():
    """_throttled_get delegates to _retry_get and returns the response."""
    r200 = _make_response(200, {"esearchresult": {"idlist": ["222"]}})
    mock_get = AsyncMock(return_value=r200)

    with patch("biocompute.data.pubmed.httpx.AsyncClient.get", mock_get):
        async with httpx.AsyncClient() as client:
            resp = await _throttled_get(client, ESEARCH_URL, {"db": "pubmed"})

    assert resp.status_code == 200
    assert resp.json()["esearchresult"]["idlist"] == ["222"]


@pytest.mark.asyncio
async def test_search_pubmed_uses_retry_on_429_then_succeeds():
    """End-to-end: search_pubmed recovers from a 429."""
    r429 = _make_response(429)
    r200 = _make_response(200, {"esearchresult": {"idlist": ["AAA", "BBB"]}})
    mock_get = AsyncMock(side_effect=[r429, r200])

    with patch("biocompute.data.pubmed.httpx.AsyncClient.get", mock_get):
        async with httpx.AsyncClient() as client:
            ids = await search_pubmed(client, "test query", max_results=5)

    assert ids == ["AAA", "BBB"]
    assert mock_get.call_count == 2
