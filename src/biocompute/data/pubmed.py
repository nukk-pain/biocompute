from __future__ import annotations

import asyncio
import time

import httpx

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

_pubmed_lock = asyncio.Lock()
_pubmed_last_request: float = 0.0


async def _retry_get(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    max_retries: int = 3,
) -> httpx.Response:
    """GET with retry on timeout / 5xx / 429 errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.get(url, params=params, timeout=15)
            if response.status_code == 429:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_exc = exc
            await asyncio.sleep(0.5 * (attempt + 1))
    raise last_exc or RuntimeError("retry exhausted")


async def _throttled_get(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    max_retries: int = 3,
) -> httpx.Response:
    """Rate-limited GET — ~3 req/s without API key."""
    global _pubmed_last_request  # noqa: PLW0603
    async with _pubmed_lock:
        elapsed = time.monotonic() - _pubmed_last_request
        if elapsed < 0.35:
            await asyncio.sleep(0.35 - elapsed)
        response = await _retry_get(client, url, params, max_retries)
        _pubmed_last_request = time.monotonic()
    return response


async def search_pubmed(
    client: httpx.AsyncClient,
    query: str,
    max_results: int = 20,
) -> list[str]:
    response = await _throttled_get(
        client,
        f"{EUTILS_BASE}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
        },
    )
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


async def fetch_abstract(
    client: httpx.AsyncClient,
    pmid: str,
) -> dict[str, str]:
    response = await _throttled_get(
        client,
        f"{EUTILS_BASE}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
        },
    )
    return {"pmid": pmid, "xml": response.text}


def _build_disease_query(disease: str) -> str:
    """Build PubMed disease query — phrase first, AND-joined keywords as fallback."""
    words = [w for w in disease.lower().split() if len(w) > 3]
    if len(words) <= 2:
        # Short disease names: use as exact phrase
        return f'"{disease}"'
    # Multi-word: use AND-joined keywords (more specific than OR)
    return "(" + " AND ".join(f'"{w}"' for w in words) + ")"


async def search_negative_evidence(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
    max_results: int = 20,
) -> dict[str, str | int | list[str]]:
    """Search PubMed for negative/failure evidence about a gene-disease pair."""
    disease_clause = _build_disease_query(disease)
    query = (
        f'"{gene}" AND {disease_clause} AND '
        f'(failed OR discontinued OR adverse OR withdrawn OR toxicity '
        f'OR "safety concern" OR "no benefit")'
    )
    pmids = await search_pubmed(client, query, max_results)
    return {
        "gene": gene,
        "disease": disease,
        "negative_count": len(pmids),
        "negative_pmids": pmids[:10],
        "source": "pubmed_negative",
    }


async def search_and_count(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
    max_results: int = 50,
) -> dict[str, str | int | list[str]]:
    disease_clause = _build_disease_query(disease)
    query = f'"{gene}" AND {disease_clause} AND (therapeutic OR target OR inhibitor)'
    pmids = await search_pubmed(client, query, max_results)
    if not pmids:
        fallback_query = f'"{gene}" AND {disease_clause}'
        pmids = await search_pubmed(client, fallback_query, max_results)
        query = fallback_query
    return {
        "gene": gene,
        "disease": disease,
        "query": query,
        "pmid_count": len(pmids),
        "pmids": pmids[:10],
        "source": "pubmed",
    }
