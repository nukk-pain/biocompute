from __future__ import annotations

import asyncio
import math
import os
import time
from typing import NotRequired, TypedDict, cast

import httpx

_s2_lock = asyncio.Lock()
_s2_last_request: float = 0.0

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_headers() -> dict[str, str]:
    key = os.environ.get("S2_API_KEY", "")
    if key:
        return {"x-api-key": key}
    return {}


class SemanticScholarPaper(TypedDict):
    paperId: str
    title: str
    citationCount: int
    influentialCitationCount: int
    year: NotRequired[int]


def _normalize_citation_metric(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        return int(value)
    return 0


async def _rate_limited_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str | int] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    global _s2_last_request  # noqa: PLW0603
    max_retries = 3
    response: httpx.Response | None = None

    for attempt in range(max_retries):
        async with _s2_lock:
            elapsed = time.monotonic() - _s2_last_request
            if elapsed < 1.1:
                await asyncio.sleep(1.1 - elapsed)
            response = await client.get(url, params=params, headers=headers)
            _s2_last_request = time.monotonic()

        if response.status_code == 429:
            # Rate limited — exponential backoff
            wait = 2.0 * (attempt + 1)
            await asyncio.sleep(wait)
            continue

        return response

    # Return last response even if still 429 (let caller handle)
    assert response is not None  # always set after at least one iteration
    return response


async def search_papers(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 10,
) -> list[SemanticScholarPaper]:
    response = await _rate_limited_get(
        client,
        f"{S2_API_BASE}/paper/search",
        params={
            "query": query,
            "limit": limit,
            "fields": "paperId,title,citationCount,influentialCitationCount,year",
        },
        headers=_s2_headers(),
    )
    _ = response.raise_for_status()
    data = cast("dict[str, object]", response.json())
    return cast("list[SemanticScholarPaper]", data.get("data", []))


async def get_citation_count(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, str | int]:
    query = f"{gene} {disease} therapeutic target"
    papers = await search_papers(client, query, limit=20)
    total_citations = sum(
        _normalize_citation_metric(paper.get("citationCount")) for paper in papers
    )
    influential_citations = sum(
        _normalize_citation_metric(paper.get("influentialCitationCount"))
        for paper in papers
    )
    return {
        "gene": gene,
        "disease": disease,
        "paper_count": len(papers),
        "total_citations": total_citations,
        "influential_citations": influential_citations,
        "source": "semantic_scholar",
    }
