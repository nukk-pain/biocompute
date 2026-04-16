from __future__ import annotations

from typing import cast

import httpx

STRING_BASE = "https://string-db.org/api/json"


async def get_interaction_partners(
    client: httpx.AsyncClient,
    gene: str,
    species: int = 9606,
    limit: int = 200,
) -> dict[str, object]:
    try:
        resp = await client.get(
            f"{STRING_BASE}/interaction_partners",
            params={
                "identifiers": gene,
                "species": species,
                "limit": limit,
            },
            timeout=30,
        )
        if resp.is_error:
            raise httpx.HTTPStatusError(
                "STRING request failed",
                request=resp.request,
                response=resp,
            )
        interactions_payload = cast(object, resp.json())
    except (httpx.HTTPError, Exception):
        return {
            "gene": gene,
            "interactions": [],
            "interaction_count": 0,
            "source": "string",
            "error": "API call failed",
        }

    interactions: list[object] = (
        interactions_payload if isinstance(interactions_payload, list) else []
    )

    return {
        "gene": gene,
        "interactions": interactions,
        "interaction_count": len(interactions),
        "source": "string",
    }
