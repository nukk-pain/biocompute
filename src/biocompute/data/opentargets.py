from __future__ import annotations

from typing import cast

import httpx

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"

TARGET_QUERY = """
query TargetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    tractability {
      modality
      value
    }
    drugAndClinicalCandidates {
      count
    }
    safetyLiabilities {
      event
      biosamples {
        tissueLabel
      }
    }
  }
}
"""

SEARCH_QUERY = """
query SearchTarget($queryString: String!) {
  search(queryString: $queryString, entityNames: ["target"], page: {size: 1, index: 0}) {
    hits {
      id
      name
    }
  }
}
"""


async def resolve_gene_to_ensembl(
    client: httpx.AsyncClient,
    gene: str,
) -> str | None:
    try:
        resp = await client.post(
            OT_GRAPHQL,
            json={"query": SEARCH_QUERY, "variables": {"queryString": gene}},
            timeout=30,
        )
        if resp.is_error:
            return None
        data_payload = cast(object, resp.json())
    except (httpx.HTTPError, Exception):
        return None

    if not isinstance(data_payload, dict):
        return None

    data_dict = cast(dict[str, object], data_payload)

    data: object = data_dict.get("data")
    if not isinstance(data, dict):
        return None

    data_section = cast(dict[str, object], data)

    search: object = data_section.get("search")
    if not isinstance(search, dict):
        return None

    search_section = cast(dict[str, object], search)

    hits: object = search_section.get("hits")
    if isinstance(hits, list) and hits:
        first_hit: object = hits[0]
        if isinstance(first_hit, dict):
            first_hit_dict = cast(dict[str, object], first_hit)
            ensembl_id: object = first_hit_dict.get("id")
            if isinstance(ensembl_id, str):
                return ensembl_id

    return None


async def get_target_info(
    client: httpx.AsyncClient,
    gene: str,
) -> dict[str, object]:
    try:
        resp = await client.post(
            OT_GRAPHQL,
            json={"query": TARGET_QUERY, "variables": {"ensemblId": gene}},
            timeout=30,
        )
        if resp.is_error:
            raise httpx.HTTPStatusError(
                "Open Targets request failed",
                request=resp.request,
                response=resp,
            )
        data_payload = cast(object, resp.json())
    except (httpx.HTTPError, Exception):
        return {
            "gene": gene,
            "source": "opentargets",
            "error": "API call failed",
        }

    target: dict[str, object] = {}
    if isinstance(data_payload, dict):
        data_dict = cast(dict[str, object], data_payload)
        data: object = data_dict.get("data")
        if isinstance(data, dict):
            data_section = cast(dict[str, object], data)
            target_payload: object = data_section.get("target")
            if isinstance(target_payload, dict):
                target = cast(dict[str, object], target_payload)

    tractability_payload = target.get("tractability")
    tractability: list[object] = (
        tractability_payload if isinstance(tractability_payload, list) else []
    )

    known_drugs_count = 0
    known_drugs = target.get("drugAndClinicalCandidates")
    if isinstance(known_drugs, dict):
        count = known_drugs.get("count")
        if isinstance(count, int):
            known_drugs_count = count

    safety_liabilities_payload = target.get("safetyLiabilities")
    safety_liabilities: list[object] = (
        safety_liabilities_payload
        if isinstance(safety_liabilities_payload, list)
        else []
    )

    ensembl_id: object = target.get("id")

    return {
        "gene": gene,
        "ensembl_id": ensembl_id if isinstance(ensembl_id, str) else None,
        "tractability": tractability,
        "known_drugs_count": known_drugs_count,
        "safety_liabilities": safety_liabilities,
        "source": "opentargets",
    }
