from __future__ import annotations

from typing import cast

import httpx

from biocompute.data.opentargets import OT_GRAPHQL

DISEASE_SEARCH_QUERY = """
query SearchDisease($queryString: String!) {
  search(queryString: $queryString, entityNames: ["disease"], page: {size: 1, index: 0}) {
    hits {
      id
      name
    }
  }
}
"""

GWAS_EVIDENCE_QUERY = """
query GwasEvidence($efoId: String!, $pageIndex: Int!, $pageSize: Int!) {
  disease(efoId: $efoId) {
    id
    associatedTargets(page: {index: $pageIndex, size: $pageSize}) {
      rows {
        target {
          id
          approvedSymbol
        }
        score
        evidences(datasourceIds: ["ot_genetics_portal"]) {
          rows {
            datasourceId
            score
          }
        }
      }
    }
  }
}
"""


async def get_gwas_evidence(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    disease_id, search_error = await _resolve_disease_id(client, disease)
    if search_error is not None:
        failed = _empty_payload(gene, disease)
        failed["error"] = "API call failed"
        return failed
    if disease_id is None:
        return _empty_payload(gene, disease)

    try:
        response = await client.post(
            OT_GRAPHQL,
            json={
                "query": GWAS_EVIDENCE_QUERY,
                "variables": {"efoId": disease_id, "pageIndex": 0, "pageSize": 200},
            },
            timeout=30,
        )
        if response.is_error:
            raise httpx.HTTPStatusError(
                "Open Targets GWAS request failed",
                request=response.request,
                response=response,
            )
        payload = cast(object, response.json())
    except (httpx.HTTPError, Exception):
        failed = _empty_payload(gene, disease)
        failed["error"] = "API call failed"
        return failed

    scores = _extract_gene_scores(payload, gene)
    result = _empty_payload(gene, disease)
    result["disease_id"] = disease_id
    result["scores"] = scores
    result["hit_count"] = len(scores)
    result["max_score"] = max(scores) if scores else 0.0
    return result


async def _resolve_disease_id(
    client: httpx.AsyncClient, disease: str
) -> tuple[str | None, str | None]:
    try:
        response = await client.post(
            OT_GRAPHQL,
            json={"query": DISEASE_SEARCH_QUERY, "variables": {"queryString": disease}},
            timeout=30,
        )
        if response.is_error:
            raise httpx.HTTPStatusError(
                "Open Targets disease search failed",
                request=response.request,
                response=response,
            )
        payload = cast(object, response.json())
    except (httpx.HTTPError, Exception):
        return None, "API call failed"

    if not isinstance(payload, dict):
        return None, None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None, None
    search = data.get("search")
    if not isinstance(search, dict):
        return None, None
    hits = search.get("hits")
    if not isinstance(hits, list) or not hits:
        return None, None
    first_hit = hits[0]
    if not isinstance(first_hit, dict):
        return None, None
    disease_id = first_hit.get("id")
    if isinstance(disease_id, str):
        return disease_id, None
    return None, None


def _extract_gene_scores(payload: object, gene: str) -> list[float]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    disease = data.get("disease")
    if not isinstance(disease, dict):
        return []
    associated_targets = disease.get("associatedTargets")
    if not isinstance(associated_targets, dict):
        return []
    rows = associated_targets.get("rows")
    if not isinstance(rows, list):
        return []

    gene_upper = gene.upper()
    for row in rows:
        if not isinstance(row, dict):
            continue
        target = row.get("target")
        if not isinstance(target, dict):
            continue
        approved_symbol = target.get("approvedSymbol")
        if (
            not isinstance(approved_symbol, str)
            or approved_symbol.upper() != gene_upper
        ):
            continue
        return _extract_evidence_scores(row.get("evidences"))
    return []


def _extract_evidence_scores(evidences: object) -> list[float]:
    if not isinstance(evidences, dict):
        return []
    rows = evidences.get("rows")
    if not isinstance(rows, list):
        return []

    scores: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        datasource_id = row.get("datasourceId")
        score = row.get("score")
        if datasource_id != "ot_genetics_portal":
            continue
        if isinstance(score, int | float):
            scores.append(float(score))
    return scores


def _empty_payload(gene: str, disease: str) -> dict[str, object]:
    return {
        "gene": gene,
        "disease": disease,
        "disease_id": None,
        "scores": [],
        "hit_count": 0,
        "max_score": 0.0,
        "source": "opentargets_gwas",
    }
