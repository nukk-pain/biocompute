"""Database-seeded initial candidates from OpenTargets.

Fetches known therapeutic targets for a disease from the OpenTargets
Platform API and converts them into TherapeuticHypothesis objects.
These supplement (not replace) LLM-generated seeds to improve coverage
of well-established targets.
"""

from __future__ import annotations

import asyncio

from biocompute.models import TherapeuticHypothesis

# Reuse the existing endpoint constant from the opentargets client
_OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"

_DISEASE_SEARCH_QUERY = """\
query SearchDisease($queryString: String!) {
  search(queryString: $queryString, entityNames: ["disease"], page: {size: 1, index: 0}) {
    hits { id name }
  }
}"""

_ASSOCIATED_TARGETS_QUERY = """\
query AssociatedTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    associatedTargets(page: {size: $size, index: 0}) {
      rows {
        target { approvedSymbol }
        score
      }
    }
  }
}"""


async def fetch_opentargets_targets(
    disease: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Fetch known therapeutic targets for a disease from OpenTargets.

    Returns a list of dicts with ``gene`` (str) and ``score`` (float) keys.
    Returns an empty list on any API or network error.
    """
    import httpx  # lazy import to keep CLI startup fast

    async with httpx.AsyncClient(timeout=15) as client:
        # Step 1: Resolve disease name to EFO ID
        try:
            resp = await client.post(
                _OT_GRAPHQL,
                json={
                    "query": _DISEASE_SEARCH_QUERY,
                    "variables": {"queryString": disease},
                },
            )
            if resp.is_error:
                return []
        except Exception:
            return []

        data = resp.json()
        hits = (
            data.get("data", {}).get("search", {}).get("hits", [])
        )
        if not isinstance(hits, list) or not hits:
            return []

        first_hit = hits[0]
        if not isinstance(first_hit, dict):
            return []

        disease_id = first_hit.get("id")
        if not isinstance(disease_id, str):
            return []

        # Step 2: Get associated targets for this disease
        try:
            resp2 = await client.post(
                _OT_GRAPHQL,
                json={
                    "query": _ASSOCIATED_TARGETS_QUERY,
                    "variables": {"efoId": disease_id, "size": limit},
                },
            )
            if resp2.is_error:
                return []
        except Exception:
            return []

        rows = (
            resp2.json()
            .get("data", {})
            .get("disease", {})
            .get("associatedTargets", {})
            .get("rows", [])
        )
        if not isinstance(rows, list):
            return []

        results: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            target = row.get("target")
            if not isinstance(target, dict):
                continue
            symbol = target.get("approvedSymbol")
            if not isinstance(symbol, str) or not symbol:
                continue
            score = row.get("score", 0.0)
            results.append({
                "gene": symbol,
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
            })

        return results


def db_seed_hypotheses(
    disease_name: str,
    n: int = 5,
) -> list[TherapeuticHypothesis]:
    """Generate seed hypotheses from OpenTargets disease-target associations.

    Returns up to ``n`` TherapeuticHypothesis objects with ``mutation_type="db_seed"``.
    Returns an empty list if the API call fails or returns no results.
    """
    try:
        targets = asyncio.run(fetch_opentargets_targets(disease_name, limit=n))
    except Exception:
        return []

    hypotheses: list[TherapeuticHypothesis] = []
    for target in targets:
        gene = target.get("gene")
        if not isinstance(gene, str) or not gene:
            continue
        h = TherapeuticHypothesis(
            target_gene=gene,
            modality="to be determined",
            delivery="to be determined",
            duration="chronic",
            tissue_context="",
            mutation_type="db_seed",
            generation=0,
        )
        hypotheses.append(h)

    return hypotheses
