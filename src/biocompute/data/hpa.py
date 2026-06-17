from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx

HPA_SEARCH = "https://www.proteinatlas.org/api/search_download.php"
HPA_GENE_JSON = "https://www.proteinatlas.org/{ensembl_id}.json"

MAX_RETRIES = 2


async def _retry_get(
    client: httpx.AsyncClient, url: str, **kwargs: Any
) -> httpx.Response:
    """GET with retry on timeout or HTTP error. Retries up to MAX_RETRIES times."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.get(url, **kwargs)
            if resp.is_error:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            return resp
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
    raise last_exc  # type: ignore[misc]


def _ntpm_to_level(ntpm: float) -> str:
    """Map normalized TPM value to HPA-style expression level."""
    if ntpm >= 400:
        return "High"
    if ntpm >= 50:
        return "Medium"
    if ntpm > 0:
        return "Low"
    return "Not detected"


def _build_tissues(gene_data: dict[str, object]) -> list[dict[str, str]]:
    """Extract tissue expression entries from the per-gene JSON payload."""
    tissues: list[dict[str, str]] = []
    seen: set[str] = set()

    # Primary source: RNA tissue specific nTPM (tissues where gene is enriched)
    ntpm_map = gene_data.get("RNA tissue specific nTPM")
    if isinstance(ntpm_map, dict):
        for tissue_name, value in ntpm_map.items():
            try:
                ntpm = float(value)
            except (TypeError, ValueError):
                continue
            level = _ntpm_to_level(ntpm)
            tissues.append({"Tissue": tissue_name, "Level": level})
            seen.add(tissue_name.lower())

    # Secondary source: RNA tissue cell type enrichment ("Tissue - CellType" entries)
    enrichment = gene_data.get("RNA tissue cell type enrichment")
    if isinstance(enrichment, list):
        for entry in enrichment:
            if not isinstance(entry, str) or " - " not in entry:
                continue
            tissue_name = entry.split(" - ", 1)[0].strip()
            key = tissue_name.lower()
            if key not in seen:
                tissues.append({"Tissue": tissue_name, "Level": "Medium"})
                seen.add(key)

    # Fallback: use overall specificity classification when no per-tissue data
    if not tissues:
        specificity = str(gene_data.get("RNA tissue specificity", ""))
        distribution = str(gene_data.get("RNA tissue distribution", ""))
        if "enriched" in specificity.lower():
            tissues.append({"Tissue": "general", "Level": "High"})
        elif "enhanced" in specificity.lower():
            tissues.append({"Tissue": "general", "Level": "Medium"})
        elif "detected in all" in distribution.lower():
            tissues.append({"Tissue": "general", "Level": "Low"})

    return tissues


async def _resolve_ensembl_id(
    client: httpx.AsyncClient, gene: str
) -> str | None:
    """Resolve a gene symbol to an Ensembl ID via the HPA search API."""
    resp = await _retry_get(
        client,
        HPA_SEARCH,
        params={
            "search": gene,
            "format": "json",
            "columns": "g,eg",
            "compress": "no",
        },
        timeout=15,
    )
    results = resp.json()
    if not isinstance(results, list):
        return None
    for entry in results:
        if isinstance(entry, dict) and entry.get("Gene") == gene:
            eid = entry.get("Ensembl")
            if isinstance(eid, str) and eid.startswith("ENSG"):
                return eid
    return None


async def get_tissue_expression(
    client: httpx.AsyncClient,
    gene: str,
) -> dict[str, object]:
    try:
        ensembl_id = await _resolve_ensembl_id(client, gene)
        if not ensembl_id:
            return {
                "gene": gene,
                "tissues": [],
                "source": "hpa",
                "error": "Could not resolve Ensembl ID",
            }

        resp = await _retry_get(
            client,
            HPA_GENE_JSON.format(ensembl_id=ensembl_id),
            timeout=30,
            follow_redirects=True,
        )
        gene_data = cast(object, resp.json())
    except (httpx.HTTPError, Exception):
        return {
            "gene": gene,
            "tissues": [],
            "source": "hpa",
            "error": "API call failed",
        }

    if not isinstance(gene_data, dict):
        return {"gene": gene, "tissues": [], "source": "hpa"}

    tissues = _build_tissues(gene_data)

    return {
        "gene": gene,
        "tissues": tissues,
        "source": "hpa",
    }
