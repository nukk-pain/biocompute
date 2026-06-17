from __future__ import annotations

from typing import cast

import httpx

GTEX_GENE_LOOKUP = "https://gtexportal.org/api/v2/reference/gene"
GTEX_MEDIAN_EXPRESSION = "https://gtexportal.org/api/v2/expression/medianGeneExpression"
GTEX_DATASET_ID = "gtex_v8"


def _tpm_to_level(tpm: float) -> str:
    """Map GTEx TPM values to HPA-style expression levels."""
    if tpm >= 100:
        return "High"
    if tpm >= 10:
        return "Medium"
    if tpm > 0:
        return "Low"
    return "Not detected"


async def _resolve_gencode_id(
    client: httpx.AsyncClient,
    gene: str,
) -> str | None:
    resp = await client.get(
        GTEX_GENE_LOOKUP,
        params={"geneId": gene, "itemsPerPage": 1, "page": 0},
        timeout=30,
    )
    if resp.is_error:
        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code}",
            request=resp.request,
            response=resp,
        )

    payload = cast(object, resp.json())
    if not isinstance(payload, dict):
        return None

    payload_dict = cast(dict[str, object], payload)
    data = payload_dict.get("data")
    if not isinstance(data, list):
        return None

    for entry in data:
        if not isinstance(entry, dict):
            continue
        entry_dict = cast(dict[str, object], entry)
        gencode_id = entry_dict.get("gencodeId")
        if isinstance(gencode_id, str) and gencode_id:
            return gencode_id

    return None


def _build_tissues(rows: list[object]) -> list[dict[str, str]]:
    tissues: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_dict = cast(dict[str, object], row)
        tissue = row_dict.get("tissueSiteDetailId")
        median = row_dict.get("median")
        if not isinstance(tissue, str):
            continue
        if not isinstance(median, int | float):
            continue
        tissues.append({"Tissue": tissue, "Level": _tpm_to_level(float(median))})
    return tissues


async def get_tissue_expression(
    client: httpx.AsyncClient,
    gene: str,
) -> dict[str, object]:
    try:
        gencode_id = await _resolve_gencode_id(client, gene)
        if not gencode_id:
            return {
                "gene": gene,
                "tissues": [],
                "source": "gtex",
                "error": "Could not resolve GTEx gencode ID",
            }

        resp = await client.get(
            GTEX_MEDIAN_EXPRESSION,
            params={"gencodeId": gencode_id, "datasetId": GTEX_DATASET_ID},
            timeout=30,
        )
        if resp.is_error:
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        payload = cast(object, resp.json())
    except (httpx.HTTPError, Exception):
        return {
            "gene": gene,
            "tissues": [],
            "source": "gtex",
            "error": "API call failed",
        }

    if not isinstance(payload, dict):
        return {"gene": gene, "tissues": [], "source": "gtex", "gencode_id": gencode_id}

    payload_dict = cast(dict[str, object], payload)
    data = payload_dict.get("data")
    rows = data if isinstance(data, list) else []

    return {
        "gene": gene,
        "gencode_id": gencode_id,
        "tissues": _build_tissues(rows),
        "source": "gtex",
    }
