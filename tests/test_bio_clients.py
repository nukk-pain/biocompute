# pyright: reportMissingImports=false

import httpx
import pytest
from unittest.mock import AsyncMock, patch

from biocompute.data.gtex import get_tissue_expression as get_gtex_tissue_expression
from biocompute.data.hpa import get_tissue_expression
from biocompute.data.opentargets import get_target_info
from biocompute.data.string_db import get_interaction_partners


@pytest.mark.asyncio
async def test_hpa_tissue_expression():
    # First call: search_download.php to resolve gene symbol -> Ensembl ID
    search_response = httpx.Response(
        200,
        json=[
            {"Gene": "CXCL12", "Ensembl": "ENSG00000107562"},
        ],
    )
    # Second call: per-gene JSON endpoint with tissue expression data
    gene_response = httpx.Response(
        200,
        json={
            "Gene": "CXCL12",
            "Ensembl": "ENSG00000107562",
            "RNA tissue specificity": "Low tissue specificity",
            "RNA tissue distribution": "Detected in all",
            "RNA tissue specific nTPM": None,
            "RNA tissue cell type enrichment": [
                "Kidney - Fibroblasts",
                "Liver - Hepatic stellate cells",
            ],
        },
    )
    mock_get = AsyncMock(side_effect=[search_response, gene_response])
    with patch(
        "biocompute.data.hpa.httpx.AsyncClient.get",
        mock_get,
    ):
        async with httpx.AsyncClient() as client:
            result = await get_tissue_expression(client, "CXCL12")

    assert isinstance(result["tissues"], list)
    assert result["gene"] == "CXCL12"
    assert len(result["tissues"]) == 2
    assert result["source"] == "hpa"


@pytest.mark.asyncio
async def test_gtex_tissue_expression_resolves_versioned_gencode_id() -> None:
    gene_lookup_response = httpx.Response(
        200,
        json={
            "data": [
                {
                    "geneSymbol": "CXCR4",
                    "gencodeId": "ENSG00000121966.10",
                }
            ]
        },
    )
    expression_response = httpx.Response(
        200,
        json={
            "data": [
                {
                    "geneSymbol": "CXCR4",
                    "tissueSiteDetailId": "SkeletalMuscle",
                    "median": 120.0,
                    "unit": "TPM",
                },
                {
                    "geneSymbol": "CXCR4",
                    "tissueSiteDetailId": "Liver",
                    "median": 12.5,
                    "unit": "TPM",
                },
                {
                    "geneSymbol": "CXCR4",
                    "tissueSiteDetailId": "BrainCortex",
                    "median": 0.2,
                    "unit": "TPM",
                },
            ]
        },
    )
    mock_get = AsyncMock(side_effect=[gene_lookup_response, expression_response])

    with patch("biocompute.data.gtex.httpx.AsyncClient.get", mock_get):
        async with httpx.AsyncClient() as client:
            result = await get_gtex_tissue_expression(client, "CXCR4")

    assert result["gene"] == "CXCR4"
    assert result["source"] == "gtex"
    assert result["gencode_id"] == "ENSG00000121966.10"
    assert result["tissues"] == [
        {"Tissue": "SkeletalMuscle", "Level": "High"},
        {"Tissue": "Liver", "Level": "Medium"},
        {"Tissue": "BrainCortex", "Level": "Low"},
    ]


@pytest.mark.asyncio
async def test_gtex_tissue_expression_returns_safe_default_on_lookup_failure() -> None:
    lookup_response = httpx.Response(200, json={"data": []})

    with patch(
        "biocompute.data.gtex.httpx.AsyncClient.get",
        new=AsyncMock(return_value=lookup_response),
    ):
        async with httpx.AsyncClient() as client:
            result = await get_gtex_tissue_expression(client, "CXCR4")

    assert result == {
        "gene": "CXCR4",
        "tissues": [],
        "source": "gtex",
        "error": "Could not resolve GTEx gencode ID",
    }


@pytest.mark.asyncio
async def test_string_interaction_partners():
    mock_response = httpx.Response(
        200,
        json=[
            {"preferredName_A": "CXCL12", "preferredName_B": "CXCR4", "score": 0.999},
            {"preferredName_A": "CXCL12", "preferredName_B": "CXCR7", "score": 0.95},
        ],
    )
    with patch(
        "biocompute.data.string_db.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        async with httpx.AsyncClient() as client:
            result = await get_interaction_partners(client, "CXCL12")

    assert result["gene"] == "CXCL12"
    assert result["interaction_count"] == 2
    assert result["source"] == "string"


@pytest.mark.asyncio
async def test_opentargets_target_info():
    mock_response = httpx.Response(
        200,
        json={
            "data": {
                "target": {
                    "id": "ENSG00000107562",
                    "approvedSymbol": "CXCL12",
                    "tractability": [
                        {"modality": "AB", "value": True},
                        {"modality": "SM", "value": False},
                    ],
                    "drugAndClinicalCandidates": {"count": 2},
                    "safetyLiabilities": [
                        {
                            "event": "immunosuppression",
                            "biosamples": [{"tissueLabel": "blood"}],
                        },
                    ],
                }
            }
        },
    )
    with patch(
        "biocompute.data.opentargets.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        async with httpx.AsyncClient() as client:
            result = await get_target_info(client, "CXCL12")

    assert result["gene"] == "CXCL12"
    assert result["source"] == "opentargets"
