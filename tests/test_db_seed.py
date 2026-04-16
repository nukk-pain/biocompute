# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from biocompute.models import TherapeuticHypothesis
from biocompute.search.db_seed import db_seed_hypotheses, fetch_opentargets_targets


# --- fetch_opentargets_targets tests ---


@pytest.mark.asyncio
async def test_fetch_opentargets_targets_success() -> None:
    """Returns gene/score dicts when API responds correctly."""
    mock_search_resp = MagicMock()
    mock_search_resp.is_error = False
    mock_search_resp.json.return_value = {
        "data": {
            "search": {"hits": [{"id": "EFO_0000305", "name": "breast carcinoma"}]}
        }
    }

    mock_targets_resp = MagicMock()
    mock_targets_resp.is_error = False
    mock_targets_resp.json.return_value = {
        "data": {
            "disease": {
                "associatedTargets": {
                    "rows": [
                        {"target": {"approvedSymbol": "ERBB2"}, "score": 0.95},
                        {"target": {"approvedSymbol": "ESR1"}, "score": 0.88},
                        {"target": {"approvedSymbol": "EGFR"}, "score": 0.82},
                    ]
                }
            }
        }
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[mock_search_resp, mock_targets_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await fetch_opentargets_targets("breast cancer", limit=3)

    assert len(results) == 3
    assert results[0]["gene"] == "ERBB2"
    assert results[0]["score"] == 0.95
    assert results[1]["gene"] == "ESR1"
    assert results[2]["gene"] == "EGFR"


@pytest.mark.asyncio
async def test_fetch_opentargets_targets_disease_not_found() -> None:
    """Returns empty list when disease search returns no hits."""
    mock_resp = MagicMock()
    mock_resp.is_error = False
    mock_resp.json.return_value = {"data": {"search": {"hits": []}}}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await fetch_opentargets_targets("nonexistent disease xyz")

    assert results == []


@pytest.mark.asyncio
async def test_fetch_opentargets_targets_api_error() -> None:
    """Returns empty list when first API call returns an error."""
    mock_resp = MagicMock()
    mock_resp.is_error = True

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await fetch_opentargets_targets("breast cancer")

    assert results == []


@pytest.mark.asyncio
async def test_fetch_opentargets_targets_network_exception() -> None:
    """Returns empty list on network failure."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=ConnectionError("network down"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await fetch_opentargets_targets("breast cancer")

    assert results == []


@pytest.mark.asyncio
async def test_fetch_opentargets_targets_second_call_error() -> None:
    """Returns empty list when targets query returns an error."""
    mock_search_resp = MagicMock()
    mock_search_resp.is_error = False
    mock_search_resp.json.return_value = {
        "data": {
            "search": {"hits": [{"id": "EFO_0000305", "name": "breast carcinoma"}]}
        }
    }

    mock_targets_resp = MagicMock()
    mock_targets_resp.is_error = True

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[mock_search_resp, mock_targets_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await fetch_opentargets_targets("breast cancer")

    assert results == []


@pytest.mark.asyncio
async def test_fetch_opentargets_targets_malformed_rows() -> None:
    """Skips rows with missing or malformed target data."""
    mock_search_resp = MagicMock()
    mock_search_resp.is_error = False
    mock_search_resp.json.return_value = {
        "data": {
            "search": {"hits": [{"id": "EFO_0000305", "name": "breast carcinoma"}]}
        }
    }

    mock_targets_resp = MagicMock()
    mock_targets_resp.is_error = False
    mock_targets_resp.json.return_value = {
        "data": {
            "disease": {
                "associatedTargets": {
                    "rows": [
                        {"target": {"approvedSymbol": "ERBB2"}, "score": 0.95},
                        {"target": {}, "score": 0.88},  # missing approvedSymbol
                        {"score": 0.82},  # missing target entirely
                        "not a dict",  # not a dict
                        {
                            "target": {"approvedSymbol": ""},
                            "score": 0.7,
                        },  # empty symbol
                        {"target": {"approvedSymbol": "TP53"}, "score": 0.6},
                    ]
                }
            }
        }
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[mock_search_resp, mock_targets_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        results = await fetch_opentargets_targets("breast cancer", limit=10)

    assert len(results) == 2
    assert results[0]["gene"] == "ERBB2"
    assert results[1]["gene"] == "TP53"


# --- db_seed_hypotheses tests ---


def test_db_seed_hypotheses_returns_therapeutic_hypotheses() -> None:
    """Converts OpenTargets results into TherapeuticHypothesis objects."""
    mock_targets = [
        {"gene": "ERBB2", "score": 0.95},
        {"gene": "ESR1", "score": 0.88},
    ]
    with patch(
        "biocompute.search.db_seed.fetch_opentargets_targets",
        new=AsyncMock(return_value=mock_targets),
    ):
        results = db_seed_hypotheses("breast cancer", n=5)

    assert len(results) == 2
    assert all(isinstance(h, TherapeuticHypothesis) for h in results)
    assert results[0].target_gene == "ERBB2"
    assert results[1].target_gene == "ESR1"
    assert all(h.mutation_type == "db_seed" for h in results)
    assert all(h.generation == 0 for h in results)


def test_db_seed_hypotheses_empty_on_failure() -> None:
    """Returns empty list when API call raises an exception."""
    with patch(
        "biocompute.search.db_seed.fetch_opentargets_targets",
        new=AsyncMock(side_effect=RuntimeError("network error")),
    ):
        results = db_seed_hypotheses("breast cancer", n=5)

    assert results == []


def test_db_seed_hypotheses_empty_api_response() -> None:
    """Returns empty list when API returns no targets."""
    with patch(
        "biocompute.search.db_seed.fetch_opentargets_targets",
        new=AsyncMock(return_value=[]),
    ):
        results = db_seed_hypotheses("unknown disease xyz", n=5)

    assert results == []


# --- Engine merge logic tests ---


def test_engine_db_seed_merge_no_duplicates() -> None:
    """DB seeds are merged into LLM seeds without duplicating genes."""
    llm_seeds = [
        TherapeuticHypothesis(
            target_gene="ERBB2",
            modality="mAb",
            delivery="IV",
            duration="chronic",
            tissue_context="breast",
            mutation_type="seed",
        ),
        TherapeuticHypothesis(
            target_gene="BRCA1",
            modality="siRNA",
            delivery="LNP",
            duration="chronic",
            tissue_context="breast",
            mutation_type="seed",
        ),
    ]

    db_seeds = [
        TherapeuticHypothesis(
            target_gene="ERBB2",
            modality="to be determined",
            delivery="to be determined",
            duration="chronic",
            tissue_context="",
            mutation_type="db_seed",
        ),
        TherapeuticHypothesis(
            target_gene="ESR1",
            modality="to be determined",
            delivery="to be determined",
            duration="chronic",
            tissue_context="",
            mutation_type="db_seed",
        ),
        TherapeuticHypothesis(
            target_gene="TP53",
            modality="to be determined",
            delivery="to be determined",
            duration="chronic",
            tissue_context="",
            mutation_type="db_seed",
        ),
    ]

    # Simulate the merge logic from engine.py
    seed_population = list(llm_seeds)
    existing_genes = {h.target_gene for h in seed_population}
    for h in db_seeds:
        if h.target_gene not in existing_genes:
            seed_population.append(h)
            existing_genes.add(h.target_gene)

    assert len(seed_population) == 4  # 2 LLM + 2 new DB (ERBB2 deduplicated)
    genes = [h.target_gene for h in seed_population]
    assert genes == ["ERBB2", "BRCA1", "ESR1", "TP53"]
    # ERBB2 should be the LLM version (has real modality), not the DB version
    assert seed_population[0].modality == "mAb"


def test_engine_db_seed_all_duplicates() -> None:
    """When all DB seeds duplicate LLM seeds, population is unchanged."""
    llm_seeds = [
        TherapeuticHypothesis(
            target_gene="ERBB2",
            modality="mAb",
            delivery="IV",
            duration="chronic",
            tissue_context="breast",
            mutation_type="seed",
        ),
    ]

    db_seeds = [
        TherapeuticHypothesis(
            target_gene="ERBB2",
            modality="to be determined",
            delivery="to be determined",
            duration="chronic",
            tissue_context="",
            mutation_type="db_seed",
        ),
    ]

    seed_population = list(llm_seeds)
    existing_genes = {h.target_gene for h in seed_population}
    for h in db_seeds:
        if h.target_gene not in existing_genes:
            seed_population.append(h)
            existing_genes.add(h.target_gene)

    assert len(seed_population) == 1
    assert seed_population[0].modality == "mAb"
