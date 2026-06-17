"""Gene name normalization for API compatibility."""

from __future__ import annotations

# Map non-standard gene symbols to API-friendly versions
GENE_ALIASES: dict[str, str] = {
    "BCR-ABL": "BCR",  # fusion gene -> use BCR component
    "CD80/86": "CD80",  # dual target -> use primary
    "PPAR-gamma": "PPARG",  # common name -> HGNC symbol
    "Endothelin-A": "EDNRA",  # receptor name -> gene symbol
    "Factor Xa": "F10",  # coagulation factor -> gene symbol
    "Myostatin": "MSTN",  # protein name -> gene symbol
    "IgE": "FCER1A",  # immunoglobulin -> receptor gene
}


def normalize_gene(gene: str) -> str:
    """Return API-compatible gene symbol."""
    return GENE_ALIASES.get(gene, gene)
