from __future__ import annotations

from typing import TypedDict, cast

import httpx

from biocompute.data.pubmed import search_pubmed
from biocompute.data.pubmed_abstracts import PubMedAbstractRecord, fetch_abstracts


STRATEGY_QUERY_CAP = 5
STRATEGY_ABSTRACT_CAP = 10


class StrategyPriorArtAssessment(TypedDict):
    prior_studies: list[str]
    modality_status: dict[str, str]
    our_differentiation: list[str]
    summary: str


def _generate_strategy_queries(gene: str, disease: str, modality: str) -> list[str]:
    """Generate up to five strategy-oriented PubMed queries using the LLM.

    Falls back to deterministic heuristic queries if the LLM fails or returns an
    unusable shape so downstream literature collection can continue.
    """
    from biocompute.data.llm import query_llm_json

    prompt = _build_query_prompt(gene, disease, modality)

    try:
        result = query_llm_json(prompt, model="haiku")
    except Exception:
        return _fallback_strategy_queries(gene, disease, modality)

    queries = _normalize_queries(result)
    if queries:
        return queries
    return _fallback_strategy_queries(gene, disease, modality)


async def _search_strategy_abstracts(
    client: httpx.AsyncClient,
    queries: list[str],
) -> list[PubMedAbstractRecord]:
    """Search PubMed and collect unique abstracts across strategy queries.

    Duplicate PMIDs are removed while preserving first-seen order. Query-level
    search/fetch failures are ignored so partial results still flow through.
    """
    unique_records: dict[str, PubMedAbstractRecord] = {}

    for query in queries[:STRATEGY_QUERY_CAP]:
        cleaned_query = query.strip()
        if not cleaned_query:
            continue

        try:
            pmids = await search_pubmed(
                client, cleaned_query, max_results=STRATEGY_ABSTRACT_CAP
            )
        except Exception:
            continue

        if not pmids:
            continue

        try:
            records = await fetch_abstracts(
                client, pmids, max_abstracts=STRATEGY_ABSTRACT_CAP
            )
        except Exception:
            continue

        for record in records:
            pmid = str(record.get("pmid", "")).strip()
            if not pmid or pmid in unique_records:
                continue
            unique_records[pmid] = record

    return list(unique_records.values())


def assess_strategy_prior_art(
    gene: str,
    disease: str,
    modality: str,
    abstracts: list[dict[str, str]],
) -> StrategyPriorArtAssessment:
    """Summarize prior therapeutic strategy evidence from PubMed abstracts.

    This is informational only. Empty abstracts, LLM exceptions, malformed JSON,
    or invalid field shapes all degrade to safe defaults instead of raising.
    """
    from biocompute.data.llm import query_llm_json

    usable_abstracts = _usable_abstracts(abstracts)
    if not usable_abstracts:
        return _safe_defaults(
            summary="No usable PubMed abstracts available for strategy prior-art assessment."
        )

    prompt = _build_assessment_prompt(gene, disease, modality, usable_abstracts)

    try:
        result = query_llm_json(prompt, model="sonnet")
    except Exception:
        return _safe_defaults(summary="Strategy prior-art assessment unavailable.")

    if not isinstance(result, dict):
        return _safe_defaults(summary="Strategy prior-art assessment unavailable.")

    normalized = _normalize_assessment(result)
    if not normalized["summary"]:
        normalized["summary"] = "Strategy prior-art assessment unavailable."
    return normalized


def _build_query_prompt(gene: str, disease: str, modality: str) -> str:
    return (
        "You are a biomedical literature strategist. "
        "Generate up to 5 PubMed-style search queries that help find prior therapeutic "
        f'strategies relevant to modulating the gene "{gene}" in the disease "{disease}".\n\n'
        "Return ONLY a JSON array of strings.\n"
        "Focus on therapeutic strategy synonyms, adjacent pathway interventions, "
        "gene overexpression/silencing, and modality-specific prior art.\n"
        "Keep each query concise and searchable in PubMed.\n\n"
        f"Gene: {gene}\n"
        f"Disease: {disease}\n"
        f"Modality of interest: {modality}\n"
    )


def _build_assessment_prompt(
    gene: str,
    disease: str,
    modality: str,
    abstracts: list[dict[str, str]],
) -> str:
    return (
        "You are a biomedical literature analyst. "
        "Use only the supplied PubMed abstracts to summarize prior therapeutic strategy evidence.\n\n"
        "Return ONLY a JSON object with exactly these keys:\n"
        "{\n"
        '  "prior_studies": ["short citation or PMID-backed study note"],\n'
        '  "modality_status": {"AAV": "status", "mRNA-LNP": "status"},\n'
        '  "our_differentiation": ["how our proposed modality differs"],\n'
        '  "summary": "one concise sentence"\n'
        "}\n\n"
        "Rules:\n"
        "- Use only information present in the provided abstracts.\n"
        "- prior_studies should capture concrete prior-art examples, not speculation.\n"
        "- modality_status should describe which modalities appear supported, absent, or unclear in the abstracts.\n"
        "- our_differentiation should describe only evidence-grounded distinctions for the proposed modality.\n"
        "- If evidence is sparse, keep lists short and say so in the summary.\n"
        "- Do not invent studies, PMIDs, or modality claims.\n\n"
        f"Gene of interest: {gene}\n"
        f"Disease of interest: {disease}\n"
        f"Proposed modality: {modality}\n"
        f"Abstract count: {len(abstracts)}\n\n"
        f"Abstracts:\n{_format_abstracts(abstracts)}"
    )


def _normalize_queries(raw: object) -> list[str]:
    if isinstance(raw, dict):
        raw_dict = cast(dict[str, object], raw)
        raw = raw_dict.get("queries")

    if not isinstance(raw, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in cast(list[object], raw):
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            continue
        query = str(item).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        normalized.append(query)
        if len(normalized) >= STRATEGY_QUERY_CAP:
            break
    return normalized


def _normalize_assessment(raw: dict[str, object]) -> StrategyPriorArtAssessment:
    modality_status = raw.get("modality_status")
    normalized_modality_status: dict[str, str] = {}
    if isinstance(modality_status, dict):
        for key, value in cast(dict[object, object], modality_status).items():
            key_text = str(key).strip()
            if (
                not key_text
                or isinstance(value, bool)
                or not isinstance(value, (str, int, float))
            ):
                continue
            value_text = str(value).strip()
            if value_text:
                normalized_modality_status[key_text] = value_text

    return {
        "prior_studies": _normalize_list(raw.get("prior_studies")),
        "modality_status": normalized_modality_status,
        "our_differentiation": _normalize_list(raw.get("our_differentiation")),
        "summary": str(raw.get("summary", "")).strip(),
    }


def _normalize_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in cast(list[object], value):
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _usable_abstracts(abstracts: list[dict[str, str]]) -> list[dict[str, str]]:
    usable: list[dict[str, str]] = []
    for abstract in abstracts:
        text = str(abstract.get("abstract", "")).strip()
        if not text:
            continue
        usable.append(
            {
                "pmid": str(abstract.get("pmid", "")).strip(),
                "title": str(abstract.get("title", "")).strip(),
                "year": str(abstract.get("year", "")).strip(),
                "abstract": text,
            }
        )
        if len(usable) >= STRATEGY_ABSTRACT_CAP:
            break
    return usable


def _format_abstracts(abstracts: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for index, abstract in enumerate(abstracts, start=1):
        lines.append(
            f"[{index}] PMID: {abstract.get('pmid') or 'unknown'} | Year: "
            + f"{abstract.get('year') or 'unknown'}\n"
            + f"Title: {abstract.get('title') or 'untitled'}\n"
            + f"Abstract: {abstract.get('abstract') or ''}"
        )
    return "\n\n".join(lines)


def _fallback_strategy_queries(gene: str, disease: str, modality: str) -> list[str]:
    disease_phrase = f'"{disease}"'
    return [
        f'"{gene}" AND {disease_phrase} AND (therapy OR therapeutic)',
        f'"{gene}" AND {disease_phrase} AND (overexpression OR silencing OR inhibition)',
        f'"{gene}" AND fibrosis AND (gene therapy OR viral vector OR AAV)',
        f'"{gene}" AND scar AND (siRNA OR antisense OR RNA therapy)',
        f'"{gene}" AND {disease_phrase} AND ("{modality}" OR nanoparticle)',
    ][:STRATEGY_QUERY_CAP]


def _safe_defaults(*, summary: str) -> StrategyPriorArtAssessment:
    return {
        "prior_studies": [],
        "modality_status": {},
        "our_differentiation": [],
        "summary": summary,
    }
