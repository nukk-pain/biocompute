from __future__ import annotations

import re
from typing import cast

from biocompute.models import (
    PRIOR_KNOWLEDGE_ABSTRACT_CAP,
    EvidenceMaturity,
    PriorKnowledge,
)


def assess_prior_knowledge(
    gene: str,
    disease: str,
    abstracts: list[dict[str, str]],
) -> PriorKnowledge:
    """Summarize prior knowledge for a gene-disease pair from PubMed abstracts.

    Returns a normalized ``PriorKnowledge`` object and fails soft on empty
    abstracts, LLM exceptions, malformed JSON, or invalid response shapes.
    """
    from biocompute.data.llm import query_llm_json

    usable_abstracts = _usable_abstracts(abstracts)
    if not usable_abstracts:
        return _safe_defaults(
            gene,
            disease,
            summary="No usable PubMed abstracts available for prior-knowledge assessment.",
        )

    prompt = _build_prompt(gene, disease, usable_abstracts)

    try:
        result = query_llm_json(prompt, model="sonnet")
    except Exception:
        return _safe_defaults(
            gene,
            disease,
            summary="Prior knowledge assessment unavailable.",
        )

    if not isinstance(result, dict):
        return _safe_defaults(
            gene,
            disease,
            summary="Prior knowledge assessment unavailable.",
        )

    return _normalize(gene, disease, result)


def _safe_defaults(gene: str, disease: str, *, summary: str) -> PriorKnowledge:
    """Return a conservative sentinel object safe for downstream rendering."""
    return PriorKnowledge(
        gene=gene,
        disease=disease,
        maturity=EvidenceMaturity.L0_HYPOTHESIS,
        known_facts=[],
        attempted_approaches=[],
        gaps=[],
        key_papers=[],
        summary=summary,
    )


def _normalize(gene: str, disease: str, raw: dict[str, object]) -> PriorKnowledge:
    """Normalize raw LLM output into the stable PriorKnowledge schema."""
    return PriorKnowledge(
        gene=gene,
        disease=disease,
        maturity=_map_maturity(raw.get("maturity")),
        known_facts=_normalize_list(raw.get("known_facts")),
        attempted_approaches=_normalize_list(raw.get("attempted_approaches")),
        gaps=_normalize_list(raw.get("gaps")),
        key_papers=_normalize_list(raw.get("key_papers")),
        summary=str(raw.get("summary", "")).strip(),
    )


def _normalize_list(value: object) -> list[str]:
    """Keep only non-empty scalar list items; wrong field types degrade to []."""
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in cast(list[object], value):
        if isinstance(item, bool):
            continue
        if isinstance(item, (str, int, float)):
            text = str(item).strip()
            if text:
                normalized.append(text)
    return normalized


def _usable_abstracts(abstracts: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter to abstracts with usable text and cap prompt size."""
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

    return usable[:PRIOR_KNOWLEDGE_ABSTRACT_CAP]


def _build_prompt(gene: str, disease: str, abstracts: list[dict[str, str]]) -> str:
    """Build a constrained JSON-only prompt for prior-knowledge classification."""
    return (
        "You are a biomedical literature analyst. "
        "Use only the provided PubMed abstracts to summarize what is already known "
        f'about the therapeutic target gene "{gene}" in the disease "{disease}".\n\n'
        "Return ONLY a JSON object with exactly these keys:\n"
        "{\n"
        '  "maturity": "L0|L1|L2|L3|L4|L5",\n'
        '  "known_facts": ["fact 1", "fact 2"],\n'
        '  "attempted_approaches": ["approach 1", "approach 2"],\n'
        '  "gaps": ["gap 1", "gap 2"],\n'
        '  "key_papers": ["PMID or short citation", "PMID or short citation"],\n'
        '  "summary": "one concise sentence"\n'
        "}\n\n"
        "Maturity definitions:\n"
        "- L0: No usable papers or hypothesis-only evidence.\n"
        "- L1: Association papers exist but no direct validation.\n"
        "- L2: In vitro validation exists (cell-level effects confirmed).\n"
        "- L3: In vivo validation exists (animal model effects confirmed).\n"
        "- L4: Clinical trial entered or human-subject evidence exists.\n"
        "- L5: Clinical trial failed or human therapeutic effort clearly failed.\n\n"
        "Rules:\n"
        "- If evidence is partial or contradictory, keep the maturity at the highest clearly supported level, "
        "but mention contradictions or uncertainty in gaps and summary.\n"
        "- If any failed human trial or failed clinical development is described, prefer L5.\n"
        "- Do not invent papers, drugs, or facts not present in the abstracts.\n"
        "- Keep each list concise and omit commentary outside the JSON object.\n\n"
        f"Abstract count: {len(abstracts)}\n"
        f"Gene: {gene}\n"
        f"Disease: {disease}\n\n"
        f"Abstracts:\n{_format_abstracts(abstracts)}"
    )


def _format_abstracts(abstracts: list[dict[str, str]]) -> str:
    """Render abstracts into a compact prompt-friendly block."""
    lines: list[str] = []
    for index, abstract in enumerate(abstracts, start=1):
        pmid = abstract.get("pmid", "")
        year = abstract.get("year", "")
        title = abstract.get("title", "")
        body = abstract.get("abstract", "")
        lines.append(
            f"[{index}] PMID: {pmid or 'unknown'} | Year: {year or 'unknown'}\n"
            + f"Title: {title or 'untitled'}\n"
            + f"Abstract: {body}"
        )
    return "\n\n".join(lines)


def _map_maturity(value: object) -> EvidenceMaturity:
    """Map LLM maturity output into the EvidenceMaturity enum conservatively."""
    if isinstance(value, EvidenceMaturity):
        return value

    if isinstance(value, int) and 0 <= value <= 5:
        return EvidenceMaturity(value)

    if not isinstance(value, str):
        return EvidenceMaturity.L0_HYPOTHESIS

    text = value.strip().upper()
    if not text:
        return EvidenceMaturity.L0_HYPOTHESIS

    raw_matches = cast(list[str], re.findall(r"L\s*([0-5])\b", text))
    matches = [int(match) for match in raw_matches]
    if matches:
        if 5 in matches:
            return EvidenceMaturity.L5_CLINICAL_FAIL
        return EvidenceMaturity(max(matches))

    keyword_mapping: list[tuple[tuple[str, ...], EvidenceMaturity]] = [
        (
            (
                "CLINICAL FAILURE",
                "FAILED CLINICAL",
                "PHASE 3 FAILURE",
                "TRIAL FAILED",
                "FAILED TRIAL",
            ),
            EvidenceMaturity.L5_CLINICAL_FAIL,
        ),
        (
            ("CLINICAL", "PHASE 1", "PHASE 2", "PHASE 3", "HUMAN"),
            EvidenceMaturity.L4_CLINICAL,
        ),
        (("IN VIVO", "ANIMAL", "MOUSE", "MURINE"), EvidenceMaturity.L3_IN_VIVO),
        (("IN VITRO", "CELL", "CELLULAR"), EvidenceMaturity.L2_IN_VITRO),
        (
            ("ASSOCIATION", "ASSOCIATIVE", "CORRELATION", "CORRELATIVE"),
            EvidenceMaturity.L1_ASSOCIATION,
        ),
        (
            ("HYPOTHESIS", "NO PAPERS", "NO USABLE PAPERS", "NO EVIDENCE"),
            EvidenceMaturity.L0_HYPOTHESIS,
        ),
    ]
    for keywords, maturity in keyword_mapping:
        if any(keyword in text for keyword in keywords):
            return maturity

    return EvidenceMaturity.L0_HYPOTHESIS
