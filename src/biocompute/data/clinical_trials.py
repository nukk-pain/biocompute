"""ClinicalTrials.gov API client for clinical outcome data.

Queries the ClinicalTrials.gov v2 API to find terminated/withdrawn and
completed studies for a gene+disease pair, then summarizes phase-level
failure signals.
"""

from __future__ import annotations

import httpx

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"


def _build_ct_query(disease: str) -> str:
    """Build ClinicalTrials.gov disease query terms.

    Short names (CML, SMA) pass through as-is.
    Longer names keep only the first 3 meaningful words (len > 2)
    to avoid over-specific queries that miss relevant trials.
    """
    words = [w for w in disease.split() if len(w) > 2]
    if len(words) <= 3:
        return disease
    return " ".join(words[:3])


async def _search_studies(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
    statuses: str,
    page_size: int = 50,
) -> list[dict[str, object]]:
    """Search ClinicalTrials.gov for studies matching gene+disease with given statuses."""
    disease_terms = _build_ct_query(disease)
    query_term = f'"{gene}" {disease_terms}'

    try:
        resp = await client.get(
            CT_BASE,
            params={
                "query.term": query_term,
                "filter.overallStatus": statuses,
                "pageSize": page_size,
                "format": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, Exception):
        return []

    studies = data.get("studies", [])
    return studies if isinstance(studies, list) else []


def _extract_phases(study: object) -> list[str]:
    """Extract phase list from a study's protocolSection.designModule.phases."""
    if not isinstance(study, dict):
        return []
    protocol = study.get("protocolSection")
    if not isinstance(protocol, dict):
        return []
    design = protocol.get("designModule")
    if not isinstance(design, dict):
        return []
    phases = design.get("phases")
    if isinstance(phases, list):
        return [str(p) for p in phases]
    return []


def _extract_title(study: object) -> str:
    """Extract brief title from a study."""
    if not isinstance(study, dict):
        return "unknown"
    protocol = study.get("protocolSection")
    if not isinstance(protocol, dict):
        return "unknown"
    id_module = protocol.get("identificationModule")
    if not isinstance(id_module, dict):
        return "unknown"
    title = id_module.get("briefTitle")
    return str(title)[:100] if isinstance(title, str) else "unknown"


def _is_phase_2_or_3(phases: list[str]) -> bool:
    """Check if any phase is Phase 2 or Phase 3."""
    for phase in phases:
        phase_lower = phase.lower()
        if "phase3" in phase_lower or "phase 3" in phase_lower:
            return True
        if "phase2" in phase_lower or "phase 2" in phase_lower:
            return True
    return False


async def get_clinical_outcome(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    """Query ClinicalTrials.gov for clinical outcome signals.

    Returns a dict with completed/failed counts, phase 3 failure count,
    failure ratio, and failed trial names.
    """
    failed_studies = await _search_studies(
        client, gene, disease, "TERMINATED,WITHDRAWN,SUSPENDED"
    )
    completed_studies = await _search_studies(
        client, gene, disease, "COMPLETED"
    )

    failed_count = len(failed_studies)
    completed_count = len(completed_studies)

    phase3_failures = 0
    failed_trial_names: list[str] = []

    for study in failed_studies:
        phases = _extract_phases(study)
        title = _extract_title(study)
        failed_trial_names.append(title)
        if _is_phase_2_or_3(phases):
            phase3_failures += 1

    total = completed_count + failed_count
    failure_ratio = failed_count / total if total > 0 else 0.0

    return {
        "gene": gene,
        "disease": disease,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "phase3_failures": phase3_failures,
        "failure_ratio": failure_ratio,
        "failed_trial_names": failed_trial_names[:5],  # cap at 5 for brevity
        "source": "clinicaltrials_gov",
    }
