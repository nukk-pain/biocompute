"""LLM-based clinical feasibility assessment.

Uses Claude CLI to determine whether a gene has an approved drug for a
specific disease -- something ClinicalTrials.gov data alone cannot
reliably distinguish (e.g. VEGF+cancer=approved vs APP+alzheimer=failed).
"""

from __future__ import annotations


def assess_clinical_feasibility(gene: str, disease: str) -> dict:
    """Ask the LLM whether an approved drug targeting *gene* exists for *disease*.

    Returns a dict with at minimum:
      - has_approved_drug: bool
      - has_phase3_failure: bool
      - feasibility_score: float (0.0-1.0)
      - approved_drugs: list[str]
      - failed_drugs: list[str]
      - rationale: str

    On any failure, returns safe defaults (feasibility_score=0.5).
    """
    from biocompute.data.llm import query_llm_json

    prompt = (
        f'You are a pharmaceutical regulatory expert. '
        f'For the therapeutic target gene "{gene}" and disease indication "{disease}":\n\n'
        f'1. Has any drug targeting {gene} been approved by FDA or EMA for {disease}?\n'
        f'2. If yes, name the approved drug(s).\n'
        f'3. If no, have any drugs targeting {gene} failed in Phase 2/3 clinical trials for {disease}?\n'
        f'4. Rate the clinical feasibility of targeting {gene} for {disease} on a scale of 0.0 to 1.0.\n\n'
        f'Return ONLY a JSON object:\n'
        f'{{\n'
        f'  "has_approved_drug": true/false,\n'
        f'  "approved_drugs": ["drug1", "drug2"],\n'
        f'  "has_phase3_failure": true/false,\n'
        f'  "failed_drugs": ["drug1"],\n'
        f'  "feasibility_score": 0.0-1.0,\n'
        f'  "rationale": "one sentence explanation"\n'
        f'}}'
    )

    try:
        result = query_llm_json(prompt, model="haiku")
    except Exception:
        return _safe_defaults()

    if not isinstance(result, dict):
        return _safe_defaults()

    normalized = _normalize(result)
    return _validate_drug_claims(normalized, gene)


def _safe_defaults() -> dict:
    """Return neutral defaults when LLM call fails or parsing fails."""
    return {
        "has_approved_drug": False,
        "approved_drugs": [],
        "has_phase3_failure": False,
        "failed_drugs": [],
        "feasibility_score": 0.5,
        "rationale": "LLM assessment unavailable",
    }


def _normalize(raw: dict) -> dict:
    """Ensure all expected keys exist with correct types."""
    return {
        "has_approved_drug": bool(raw.get("has_approved_drug", False)),
        "approved_drugs": (
            raw["approved_drugs"]
            if isinstance(raw.get("approved_drugs"), list)
            else []
        ),
        "has_phase3_failure": bool(raw.get("has_phase3_failure", False)),
        "failed_drugs": (
            raw["failed_drugs"]
            if isinstance(raw.get("failed_drugs"), list)
            else []
        ),
        "feasibility_score": _clamp_float(raw.get("feasibility_score", 0.5)),
        "rationale": str(raw.get("rationale", "")),
    }


def _validate_drug_claims(result: dict, gene: str) -> dict:
    """Flag potentially hallucinated drug claims."""
    approved_drugs = result.get("approved_drugs", [])
    if not isinstance(approved_drugs, list):
        approved_drugs = []

    # Known real drugs for common targets (quick sanity check)
    KNOWN_DRUGS: dict[str, set[str]] = {
        "VEGF": {"bevacizumab", "ranibizumab", "aflibercept", "ramucirumab"},
        "TNF": {"adalimumab", "infliximab", "etanercept", "certolizumab", "golimumab"},
        "PD1": {"pembrolizumab", "nivolumab", "cemiplimab"},
        "PDL1": {"atezolizumab", "durvalumab", "avelumab"},
        "HER2": {"trastuzumab", "pertuzumab", "ado-trastuzumab"},
        "EGFR": {"erlotinib", "gefitinib", "osimertinib", "cetuximab"},
        "CXCR4": {"plerixafor"},
        "CGRP": {"erenumab", "fremanezumab", "galcanezumab"},
        "CD20": {"rituximab", "obinutuzumab", "ofatumumab"},
        "PCSK9": {"evolocumab", "alirocumab"},
        "CCR5": {"maraviroc"},
        "IL17A": {"secukinumab", "ixekizumab"},
        "JAK1": {"tofacitinib", "baricitinib", "upadacitinib"},
        "RANKL": {"denosumab"},
        "GLP1R": {"semaglutide", "liraglutide", "dulaglutide"},
    }

    known = KNOWN_DRUGS.get(gene.upper(), set())
    if known and approved_drugs:
        # Check if any claimed drug matches known drugs (case-insensitive)
        claimed_lower = {d.lower() for d in approved_drugs if isinstance(d, str)}
        verified = claimed_lower & {d.lower() for d in known}
        if not verified and claimed_lower:
            # LLM claimed drugs but none match known list — suspicious
            result["drug_verification"] = "unverified"
            result["verified_drugs"] = []
        else:
            result["drug_verification"] = "verified"
            result["verified_drugs"] = list(verified)
    else:
        result["drug_verification"] = "no_reference"

    return result


def _clamp_float(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi], defaulting to 0.5 for non-numeric input."""
    if isinstance(value, (int, float)):
        return max(lo, min(float(value), hi))
    return 0.5
