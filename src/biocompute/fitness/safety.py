from __future__ import annotations

from collections.abc import Mapping

from biocompute.models import Evidence

EXPOSURE_FACTORS = {
    ("local", "single-dose"): 0.1,
    ("local", "acute"): 0.2,
    ("local", "chronic"): 0.3,
    ("systemic", "single-dose"): 0.5,
    ("systemic", "acute"): 0.7,
    ("systemic", "chronic"): 1.0,
}

SEVERE_LIABILITY_KEYWORDS = {
    "cardiotoxicity",
    "hepatotoxicity",
    "neurotoxicity",
    "myelosuppression",
    "renal toxicity",
}

CLASS_EFFECTS: dict[str, dict[str, object]] = {
    "NGF": {"event": "RPOA (rapidly progressive osteoarthropathy)", "penalty": 0.3},
    "VEGF": {"event": "hypertension/bleeding risk", "penalty": 0.15},
    "TNF": {"event": "infection susceptibility", "penalty": 0.1},
    "IL6": {"event": "infection susceptibility", "penalty": 0.1},
    "IL2": {"event": "capillary leak syndrome / cytokine storm", "penalty": 0.25},
    "CTLA4": {"event": "immune-related adverse events (irAEs)", "penalty": 0.15},
    "PD1": {"event": "immune-related adverse events (irAEs)", "penalty": 0.1},
    "PDL1": {"event": "immune-related adverse events (irAEs)", "penalty": 0.1},
    "JAK1": {"event": "infection risk / thrombosis (FDA black box)", "penalty": 0.15},
    "JAK2": {"event": "infection risk / thrombosis / myelosuppression", "penalty": 0.2},
    "BRAF": {"event": "cutaneous squamous cell carcinoma", "penalty": 0.1},
    "CETP": {"event": "cardiovascular mortality (torcetrapib)", "penalty": 0.3},
    "PPAR-GAMMA": {"event": "cardiovascular risk / fluid retention", "penalty": 0.2},
    "PPARG": {"event": "cardiovascular risk / fluid retention", "penalty": 0.2},
    "RANKL": {"event": "osteonecrosis of jaw / atypical fractures", "penalty": 0.1},
}


def classify_delivery(delivery: str) -> str:
    delivery_lower = delivery.lower()
    if any(
        keyword in delivery_lower
        for keyword in ["local", "topical", "intratumoral", "intra-articular"]
    ):
        return "local"
    return "systemic"


def _normalize_duration(duration: str) -> str:
    normalized = duration.lower().replace(" ", "-")
    if normalized in {"single", "single-dose", "single-dos", "single_dose"}:
        return "single-dose"
    return normalized


def _liability_weight(liability: object) -> float:
    if not isinstance(liability, dict):
        return 1.0

    event = str(liability.get("event", "")).lower()
    return 1.5 if event in SEVERE_LIABILITY_KEYWORDS else 1.0


def score_safety(
    data: Mapping[str, object],
    delivery: str = "systemic",
    duration: str = "chronic",
    gene: str = "",
) -> tuple[float, list[Evidence]]:
    liabilities_value = data.get("safety_liabilities", [])
    liabilities = liabilities_value if isinstance(liabilities_value, list) else []
    delivery_class = classify_delivery(delivery)

    if not liabilities:
        systemic_risk = 0.1
    else:
        weighted_liabilities = sum(
            _liability_weight(liability) for liability in liabilities
        )
        systemic_risk = min(weighted_liabilities / 5, 1.0)

    duration_norm = _normalize_duration(duration)
    exposure = EXPOSURE_FACTORS.get((delivery_class, duration_norm), 0.5)
    adjusted_risk = systemic_risk * exposure
    score = max(0.0, min(1.0 - adjusted_risk, 1.0))

    class_effect = CLASS_EFFECTS.get(gene.upper()) if gene else None
    if class_effect is not None:
        penalty = float(class_effect["penalty"])
        score = max(0.0, score - penalty)

    events = [
        str(liability.get("event", "unknown"))
        for liability in liabilities[:3]
        if isinstance(liability, dict)
    ]
    evidence = [
        Evidence(
            str(data.get("source", "opentargets")),
            f"safety:{delivery_class}/{duration_norm}",
            f"Liabilities: {events}, exposure factor: {exposure}",
            score,
        )
    ]
    if class_effect is not None:
        evidence.append(
            Evidence(
                "class_effect",
                f"safety:class_effect/{gene.upper()}",
                f"Known class effect: {class_effect['event']} (penalty: {class_effect['penalty']})",
                score,
            )
        )
    return score, evidence
