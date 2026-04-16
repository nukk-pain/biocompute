from __future__ import annotations

from collections.abc import Mapping
import re

from biocompute.models import Evidence

LEVEL_MAP = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Not detected": 0,
}

# Map pathological/clinical tissue contexts to HPA anatomical terms
_TISSUE_ALIASES: dict[str, list[str]] = {
    "scar": ["skin", "skeletal muscle", "connective"],
    "fibrotic": ["skin", "liver", "lung"],
    "fibroblast": ["skin", "connective"],
    "myofascial": ["skeletal muscle"],
    "nociceptor": ["brain", "spinal cord", "nerve"],
    "nerve": ["brain", "spinal cord", "nerve"],
    "dorsal root": ["brain", "spinal cord"],
    "spinal": ["spinal cord", "brain"],
    "synovial": ["bone marrow", "skeletal muscle"],
    "endometri": ["endometrium", "ovary", "fallopian"],
    "liver": ["liver"],
    "brain": ["brain", "cerebral cortex"],
    "lung": ["lung"],
    "bone": ["bone marrow"],
    "immune": ["bone marrow", "lymph node", "spleen"],
    "arthritis": ["bone marrow", "skeletal muscle", "synovial"],
    "joint": ["bone marrow", "skeletal muscle"],
    "cancer": ["bone marrow", "lymph node", "liver", "lung", "breast"],
    "tumor": ["bone marrow", "lymph node", "liver", "lung"],
    "melanoma": ["skin"],
    "lymphoma": ["bone marrow", "lymph node", "spleen"],
    "leukemia": ["bone marrow"],
    "breast": ["breast"],
    "vascular": ["heart", "blood vessel", "endothelial"],
    "cardiac": ["heart", "cardiac muscle"],
    "heart": ["heart", "cardiac muscle"],
    "renal": ["kidney"],
    "kidney": ["kidney"],
    "hepatic": ["liver"],
    "pulmonary": ["lung"],
    "pancrea": ["pancreas"],
    "intestin": ["small intestine", "colon"],
    "colon": ["colon", "rectum"],
    "prostate": ["prostate"],
    "ovary": ["ovary"],
    "uterus": ["uterus", "endometrium"],
    "thyroid": ["thyroid"],
    "adrenal": ["adrenal"],
    "retina": ["retina", "eye"],
    "skin": ["skin"],
    "muscle": ["skeletal muscle"],
    "diabetes": ["pancreas", "liver", "adipose"],
    "asthma": ["lung", "bronchus"],
    "psoriasis": ["skin"],
    "fibrosis": ["liver", "lung", "kidney"],
    "inflamma": ["bone marrow", "lymph node", "spleen"],
    "autoimmun": ["bone marrow", "lymph node", "spleen", "thymus"],
    "neurodegen": ["brain", "cerebral cortex", "hippocampus"],
    "depression": ["brain", "cerebral cortex", "hippocampus"],
    "schizophren": ["brain", "cerebral cortex"],
    "epilep": ["brain", "cerebral cortex", "hippocampus"],
    "pain": ["spinal cord", "brain", "nerve", "skeletal muscle"],
    "migraine": ["brain", "cerebral cortex"],
    "osteoporosis": ["bone marrow"],
}


def _resolve_target_tissues(target_tissue: str) -> list[str]:
    """Map clinical tissue context to HPA anatomical tissue names."""
    target_lower = target_tissue.lower()
    resolved: list[str] = []
    for keyword, hpa_names in _TISSUE_ALIASES.items():
        if keyword in target_lower:
            resolved.extend(hpa_names)
    return list(dict.fromkeys(resolved))  # deduplicate preserving order


def _coerce_tissue_entries(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def _normalize_tissue_name(tissue_name: str) -> str:
    """Normalize HPA and GTEx tissue labels to a shared comparison format."""
    with_spaces = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tissue_name)
    normalized = re.sub(r"[^a-z0-9]+", " ", with_spaces.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def merge_expression_tissues(
    hpa_tissues: list[Mapping[str, object]],
    gtex_tissues: list[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Merge HPA and GTEx tissues, keeping the higher level per tissue."""
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for tissue in [*hpa_tissues, *gtex_tissues]:
        tissue_name = tissue.get("Tissue")
        level_name = tissue.get("Level")
        if not isinstance(tissue_name, str) or not isinstance(level_name, str):
            continue

        key = _normalize_tissue_name(tissue_name)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {"Tissue": tissue_name, "Level": level_name}
            order.append(key)
            continue

        if LEVEL_MAP.get(level_name, 0) > LEVEL_MAP.get(existing.get("Level", ""), 0):
            existing["Level"] = level_name

    return [merged[key] for key in order]


def score_expression(
    data: Mapping[str, object],
    target_tissue: str = "",
) -> tuple[float, list[Evidence]]:
    tissues = _coerce_tissue_entries(data.get("tissues", []))
    gtex_tissues = _coerce_tissue_entries(data.get("gtex_tissues", []))
    if gtex_tissues:
        tissues = merge_expression_tissues(tissues, gtex_tissues)
    if not tissues:
        source = data.get("source")
        source_name = source if isinstance(source, str) else "hpa"
        return 0.0, [Evidence(source_name, "none", "No expression data available", 0.0)]

    # Resolve pathological tissue names to HPA anatomical terms
    hpa_targets = _resolve_target_tissues(target_tissue) if target_tissue else []

    target_level = 0
    other_levels: list[int] = []

    for tissue in tissues:
        tissue_name = _normalize_tissue_name(str(tissue.get("Tissue", "")))
        level = LEVEL_MAP.get(str(tissue.get("Level", "")), 0)
        matched = False
        if hpa_targets:
            for hpa_name in hpa_targets:
                normalized_target = _normalize_tissue_name(hpa_name)
                if normalized_target in tissue_name or tissue_name in normalized_target:
                    matched = True
                    break
        elif target_tissue:
            matched = _normalize_tissue_name(target_tissue) in tissue_name

        if matched:
            target_level = max(target_level, level)
        else:
            other_levels.append(level)

    # Fallback: if no tissue matched, use max expression level
    if target_level == 0:
        target_level = max(
            (LEVEL_MAP.get(str(tissue.get("Level", "")), 0) for tissue in tissues),
            default=0,
        )
        other_levels = [
            LEVEL_MAP.get(str(tissue.get("Level", "")), 0) for tissue in tissues
        ]

    avg_other = sum(other_levels) / len(other_levels) if other_levels else 0.0

    # Two components: presence (is the gene expressed?) + specificity (preferential?)
    presence = target_level / 3.0  # 0.0 to 1.0
    specificity = (
        max((target_level - avg_other) / 3.0, 0.0) if target_level > 0 else 0.0
    )

    # Weighted blend: 60% presence, 40% specificity bonus
    score = max(0.0, min(presence * 0.6 + specificity * 0.4, 1.0))

    evidence = [
        Evidence(
            str(data.get("source", "hpa")),
            f"tissue:{target_tissue or 'general'}",
            f"Target tissue level: {target_level}/3, avg other: {avg_other:.1f}/3",
            score,
        )
    ]
    return score, evidence
