from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import uuid


class EvidenceMaturity(IntEnum):
    L0_HYPOTHESIS = 0
    L1_ASSOCIATION = 1
    L2_IN_VITRO = 2
    L3_IN_VIVO = 3
    L4_CLINICAL = 4
    L5_CLINICAL_FAIL = 5


@dataclass
class PriorKnowledge:
    gene: str
    disease: str
    maturity: EvidenceMaturity
    known_facts: list[str]
    attempted_approaches: list[str]
    gaps: list[str]
    key_papers: list[str]
    summary: str


@dataclass
class StrategyPriorArt:
    strategy: str
    disease_class: str
    prior_studies: list[str]
    modality_status: dict[str, str]
    our_differentiation: list[str]
    key_papers: list[str]
    summary: str


@dataclass
class DiseaseQuery:
    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    known_targets: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)


@dataclass
class TherapeuticHypothesis:
    target_gene: str
    modality: str
    delivery: str
    duration: str
    tissue_context: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    mutation_type: str = "seed"
    generation: int = 0


@dataclass
class Evidence:
    source_type: str
    source_id: str
    summary: str
    relevance_score: float = 0.0


@dataclass
class FitnessScores:
    literature_strength: float = 0.0
    expression_specificity: float = 0.0
    pathway_centrality: float = 0.0
    druggability: float = 0.0
    safety_profile: float = 0.0
    ip_freedom: float = 0.0

    def dimensions(self) -> dict[str, float]:
        return {
            "literature_strength": self.literature_strength,
            "expression_specificity": self.expression_specificity,
            "pathway_centrality": self.pathway_centrality,
            "druggability": self.druggability,
            "safety_profile": self.safety_profile,
            "ip_freedom": self.ip_freedom,
        }


@dataclass
class ScoredHypothesis:
    hypothesis: TherapeuticHypothesis
    fitness: float
    scores: FitnessScores
    evidence: list[Evidence] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    api_errors: list[str] = field(default_factory=list)
    prior_knowledge: PriorKnowledge | None = None
    strategy_prior_art: StrategyPriorArt | None = None


# Internal configuration for prior knowledge
PRIOR_KNOWLEDGE_TOP_N = 5
PRIOR_KNOWLEDGE_ABSTRACT_CAP = 10


# Weights tuned from 55-entry live calibration (2026-04-10)
# Live separation: 0.523 (default) -> 0.611 (tuned)
# Demo separation: 1.000 (preserved)
@dataclass
class Weights:
    literature_strength: float = 0.0899
    expression_specificity: float = 0.1370
    pathway_centrality: float = 0.2915
    druggability: float = 0.0831
    safety_profile: float = 0.2935
    ip_freedom: float = 0.1050
    safety_threshold: float = 0.1947

    def dimensions(self) -> list[str]:
        return [
            "literature_strength",
            "expression_specificity",
            "pathway_centrality",
            "druggability",
            "safety_profile",
            "ip_freedom",
        ]


def compute_fitness(scores: FitnessScores, weights: Weights, gene: str = "") -> float:
    # Safety veto: low safety kills fitness UNLESS target has approved drugs
    # AND no severe class-effect penalty.
    #
    # Why check class effects? druggability > 0.5 alone is too permissive:
    # NGF has druggability > 0.5 (mAb modality = tractable) but tanezumab was
    # FDA-rejected due to RPOA. We allow bypass only when class-effect penalty
    # is mild (< 0.2), covering approved-drug successes like TNF/Adalimumab
    # (penalty 0.1) and VEGF/Bevacizumab (penalty 0.15).
    if scores.safety_profile < weights.safety_threshold:
        from biocompute.fitness.safety import CLASS_EFFECTS

        gene_upper = gene.upper() if gene else ""
        class_effect = CLASS_EFFECTS.get(gene_upper, {})
        penalty = (
            class_effect.get("penalty", 0) if isinstance(class_effect, dict) else 0
        )
        severe_class_effect = isinstance(penalty, (int, float)) and penalty >= 0.2

        if scores.druggability <= 0.5 or severe_class_effect:
            # No approved drugs OR severe class effect → veto stands
            return scores.safety_profile * 0.1

    weight_values = {
        "literature_strength": weights.literature_strength,
        "expression_specificity": weights.expression_specificity,
        "pathway_centrality": weights.pathway_centrality,
        "druggability": weights.druggability,
        "safety_profile": weights.safety_profile,
        "ip_freedom": weights.ip_freedom,
    }
    score_values = scores.dimensions()
    total: float = 0.0
    for dim in weights.dimensions():
        total += score_values[dim] * weight_values[dim]
    return total


@dataclass
class RunMetadata:
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    generations_run: int = 0
    total_hypotheses: int = 0
    total_api_calls: int = 0


def deduplicate_by_gene(
    candidates: list[ScoredHypothesis],
) -> list[ScoredHypothesis]:
    """Keep only the best-scoring hypothesis per gene, preserving order."""
    seen: dict[str, ScoredHypothesis] = {}
    for candidate in candidates:
        gene = candidate.hypothesis.target_gene
        if gene not in seen or candidate.fitness > seen[gene].fitness:
            seen[gene] = candidate
    return sorted(seen.values(), key=lambda s: s.fitness, reverse=True)


@dataclass
class DiscoveryResult:
    query: DiseaseQuery
    candidates: list[ScoredHypothesis]
    metadata: RunMetadata
    db_path: str | None = None
