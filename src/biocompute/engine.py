# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportAny=false

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import httpx

from biocompute.archive.store import ArchiveStore
from biocompute.fitness.evaluator import evaluate_all_dimensions
from biocompute.models import (
    DiseaseQuery,
    DiscoveryResult,
    Evidence,
    FitnessScores,
    PRIOR_KNOWLEDGE_TOP_N,
    RunMetadata,
    ScoredHypothesis,
    TherapeuticHypothesis,
    Weights,
)
from biocompute.data.gene_names import normalize_gene
from biocompute.search.critique import critique_hypothesis
from biocompute.search.mutate import mutate_hypothesis
from biocompute.search.seed import generate_seed_population
from biocompute.search.select import select_survivors

CollectDataFn = Callable[[TherapeuticHypothesis, DiseaseQuery], dict[str, object]]


def _cap_generation_children(
    children_by_parent: Sequence[Sequence[TherapeuticHypothesis]],
    limit: int,
) -> list[TherapeuticHypothesis]:
    if limit <= 0:
        return []

    capped: list[TherapeuticHypothesis] = []
    index = 0
    while len(capped) < limit:
        added_in_round = False
        for children in children_by_parent:
            if index >= len(children):
                continue
            capped.append(children[index])
            added_in_round = True
            if len(capped) >= limit:
                break

        if not added_in_round:
            break
        index += 1

    return capped


def collect_bio_data(
    hypothesis: TherapeuticHypothesis,
    query: DiseaseQuery,
) -> dict[str, object]:
    return asyncio.run(_collect_bio_data_async(hypothesis, query))


async def _collect_bio_data_async(
    hypothesis: TherapeuticHypothesis,
    query: DiseaseQuery,
) -> dict[str, object]:
    import httpx as _httpx  # lazy import (M6)

    gene = normalize_gene(hypothesis.target_gene)
    disease = query.name

    async with _httpx.AsyncClient() as client:
        literature_task = asyncio.create_task(
            _collect_literature_data(client, gene, disease)
        )
        hpa_expression_task = asyncio.create_task(
            _safe_get_tissue_expression(client, gene)
        )
        gtex_expression_task = asyncio.create_task(
            _safe_get_gtex_expression(client, gene)
        )
        pathway_task = asyncio.create_task(_safe_get_interaction_partners(client, gene))
        target_task = asyncio.create_task(_collect_target_data(client, gene))
        clinical_task = asyncio.create_task(
            _safe_get_clinical_outcome(client, gene, disease)
        )
        gwas_task = asyncio.create_task(_safe_get_gwas_evidence(client, gene, disease))
        (
            literature,
            hpa_expression,
            gtex_expression,
            pathway,
            target,
            clinical,
            gwas,
        ) = await asyncio.gather(
            literature_task,
            hpa_expression_task,
            gtex_expression_task,
            pathway_task,
            target_task,
            clinical_task,
            gwas_task,
        )
        expression = _merge_expression_sections(gene, hpa_expression, gtex_expression)
        llm_clinical = await _safe_get_llm_feasibility(gene, disease)

    raw_data: dict[str, object] = {
        "literature": literature,
        "expression": expression,
        "pathway": pathway,
        "druggability": target,
        "safety": target,
        "ip": _build_ip_section(target),
        "clinical": clinical,
        "gwas": gwas,
        "llm_clinical": llm_clinical,
    }
    api_errors = _gather_api_errors(raw_data)
    if api_errors:
        raw_data["api_errors"] = api_errors
    return raw_data


def _collect_errors(*sections: dict[str, object]) -> list[str]:
    """Extract error strings from section dicts returned by _safe_* functions."""
    errors: list[str] = []
    for section in sections:
        err = section.get("error")
        if isinstance(err, str):
            errors.append(err)
    return errors


def _gather_api_errors(raw_data: dict[str, object]) -> list[str]:
    """Walk all top-level sections and collect any error strings into a list."""
    errors: list[str] = []
    for value in raw_data.values():
        if isinstance(value, dict):
            err = value.get("error")
            if isinstance(err, str):
                errors.append(err)
    return errors


def _format_prior_knowledge_error(exc: Exception) -> str:
    return f"prior_knowledge: {type(exc).__name__}: {str(exc)[:100]}"


def _format_strategy_prior_art_error(exc: Exception) -> str:
    return f"strategy_prior_art: {type(exc).__name__}: {str(exc)[:100]}"


async def _collect_literature_data(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    pubmed_result, citation_result, negative_result = await asyncio.gather(
        _safe_search_and_count(client, gene, disease),
        _safe_get_citation_count(client, gene, disease),
        _safe_search_negative_evidence(client, gene, disease),
    )
    pmids = pubmed_result.get("pmids", [])
    errors = _collect_errors(pubmed_result, citation_result, negative_result)
    result: dict[str, object] = {
        "gene": gene,
        "disease": disease,
        "pmid_count": _as_int(pubmed_result.get("pmid_count")),
        "pmids": pmids if isinstance(pmids, list) else [],
        "total_citations": _as_int(citation_result.get("total_citations")),
        "influential_citations": _as_int(citation_result.get("influential_citations")),
        "negative_count": _as_int(negative_result.get("negative_count")),
        "source": "pubmed+semantic_scholar",
    }
    if errors:
        result["error"] = "; ".join(errors)
    return result


async def _collect_target_data(
    client: httpx.AsyncClient,
    gene: str,
) -> dict[str, object]:
    identifier = gene
    resolved, resolve_error = await _safe_resolve_gene_to_ensembl(client, gene)
    if resolved is not None:
        identifier = resolved

    target = await _safe_get_target_info(client, identifier)
    target["gene"] = gene

    # Propagate resolve error into target dict
    if resolve_error is not None:
        existing = target.get("error")
        if isinstance(existing, str):
            target["error"] = f"{existing}; {resolve_error}"
        else:
            target["error"] = resolve_error

    return target


async def _safe_search_and_count(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.pubmed import search_and_count  # lazy import (M6)

    ck = cache_key("pubmed", gene, disease)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await search_and_count(client, gene, disease)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "disease": disease,
            "pmid_count": 0,
            "pmids": [],
            "source": "pubmed",
            "error": f"pubmed: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, cast("dict[str, object]", result))
    return cast("dict[str, object]", result)


async def _safe_search_negative_evidence(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.pubmed import search_negative_evidence  # lazy import (M6)

    ck = cache_key("pubmed_neg", gene, disease)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await search_negative_evidence(client, gene, disease)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "disease": disease,
            "negative_count": 0,
            "source": "pubmed_negative",
            "error": f"pubmed_negative: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, cast("dict[str, object]", result))
    return cast("dict[str, object]", result)


async def _safe_get_citation_count(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.semantic_scholar import get_citation_count  # lazy import (M6)

    ck = cache_key("s2", gene, disease)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await get_citation_count(client, gene, disease)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "disease": disease,
            "paper_count": 0,
            "total_citations": 0,
            "influential_citations": 0,
            "source": "semantic_scholar",
            "error": f"semantic_scholar: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, cast("dict[str, object]", result))
    return cast("dict[str, object]", result)


async def _safe_get_tissue_expression(
    client: httpx.AsyncClient,
    gene: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.hpa import get_tissue_expression  # lazy import (M6)

    ck = cache_key("hpa", gene)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await get_tissue_expression(client, gene)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "tissues": [],
            "source": "hpa",
            "error": f"hpa: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, result)
    return result


async def _safe_get_gtex_expression(
    client: httpx.AsyncClient,
    gene: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.gtex import get_tissue_expression  # lazy import (M6)

    ck = cache_key("gtex", gene)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await get_tissue_expression(client, gene)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "tissues": [],
            "source": "gtex",
            "error": f"gtex: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, result)
    return result


async def _safe_get_interaction_partners(
    client: httpx.AsyncClient,
    gene: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.string_db import get_interaction_partners  # lazy import (M6)

    ck = cache_key("string", gene)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await get_interaction_partners(client, gene)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "interactions": [],
            "interaction_count": 0,
            "source": "string",
            "error": f"string_db: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, result)
    return result


async def _safe_resolve_gene_to_ensembl(
    client: httpx.AsyncClient,
    gene: str,
) -> tuple[str | None, str | None]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.opentargets import resolve_gene_to_ensembl  # lazy import (M6)

    ck = cache_key("ensembl_resolve", gene)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached.get("ensembl_id"), cached.get("error")

    try:
        result = await resolve_gene_to_ensembl(client, gene)  # type: ignore[arg-type]
    except Exception as exc:
        error_msg = f"opentargets_resolve: {type(exc).__name__}: {str(exc)[:100]}"
        # Don't cache errors — let retries hit the API
        return None, error_msg
    ensembl_id = result if isinstance(result, str) else None
    get_cache().set(ck, {"ensembl_id": ensembl_id, "error": None})
    return ensembl_id, None


async def _safe_get_target_info(
    client: httpx.AsyncClient,
    identifier: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.opentargets import get_target_info  # lazy import (M6)

    ck = cache_key("opentargets", identifier)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await get_target_info(client, identifier)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": identifier,
            "tractability": [],
            "known_drugs_count": 0,
            "safety_liabilities": [],
            "source": "opentargets",
            "error": f"opentargets: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, result)
    return result


async def _safe_get_clinical_outcome(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.clinical_trials import get_clinical_outcome  # lazy import (M6)

    ck = cache_key("clinical_trials", gene, disease)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await get_clinical_outcome(client, gene, disease)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "disease": disease,
            "completed_count": 0,
            "failed_count": 0,
            "phase3_failures": 0,
            "failure_ratio": 0.0,
            "failed_trial_names": [],
            "source": "clinicaltrials_gov",
            "error": f"clinicaltrials: {type(exc).__name__}: {str(exc)[:100]}",
        }
    get_cache().set(ck, result)
    return result


async def _safe_get_gwas_evidence(
    client: httpx.AsyncClient,
    gene: str,
    disease: str,
) -> dict[str, object]:
    from biocompute.data.cache import cache_key, get_cache  # lazy import
    from biocompute.data.gwas import get_gwas_evidence  # lazy import

    ck = cache_key("gwas", gene, disease)
    cached = get_cache().get(ck)
    if cached is not None:
        return cached

    try:
        result = await get_gwas_evidence(client, gene, disease)  # type: ignore[arg-type]
    except Exception as exc:
        return {
            "gene": gene,
            "disease": disease,
            "disease_id": None,
            "scores": [],
            "hit_count": 0,
            "max_score": 0.0,
            "source": "opentargets_gwas",
            "error": f"gwas: {type(exc).__name__}: {str(exc)[:100]}",
        }

    if not isinstance(result.get("error"), str):
        get_cache().set(ck, result)
        return result

    returned = dict(result)
    returned["error"] = f"gwas: {result['error']}"
    return returned


async def _safe_get_llm_feasibility(gene: str, disease: str) -> dict[str, object]:
    """Call LLM clinical feasibility assessment in a thread (sync subprocess)."""
    try:
        from biocompute.fitness.llm_clinical import assess_clinical_feasibility

        # Run sync LLM call in thread to not block event loop
        result = await asyncio.to_thread(assess_clinical_feasibility, gene, disease)
        return result
    except Exception as exc:
        return {
            "feasibility_score": 0.5,
            "has_approved_drug": False,
            "error": f"llm_feasibility: {type(exc).__name__}: {str(exc)[:100]}",
        }


def _build_ip_section(target: dict[str, object]) -> dict[str, object]:
    known_drugs_count = _as_int(target.get("known_drugs_count"))
    liabilities = target.get("safety_liabilities", [])
    liability_count = len(liabilities) if isinstance(liabilities, list) else 0
    freedom_estimate = (
        0.8 - min(known_drugs_count * 0.1, 0.5) - min(liability_count * 0.05, 0.2)
    )
    return {
        "source": "opentargets_heuristic",
        "freedom_estimate": max(0.1, min(freedom_estimate, 0.9)),
        "known_drugs_count": known_drugs_count,
    }


def _merge_expression_sections(
    gene: str,
    hpa_expression: dict[str, object],
    gtex_expression: dict[str, object],
) -> dict[str, object]:
    from biocompute.fitness.expression import merge_expression_tissues

    hpa_tissues_value = hpa_expression.get("tissues", [])
    hpa_tissues = hpa_tissues_value if isinstance(hpa_tissues_value, list) else []
    gtex_tissues_value = gtex_expression.get("tissues", [])
    gtex_tissues = gtex_tissues_value if isinstance(gtex_tissues_value, list) else []

    merged: dict[str, object] = {
        "gene": gene,
        "tissues": merge_expression_tissues(hpa_tissues, gtex_tissues),
        "source": "hpa+gtex",
    }

    errors = _collect_errors(hpa_expression, gtex_expression)
    if errors:
        merged["error"] = "; ".join(errors)

    return merged


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _extract_dimension_sources(raw_data: dict[str, object]) -> dict[str, str]:
    """Extract source strings from raw_data sections for each fitness dimension."""
    dimension_to_section = {
        "literature_strength": "literature",
        "expression_specificity": "expression",
        "pathway_centrality": "pathway",
        "druggability": "druggability",
        "safety_profile": "safety",
        "ip_freedom": "ip",
    }
    sources: dict[str, str] = {}
    for dimension, section_key in dimension_to_section.items():
        section = raw_data.get(section_key)
        if isinstance(section, dict):
            source = section.get("source")
            if isinstance(source, str):
                sources[dimension] = source
    return sources


def _extract_dimension_raw_data(raw_data: dict[str, object]) -> dict[str, object]:
    """Map section-keyed raw data into score-dimension keys for archive storage."""
    dimension_to_section = {
        "literature_strength": "literature",
        "expression_specificity": "expression",
        "pathway_centrality": "pathway",
        "druggability": "druggability",
        "safety_profile": "safety",
        "ip_freedom": "ip",
    }
    dimension_raw_data: dict[str, object] = {}
    for dimension, section_key in dimension_to_section.items():
        section = raw_data.get(section_key)
        if isinstance(section, dict):
            dimension_raw_data[dimension] = section
    return dimension_raw_data


def _collect_all_bio_data_batch(
    hypotheses: list[TherapeuticHypothesis],
    query: DiseaseQuery,
    collect_data_fn: CollectDataFn | None,
) -> list[dict[str, object]]:
    """Collect bio data for all hypotheses. Uses asyncio.run() once for the
    entire batch when using the default collector, avoiding per-hypothesis
    asyncio.run() overhead (M3 fix)."""
    if collect_data_fn is not None and collect_data_fn is not collect_bio_data:
        return [collect_data_fn(h, query) for h in hypotheses]

    if not hypotheses:
        return []

    return asyncio.run(_collect_batch_async(hypotheses, query))


async def _collect_batch_async(
    hypotheses: list[TherapeuticHypothesis],
    query: DiseaseQuery,
) -> list[dict[str, object]]:
    """Run bio-data collection for all hypotheses concurrently in a single
    async event loop."""
    import httpx as _httpx  # lazy import (M6)

    async with _httpx.AsyncClient() as client:
        tasks = [_collect_single_hypothesis(client, h, query) for h in hypotheses]
        return await asyncio.gather(*tasks)


async def _collect_single_hypothesis(
    client: httpx.AsyncClient,
    hypothesis: TherapeuticHypothesis,
    query: DiseaseQuery,
) -> dict[str, object]:
    """Collect all bio-data sections for a single hypothesis using a shared
    async client."""
    gene = normalize_gene(hypothesis.target_gene)
    disease = query.name

    literature_task = asyncio.create_task(
        _collect_literature_data(client, gene, disease)
    )
    hpa_expression_task = asyncio.create_task(_safe_get_tissue_expression(client, gene))
    gtex_expression_task = asyncio.create_task(_safe_get_gtex_expression(client, gene))
    pathway_task = asyncio.create_task(_safe_get_interaction_partners(client, gene))
    target_task = asyncio.create_task(_collect_target_data(client, gene))
    clinical_task = asyncio.create_task(
        _safe_get_clinical_outcome(client, gene, disease)
    )
    gwas_task = asyncio.create_task(_safe_get_gwas_evidence(client, gene, disease))
    (
        literature,
        hpa_expression,
        gtex_expression,
        pathway,
        target,
        clinical,
        gwas,
    ) = await asyncio.gather(
        literature_task,
        hpa_expression_task,
        gtex_expression_task,
        pathway_task,
        target_task,
        clinical_task,
        gwas_task,
    )
    expression = _merge_expression_sections(gene, hpa_expression, gtex_expression)

    # LLM clinical feasibility — sync subprocess call run in a thread
    llm_clinical = await _safe_get_llm_feasibility(gene, disease)

    raw_data: dict[str, object] = {
        "literature": literature,
        "expression": expression,
        "pathway": pathway,
        "druggability": target,
        "safety": target,
        "ip": _build_ip_section(target),
        "clinical": clinical,
        "gwas": gwas,
        "llm_clinical": llm_clinical,
    }
    api_errors = _gather_api_errors(raw_data)
    if api_errors:
        raw_data["api_errors"] = api_errors
    return raw_data


@dataclass
class EngineConfig:
    max_generations: int = 10
    population_size: int = 30
    top_n: int = 10
    diverse_n: int = 5
    critique_top_k: int = 5
    seed_model: str = "sonnet"
    mutate_model: str = "haiku"
    critique_model: str = "sonnet"
    weights: Weights = field(default_factory=Weights)
    use_db_seed: bool = True


class EvolutionEngine:
    def __init__(self, config: EngineConfig, db_path: str):
        self.config: EngineConfig = config
        self.db_path: str = db_path
        self.store: ArchiveStore = ArchiveStore(db_path)
        self.generation: int = 0
        self.population: list[ScoredHypothesis] = []
        self.best_scores_history: list[float] = []
        self.metadata: RunMetadata = RunMetadata()

    def should_stop(self) -> bool:
        if self.generation >= self.config.max_generations:
            return True

        if len(self.best_scores_history) >= 3:
            last_three = self.best_scores_history[-3:]
            if max(last_three) - min(last_three) < 0.01:
                return True

        return False

    def run(
        self,
        query: DiseaseQuery,
        collect_data_fn: CollectDataFn | None = None,
    ) -> DiscoveryResult:
        self.metadata.started_at = datetime.now()
        collector = collect_data_fn or collect_bio_data

        try:
            seed_population = generate_seed_population(
                query,
                n=self.config.population_size,
                model=self.config.seed_model,
            )
            if not seed_population:
                raise RuntimeError("Seed generation produced no hypotheses")

            # Supplement LLM seeds with database-sourced candidates
            if self.config.use_db_seed:
                from biocompute.search.db_seed import db_seed_hypotheses  # lazy import

                db_n = min(5, self.config.population_size // 2)
                db_seeds = db_seed_hypotheses(query.name, n=db_n)
                existing_genes = {h.target_gene for h in seed_population}
                for h in db_seeds:
                    if h.target_gene not in existing_genes:
                        seed_population.append(h)
                        existing_genes.add(h.target_gene)

            self.population = self._evaluate_batch(
                seed_population,
                query,
                collector,
            )
            self._archive_generation()
            self._record_best()
            self._print_top(5)

            while not self.should_stop():
                self.generation += 1

                survivors = select_survivors(
                    self.population,
                    top_n=self.config.top_n,
                    diverse_n=self.config.diverse_n,
                )

                top_k = sorted(
                    survivors,
                    key=lambda scored: scored.fitness,
                    reverse=True,
                )[: self.config.critique_top_k]
                for scored in top_k:
                    critiques = critique_hypothesis(
                        scored,
                        query,
                        model=self.config.critique_model,
                    )
                    scored.critiques = critiques
                    for critique in critiques:
                        self.store.save_critique(
                            scored.hypothesis.id,
                            critique,
                            self.config.critique_model,
                        )

                children_by_parent: list[list[TherapeuticHypothesis]] = []
                for parent in survivors:
                    children = mutate_hypothesis(
                        parent,
                        query,
                        generation=self.generation,
                        model=self.config.mutate_model,
                    )
                    children_by_parent.append(children)

                new_hypotheses = _cap_generation_children(
                    children_by_parent,
                    limit=self.config.population_size,
                )

                new_scored = self._evaluate_batch(
                    new_hypotheses,
                    query,
                    collector,
                )
                self.population = survivors + new_scored
                self._archive_generation()
                self._record_best()
                self._print_top(5)

            final_candidates = self._build_final_candidates()
            self._enrich_prior_knowledge(query, final_candidates)
            self._enrich_strategy_prior_art(query, final_candidates)

            self.metadata.finished_at = datetime.now()
            self.metadata.generations_run = self.generation
            self.metadata.total_hypotheses = len(self.store.get_top_hypotheses(n=9999))

            return DiscoveryResult(
                query=query,
                candidates=final_candidates,
                metadata=self.metadata,
                db_path=self.db_path,
            )
        finally:
            self.store.close()

    def _evaluate_batch(
        self,
        hypotheses: list[TherapeuticHypothesis],
        query: DiseaseQuery,
        collect_data_fn: CollectDataFn | None,
    ) -> list[ScoredHypothesis]:
        all_raw_data = _collect_all_bio_data_batch(hypotheses, query, collect_data_fn)

        scored_population: list[ScoredHypothesis] = []
        for hypothesis, raw_data in zip(hypotheses, all_raw_data):
            scored = evaluate_all_dimensions(
                hypothesis,
                query,
                raw_data,
                self.config.weights,
            )
            dimension_sources = _extract_dimension_sources(raw_data)
            self.store.save_hypothesis(
                hypothesis,
                scored.scores,
                scored.fitness,
                dimension_sources=dimension_sources,
                dimension_raw_data=_extract_dimension_raw_data(raw_data),
            )
            # Propagate API errors from raw_data to scored hypothesis
            api_errors = raw_data.get("api_errors")
            if isinstance(api_errors, list):
                scored.api_errors = [str(e) for e in api_errors]

            for evidence in scored.evidence:
                self.store.save_evidence(hypothesis.id, evidence)
            scored_population.append(scored)

        return scored_population

    def _archive_generation(self) -> None:
        """Generation archival currently happens during _evaluate_batch().

        Each hypothesis is persisted with its generation number when
        save_hypothesis() is called. This hook exists to mark the logical
        generation boundary and leaves room for future per-generation snapshot
        artifacts without changing the current persistence contract.
        """
        return None

    def _record_best(self) -> None:
        if not self.population:
            return

        best = max(self.population, key=lambda scored: scored.fitness)
        self.best_scores_history.append(best.fitness)

    def _print_top(self, n: int) -> None:
        top = sorted(self.population, key=lambda scored: scored.fitness, reverse=True)[
            :n
        ]
        for index, scored in enumerate(top, start=1):
            hypothesis = scored.hypothesis
            line = f"#{index} {hypothesis.target_gene} {hypothesis.modality} {hypothesis.delivery} fitness={scored.fitness:.3f}"
            if scored.api_errors:
                # Extract short API names from error strings (e.g. "pubmed" from "pubmed: RuntimeError: ...")
                api_names = [e.split(":")[0] for e in scored.api_errors]
                line += f" [!] {len(scored.api_errors)} API error(s) ({', '.join(api_names)})"
            print(line)

    def _build_final_candidates(self) -> list[ScoredHypothesis]:
        current_by_id = {scored.hypothesis.id: scored for scored in self.population}
        candidates: list[ScoredHypothesis] = []

        for row in self.store.get_top_hypotheses(n=20):
            hypothesis_id = row.get("id")
            if not isinstance(hypothesis_id, str):
                continue

            evidence = self._load_evidence(hypothesis_id)
            critiques = self._load_critiques(hypothesis_id)
            archive_scores = self.store.get_scores(hypothesis_id)
            current = current_by_id.get(hypothesis_id)
            if current is not None:
                candidates.append(
                    ScoredHypothesis(
                        hypothesis=current.hypothesis,
                        fitness=current.fitness,
                        scores=archive_scores or current.scores,
                        evidence=evidence or current.evidence,
                        critiques=critiques or current.critiques,
                        api_errors=list(current.api_errors),
                        prior_knowledge=current.prior_knowledge,
                    )
                )
                continue

            candidates.append(
                ScoredHypothesis(
                    hypothesis=self._row_to_hypothesis(row),
                    fitness=self._row_float(row, "fitness_total"),
                    scores=archive_scores or FitnessScores(),
                    evidence=evidence,
                    critiques=critiques,
                )
            )

        return candidates

    def _enrich_prior_knowledge(
        self,
        query: DiseaseQuery,
        candidates: list[ScoredHypothesis],
    ) -> None:
        top_candidates = self._top_unique_gene_candidates_for_prior_knowledge(
            candidates
        )
        if not top_candidates:
            return

        asyncio.run(self._enrich_prior_knowledge_async(query, top_candidates))

    def _enrich_strategy_prior_art(
        self,
        query: DiseaseQuery,
        candidates: list[ScoredHypothesis],
    ) -> None:
        top_candidates = candidates[:3]
        if not top_candidates:
            return

        asyncio.run(self._enrich_strategy_prior_art_async(query, top_candidates))

    def _top_unique_gene_candidates_for_prior_knowledge(
        self,
        candidates: list[ScoredHypothesis],
    ) -> list[ScoredHypothesis]:
        top_candidates: list[ScoredHypothesis] = []
        seen_genes: set[str] = set()

        for candidate in candidates:
            gene = candidate.hypothesis.target_gene
            if gene in seen_genes:
                continue
            top_candidates.append(candidate)
            seen_genes.add(gene)
            if len(top_candidates) >= PRIOR_KNOWLEDGE_TOP_N:
                break

        return top_candidates

    async def _enrich_prior_knowledge_async(
        self,
        query: DiseaseQuery,
        candidates: list[ScoredHypothesis],
    ) -> None:
        from biocompute.data.pubmed_abstracts import fetch_abstracts  # lazy import
        from biocompute.fitness.prior_knowledge import (
            assess_prior_knowledge,
        )  # lazy import

        async with httpx.AsyncClient() as client:
            for candidate in candidates:
                try:
                    pmids = self._load_literature_pmids(candidate.hypothesis.id)
                    abstracts = await fetch_abstracts(client, pmids)
                    candidate.prior_knowledge = await asyncio.to_thread(
                        assess_prior_knowledge,
                        candidate.hypothesis.target_gene,
                        query.name,
                        cast(list[dict[str, str]], abstracts),
                    )
                    self.store.save_prior_knowledge(
                        candidate.hypothesis.id,
                        candidate.prior_knowledge,
                    )
                except Exception as exc:
                    candidate.api_errors.append(_format_prior_knowledge_error(exc))

    async def _enrich_strategy_prior_art_async(
        self,
        query: DiseaseQuery,
        candidates: list[ScoredHypothesis],
    ) -> None:
        from biocompute.fitness import strategy_prior_art as strategy_prior_art_module
        from biocompute.models import StrategyPriorArt

        generate_strategy_queries = cast(
            Callable[[str, str, str], list[str]],
            getattr(strategy_prior_art_module, "_generate_strategy_queries"),
        )
        search_strategy_abstracts = cast(
            Callable[[httpx.AsyncClient, list[str]], Awaitable[list[dict[str, str]]]],
            getattr(strategy_prior_art_module, "_search_strategy_abstracts"),
        )
        assess_strategy_prior_art = strategy_prior_art_module.assess_strategy_prior_art

        async with httpx.AsyncClient() as client:
            for candidate in candidates:
                try:
                    queries = await asyncio.to_thread(
                        generate_strategy_queries,
                        candidate.hypothesis.target_gene,
                        query.name,
                        candidate.hypothesis.modality,
                    )
                    abstracts = await search_strategy_abstracts(client, queries)
                    assessment = await asyncio.to_thread(
                        assess_strategy_prior_art,
                        candidate.hypothesis.target_gene,
                        query.name,
                        candidate.hypothesis.modality,
                        abstracts,
                    )
                    key_papers = [
                        f"PMID:{pmid}"
                        for abstract in abstracts
                        for pmid in [str(abstract.get("pmid", "")).strip()]
                        if pmid
                    ]
                    candidate.strategy_prior_art = StrategyPriorArt(
                        strategy=f"{candidate.hypothesis.target_gene} modulation",
                        disease_class=query.name,
                        prior_studies=assessment["prior_studies"],
                        modality_status=assessment["modality_status"],
                        our_differentiation=assessment["our_differentiation"],
                        key_papers=key_papers,
                        summary=assessment["summary"],
                    )
                    self.store.save_strategy_prior_art(
                        candidate.hypothesis.id,
                        candidate.strategy_prior_art,
                    )
                except Exception as exc:
                    candidate.api_errors.append(_format_strategy_prior_art_error(exc))

    def _load_literature_pmids(self, hypothesis_id: str) -> list[str]:
        for dimension in self.store.get_scores_with_metadata(hypothesis_id):
            if dimension.get("dimension") != "literature_strength":
                continue

            raw_data = dimension.get("raw_data")
            if not isinstance(raw_data, dict):
                return []

            pmids = raw_data.get("pmids")
            if not isinstance(pmids, list):
                return []

            normalized_pmids: list[str] = []
            for pmid in pmids:
                text = str(pmid).strip()
                if text:
                    normalized_pmids.append(text)
            return normalized_pmids

        return []

    def _load_evidence(self, hypothesis_id: str) -> list[Evidence]:
        evidence_rows = self.store.get_evidence(hypothesis_id)
        evidence: list[Evidence] = []
        for row in evidence_rows:
            source_type = row.get("source_type")
            source_id = row.get("source_id")
            summary = row.get("summary")
            relevance_score = row.get("relevance_score")
            if not isinstance(source_type, str):
                continue
            if not isinstance(source_id, str):
                continue
            if not isinstance(summary, str):
                continue

            evidence.append(
                Evidence(
                    source_type=source_type,
                    source_id=source_id,
                    summary=summary,
                    relevance_score=(
                        float(relevance_score)
                        if isinstance(relevance_score, int | float)
                        else 0.0
                    ),
                )
            )

        return evidence

    def _load_critiques(self, hypothesis_id: str) -> list[str]:
        critiques: list[str] = []
        for row in self.store.get_critiques(hypothesis_id):
            critique_text = row.get("critique_text")
            if isinstance(critique_text, str):
                critiques.append(critique_text)
        return critiques

    def _row_to_hypothesis(
        self,
        row: dict[str, object],
    ) -> TherapeuticHypothesis:
        return TherapeuticHypothesis(
            target_gene=self._row_str(row, "target_gene"),
            modality=self._row_str(row, "modality"),
            delivery=self._row_str(row, "delivery"),
            duration=self._row_str(row, "duration"),
            tissue_context=self._row_str(row, "tissue_context"),
            id=self._row_str(row, "id"),
            parent_id=self._row_optional_str(row, "parent_id"),
            mutation_type=self._row_str(row, "mutation_type", default="seed"),
            generation=self._row_int(row, "generation"),
        )

    def _row_str(
        self,
        row: dict[str, object],
        key: str,
        default: str = "unknown",
    ) -> str:
        value = row.get(key)
        return value if isinstance(value, str) else default

    def _row_optional_str(self, row: dict[str, object], key: str) -> str | None:
        value = row.get(key)
        return value if isinstance(value, str) else None

    def _row_int(self, row: dict[str, object], key: str, default: int = 0) -> int:
        value = row.get(key)
        return int(value) if isinstance(value, int | float) else default

    def _row_float(
        self, row: dict[str, object], key: str, default: float = 0.0
    ) -> float:
        value = row.get(key)
        return float(value) if isinstance(value, int | float) else default
