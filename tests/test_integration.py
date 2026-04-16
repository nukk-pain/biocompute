# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import json
import sqlite3
import tempfile
from unittest.mock import AsyncMock, patch

from biocompute.archive.export import export_for_neuroregen
from biocompute.archive.report import generate_report
from biocompute.engine import EngineConfig, EvolutionEngine
from biocompute.models import DiseaseQuery, EvidenceMaturity, PriorKnowledge
from biocompute.verification.literature import verify_targets

MOCK_SEED_RESPONSE = json.dumps(
    {
        "hypotheses": [
            {
                "target_gene": "CXCL12",
                "modality": "VHH",
                "delivery": "local injection",
                "duration": "single-dose",
                "tissue_context": "scar tissue",
                "rationale": "CXCL12 drives nociceptor sprouting",
            },
            {
                "target_gene": "NGF",
                "modality": "mAb",
                "delivery": "systemic IV",
                "duration": "chronic",
                "tissue_context": "joint",
                "rationale": "NGF promotes pain signaling",
            },
            {
                "target_gene": "BDNF",
                "modality": "siRNA",
                "delivery": "topical LNP",
                "duration": "acute",
                "tissue_context": "scar tissue",
                "rationale": "BDNF involved in central sensitization",
            },
        ]
    }
)

MOCK_MUTATE_RESPONSE = json.dumps(
    {
        "mutations": [
            {
                "target_gene": "CXCR4",
                "modality": "VHH",
                "delivery": "local injection",
                "duration": "single-dose",
                "tissue_context": "scar tissue",
                "mutation_type": "pathway_neighbor",
                "rationale": "Target receptor instead",
            }
        ]
    }
)

MOCK_CRITIQUE_RESPONSE = json.dumps(
    {
        "critiques": [
            "No in vivo evidence for fascial CXCL12 overexpression",
            "VHH penetration into deep fascia is unproven",
        ]
    }
)


def mock_query_llm(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    del model, max_tokens

    if "Propose exactly" in prompt and "hypotheses" in (system_prompt or ""):
        return MOCK_SEED_RESPONSE
    if "PATHWAY_NEIGHBOR" in prompt:
        return MOCK_MUTATE_RESPONSE
    if (
        "skeptical" in (system_prompt or "").lower()
        or "critic" in (system_prompt or "").lower()
    ):
        return MOCK_CRITIQUE_RESPONSE
    return "{}"


def mock_collect_data(*_args, **_kwargs) -> dict[str, object]:
    return {
        "literature": {
            "pmid_count": 10,
            "total_citations": 120,
            "influential_citations": 4,
            "source": "pubmed+semantic_scholar",
        },
        "expression": {
            "tissues": [
                {"Tissue": "scar tissue", "Level": "High"},
                {"Tissue": "liver", "Level": "Low"},
            ]
        },
        "pathway": {
            "interaction_count": 12,
            "interactions": [{"score": 0.95}] * 12,
        },
        "druggability": {
            "tractability": [{"modality": "AB", "value": True}],
            "known_drugs_count": 1,
            "source": "opentargets",
        },
        "safety": {
            "safety_liabilities": [{"event": "cardiotoxicity"}],
            "source": "opentargets",
        },
        "ip": {"source": "opentargets_heuristic", "freedom_estimate": 0.6},
    }


def mock_collect_data_with_pmids(*_args, **_kwargs) -> dict[str, object]:
    raw_data = mock_collect_data()
    raw_data["literature"] = {
        "pmid_count": 10,
        "pmids": ["12345", "67890"],
        "total_citations": 120,
        "influential_citations": 4,
        "source": "pubmed+semantic_scholar",
    }
    return raw_data


@patch("biocompute.search.seed.query_llm", side_effect=mock_query_llm)
@patch("biocompute.search.mutate.query_llm", side_effect=mock_query_llm)
@patch("biocompute.search.critique.query_llm", side_effect=mock_query_llm)
def test_full_evolution_loop(mock_crit, mock_mut, mock_seed) -> None:
    query = DiseaseQuery(
        name="Myofascial Pain Syndrome",
        description="Chronic pain from nerve hyperinnervation in fascial scar tissue",
        keywords=["scar", "hyperinnervation", "fascia"],
    )
    config = EngineConfig(
        max_generations=2,
        population_size=3,
        top_n=2,
        diverse_n=1,
        critique_top_k=1,
        use_db_seed=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = EvolutionEngine(config, f"{tmp_dir}/test.db")
        result = engine.run(query, collect_data_fn=mock_collect_data)

    assert len(result.candidates) > 0
    assert result.metadata.generations_run >= 1
    assert mock_seed.called
    assert mock_mut.called
    assert mock_crit.called

    report = generate_report(result)
    assert "Myofascial Pain Syndrome" in report
    assert len(report) > 100

    assert any(scored.evidence or scored.critiques for scored in result.candidates)


@patch("biocompute.search.seed.query_llm", side_effect=mock_query_llm)
@patch("biocompute.search.mutate.query_llm", side_effect=mock_query_llm)
@patch("biocompute.search.critique.query_llm", side_effect=mock_query_llm)
def test_full_evolution_loop_persists_generation_history(
    mock_crit, mock_mut, mock_seed
) -> None:
    query = DiseaseQuery(
        name="Myofascial Pain Syndrome",
        description="Chronic pain from nerve hyperinnervation in fascial scar tissue",
        keywords=["scar", "hyperinnervation", "fascia"],
    )
    config = EngineConfig(
        max_generations=2,
        population_size=3,
        top_n=2,
        diverse_n=1,
        critique_top_k=1,
        use_db_seed=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = f"{tmp_dir}/test.db"
        engine = EvolutionEngine(config, db_path)
        _ = engine.run(query, collect_data_fn=mock_collect_data)

        conn = sqlite3.connect(db_path)
        generations = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT generation FROM hypotheses ORDER BY generation"
            ).fetchall()
        ]
        conn.close()

    assert generations == [0, 1, 2]
    assert mock_seed.called
    assert mock_mut.called
    assert mock_crit.called


MOCK_MUTATE_3_CHILDREN_RESPONSE = json.dumps(
    {
        "mutations": [
            {
                "target_gene": "CXCR4",
                "modality": "VHH",
                "delivery": "local injection",
                "duration": "single-dose",
                "tissue_context": "scar tissue",
                "mutation_type": "pathway_neighbor",
                "rationale": "Target receptor instead",
            },
            {
                "target_gene": "CXCR5",
                "modality": "VHH",
                "delivery": "local injection",
                "duration": "single-dose",
                "tissue_context": "scar tissue",
                "mutation_type": "pathway_neighbor",
                "rationale": "Target receptor instead",
            },
            {
                "target_gene": "CXCR6",
                "modality": "VHH",
                "delivery": "local injection",
                "duration": "single-dose",
                "tissue_context": "scar tissue",
                "mutation_type": "pathway_neighbor",
                "rationale": "Target receptor instead",
            },
        ]
    }
)


def mock_query_llm_3_children(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    del model, max_tokens

    if "Propose exactly" in prompt and "hypotheses" in (system_prompt or ""):
        return MOCK_SEED_RESPONSE
    if "PATHWAY_NEIGHBOR" in prompt:
        return MOCK_MUTATE_3_CHILDREN_RESPONSE
    if (
        "skeptical" in (system_prompt or "").lower()
        or "critic" in (system_prompt or "").lower()
    ):
        return MOCK_CRITIQUE_RESPONSE
    return "{}"


@patch("biocompute.search.seed.query_llm", side_effect=mock_query_llm)
@patch("biocompute.search.mutate.query_llm", side_effect=mock_query_llm_3_children)
@patch("biocompute.search.critique.query_llm", side_effect=mock_query_llm)
def test_hypothesis_growth_is_bounded(mock_crit, mock_mut, mock_seed) -> None:
    query = DiseaseQuery(
        name="Myofascial Pain Syndrome",
        description="Chronic pain from nerve hyperinnervation in fascial scar tissue",
        keywords=["scar", "hyperinnervation", "fascia"],
    )
    config = EngineConfig(
        max_generations=2,
        population_size=3,
        top_n=2,
        diverse_n=1,
        critique_top_k=1,
        use_db_seed=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = f"{tmp_dir}/test.db"
        engine = EvolutionEngine(config, db_path)
        _ = engine.run(query, collect_data_fn=mock_collect_data)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        conn.close()

    # seed_count (3) + generations (2) * population_size (3) = 9
    assert count == 9, f"Expected 9 hypotheses, but found {count}"


@patch("biocompute.search.seed.query_llm", side_effect=mock_query_llm)
@patch("biocompute.search.mutate.query_llm", side_effect=mock_query_llm)
@patch("biocompute.search.critique.query_llm", side_effect=mock_query_llm)
def test_full_evolution_loop_propagates_prior_knowledge_to_archive_surfaces(
    mock_crit, mock_mut, mock_seed
) -> None:
    query = DiseaseQuery(
        name="Hypertrophic Scarring",
        description="Persistent fibrotic signaling in mature scar tissue",
        keywords=["scar", "fibrosis", "tgfb"],
    )
    config = EngineConfig(
        max_generations=1,
        population_size=3,
        top_n=2,
        diverse_n=1,
        critique_top_k=1,
    )

    def mock_assess_prior_knowledge(
        gene: str, disease: str, abstracts: list[dict[str, str]]
    ) -> PriorKnowledge:
        assert abstracts
        return PriorKnowledge(
            gene=gene,
            disease=disease,
            maturity=EvidenceMaturity.L2_IN_VITRO,
            known_facts=[
                f"{gene} has reproducible mechanistic support in fibrosis models."
            ],
            attempted_approaches=[
                f"{gene}-directed modulation has been explored preclinically."
            ],
            gaps=[f"Disease-specific delivery for {gene} remains unresolved."],
            key_papers=["PMID:12345"],
            summary=f"{gene} prior knowledge was attached during discovery.",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = f"{tmp_dir}/test.db"
        with open(f"{tmp_dir}/config.json", "w", encoding="utf-8") as config_file:
            json.dump(
                {
                    "disease": query.name,
                    "description": query.description,
                },
                config_file,
            )

        engine = EvolutionEngine(config, db_path)
        with (
            patch(
                "biocompute.data.pubmed_abstracts.fetch_abstracts",
                AsyncMock(
                    return_value=[
                        {
                            "pmid": "12345",
                            "title": "SMAD3 fibrosis biology",
                            "abstract": "Prior work establishes fibrotic signaling.",
                        }
                    ]
                ),
            ),
            patch(
                "biocompute.fitness.prior_knowledge.assess_prior_knowledge",
                side_effect=mock_assess_prior_knowledge,
            ),
            patch(
                "biocompute.verification.literature.search_pubmed",
                AsyncMock(return_value=[]),
            ),
            patch(
                "biocompute.verification.literature.search_papers",
                AsyncMock(return_value=[]),
            ),
            patch(
                "biocompute.verification.literature.get_clinical_outcome",
                AsyncMock(
                    return_value={
                        "completed_count": 0,
                        "failed_count": 0,
                        "phase3_failures": 0,
                        "failure_ratio": 0.0,
                        "failed_trial_names": [],
                        "source": "clinicaltrials_gov",
                    }
                ),
            ),
            patch(
                "biocompute.verification.literature.assess_clinical_feasibility",
                return_value={
                    "has_approved_drug": False,
                    "approved_drugs": [],
                    "has_phase3_failure": False,
                    "failed_drugs": [],
                    "feasibility_score": 0.5,
                    "rationale": "LLM assessment unavailable",
                    "drug_verification": "no_reference",
                    "verified_drugs": [],
                },
            ),
        ):
            result = engine.run(query, collect_data_fn=mock_collect_data_with_pmids)
            report = generate_report(result)
            exported = export_for_neuroregen(db_path, top_n=1)
            verifications = verify_targets(db_path, top_n=1)

    assert result.candidates
    top_candidate = result.candidates[0]
    assert top_candidate.prior_knowledge is not None
    assert top_candidate.prior_knowledge.gene == top_candidate.hypothesis.target_gene
    assert top_candidate.prior_knowledge.summary.endswith("attached during discovery.")
    assert "Prior Knowledge" in report
    assert top_candidate.prior_knowledge.summary in report

    exported_candidate = exported["candidates"][0]
    assert exported_candidate["gene"]["symbol"] == top_candidate.hypothesis.target_gene
    assert exported_candidate["prior_knowledge"] is not None
    assert exported_candidate["prior_knowledge"]["summary"] == (
        top_candidate.prior_knowledge.summary
    )

    assert len(verifications) == 1
    assert verifications[0].gene == top_candidate.hypothesis.target_gene
    assert verifications[0].prior_knowledge is not None
    assert (
        verifications[0].prior_knowledge.summary
        == top_candidate.prior_knowledge.summary
    )
    assert mock_seed.called
    assert mock_mut.called
    assert mock_crit.called
