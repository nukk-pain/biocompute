"""Tests for cross-indication batch analysis."""

from __future__ import annotations

import json
import os
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from biocompute.archive.batch_analysis import (
    CompetitiveLandscape,
    IndicationSummary,
    IndicationTarget,
    PlatformTarget,
    _abbreviate,
    _format_report,
    check_competitive_landscape,
    extract_platform_targets,
    generate_batch_analysis,
    load_indication_summary,
)


def _create_run_dir(
    tmp_path: str,
    disease_name: str,
    targets: list[tuple[str, str, float, dict[str, float]]],
    dir_suffix: str = "",
) -> str:
    """Create a mock run directory with SQLite DB and config.json.

    Args:
        tmp_path: Base temp directory.
        disease_name: Name of the disease.
        targets: List of (gene, modality, fitness_total, dimension_scores) tuples.
            dimension_scores maps dimension name -> score value.
        dir_suffix: Optional suffix for the run directory name.

    Returns:
        Path to the created run directory.
    """
    slug = disease_name.lower().replace(" ", "-") + dir_suffix
    run_dir = os.path.join(tmp_path, slug)
    os.makedirs(run_dir, exist_ok=True)

    # Write config.json
    config = {"disease": disease_name, "description": f"Description of {disease_name}"}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Create SQLite DB
    db_path = os.path.join(run_dir, "run.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS hypotheses (
            id TEXT PRIMARY KEY,
            generation INTEGER,
            target_gene TEXT,
            modality TEXT,
            delivery TEXT,
            duration TEXT,
            tissue_context TEXT,
            fitness_total REAL,
            parent_id TEXT,
            mutation_type TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS scores (
            hypothesis_id TEXT,
            dimension TEXT,
            score REAL,
            source TEXT,
            raw_data JSON,
            FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
        );
    """)

    default_dims = {
        "literature_strength": 0.5,
        "expression_specificity": 0.5,
        "pathway_centrality": 0.5,
        "druggability": 0.5,
        "safety_profile": 0.5,
        "ip_freedom": 0.5,
    }

    for i, (gene, modality, fitness_total, dim_scores) in enumerate(targets):
        hyp_id = f"{slug}-{gene}-{i}"
        conn.execute(
            """INSERT INTO hypotheses
               (id, generation, target_gene, modality, delivery, duration,
                tissue_context, fitness_total, parent_id, mutation_type, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (hyp_id, 1, gene, modality, "IV", "chronic", "tissue", fitness_total,
             None, "seed", "2026-01-01T00:00:00"),
        )
        dims = {**default_dims, **dim_scores}
        for dim_name, score in dims.items():
            conn.execute(
                "INSERT INTO scores (hypothesis_id, dimension, score, source, raw_data) "
                "VALUES (?, ?, ?, ?, ?)",
                (hyp_id, dim_name, score, "test", None),
            )
    conn.commit()
    conn.close()

    return run_dir


class TestLoadIndicationSummary:
    def test_loads_targets_from_db(self, tmp_path: str) -> None:
        run_dir = _create_run_dir(
            str(tmp_path),
            "Test Disease",
            [
                ("GENE1", "mAb", 0.8, {"literature_strength": 0.9, "pathway_centrality": 0.8}),
                ("GENE2", "siRNA", 0.6, {"literature_strength": 0.5}),
            ],
        )

        summary = load_indication_summary(run_dir)

        assert summary.disease_name == "Test Disease"
        assert len(summary.targets) == 2
        assert summary.targets[0].gene == "GENE1"
        assert summary.targets[0].fitness > summary.targets[1].fitness

    def test_deduplicates_by_gene(self, tmp_path: str) -> None:
        # Same gene appearing twice with different fitness
        run_dir = _create_run_dir(
            str(tmp_path),
            "Dedup Test",
            [
                ("SMAD3", "mAb", 0.9, {"literature_strength": 0.95}),
                ("SMAD3", "siRNA", 0.7, {"literature_strength": 0.5}),
                ("CXCR4", "peptide", 0.6, {}),
            ],
        )

        summary = load_indication_summary(run_dir, top_n=5)

        genes = [t.gene for t in summary.targets]
        assert genes.count("SMAD3") == 1  # deduplicated
        assert "CXCR4" in genes

    def test_respects_top_n(self, tmp_path: str) -> None:
        targets = [(f"GENE{i}", "mAb", 0.5 + i * 0.01, {}) for i in range(10)]
        run_dir = _create_run_dir(str(tmp_path), "TopN Test", targets)

        summary = load_indication_summary(run_dir, top_n=3)
        assert len(summary.targets) == 3

    def test_fallback_disease_name(self, tmp_path: str) -> None:
        """Uses directory basename when config.json is missing."""
        run_dir = _create_run_dir(str(tmp_path), "Fallback", [("X", "mAb", 0.5, {})])
        # Remove config.json
        os.remove(os.path.join(run_dir, "config.json"))

        summary = load_indication_summary(run_dir)
        assert summary.disease_name == os.path.basename(run_dir)


class TestExtractPlatformTargets:
    def test_finds_common_targets(self) -> None:
        summaries = [
            IndicationSummary(
                disease_name="Disease A",
                run_dir="/a",
                targets=[
                    IndicationTarget("SMAD3", 0.8, "mAb"),
                    IndicationTarget("CXCR4", 0.7, "siRNA"),
                ],
            ),
            IndicationSummary(
                disease_name="Disease B",
                run_dir="/b",
                targets=[
                    IndicationTarget("SMAD3", 0.85, "mAb"),
                    IndicationTarget("GENE2", 0.6, "peptide"),
                ],
            ),
        ]

        platform = extract_platform_targets(summaries)

        assert len(platform) == 1
        assert platform[0].gene == "SMAD3"
        assert platform[0].hit_count == 2
        assert platform[0].total_indications == 2
        assert abs(platform[0].avg_fitness - 0.825) < 0.01

    def test_no_common_targets(self) -> None:
        summaries = [
            IndicationSummary(
                disease_name="A",
                run_dir="/a",
                targets=[IndicationTarget("GENE1", 0.8, "mAb")],
            ),
            IndicationSummary(
                disease_name="B",
                run_dir="/b",
                targets=[IndicationTarget("GENE2", 0.7, "mAb")],
            ),
        ]

        platform = extract_platform_targets(summaries)
        assert len(platform) == 0

    def test_sorting_by_hit_count(self) -> None:
        summaries = [
            IndicationSummary("A", "/a", [
                IndicationTarget("G1", 0.9, "mAb"),
                IndicationTarget("G2", 0.8, "mAb"),
            ]),
            IndicationSummary("B", "/b", [
                IndicationTarget("G1", 0.85, "mAb"),
                IndicationTarget("G2", 0.75, "mAb"),
            ]),
            IndicationSummary("C", "/c", [
                IndicationTarget("G1", 0.88, "mAb"),
            ]),
        ]

        platform = extract_platform_targets(summaries)
        # G1 appears in 3 indications, G2 in 2
        assert platform[0].gene == "G1"
        assert platform[0].hit_count == 3
        assert platform[1].gene == "G2"
        assert platform[1].hit_count == 2


class TestCheckCompetitiveLandscape:
    @pytest.mark.asyncio
    async def test_returns_landscape_data(self) -> None:
        mock_resolve = AsyncMock(return_value="ENSG00000123456")
        mock_info = AsyncMock(return_value={
            "gene": "ENSG00000123456",
            "known_drugs_count": 5,
            "tractability": [
                {"modality": "SM", "value": True},
                {"modality": "AB", "value": True},
                {"modality": "Other", "value": False},
            ],
            "safety_liabilities": [],
            "source": "opentargets",
        })

        with (
            patch("biocompute.data.opentargets.resolve_gene_to_ensembl", mock_resolve),
            patch("biocompute.data.opentargets.get_target_info", mock_info),
        ):
            results = await check_competitive_landscape(["SMAD3"])

        assert len(results) == 1
        assert results[0].gene == "SMAD3"
        assert results[0].known_drugs_count == 5
        assert results[0].assessment == "Crowded (5 known drugs)"
        assert "SM" in results[0].tractability
        assert "AB" in results[0].tractability
        assert "Other" not in results[0].tractability

    @pytest.mark.asyncio
    async def test_blue_ocean_assessment(self) -> None:
        mock_resolve = AsyncMock(return_value="ENSG00000123456")
        mock_info = AsyncMock(return_value={
            "gene": "ENSG00000123456",
            "known_drugs_count": 0,
            "tractability": [],
            "safety_liabilities": [],
            "source": "opentargets",
        })

        with (
            patch("biocompute.data.opentargets.resolve_gene_to_ensembl", mock_resolve),
            patch("biocompute.data.opentargets.get_target_info", mock_info),
        ):
            results = await check_competitive_landscape(["NOVEL1"])

        assert results[0].assessment == "Blue ocean (no known drugs)"

    @pytest.mark.asyncio
    async def test_unresolved_gene(self) -> None:
        mock_resolve = AsyncMock(return_value=None)

        with patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl", mock_resolve,
        ):
            results = await check_competitive_landscape(["UNKNOWN"])

        assert results[0].assessment == "Unknown (gene not resolved)"

    @pytest.mark.asyncio
    async def test_api_error_handled(self) -> None:
        mock_resolve = AsyncMock(side_effect=Exception("network error"))

        with patch(
            "biocompute.data.opentargets.resolve_gene_to_ensembl", mock_resolve,
        ):
            results = await check_competitive_landscape(["BROKEN"])

        assert results[0].assessment == "Unknown (API error)"


class TestFormatReport:
    def test_markdown_structure(self) -> None:
        summaries = [
            IndicationSummary("Disease A", "/a", [
                IndicationTarget("SMAD3", 0.812, "mAb"),
                IndicationTarget("CXCR4", 0.755, "siRNA"),
            ]),
            IndicationSummary("Disease B", "/b", [
                IndicationTarget("SMAD3", 0.800, "mAb"),
                IndicationTarget("GENE2", 0.600, "peptide"),
            ]),
        ]
        platform = [
            PlatformTarget(
                gene="SMAD3",
                indications=["Disease A", "Disease B"],
                fitness_by_indication={"Disease A": 0.812, "Disease B": 0.800},
                avg_fitness=0.806,
                hit_count=2,
                total_indications=2,
            ),
        ]
        landscape = [
            CompetitiveLandscape("SMAD3", 0, ["SM"], "Blue ocean (no known drugs)"),
        ]

        report = _format_report(summaries, platform, landscape)

        assert "# Cross-Indication Batch Analysis" in report
        assert "Disease A" in report
        assert "Disease B" in report
        assert "SMAD3" in report
        assert "## Per-Indication Top Targets" in report
        assert "## Cross-Indication Target Map" in report
        assert "## Platform Targets" in report
        assert "## Competitive Landscape" in report
        assert "## Strategic Summary" in report
        assert "Blue ocean" in report
        assert "2/2" in report

    def test_no_platform_targets(self) -> None:
        summaries = [
            IndicationSummary("A", "/a", [IndicationTarget("G1", 0.8, "mAb")]),
            IndicationSummary("B", "/b", [IndicationTarget("G2", 0.7, "mAb")]),
        ]

        report = _format_report(summaries, [], None)

        assert "No targets found across multiple indications" in report
        assert "## Platform Targets" not in report


class TestAbbreviate:
    def test_short_name(self) -> None:
        assert _abbreviate("HIV") == "HIV"

    def test_multi_word(self) -> None:
        result = _abbreviate("Hypertrophic Scarring")
        assert result == "HS"

    def test_single_long_word(self) -> None:
        result = _abbreviate("Osteoarthritis", max_len=6)
        assert len(result) <= 6


class TestGenerateBatchAnalysis:
    def test_full_pipeline(self, tmp_path: str) -> None:
        run_dir_a = _create_run_dir(
            str(tmp_path),
            "Disease Alpha",
            [
                ("SMAD3", "mAb", 0.8, {"literature_strength": 0.9}),
                ("CXCR4", "siRNA", 0.7, {"literature_strength": 0.7}),
            ],
        )
        run_dir_b = _create_run_dir(
            str(tmp_path),
            "Disease Beta",
            [
                ("SMAD3", "peptide", 0.85, {"literature_strength": 0.95}),
                ("GENE3", "ASO", 0.6, {"literature_strength": 0.4}),
            ],
        )

        output_path = os.path.join(str(tmp_path), "analysis.md")
        report = generate_batch_analysis(
            [run_dir_a, run_dir_b],
            output_path,
            skip_competitive=True,
        )

        assert os.path.exists(output_path)
        assert "SMAD3" in report
        assert "Platform Targets" in report
        assert "Disease Alpha" in report
        assert "Disease Beta" in report

        # Verify file content matches return value
        with open(output_path) as f:
            assert f.read() == report

    def test_insufficient_runs(self, tmp_path: str) -> None:
        run_dir = _create_run_dir(
            str(tmp_path),
            "Only One",
            [("GENE1", "mAb", 0.8, {})],
        )

        output_path = os.path.join(str(tmp_path), "analysis.md")
        report = generate_batch_analysis([run_dir], output_path)

        assert "Insufficient data" in report

    def test_missing_db_skipped(self, tmp_path: str) -> None:
        run_dir_a = _create_run_dir(
            str(tmp_path),
            "Real Run",
            [("GENE1", "mAb", 0.8, {})],
        )
        fake_dir = os.path.join(str(tmp_path), "fake-run")
        os.makedirs(fake_dir, exist_ok=True)

        output_path = os.path.join(str(tmp_path), "analysis.md")
        report = generate_batch_analysis(
            [run_dir_a, fake_dir],
            output_path,
            skip_competitive=True,
        )

        # Only 1 valid run, so insufficient
        assert "Insufficient data" in report

    def test_competitive_landscape_integration(self, tmp_path: str) -> None:
        """Verify competitive landscape is included when not skipped."""
        run_dir_a = _create_run_dir(
            str(tmp_path),
            "Disease X",
            [("TARGET1", "mAb", 0.8, {})],
        )
        run_dir_b = _create_run_dir(
            str(tmp_path),
            "Disease Y",
            [("TARGET1", "peptide", 0.75, {})],
        )

        mock_landscape = [
            CompetitiveLandscape("TARGET1", 2, ["SM"], "Emerging (2 known drugs)"),
        ]

        with patch(
            "biocompute.archive.batch_analysis.check_competitive_landscape",
            new_callable=AsyncMock,
            return_value=mock_landscape,
        ) as mock_check:
            output_path = os.path.join(str(tmp_path), "analysis.md")
            report = generate_batch_analysis(
                [run_dir_a, run_dir_b],
                output_path,
                skip_competitive=False,
            )

        mock_check.assert_called_once()
        assert "Emerging (2 known drugs)" in report
