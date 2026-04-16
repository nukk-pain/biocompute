# pyright: reportMissingImports=false

import json
import os
import tempfile
from datetime import datetime
from subprocess import CompletedProcess
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from biocompute.archive.store import ArchiveStore
from biocompute.cli import main
from biocompute.models import (
    DiscoveryResult,
    EvidenceMaturity,
    FitnessScores,
    PriorKnowledge,
    RunMetadata,
    ScoredHypothesis,
    TherapeuticHypothesis,
)


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls(2026, 4, 10, 12, 30, 45, 123456, tzinfo=tz)


class FakeEvolutionEngine:
    def __init__(self, _config, db_path: str):
        self.db_path = db_path
        with open(db_path, "w", encoding="utf-8"):
            pass

    def run(self, query):
        return DiscoveryResult(
            query=query,
            candidates=[],
            metadata=RunMetadata(
                started_at=datetime(2026, 4, 10, 12, 0),
                finished_at=datetime(2026, 4, 10, 12, 5),
                generations_run=1,
                total_hypotheses=1,
            ),
        )


class FakePipelineEngine(FakeEvolutionEngine):
    def run(self, query):
        candidate = ScoredHypothesis(
            hypothesis=TherapeuticHypothesis(
                "CXCL12",
                "VHH",
                "local injection",
                "single-dose",
                "scar tissue",
            ),
            fitness=0.711,
            scores=FitnessScores(
                literature_strength=0.9,
                expression_specificity=0.7,
                pathway_centrality=0.8,
                druggability=0.6,
                safety_profile=0.95,
                ip_freedom=0.8,
            ),
        )
        return DiscoveryResult(
            query=query,
            candidates=[candidate],
            metadata=RunMetadata(
                started_at=datetime(2026, 4, 10, 12, 0),
                finished_at=datetime(2026, 4, 10, 12, 5),
                generations_run=1,
                total_hypotheses=1,
            ),
        )


class FakePriorKnowledgeEngine(FakeEvolutionEngine):
    def run(self, query):
        hypothesis = TherapeuticHypothesis(
            "SMAD3",
            "siRNA",
            "local injection",
            "repeat-dose",
            "scar tissue",
        )
        prior_knowledge = PriorKnowledge(
            gene="SMAD3",
            disease=query.name,
            maturity=EvidenceMaturity.L3_IN_VIVO,
            known_facts=["Fibrotic signaling is elevated in preclinical scar models."],
            attempted_approaches=[
                "Upstream TGF-β blockade has been tested clinically."
            ],
            gaps=["Targeted local delivery remains underdefined."],
            key_papers=["PMID:12345"],
            summary=(
                "SMAD3 biology is validated, but scar-targeted intervention design remains open."
            ),
        )
        candidate = ScoredHypothesis(
            hypothesis=hypothesis,
            fitness=0.842,
            scores=FitnessScores(
                literature_strength=0.88,
                expression_specificity=0.73,
                pathway_centrality=0.79,
                druggability=0.42,
                safety_profile=0.93,
                ip_freedom=0.71,
            ),
            prior_knowledge=prior_knowledge,
        )

        store = ArchiveStore(self.db_path)
        try:
            store.save_hypothesis(
                hypothesis,
                candidate.scores,
                fitness_total=candidate.fitness,
                dimension_raw_data={
                    "literature_strength": {"pmids": ["12345", "67890"]},
                    "pathway_centrality": {
                        "interactions": [{"partner": "TGFB1", "score": 0.91}]
                    },
                },
            )
            store.save_prior_knowledge(hypothesis.id, prior_knowledge)
        finally:
            store.close()

        return DiscoveryResult(
            query=query,
            candidates=[candidate],
            metadata=RunMetadata(
                started_at=datetime(2026, 4, 10, 12, 0),
                finished_at=datetime(2026, 4, 10, 12, 5),
                generations_run=1,
                total_hypotheses=1,
            ),
            db_path=self.db_path,
        )


def _extract_output_path(command_output: str) -> str:
    for line in command_output.splitlines():
        if line.startswith("Output: "):
            return line.removeprefix("Output: ")
    raise AssertionError(f"Output path line missing from CLI output: {command_output}")


def _extract_results_path(command_output: str) -> str:
    for line in command_output.splitlines():
        if line.startswith("  Results saved to: "):
            return line.removeprefix("  Results saved to: ")
    raise AssertionError(f"Results path line missing from CLI output: {command_output}")


def test_discover_uses_unique_run_directories_for_same_timestamp() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakeEvolutionEngine),
        ):
            first = runner.invoke(
                main,
                ["discover", "MPS", "-d", "Myofascial pain", "-o", tmp_dir],
            )
            second = runner.invoke(
                main,
                ["discover", "MPS", "-d", "Myofascial pain", "-o", tmp_dir],
            )

        assert first.exit_code == 0
        assert second.exit_code == 0

        first_dir = _extract_output_path(first.output)
        second_dir = _extract_output_path(second.output)

        assert first_dir != second_dir
        assert os.path.basename(first_dir) == "2026-04-10_123045_123456_mps"
        assert os.path.basename(second_dir) == "2026-04-10_123045_123456_mps-2"

        for run_dir in (first_dir, second_dir):
            assert os.path.isdir(run_dir)
            assert os.path.exists(os.path.join(run_dir, "run.db"))
            assert os.path.exists(os.path.join(run_dir, "config.json"))
            assert os.path.exists(os.path.join(run_dir, "report.md"))

            with open(
                os.path.join(run_dir, "config.json"), encoding="utf-8"
            ) as config_file:
                config = json.load(config_file)
            assert config["disease"] == "MPS"
            assert config["description"] == "Myofascial pain"


def test_calibrate_without_demo_shows_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["calibrate"])

    assert result.exit_code == 0
    assert "Calibration set:" in result.output
    assert "Use --demo" in result.output


def test_calibrate_demo_prints_separation_and_tuned_weights() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["calibrate", "--demo"])

    assert result.exit_code == 0
    assert "Running demo calibration" in result.output
    assert "Separation score:" in result.output
    assert "Success mean fitness:" in result.output
    assert "Fail mean fitness:" in result.output
    assert "Tuning weights..." in result.output
    assert "Tuned separation score:" in result.output
    assert "literature_strength:" in result.output
    assert "safety_threshold:" in result.output


def test_pipeline_help_shows_usage() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pipeline", "--help"])

    assert result.exit_code == 0
    assert "End-to-end" in result.output
    assert "--description" in result.output
    assert "--generations" in result.output
    assert "--population-size" in result.output
    assert "--top" in result.output
    assert "--neuroregen-dir" in result.output
    assert "--cds-mode" in result.output
    assert "--no-cache" in result.output


def test_pipeline_runs_end_to_end_and_saves_mrna_output() -> None:
    runner = CliRunner()
    exported_candidates = [
        {
            "gene": {"symbol": "CXCL12", "ncbi_id": 6387, "uniprot_id": None},
            "score": 0.711,
            "evidence": [],
            "pathway": ["CXCR4"],
        }
    ]
    neuroregen_output = {
        "designs": [{"gene": {"symbol": "CXCL12"}, "sequence": "AUGC"}]
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=exported_candidates,
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=0,
                    stdout=json.dumps(neuroregen_output),
                    stderr="",
                ),
            ) as mock_run,
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 0
        assert "[1/4] Discovering targets" in result.output
        assert "[2/4] Verifying top 1 targets" in result.output
        assert "[3/4] Exporting top 1 targets" in result.output
        assert "[4/4] Running neuroregen mRNA design" in result.output
        assert "Designed mRNA for: CXCL12" in result.output

        results_path = _extract_results_path(result.output)
        run_dir = os.path.dirname(results_path)
        assert os.path.exists(os.path.join(run_dir, "run.db"))
        assert os.path.exists(os.path.join(run_dir, "config.json"))
        assert os.path.exists(os.path.join(run_dir, "report.md"))
        export_path = os.path.join(run_dir, "neuroregen_targets.json")
        assert os.path.exists(export_path)
        assert os.path.exists(results_path)

        with open(results_path, encoding="utf-8") as output_file:
            saved = json.load(output_file)
        assert saved == neuroregen_output

        with open(export_path, encoding="utf-8") as export_file:
            saved_candidates = json.load(export_file)
        assert saved_candidates == exported_candidates

        with open(
            os.path.join(run_dir, "config.json"), encoding="utf-8"
        ) as config_file:
            config = json.load(config_file)
        assert config["command"] == "pipeline"
        assert config["top"] == 1
        assert config["cds_mode"] == "auto"
        assert config["no_cache"] is False

        command = mock_run.call_args.args[0]
        assert command[:6] == ["cargo", "run", "-p", "pipeline", "--", "run"]
        command_export_path = command[command.index("--targets-file") + 1]
        assert command_export_path == export_path


def test_pipeline_accepts_prefixed_json_array_output() -> None:
    runner = CliRunner()
    prefixed_stdout = "Loaded 1 targets...\n" + json.dumps(
        [{"gene": {"symbol": "CXCL12"}, "sequence": "AUGC"}]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=[
                    {
                        "gene": {
                            "symbol": "CXCL12",
                            "ncbi_id": 6387,
                            "uniprot_id": None,
                        },
                        "score": 0.711,
                        "evidence": [],
                        "pathway": [],
                    }
                ],
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=0,
                    stdout=prefixed_stdout,
                    stderr="",
                ),
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 0
        assert "Designed mRNA for: CXCL12" in result.output

        results_path = _extract_results_path(result.output)
        with open(results_path, encoding="utf-8") as output_file:
            saved = json.load(output_file)
        assert isinstance(saved, list)
        assert saved[0]["gene"]["symbol"] == "CXCL12"


def test_pipeline_accepts_prefixed_json_object_output() -> None:
    runner = CliRunner()
    prefixed_stdout = "Loaded 1 targets...\n" + json.dumps(
        {"designs": [{"gene": {"symbol": "CXCL12"}, "sequence": "AUGC"}]}
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=[
                    {
                        "gene": {
                            "symbol": "CXCL12",
                            "ncbi_id": 6387,
                            "uniprot_id": None,
                        },
                        "score": 0.711,
                        "evidence": [],
                        "pathway": [],
                    }
                ],
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=0,
                    stdout=prefixed_stdout,
                    stderr="",
                ),
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 0
        assert "Designed mRNA for: CXCL12" in result.output

        results_path = _extract_results_path(result.output)
        with open(results_path, encoding="utf-8") as output_file:
            saved = json.load(output_file)
        assert isinstance(saved, dict)
        assert saved["designs"][0]["gene"]["symbol"] == "CXCL12"


def test_pipeline_accepts_bracketed_log_prefix_before_json() -> None:
    runner = CliRunner()
    prefixed_stdout = "[INFO] Loaded 1 targets\n" + json.dumps(
        {"designs": [{"gene": {"symbol": "CXCL12"}, "sequence": "AUGC"}]}
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=[
                    {
                        "gene": {
                            "symbol": "CXCL12",
                            "ncbi_id": 6387,
                            "uniprot_id": None,
                        },
                        "score": 0.711,
                        "evidence": [],
                        "pathway": [],
                    }
                ],
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=0,
                    stdout=prefixed_stdout,
                    stderr="",
                ),
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 0
        assert "Designed mRNA for: CXCL12" in result.output


def test_pipeline_accepts_candidates_object_with_trailing_log() -> None:
    """neuroregen outputs {"query":"external","candidates":[...]} with surrounding log lines."""
    runner = CliRunner()
    neuroregen_json = json.dumps(
        {
            "query": "external",
            "candidates": [
                {
                    "gene_symbol": "SMAD3",
                    "discovery_score": 0.807,
                    "design": {"sequence": "AUGC"},
                    "design_error": None,
                    "cds_source": "NCBI:SMAD3",
                }
            ],
            "poc_result": None,
            "doe_matrix": None,
        }
    )
    prefixed_stdout = (
        "Loaded 5 external targets from /tmp/neuroregen_targets.json\n"
        + neuroregen_json
        + "\npipeline reports written to /path/to/reports"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=[
                    {
                        "gene": {
                            "symbol": "SMAD3",
                            "ncbi_id": 4088,
                            "uniprot_id": None,
                        },
                        "score": 0.807,
                        "evidence": [],
                        "pathway": [],
                    }
                ],
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=0,
                    stdout=prefixed_stdout,
                    stderr="",
                ),
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Designed mRNA for: SMAD3" in result.output

        results_path = _extract_results_path(result.output)
        with open(results_path, encoding="utf-8") as output_file:
            saved = json.load(output_file)
        assert isinstance(saved, dict)
        assert saved["candidates"][0]["gene_symbol"] == "SMAD3"


def test_pipeline_surfaces_neuroregen_failure() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=[
                    {
                        "gene": {
                            "symbol": "CXCL12",
                            "ncbi_id": 6387,
                            "uniprot_id": None,
                        },
                        "score": 0.711,
                        "evidence": [],
                        "pathway": [],
                    }
                ],
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=7,
                    stdout="",
                    stderr="pipeline failed badly",
                ),
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 1
        assert "Error: neuroregen exited with code 7" in result.output
        assert "stderr: pipeline failed badly" in result.output

        run_dir = os.path.join(tmp_dir, "2026-04-10_123045_123456_mps")
        assert os.path.exists(os.path.join(run_dir, "run.db"))
        assert os.path.exists(os.path.join(run_dir, "config.json"))
        assert os.path.exists(os.path.join(run_dir, "report.md"))
        assert os.path.exists(os.path.join(run_dir, "neuroregen_targets.json"))
        assert not os.path.exists(os.path.join(run_dir, "mrna_designs.json"))


def test_pipeline_rejects_invalid_neuroregen_json_output() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=[
                    {
                        "gene": {
                            "symbol": "CXCL12",
                            "ncbi_id": 6387,
                            "uniprot_id": None,
                        },
                        "score": 0.711,
                        "evidence": [],
                        "pathway": [],
                    }
                ],
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=0,
                    stdout="not-json",
                    stderr="",
                ),
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 1
        assert "invalid JSON output" in result.output
        run_dir = os.path.join(tmp_dir, "2026-04-10_123045_123456_mps")
        assert os.path.exists(os.path.join(run_dir, "neuroregen_targets.json"))
        assert not os.path.exists(os.path.join(run_dir, "mrna_designs.json"))


def test_pipeline_rejects_unsupported_neuroregen_json_shape() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=[
                    {
                        "gene": {
                            "symbol": "CXCL12",
                            "ncbi_id": 6387,
                            "uniprot_id": None,
                        },
                        "score": 0.711,
                        "evidence": [],
                        "pathway": [],
                    }
                ],
            ),
            patch(
                "subprocess.run",
                return_value=CompletedProcess(
                    args=["cargo"],
                    returncode=0,
                    stdout=json.dumps({"designs": [{}]}),
                    stderr="",
                ),
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                ],
            )

        assert result.exit_code == 1
        assert "unsupported JSON shape" in result.output
        run_dir = os.path.join(tmp_dir, "2026-04-10_123045_123456_mps")
        assert os.path.exists(os.path.join(run_dir, "neuroregen_targets.json"))
        assert not os.path.exists(os.path.join(run_dir, "mrna_designs.json"))


def test_pipeline_validates_positive_numeric_options() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = runner.invoke(
            main,
            [
                "pipeline",
                "MPS",
                "-d",
                "Myofascial pain",
                "--top",
                "0",
                "--neuroregen-dir",
                tmp_dir,
                "-o",
                tmp_dir,
            ],
        )

        assert result.exit_code == 2
        assert "0 is not in the range x>=1" in result.output
        assert os.listdir(tmp_dir) == []


def test_pipeline_validates_neuroregen_dir() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_dir = os.path.join(tmp_dir, "missing")
        result = runner.invoke(
            main,
            [
                "pipeline",
                "MPS",
                "-d",
                "Myofascial pain",
                "--neuroregen-dir",
                missing_dir,
                "-o",
                tmp_dir,
            ],
        )

        assert result.exit_code == 2
        assert "does not exist" in result.output
        assert os.listdir(tmp_dir) == []


def test_pipeline_validates_cds_mode_path() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_cds = os.path.join(tmp_dir, "missing.fasta")
        result = runner.invoke(
            main,
            [
                "pipeline",
                "MPS",
                "-d",
                "Myofascial pain",
                "--neuroregen-dir",
                tmp_dir,
                "--cds-mode",
                missing_cds,
                "-o",
                tmp_dir,
            ],
        )

        assert result.exit_code == 2
        assert "existing file path" in result.output
        assert os.listdir(tmp_dir) == []


def test_pipeline_uses_absolute_targets_file_for_relative_output_dir() -> None:
    runner = CliRunner()
    exported_candidates = [
        {
            "gene": {"symbol": "CXCL12", "ncbi_id": 6387, "uniprot_id": None},
            "score": 0.711,
            "evidence": [],
            "pathway": ["CXCR4"],
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        relative_output_dir = "relative-runs"
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            with (
                patch("biocompute.cli.datetime", FrozenDateTime),
                patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
                patch(
                    "biocompute.verification.literature.verify_targets",
                    return_value=[],
                ),
                patch(
                    "biocompute.archive.export.export_for_neuroregen_pipeline",
                    return_value=exported_candidates,
                ),
                patch(
                    "subprocess.run",
                    return_value=CompletedProcess(
                        args=["cargo"],
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "designs": [
                                    {
                                        "gene": {"symbol": "CXCL12"},
                                        "sequence": "AUGC",
                                    }
                                ]
                            }
                        ),
                        stderr="",
                    ),
                ) as mock_run,
            ):
                result = runner.invoke(
                    main,
                    [
                        "pipeline",
                        "MPS",
                        "-d",
                        "Myofascial pain",
                        "--top",
                        "1",
                        "--neuroregen-dir",
                        tmp_dir,
                        "-o",
                        relative_output_dir,
                    ],
                )
        finally:
            os.chdir(cwd)

        assert result.exit_code == 0
        command = mock_run.call_args.args[0]
        export_path = command[command.index("--targets-file") + 1]
        assert os.path.isabs(export_path)


def test_pipeline_normalizes_relative_cds_mode_path_for_subprocess() -> None:
    runner = CliRunner()
    exported_candidates = [
        {
            "gene": {"symbol": "CXCL12", "ncbi_id": 6387, "uniprot_id": None},
            "score": 0.711,
            "evidence": [],
            "pathway": ["CXCR4"],
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        cds_path = os.path.join(tmp_dir, "target.fasta")
        with open(cds_path, "w", encoding="utf-8") as cds_file:
            cds_file.write(">CXCL12\nATGG")

        relative_cds = os.path.basename(cds_path)
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            with (
                patch("biocompute.cli.datetime", FrozenDateTime),
                patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
                patch(
                    "biocompute.verification.literature.verify_targets",
                    return_value=[],
                ),
                patch(
                    "biocompute.archive.export.export_for_neuroregen_pipeline",
                    return_value=exported_candidates,
                ),
                patch(
                    "subprocess.run",
                    return_value=CompletedProcess(
                        args=["cargo"],
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "designs": [
                                    {
                                        "gene": {"symbol": "CXCL12"},
                                        "sequence": "AUGC",
                                    }
                                ]
                            }
                        ),
                        stderr="",
                    ),
                ) as mock_run,
            ):
                result = runner.invoke(
                    main,
                    [
                        "pipeline",
                        "MPS",
                        "-d",
                        "Myofascial pain",
                        "--top",
                        "1",
                        "--neuroregen-dir",
                        tmp_dir,
                        "--cds-mode",
                        relative_cds,
                        "-o",
                        tmp_dir,
                    ],
                )
        finally:
            os.chdir(cwd)

        assert result.exit_code == 0
        command = mock_run.call_args.args[0]
        cds_value = command[command.index("--cds") + 1]
        assert os.path.isabs(cds_value)
        assert os.path.samefile(cds_value, cds_path)


def test_pipeline_help_shows_poc_options() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pipeline", "--help"])

    assert result.exit_code == 0
    assert "--poc" in result.output
    assert "--poc-dose" in result.output
    assert "--poc-mode" in result.output


def test_pipeline_runs_poc_simulation_when_flag_set() -> None:
    runner = CliRunner()
    exported_candidates = [
        {
            "gene": {"symbol": "CXCL12", "ncbi_id": 6387, "uniprot_id": None},
            "score": 0.711,
            "evidence": [],
            "pathway": ["CXCR4"],
        }
    ]
    neuroregen_output = {
        "designs": [{"gene": {"symbol": "CXCL12"}, "sequence": "AUGC"}]
    }
    poc_output = {
        "go_probability": 0.82,
        "conditional_go_probability": 0.91,
        "mean_effect_size_cohens_d": 1.3,
    }

    def fake_subprocess_run(cmd, **kwargs):
        if "poc-analyzer" in cmd:
            if "simulate" in cmd:
                return CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=json.dumps(poc_output),
                    stderr="",
                )
            if "protocol" in cmd:
                return CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="# PoC Protocol\n\nStudy design...\n",
                    stderr="",
                )
        # mRNA design
        return CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps(neuroregen_output),
            stderr="",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePipelineEngine),
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[],
            ),
            patch(
                "biocompute.archive.export.export_for_neuroregen_pipeline",
                return_value=exported_candidates,
            ),
            patch("subprocess.run", side_effect=fake_subprocess_run),
        ):
            result = runner.invoke(
                main,
                [
                    "pipeline",
                    "MPS",
                    "-d",
                    "Myofascial pain",
                    "--top",
                    "1",
                    "--neuroregen-dir",
                    tmp_dir,
                    "-o",
                    tmp_dir,
                    "--poc",
                    "--poc-dose",
                    "15.0",
                    "--poc-mode",
                    "both",
                ],
            )

        assert result.exit_code == 0
        assert "[1/5] Discovering targets" in result.output
        assert "[2/5] Verifying top 1 targets" in result.output
        assert "[3/5] Exporting top 1 targets" in result.output
        assert "[4/5] Running neuroregen mRNA design" in result.output
        assert "[5/5] Running PoC simulation" in result.output
        assert "dose=15.0ug" in result.output
        assert "mode=both" in result.output
        assert "Go: 82%" in result.output
        assert "Effect: d=1.3" in result.output

        run_dir = os.path.join(tmp_dir, "2026-04-10_123045_123456_mps")
        assert os.path.exists(os.path.join(run_dir, "poc_simulation.json"))
        assert os.path.exists(os.path.join(run_dir, "poc_protocol.md"))

        with open(os.path.join(run_dir, "poc_simulation.json"), encoding="utf-8") as f:
            saved_poc = json.load(f)
        assert saved_poc["go_probability"] == 0.82

        with open(os.path.join(run_dir, "poc_protocol.md"), encoding="utf-8") as f:
            assert "PoC Protocol" in f.read()


def test_calibrate_live_dispatches_to_live_runner() -> None:
    runner = CliRunner()

    with patch("biocompute.cli._run_live_calibration") as mock_live:
        result = runner.invoke(main, ["calibrate", "--live"])

    assert result.exit_code == 0
    mock_live.assert_called_once()


def test_verify_writes_report_file() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = os.path.join(tmp_dir, "run")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "run.db"), "w", encoding="utf-8"):
            pass
        output_path = os.path.join(tmp_dir, "verification.md")

        with (
            patch(
                "biocompute.verification.literature.verify_targets",
                return_value=[object()],
            ),
            patch(
                "biocompute.verification.report.generate_verification_report",
                return_value="# Literature Verification Report\n",
            ),
        ):
            result = runner.invoke(
                main,
                ["verify", run_dir, "-o", output_path],
            )

        assert result.exit_code == 0
        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as report_file:
            assert report_file.read() == "# Literature Verification Report\n"


def test_discover_report_export_and_verify_propagate_prior_knowledge() -> None:
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        with (
            patch("biocompute.cli.datetime", FrozenDateTime),
            patch("biocompute.engine.EvolutionEngine", FakePriorKnowledgeEngine),
        ):
            discover_result = runner.invoke(
                main,
                [
                    "discover",
                    "Hypertrophic Scarring",
                    "-d",
                    "Pathologic scar fibrosis with persistent TGF-β signaling",
                    "-o",
                    tmp_dir,
                ],
            )

        assert discover_result.exit_code == 0
        run_dir = _extract_output_path(discover_result.output)

        report_result = runner.invoke(main, ["report", run_dir])
        assert report_result.exit_code == 0
        assert "Prior Knowledge" in report_result.output
        assert "SMAD3 biology is validated" in report_result.output

        export_result = runner.invoke(main, ["export", run_dir, "-n", "1"])
        assert export_result.exit_code == 0
        exported = json.loads(export_result.output)
        export_candidate = exported["candidates"][0]
        assert export_candidate["gene"]["symbol"] == "SMAD3"
        assert export_candidate["prior_knowledge"]["maturity"] == "L3_IN_VIVO"
        assert export_candidate["prior_knowledge"]["summary"].startswith(
            "SMAD3 biology is validated"
        )

        verification_path = os.path.join(tmp_dir, "verification.md")
        with (
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
            verify_result = runner.invoke(
                main,
                ["verify", run_dir, "-n", "1", "-o", verification_path],
            )

        assert verify_result.exit_code == 0
        assert os.path.exists(verification_path)
        with open(verification_path, encoding="utf-8") as report_file:
            verification_report = report_file.read()

        assert "Prior Knowledge Framing" in verification_report
        assert "Evidence maturity:" in verification_report
        assert "SMAD3 biology is validated" in verification_report
        assert "Targeted local delivery remains underdefined." in verification_report
