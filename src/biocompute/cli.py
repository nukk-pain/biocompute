# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click

from biocompute.archive.report import generate_report
from biocompute.models import DiseaseQuery

if TYPE_CHECKING:
    from biocompute.calibration.ground_truth import CalibrationEntry
    from biocompute.models import FitnessScores


def _load_env(path: str = ".env") -> None:
    """Load .env file into os.environ without external dependencies."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass


def _create_run_directory(output_dir: str, disease_name: str) -> str:
    timestamp_slug = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    name_slug = disease_name.lower().replace(" ", "-")[:30]
    base_name = f"{timestamp_slug}_{name_slug}"
    run_dir = os.path.join(output_dir, base_name)
    suffix = 1

    while True:
        try:
            os.makedirs(run_dir)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = os.path.join(output_dir, f"{base_name}-{suffix}")


def _extract_json_payload(stdout: str) -> str:
    """Strip non-JSON log prefixes from neuroregen stdout.

    neuroregen may print a short informational prefix before the JSON payload.
    Accept either an object or array payload and return the substring starting at
    the first JSON delimiter.
    """
    decoder = json.JSONDecoder()
    best: str | None = None
    for index, char in enumerate(stdout):
        if char not in "[{":
            continue
        candidate = stdout[index:].lstrip()
        try:
            _, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        # Prefer a payload with no trailing text; accept trailing log lines
        # (e.g. "pipeline reports written to ...") as a fallback.
        if not candidate[end:].strip():
            return candidate
        if best is None:
            best = candidate[:end]
    return best if best is not None else stdout


def _validate_cds_mode(_ctx: click.Context, _param: click.Parameter, value: str) -> str:
    if value in {"auto", "mock"}:
        return value
    if os.path.isfile(value):
        return os.path.abspath(value)
    raise click.BadParameter("must be 'auto', 'mock', or an existing file path")


def _parse_neuroregen_designs(stdout: str) -> tuple[str, list[str]]:
    json_content = _extract_json_payload(stdout)

    try:
        payload = json.loads(json_content)
    except json.JSONDecodeError as exc:
        raise click.ClickException("neuroregen produced invalid JSON output") from exc

    if isinstance(payload, list):
        designs = payload
    elif isinstance(payload, dict) and isinstance(payload.get("designs"), list):
        designs = payload["designs"]
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        designs = payload["candidates"]
    else:
        raise click.ClickException("neuroregen produced unsupported JSON shape")

    gene_names: list[str] = []
    for design in designs:
        if not isinstance(design, dict):
            raise click.ClickException("neuroregen produced unsupported JSON shape")
        # Support both nested {"gene": {"symbol": "X"}} and flat {"gene_symbol": "X"}
        gene = design.get("gene")
        if isinstance(gene, dict):
            symbol = gene.get("symbol")
        else:
            symbol = design.get("gene_symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise click.ClickException("neuroregen produced unsupported JSON shape")
        gene_names.append(symbol)

    return json.dumps(payload, indent=2), gene_names


def _run_discovery_workflow(
    *,
    disease_name: str,
    description: str,
    keywords: tuple[str, ...],
    generations: int,
    population_size: int,
    output_dir: str,
    no_cache: bool,
    no_db_seed: bool,
    extra_config: dict[str, Any] | None = None,
) -> tuple[str, str, Any]:
    had_no_cache_env = "BIOCOMPUTE_NO_CACHE" in os.environ
    previous_no_cache = os.environ.get("BIOCOMPUTE_NO_CACHE")

    if no_cache:
        os.environ["BIOCOMPUTE_NO_CACHE"] = "1"
    try:
        from biocompute.engine import EngineConfig, EvolutionEngine

        query = DiseaseQuery(
            name=disease_name,
            description=description,
            keywords=list(keywords),
        )

        config = EngineConfig(
            max_generations=generations,
            population_size=population_size,
            use_db_seed=not no_db_seed,
        )

        run_dir = os.path.abspath(_create_run_directory(output_dir, disease_name))
        db_path = os.path.join(run_dir, "run.db")
        config_path = os.path.join(run_dir, "config.json")
        report_path = os.path.join(run_dir, "report.md")

        config_payload: dict[str, Any] = {
            "disease": disease_name,
            "description": description,
            "keywords": list(keywords),
            "generations": generations,
            "population_size": population_size,
        }
        if extra_config:
            config_payload.update(extra_config)

        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_payload, config_file, indent=2)

        engine = EvolutionEngine(config, db_path)
        result = engine.run(query)

        report = generate_report(result)
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(report)

        return run_dir, db_path, result
    finally:
        if no_cache:
            if had_no_cache_env and previous_no_cache is not None:
                os.environ["BIOCOMPUTE_NO_CACHE"] = previous_no_cache
            else:
                os.environ.pop("BIOCOMPUTE_NO_CACHE", None)


@click.group()
def main() -> None:
    """BioCompute: Disease-agnostic therapeutic target discovery."""
    _load_env()


@main.command()
@click.argument("disease_name")
@click.option(
    "--description",
    "-d",
    required=True,
    help="Disease pathophysiology description",
)
@click.option("--keywords", "-k", multiple=True, help="Search keywords")
@click.option(
    "--generations",
    "-g",
    type=click.IntRange(min=1),
    default=10,
    help="Number of evolution generations",
)
@click.option(
    "--population-size",
    "-p",
    type=click.IntRange(min=1),
    default=30,
    help="Initial population size",
)
@click.option("--output-dir", "-o", default="archive/runs", help="Output directory")
@click.option("--no-cache", is_flag=True, default=False, help="Bypass API result cache")
@click.option(
    "--no-db-seed",
    is_flag=True,
    default=False,
    help="Disable database-seeded initial candidates from OpenTargets",
)
def discover(
    disease_name: str,
    description: str,
    keywords: tuple[str, ...],
    generations: int,
    population_size: int,
    output_dir: str,
    no_cache: bool,
    no_db_seed: bool,
) -> None:
    """Run therapeutic target discovery for a disease."""
    click.echo(f"Starting discovery for: {disease_name}")

    run_dir, _db_path, result = _run_discovery_workflow(
        disease_name=disease_name,
        description=description,
        keywords=keywords,
        generations=generations,
        population_size=population_size,
        output_dir=output_dir,
        no_cache=no_cache,
        no_db_seed=no_db_seed,
    )
    report_path = os.path.join(run_dir, "report.md")

    click.echo(f"Output: {run_dir}")

    click.echo("\nDiscovery complete!")
    click.echo(f"Report saved to: {report_path}")
    click.echo("\nTop 5 candidates (best per gene):")
    from biocompute.models import deduplicate_by_gene

    deduped = deduplicate_by_gene(result.candidates)
    for index, scored in enumerate(deduped[:5], start=1):
        hypothesis = scored.hypothesis
        click.echo(
            f"  #{index} {hypothesis.target_gene:10s} via {hypothesis.modality:10s} "
            f"({hypothesis.delivery}) fitness={scored.fitness:.3f}"
        )


@main.command()
@click.argument("run_dir")
def report(run_dir: str) -> None:
    """Display report from a previous run."""
    report_path = os.path.join(run_dir, "report.md")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as report_file:
            click.echo(report_file.read())
        return

    click.echo(f"No report found at {report_path}")


@main.command()
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Run calibration with hardcoded demo scores to demonstrate the workflow",
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Run calibration against real bio APIs (PubMed, S2, HPA, String DB, OpenTargets)",
)
def calibrate(demo: bool, live: bool) -> None:
    """Verify fitness function against calibration set."""
    from biocompute.calibration.ground_truth import CALIBRATION_SET, CalibrationEntry
    from biocompute.calibration.tune import (
        _calibration_key,
        evaluate_calibration,
        tune_weights,
    )
    from biocompute.models import FitnessScores, Weights, compute_fitness

    click.echo(f"Calibration set: {len(CALIBRATION_SET)} entries")

    if not demo and not live:
        click.echo(
            "(Full calibration requires running fitness evaluation on each entry)"
        )
        click.echo("Use --demo to run with hardcoded scores, --live for real API data,")
        click.echo("or run `biocompute discover` first.")
        return

    # Build demo scores (used by --demo and for comparison in --live)
    demo_scores: dict[tuple[str, str], FitnessScores] = {
        ("VEGF", "cancer"): FitnessScores(
            literature_strength=0.9,
            expression_specificity=0.7,
            pathway_centrality=0.8,
            druggability=0.8,
            safety_profile=0.6,
            ip_freedom=0.3,
        ),
        ("VEGF", "heart failure"): FitnessScores(
            literature_strength=0.4,
            expression_specificity=0.3,
            pathway_centrality=0.5,
            druggability=0.5,
            safety_profile=0.3,
            ip_freedom=0.2,
        ),
        ("CXCR4", "stem cell mobilization"): FitnessScores(
            literature_strength=0.8,
            expression_specificity=0.6,
            pathway_centrality=0.7,
            druggability=0.9,
            safety_profile=0.7,
            ip_freedom=0.4,
        ),
        ("NGF", "osteoarthritis pain"): FitnessScores(
            literature_strength=0.7,
            expression_specificity=0.5,
            pathway_centrality=0.6,
            druggability=0.6,
            safety_profile=0.1,
            ip_freedom=0.3,
        ),
        ("CGRP", "migraine"): FitnessScores(
            literature_strength=0.9,
            expression_specificity=0.8,
            pathway_centrality=0.7,
            druggability=0.9,
            safety_profile=0.9,
            ip_freedom=0.3,
        ),
        ("TNF", "rheumatoid arthritis"): FitnessScores(
            literature_strength=0.95,
            expression_specificity=0.7,
            pathway_centrality=0.9,
            druggability=0.9,
            safety_profile=0.6,
            ip_freedom=0.2,
        ),
        ("PCSK9", "hyperlipidemia"): FitnessScores(
            literature_strength=0.85,
            expression_specificity=0.8,
            pathway_centrality=0.6,
            druggability=0.9,
            safety_profile=0.8,
            ip_freedom=0.4,
        ),
        ("CCR5", "HIV"): FitnessScores(
            literature_strength=0.8,
            expression_specificity=0.7,
            pathway_centrality=0.6,
            druggability=0.8,
            safety_profile=0.7,
            ip_freedom=0.5,
        ),
        ("APP", "alzheimer"): FitnessScores(
            literature_strength=0.6,
            expression_specificity=0.4,
            pathway_centrality=0.5,
            druggability=0.4,
            safety_profile=0.3,
            ip_freedom=0.2,
        ),
        ("CXCL12", "CLL"): FitnessScores(
            literature_strength=0.75,
            expression_specificity=0.55,
            pathway_centrality=0.65,
            druggability=0.80,
            safety_profile=0.65,
            ip_freedom=0.40,
        ),
        ("NGF", "chronic low back pain"): FitnessScores(
            literature_strength=0.70,
            expression_specificity=0.50,
            pathway_centrality=0.60,
            druggability=0.55,
            safety_profile=0.15,
            ip_freedom=0.30,
        ),
        ("CXCL12", "scar tissue hyperinnervation"): FitnessScores(
            literature_strength=0.895,
            expression_specificity=0.75,
            pathway_centrality=1.0,
            druggability=0.60,
            safety_profile=0.97,
            ip_freedom=0.80,
        ),
    }

    if live:
        _run_live_calibration(CALIBRATION_SET, demo_scores)
        return

    # --demo mode
    click.echo("Running demo calibration with hardcoded scores...\n")

    weights = Weights()
    result = evaluate_calibration(CALIBRATION_SET, demo_scores, weights)

    click.echo(f"Separation score: {result['separation_score']:.3f}")
    click.echo(f"Success mean fitness: {result['success_mean']:.3f}")
    click.echo(f"Fail mean fitness: {result['fail_mean']:.3f}")
    click.echo(f"Success scores: {[f'{s:.3f}' for s in result['success_scores']]}")
    click.echo(f"Fail scores: {[f'{s:.3f}' for s in result['fail_scores']]}")

    click.echo("\nTuning weights...")
    tuned = tune_weights(CALIBRATION_SET, demo_scores, steps=100)
    tuned_result = evaluate_calibration(CALIBRATION_SET, demo_scores, tuned)

    click.echo(f"\nTuned separation score: {tuned_result['separation_score']:.3f}")
    click.echo("Tuned weights:")
    for dim in tuned.dimensions():
        click.echo(f"  {dim}: {getattr(tuned, dim):.4f}")
    click.echo(f"  safety_threshold: {tuned.safety_threshold:.4f}")


def _run_live_calibration(
    entries: list[CalibrationEntry],
    demo_scores: dict[tuple[str, str], FitnessScores],
) -> None:
    """Execute live calibration and print results with demo comparison."""
    from biocompute.calibration.live import evaluate_calibration_live
    from biocompute.calibration.tune import _calibration_key
    from biocompute.models import FitnessScores, Weights, compute_fitness

    click.echo("Running LIVE calibration against real bio APIs...")
    click.echo("(This may take a few minutes due to API rate limits)\n")

    def on_progress(i: int, total: int, entry: object) -> None:
        from biocompute.calibration.ground_truth import CalibrationEntry

        if isinstance(entry, CalibrationEntry):
            click.echo(f"  [{i + 1}/{total}] {entry.target_gene} / {entry.disease}...")

    result = evaluate_calibration_live(entries, on_progress=on_progress)

    # --- Live results ---
    evaluation = result["evaluation"]
    click.echo(f"\n{'=' * 60}")
    click.echo("LIVE CALIBRATION RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"Entries scored:  {len(result['scores_map'])}/{len(entries)}")
    click.echo(f"Entries skipped: {len(result['skipped'])}")
    click.echo(f"Elapsed time:    {result['elapsed_seconds']:.1f}s")
    click.echo(f"\nSeparation score:    {evaluation['separation_score']:.3f}")
    click.echo(f"Success mean fitness: {evaluation['success_mean']:.3f}")
    click.echo(f"Fail mean fitness:    {evaluation['fail_mean']:.3f}")

    if result["skipped"]:
        click.echo("\nSkipped entries:")
        for gene, disease, err in result["skipped"]:
            click.echo(f"  {gene}/{disease}: {err[:80]}")

    # --- Tuned weights ---
    tuned = result["tuned_weights"]
    tuned_eval = result["tuned_evaluation"]
    click.echo(f"\n{'─' * 60}")
    click.echo("TUNED WEIGHTS (from live data)")
    click.echo(f"{'─' * 60}")
    click.echo(f"Tuned separation score: {tuned_eval['separation_score']:.3f}")
    for dim in tuned.dimensions():
        click.echo(f"  {dim}: {getattr(tuned, dim):.4f}")
    click.echo(f"  safety_threshold: {tuned.safety_threshold:.4f}")

    # --- Side-by-side comparison ---
    click.echo(f"\n{'─' * 60}")
    click.echo("DEMO vs LIVE SCORE COMPARISON")
    click.echo(f"{'─' * 60}")
    weights = Weights()
    click.echo(
        f"  {'Target':<10} {'Disease':<28} {'Outcome':<8} "
        f"{'Demo':>6} {'Live':>6} {'Delta':>7}"
    )
    click.echo(f"  {'─' * 10} {'─' * 28} {'─' * 8} {'─' * 6} {'─' * 6} {'─' * 7}")

    for entry in entries:
        key = _calibration_key(entry)
        demo_fs = demo_scores.get(key)
        live_val = result["fitness_map"].get(key)

        demo_val = (
            compute_fitness(demo_fs, weights, gene=entry.target_gene)
            if demo_fs
            else None
        )

        demo_str = f"{demo_val:.3f}" if demo_val is not None else "  N/A"
        live_str = f"{live_val:.3f}" if live_val is not None else "  N/A"

        if demo_val is not None and live_val is not None:
            delta = live_val - demo_val
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "    N/A"

        click.echo(
            f"  {entry.target_gene:<10} {entry.disease:<28} {entry.outcome:<8} "
            f"{demo_str:>6} {live_str:>6} {delta_str:>7}"
        )


@main.command()
@click.argument("run_dirs", nargs=-1, required=True)
@click.option("--top", "-n", default=10, help="Number of top candidates per run")
@click.option("--output", "-o", default=None, help="Save report to file")
@click.option(
    "--analysis",
    is_flag=True,
    default=False,
    help="Also generate cross-indication analysis report",
)
def compare(
    run_dirs: tuple[str, ...], top: int, output: str | None, analysis: bool
) -> None:
    """Compare results across multiple discovery runs."""
    from biocompute.archive.compare import compare_runs, load_run

    runs = []
    valid_run_dirs: list[str] = []
    for i, run_dir in enumerate(run_dirs, 1):
        db_path = os.path.join(run_dir, "run.db")
        if not os.path.exists(db_path):
            click.echo(f"Warning: {db_path} not found, skipping")
            continue
        label = f"Run{i}"
        runs.append(load_run(db_path, label=label, top_n=top))
        valid_run_dirs.append(run_dir)

    if len(runs) < 2:
        click.echo("Need at least 2 valid run directories to compare.")
        return

    report = compare_runs(runs)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(report)
        click.echo(f"Comparison report saved to: {output}")
    else:
        click.echo(report)

    if analysis and len(valid_run_dirs) >= 2:
        from biocompute.archive.batch_analysis import generate_batch_analysis

        if output:
            # Place analysis alongside the comparison report
            analysis_path = output.replace(".md", "-analysis.md")
            if analysis_path == output:
                analysis_path = output + ".analysis.md"
        else:
            analysis_path = "batch-analysis.md"

        try:
            generate_batch_analysis(valid_run_dirs, analysis_path)
            click.echo(f"Cross-indication analysis saved to: {analysis_path}")
        except Exception as exc:
            click.echo(f"Warning: cross-indication analysis failed: {exc}")


@main.command(name="export")
@click.argument("run_dir")
@click.option("--top", "-n", default=5, help="Number of top candidates to export")
@click.option("--output", "-o", default=None, help="Save to file instead of stdout")
def export_cmd(run_dir: str, top: int, output: str | None) -> None:
    """Export results as neuroregen-compatible JSON."""
    from biocompute.archive.export import export_for_neuroregen

    db_path = os.path.join(run_dir, "run.db")
    if not os.path.exists(db_path):
        click.echo(f"Error: {db_path} not found")
        raise SystemExit(1)

    result = export_for_neuroregen(db_path, top_n=top)
    json_str = json.dumps(result, indent=2)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(json_str)
        click.echo(f"Exported to: {output}")
    else:
        click.echo(json_str)


@main.command()
@click.argument("config_path")
@click.option("--output-dir", "-o", default="archive/runs", help="Output directory")
@click.option("--compare/--no-compare", default=True, help="Generate comparison report")
def batch(config_path: str, output_dir: str, compare: bool) -> None:
    """Run discovery for multiple diseases from a config file."""
    from biocompute.batch import run_batch

    run_dirs = run_batch(config_path, output_dir)

    click.echo(f"\nBatch complete: {len(run_dirs)} successful runs")

    if compare and len(run_dirs) >= 2:
        from biocompute.archive.compare import compare_runs, load_run

        runs = []
        for i, run_dir in enumerate(run_dirs, 1):
            db_path = os.path.join(run_dir, "run.db")
            if os.path.exists(db_path):
                runs.append(load_run(db_path, label=f"Run{i}"))

        if len(runs) >= 2:
            report = compare_runs(runs)
            report_path = os.path.join(output_dir, "batch-comparison.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            click.echo(f"Comparison report saved to: {report_path}")

    # Generate cross-indication analysis
    if len(run_dirs) >= 2:
        from biocompute.archive.batch_analysis import generate_batch_analysis

        analysis_path = os.path.join(output_dir, "batch-analysis.md")
        try:
            generate_batch_analysis(run_dirs, analysis_path)
            click.echo(f"Cross-indication analysis saved to: {analysis_path}")
        except Exception as exc:
            click.echo(f"Warning: cross-indication analysis failed: {exc}")

    # Summary table
    click.echo("\nSummary:")
    click.echo(f"  {'#':<4} {'Disease':<35} {'Directory'}")
    click.echo(f"  {'─' * 4} {'─' * 35} {'─' * 40}")
    for i, run_dir in enumerate(run_dirs, 1):
        base = os.path.basename(run_dir)
        parts = base.split("_", 4)
        disease_slug = parts[-1] if len(parts) > 4 else base
        click.echo(f"  {i:<4} {disease_slug:<35} {run_dir}")


@main.command()
@click.argument("run_dir")
@click.option("--top", "-n", default=5, help="Number of top targets to verify")
@click.option("--output", "-o", default=None, help="Save verification report to file")
def verify(run_dir: str, top: int, output: str | None) -> None:
    """Verify top targets against literature evidence."""
    from biocompute.verification.literature import verify_targets
    from biocompute.verification.report import generate_verification_report

    db_path = os.path.join(run_dir, "run.db")
    if not os.path.exists(db_path):
        click.echo(f"Error: {db_path} not found")
        raise SystemExit(1)

    click.echo(f"Verifying top {top} targets from: {run_dir}")
    verifications = verify_targets(db_path, top_n=top)

    if not verifications:
        click.echo("No targets found to verify.")
        return

    report_text = generate_verification_report(verifications)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(report_text)
        click.echo(f"Verification report saved to: {output}")
    else:
        click.echo(report_text)


@main.command()
@click.argument("disease_name")
@click.option(
    "--description", "-d", required=True, help="Disease pathophysiology description"
)
@click.option("--keywords", "-k", multiple=True, help="Search keywords")
@click.option(
    "--generations",
    "-g",
    type=click.IntRange(min=1),
    default=3,
    help="Number of evolution generations",
)
@click.option(
    "--population-size",
    "-p",
    type=click.IntRange(min=1),
    default=8,
    help="Initial population size",
)
@click.option(
    "--top",
    "-n",
    type=click.IntRange(min=1),
    default=3,
    help="Top targets for mRNA design",
)
@click.option(
    "--neuroregen-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    required=True,
    help="Path to neuroregen repo",
)
@click.option(
    "--cds-mode",
    default="auto",
    callback=_validate_cds_mode,
    help="CDS source: auto (NCBI fetch), mock, or file path",
)
@click.option(
    "--utr5",
    default="alpha-globin",
    help="5' UTR template: alpha-globin (default) or beta-globin-minimal",
)
@click.option(
    "--utr3",
    default="alpha-globin",
    help="3' UTR template: alpha-globin (default) or bgh-minimal",
)
@click.option("--output-dir", "-o", default="archive/runs", help="Output directory")
@click.option("--no-cache", is_flag=True, default=False, help="Bypass API result cache")
@click.option(
    "--no-db-seed",
    is_flag=True,
    default=False,
    help="Disable database-seeded initial candidates from OpenTargets",
)
@click.option(
    "--poc/--no-poc", default=False, help="Run Phase C PoC simulation after mRNA design"
)
@click.option(
    "--poc-dose", default=10.0, type=float, help="mRNA-LNP dose for PoC simulation (ug)"
)
@click.option(
    "--poc-mode",
    default="pain",
    type=click.Choice(["pain", "fibrosis", "both"]),
    help="PoC mode: pain, fibrosis, or both",
)
def pipeline(
    disease_name: str,
    description: str,
    keywords: tuple[str, ...],
    generations: int,
    population_size: int,
    top: int,
    neuroregen_dir: str,
    cds_mode: str,
    utr5: str,
    utr3: str,
    output_dir: str,
    no_cache: bool,
    no_db_seed: bool,
    poc: bool,
    poc_dose: float,
    poc_mode: str,
) -> None:
    """End-to-end: discover targets -> export -> neuroregen mRNA design."""
    import subprocess

    from biocompute.archive.export import export_for_neuroregen_pipeline
    from biocompute.models import deduplicate_by_gene

    total_steps = 5 if poc else 4

    # Step 1: Discover targets
    click.echo(f"[1/{total_steps}] Discovering targets for: {disease_name}...")

    run_dir, db_path, result = _run_discovery_workflow(
        disease_name=disease_name,
        description=description,
        keywords=keywords,
        generations=generations,
        population_size=population_size,
        output_dir=output_dir,
        no_cache=no_cache,
        no_db_seed=no_db_seed,
        extra_config={
            "command": "pipeline",
            "top": top,
            "cds_mode": cds_mode,
            "utr5": utr5,
            "utr3": utr3,
            "neuroregen_dir": neuroregen_dir,
            "no_cache": no_cache,
        },
    )

    deduped = deduplicate_by_gene(result.candidates)
    summary_parts = [
        f"{s.hypothesis.target_gene} ({s.fitness:.3f})" for s in deduped[:5]
    ]
    click.echo(f"  Found {len(deduped)} targets: {', '.join(summary_parts)}")

    # Step 2: Verify targets against literature
    click.echo(f"[2/{total_steps}] Verifying top {top} targets against literature...")

    from biocompute.verification.literature import verify_targets
    from biocompute.verification.report import generate_verification_report

    verifications = verify_targets(db_path, top_n=top)
    if verifications:
        verification_text = generate_verification_report(verifications)
        verification_path = os.path.join(run_dir, "verification.md")
        with open(verification_path, "w", encoding="utf-8") as f:
            f.write(verification_text)
        click.echo(f"  Verified {len(verifications)} targets → {verification_path}")
    else:
        click.echo("  Warning: no targets found to verify, skipping")

    # Step 3: Export for neuroregen pipeline
    click.echo(f"[3/{total_steps}] Exporting top {top} targets for mRNA design...")

    candidates = export_for_neuroregen_pipeline(db_path, top_n=top)
    targets_path = os.path.join(run_dir, "neuroregen_targets.json")
    with open(targets_path, "w", encoding="utf-8") as targets_file:
        json.dump(candidates, targets_file, indent=2)

    click.echo(f"  Exported to: {targets_path}")

    # Step 4: Run neuroregen mRNA design
    click.echo(f"[4/{total_steps}] Running neuroregen mRNA design...")

    cmd = [
        "cargo",
        "run",
        "-p",
        "pipeline",
        "--",
        "run",
        "--targets-file",
        targets_path,
        "--top-n",
        str(top),
        "--cds",
        cds_mode,
        "--utr5",
        utr5,
        "--utr3",
        utr3,
        "--format",
        "json",
    ]

    try:
        proc = subprocess.run(
            cmd, cwd=neuroregen_dir, capture_output=True, text=True, timeout=300
        )
    except FileNotFoundError:
        click.echo("  Error: cargo or the neuroregen pipeline executable was not found")
        raise SystemExit(1)
    except subprocess.TimeoutExpired:
        click.echo("  Error: neuroregen timed out after 300s")
        raise SystemExit(1)

    if proc.returncode != 0:
        click.echo(f"  Error: neuroregen exited with code {proc.returncode}")
        if proc.stderr:
            click.echo(f"  stderr: {proc.stderr[:500]}")
        raise SystemExit(1)

    # Validate and save neuroregen output
    mrna_output_path = os.path.join(run_dir, "mrna_designs.json")
    json_content, gene_names = _parse_neuroregen_designs(proc.stdout)
    with open(mrna_output_path, "w", encoding="utf-8") as f:
        f.write(json_content)

    click.echo(f"  Designed mRNA for: {', '.join(gene_names)}")

    # Step 5 (optional): PoC simulation
    if poc:
        click.echo(
            f"[5/{total_steps}] Running PoC simulation "
            f"(dose={poc_dose}ug, mode={poc_mode})..."
        )

        poc_cmd = [
            "cargo",
            "run",
            "-p",
            "poc-analyzer",
            "--",
            "simulate",
            "--dose",
            str(poc_dose),
            "--repeats",
            "5000",
            "--calibrated",
            "--mode",
            poc_mode,
            "--format",
            "json",
        ]

        try:
            poc_result = subprocess.run(
                poc_cmd, cwd=neuroregen_dir, capture_output=True, text=True, timeout=60
            )
        except FileNotFoundError:
            click.echo("  Error: cargo or poc-analyzer not found")
            raise SystemExit(1)
        except subprocess.TimeoutExpired:
            click.echo("  Error: PoC simulation timed out after 60s")
            raise SystemExit(1)

        if poc_result.returncode != 0:
            click.echo(
                f"  Error: poc-analyzer exited with code {poc_result.returncode}"
            )
            if poc_result.stderr:
                click.echo(f"  stderr: {poc_result.stderr[:500]}")
            raise SystemExit(1)

        # Parse and save PoC results
        poc_output_path = os.path.join(run_dir, "poc_simulation.json")
        poc_stdout = poc_result.stdout
        json_start = poc_stdout.find("{")
        if json_start >= 0:
            poc_json = poc_stdout[json_start:]
            with open(poc_output_path, "w", encoding="utf-8") as f:
                f.write(poc_json)

            try:
                poc_data = json.loads(poc_json)
                go_prob = poc_data.get("go_probability", 0)
                cond_go = poc_data.get("conditional_go_probability", 0)
                effect = poc_data.get("mean_effect_size_cohens_d", 0)
                click.echo(
                    f"  Go: {go_prob:.0%} | Conditional: {cond_go:.0%} "
                    f"| Effect: d={effect:.1f}"
                )
            except json.JSONDecodeError:
                click.echo("  Warning: could not parse PoC simulation summary")
        else:
            click.echo("  Warning: no JSON found in PoC simulation output")

        # Generate protocol document
        protocol_cmd = [
            "cargo",
            "run",
            "-p",
            "poc-analyzer",
            "--",
            "protocol",
            "--format",
            "markdown",
        ]

        try:
            protocol_result = subprocess.run(
                protocol_cmd,
                cwd=neuroregen_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if protocol_result.returncode == 0 and protocol_result.stdout.strip():
                protocol_path = os.path.join(run_dir, "poc_protocol.md")
                with open(protocol_path, "w", encoding="utf-8") as f:
                    f.write(protocol_result.stdout)
                click.echo(f"  Protocol saved to: {protocol_path}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            click.echo("  Warning: protocol generation skipped")

    click.echo(f"  Results saved to: {mrna_output_path}")


if __name__ == "__main__":
    main()
