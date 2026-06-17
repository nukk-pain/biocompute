"""Multi-disease batch execution for biocompute."""

from __future__ import annotations

import json
import os
import time

import click

from biocompute.archive.report import generate_report
from biocompute.cli import _create_run_directory
from biocompute.engine import EngineConfig, EvolutionEngine
from biocompute.models import DiseaseQuery


DEFAULT_SETTINGS: dict[str, int] = {
    "generations": 10,
    "population_size": 30,
}


def load_batch_config(config_path: str) -> dict:
    """Load and validate a batch config JSON file.

    Ensures diseases list is non-empty and fills in default settings
    if missing.
    """
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config.get("diseases"), list) or len(config["diseases"]) == 0:
        raise ValueError("Batch config must contain a non-empty 'diseases' list")

    if "settings" not in config:
        config["settings"] = {}

    for key, default_value in DEFAULT_SETTINGS.items():
        if key not in config["settings"]:
            config["settings"][key] = default_value

    return config


def run_batch(config_path: str, output_dir: str = "archive/runs") -> list[str]:
    """Run discovery for multiple diseases from a batch config file.

    Returns list of run_dir paths for successful runs.
    """
    config = load_batch_config(config_path)
    diseases = config["diseases"]
    settings = config["settings"]
    total = len(diseases)
    run_dirs: list[str] = []

    for index, disease_entry in enumerate(diseases, start=1):
        name = disease_entry["name"]
        click.echo(f"[{index}/{total}] Discovering: {name}...")

        try:
            query = DiseaseQuery(
                name=name,
                description=disease_entry["description"],
                keywords=disease_entry.get("keywords", []),
            )

            engine_config = EngineConfig(
                max_generations=settings["generations"],
                population_size=settings["population_size"],
            )

            run_dir = _create_run_directory(output_dir, name)

            db_path = os.path.join(run_dir, "run.db")
            config_file_path = os.path.join(run_dir, "config.json")
            report_path = os.path.join(run_dir, "report.md")

            with open(config_file_path, "w", encoding="utf-8") as cf:
                json.dump(
                    {
                        "disease": name,
                        "description": disease_entry["description"],
                        "keywords": disease_entry.get("keywords", []),
                        "generations": settings["generations"],
                        "population_size": settings["population_size"],
                    },
                    cf,
                    indent=2,
                )

            engine = EvolutionEngine(engine_config, db_path)
            result = engine.run(query)

            report = generate_report(result)
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write(report)

            top = result.candidates[0] if result.candidates else None
            if top:
                click.echo(
                    f"  Done: {top.hypothesis.target_gene} "
                    f"fitness={top.fitness:.3f}"
                )
            else:
                click.echo("  Done: no candidates found")

            run_dirs.append(run_dir)

        except Exception as exc:
            click.echo(f"  Error: {exc}")
            continue

        # API cooldown between diseases (skip after the last one)
        if index < total:
            click.echo("  Waiting 30s for API cooldown...")
            time.sleep(30)

    return run_dirs
