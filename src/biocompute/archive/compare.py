"""Cross-run comparison framework for biocompute discovery results."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass


@dataclass
class RunSummary:
    path: str
    label: str
    generations: int
    total_hypotheses: int
    top_targets: list[TargetScore]


@dataclass
class TargetScore:
    gene: str
    modality: str
    generation: int
    fitness: float
    literature: float
    expression: float
    pathway: float
    druggability: float
    safety: float
    ip: float


# neuroregen reference scores for scar tissue hyperinnervation
NEUROREGEN_REFERENCE: dict[str, dict[str, float]] = {
    "CXCL12": {
        "composite": 0.634,
        "literature": 0.80,
        "expression": 0.75,
        "outcome": 1,  # 1=lead target
    },
    "NGF": {
        "composite": 0.572,
        "literature": 0.70,
        "expression": 0.55,
        "outcome": 0,  # 0=rejected (RPOA)
    },
    "BDNF": {
        "composite": 0.541,
        "literature": 0.50,
        "expression": 0.45,
        "outcome": 0.5,  # 0.5=tertiary
    },
}


def load_run(db_path: str, label: str = "", top_n: int = 10) -> RunSummary:
    """Load a run's top results from its SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    meta = conn.execute(
        "SELECT MAX(generation) as gens, COUNT(*) as total FROM hypotheses"
    ).fetchone()

    rows = conn.execute("""
        SELECT h.target_gene, h.modality, h.generation,
            SUM(CASE WHEN s.dimension='literature_strength' THEN s.score ELSE 0 END) as lit,
            SUM(CASE WHEN s.dimension='expression_specificity' THEN s.score ELSE 0 END) as expr,
            SUM(CASE WHEN s.dimension='pathway_centrality' THEN s.score ELSE 0 END) as path,
            SUM(CASE WHEN s.dimension='druggability' THEN s.score ELSE 0 END) as drug,
            SUM(CASE WHEN s.dimension='safety_profile' THEN s.score ELSE 0 END) as safe,
            SUM(CASE WHEN s.dimension='ip_freedom' THEN s.score ELSE 0 END) as ip,
            SUM(CASE WHEN s.dimension='literature_strength' THEN s.score*0.25
                     WHEN s.dimension='expression_specificity' THEN s.score*0.20
                     WHEN s.dimension='pathway_centrality' THEN s.score*0.15
                     WHEN s.dimension='druggability' THEN s.score*0.15
                     WHEN s.dimension='safety_profile' THEN s.score*0.15
                     WHEN s.dimension='ip_freedom' THEN s.score*0.10 END) as fitness
        FROM hypotheses h
        JOIN scores s ON h.id = s.hypothesis_id
        GROUP BY h.id
        ORDER BY fitness DESC
        LIMIT ?
    """, (top_n,)).fetchall()

    targets = [
        TargetScore(
            gene=r["target_gene"], modality=r["modality"], generation=r["generation"],
            fitness=round(r["fitness"], 3),
            literature=round(r["lit"], 3), expression=round(r["expr"], 3),
            pathway=round(r["path"], 3), druggability=round(r["drug"], 3),
            safety=round(r["safe"], 3), ip=round(r["ip"], 3),
        )
        for r in rows
    ]
    conn.close()

    return RunSummary(
        path=db_path,
        label=label or os.path.basename(os.path.dirname(db_path)),
        generations=meta["gens"] or 0,
        total_hypotheses=meta["total"] or 0,
        top_targets=targets,
    )


def compare_runs(runs: list[RunSummary]) -> str:
    """Generate a markdown comparison report across runs."""
    lines: list[str] = ["# Cross-Run Comparison Report", ""]

    # Summary table
    lines.append("## Run Summary")
    lines.append("| Run | Generations | Hypotheses | Top Target | Top Fitness |")
    lines.append("|-----|------------|------------|------------|-------------|")
    for run in runs:
        top = run.top_targets[0] if run.top_targets else None
        lines.append(
            f"| {run.label} | {run.generations} | {run.total_hypotheses} "
            f"| {top.gene if top else '-'} | {top.fitness if top else '-'} |"
        )
    lines.append("")

    # Per-dimension comparison for top 5
    lines.append("## Top 5 Targets by Run")
    for run in runs:
        lines.append(f"\n### {run.label}")
        lines.append("| # | Gene | Fitness | Lit | Expr | Path | Drug | Safe | IP |")
        lines.append("|---|------|---------|-----|------|------|------|------|-----|")
        for i, t in enumerate(run.top_targets[:5], 1):
            lines.append(
                f"| {i} | {t.gene} | {t.fitness} | {t.literature} | {t.expression} "
                f"| {t.pathway} | {t.druggability} | {t.safety} | {t.ip} |"
            )
    lines.append("")

    # neuroregen alignment
    lines.append("## neuroregen Alignment")
    lines.append("| Target | neuroregen | " + " | ".join(r.label for r in runs) + " |")
    lines.append("|--------|------------|" + "|".join("----" for _ in runs) + "|")
    for gene, ref in NEUROREGEN_REFERENCE.items():
        row = f"| {gene} | {ref['composite']:.3f} |"
        for run in runs:
            match = next((t for t in run.top_targets if t.gene == gene), None)
            row += f" {match.fitness if match else '-'} |"
        lines.append(row)
    lines.append("")

    # CXCL12/CXCR4 ranking consistency
    lines.append("## CXCL12/CXCR4 Axis Ranking")
    for run in runs:
        cxcl12_rank = next(
            (i for i, t in enumerate(run.top_targets, 1) if t.gene == "CXCL12"), None
        )
        cxcr4_rank = next(
            (i for i, t in enumerate(run.top_targets, 1) if t.gene == "CXCR4"), None
        )
        lines.append(
            f"- **{run.label}**: CXCR4 #{cxcr4_rank or '-'}, CXCL12 #{cxcl12_rank or '-'}"
        )

    return "\n".join(lines)
