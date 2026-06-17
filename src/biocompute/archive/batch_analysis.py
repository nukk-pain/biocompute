"""Cross-indication analysis for batch discovery runs."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IndicationTarget:
    gene: str
    fitness: float
    modality: str


@dataclass
class IndicationSummary:
    disease_name: str
    run_dir: str
    targets: list[IndicationTarget]


@dataclass
class PlatformTarget:
    gene: str
    indications: list[str]
    fitness_by_indication: dict[str, float]
    avg_fitness: float
    hit_count: int
    total_indications: int


@dataclass
class CompetitiveLandscape:
    gene: str
    known_drugs_count: int
    tractability: list[str]
    assessment: str


def load_indication_summary(run_dir: str, top_n: int = 5) -> IndicationSummary:
    """Load top unique-gene targets from a run directory."""
    db_path = os.path.join(run_dir, "run.db")
    config_path = os.path.join(run_dir, "config.json")

    # Load disease name from config.json
    disease_name = os.path.basename(run_dir)
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        disease_name = config.get("disease", disease_name)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get top unique genes by fitness (deduplicating by gene, keeping best)
    rows = conn.execute(
        """
        SELECT h.target_gene, h.modality,
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
        """,
    ).fetchall()
    conn.close()

    # Deduplicate by gene, keeping highest fitness
    seen: dict[str, IndicationTarget] = {}
    for r in rows:
        gene = r["target_gene"]
        fitness = round(r["fitness"], 3)
        if gene not in seen or fitness > seen[gene].fitness:
            seen[gene] = IndicationTarget(
                gene=gene,
                fitness=fitness,
                modality=r["modality"],
            )

    targets = sorted(seen.values(), key=lambda t: t.fitness, reverse=True)[:top_n]

    return IndicationSummary(
        disease_name=disease_name,
        run_dir=run_dir,
        targets=targets,
    )


def extract_platform_targets(
    summaries: list[IndicationSummary],
    min_indications: int = 2,
) -> list[PlatformTarget]:
    """Find targets appearing across multiple indications."""
    total = len(summaries)

    # gene -> {indication: fitness}
    gene_map: dict[str, dict[str, float]] = {}
    for summary in summaries:
        for target in summary.targets:
            if target.gene not in gene_map:
                gene_map[target.gene] = {}
            gene_map[target.gene][summary.disease_name] = target.fitness

    platform: list[PlatformTarget] = []
    for gene, fitness_by_indication in gene_map.items():
        if len(fitness_by_indication) >= min_indications:
            fitnesses = list(fitness_by_indication.values())
            platform.append(
                PlatformTarget(
                    gene=gene,
                    indications=list(fitness_by_indication.keys()),
                    fitness_by_indication=fitness_by_indication,
                    avg_fitness=round(sum(fitnesses) / len(fitnesses), 3),
                    hit_count=len(fitness_by_indication),
                    total_indications=total,
                )
            )

    # Sort by hit count desc, then avg fitness desc
    platform.sort(key=lambda p: (p.hit_count, p.avg_fitness), reverse=True)
    return platform


async def check_competitive_landscape(
    genes: list[str],
) -> list[CompetitiveLandscape]:
    """Check OpenTargets for known drugs per gene."""
    import httpx

    from biocompute.data.opentargets import get_target_info, resolve_gene_to_ensembl

    results: list[CompetitiveLandscape] = []

    async with httpx.AsyncClient(timeout=15) as client:
        for gene in genes:
            try:
                ensembl_id = await resolve_gene_to_ensembl(client, gene)
                if not ensembl_id:
                    results.append(
                        CompetitiveLandscape(
                            gene=gene,
                            known_drugs_count=0,
                            tractability=[],
                            assessment="Unknown (gene not resolved)",
                        )
                    )
                    continue

                info = await get_target_info(client, ensembl_id)

                drug_count = info.get("known_drugs_count", 0)
                if not isinstance(drug_count, int):
                    drug_count = 0

                tract_raw = info.get("tractability", [])
                tract_labels: list[str] = []
                if isinstance(tract_raw, list):
                    for entry in tract_raw:
                        if isinstance(entry, dict):
                            modality = entry.get("modality")
                            value = entry.get("value")
                            if (
                                isinstance(modality, str)
                                and isinstance(value, bool)
                                and value
                            ):
                                tract_labels.append(modality)

                if drug_count == 0:
                    assessment = "Blue ocean (no known drugs)"
                elif drug_count <= 3:
                    assessment = f"Emerging ({drug_count} known drugs)"
                else:
                    assessment = f"Crowded ({drug_count} known drugs)"

                results.append(
                    CompetitiveLandscape(
                        gene=gene,
                        known_drugs_count=drug_count,
                        tractability=tract_labels,
                        assessment=assessment,
                    )
                )
            except Exception:
                results.append(
                    CompetitiveLandscape(
                        gene=gene,
                        known_drugs_count=0,
                        tractability=[],
                        assessment="Unknown (API error)",
                    )
                )

    return results


def _abbreviate(name: str, max_len: int = 6) -> str:
    """Create a short column-header abbreviation for a disease name."""
    words = name.split()
    if len(name) <= max_len:
        return name
    if len(words) == 1:
        return name[:max_len]
    # Use first letters of each word
    abbrev = "".join(w[0].upper() for w in words if w)
    return abbrev if len(abbrev) <= max_len else abbrev[:max_len]


def _format_report(
    summaries: list[IndicationSummary],
    platform_targets: list[PlatformTarget],
    landscape: list[CompetitiveLandscape] | None = None,
) -> str:
    """Format the cross-indication analysis as markdown."""
    lines: list[str] = []
    disease_names = [s.disease_name for s in summaries]
    n = len(summaries)
    today = datetime.now().strftime("%Y-%m-%d")

    lines.append("# Cross-Indication Batch Analysis")
    lines.append(f"> Generated: {today}")
    lines.append(f"> Diseases: {', '.join(disease_names)}")
    lines.append("")

    # Per-Indication Top Targets
    lines.append("## Per-Indication Top Targets")
    lines.append("| Indication | #1 | Fitness | #2 | Fitness | #3 | Fitness |")
    lines.append("|------------|----|---------|----|---------|----|---------| ")
    for s in summaries:
        parts = [f"| {s.disease_name}"]
        for i in range(3):
            if i < len(s.targets):
                t = s.targets[i]
                parts.append(f" {t.gene}")
                parts.append(f" {t.fitness:.3f}")
            else:
                parts.append(" -")
                parts.append(" -")
        parts.append("")
        lines.append(" |".join(parts))
    lines.append("")

    # Cross-Indication Target Map
    lines.append("## Cross-Indication Target Map")
    abbrevs = [_abbreviate(name) for name in disease_names]
    header = "| Target | " + " | ".join(abbrevs) + " | Count |"
    sep = "|--------|" + "|".join("------" for _ in abbrevs) + "|-------|"
    lines.append(header)
    lines.append(sep)

    # Collect all genes across all summaries
    all_genes: dict[str, dict[str, float]] = {}
    for s in summaries:
        for t in s.targets:
            if t.gene not in all_genes:
                all_genes[t.gene] = {}
            all_genes[t.gene][s.disease_name] = t.fitness

    # Sort by hit count desc, then avg fitness desc
    sorted_genes = sorted(
        all_genes.items(),
        key=lambda item: (len(item[1]), sum(item[1].values()) / len(item[1])),
        reverse=True,
    )

    for gene, fitness_map in sorted_genes:
        row = f"| {gene}"
        hit_count = 0
        for name in disease_names:
            if name in fitness_map:
                row += f" | {fitness_map[name]:.3f}"
                hit_count += 1
            else:
                row += " | -"
        row += f" | {hit_count}/{n} |"
        lines.append(row)
    lines.append("")

    # Platform Targets
    if platform_targets:
        lines.append("## Platform Targets (appearing in 2+ indications)")
        lines.append("| Target | Count | Avg Fitness | Indications |")
        lines.append("|--------|-------|-------------|-------------|")
        for pt in platform_targets:
            if pt.hit_count == pt.total_indications:
                ind_str = "All"
            else:
                ind_str = ", ".join(pt.indications)
            lines.append(
                f"| {pt.gene} | {pt.hit_count}/{pt.total_indications} "
                f"| {pt.avg_fitness:.3f} | {ind_str} |"
            )
        lines.append("")

    # Competitive Landscape
    if landscape:
        lines.append("## Competitive Landscape")
        lines.append("| Target | Known Drugs | Tractability | Assessment |")
        lines.append("|--------|------------|-------------|------------|")
        for cl in landscape:
            tract_str = ", ".join(cl.tractability) if cl.tractability else "Unknown"
            lines.append(
                f"| {cl.gene} | {cl.known_drugs_count} "
                f"| {tract_str} | {cl.assessment} |"
            )
        lines.append("")

    # Strategic Summary
    lines.append("## Strategic Summary")
    if platform_targets:
        primary = platform_targets[0]
        lines.append(
            f"- **Primary platform target**: {primary.gene} "
            f"({primary.hit_count}/{primary.total_indications} indications, "
            f"avg fitness {primary.avg_fitness:.3f})"
        )
        if landscape:
            primary_landscape = next(
                (cl for cl in landscape if cl.gene == primary.gene), None
            )
            if primary_landscape:
                lines.append(
                    f"  - Competitive status: {primary_landscape.assessment}"
                )
        for pt in platform_targets[1:3]:
            lines.append(
                f"- **Secondary target**: {pt.gene} "
                f"({pt.hit_count}/{pt.total_indications} indications, "
                f"avg fitness {pt.avg_fitness:.3f})"
            )
    else:
        lines.append("- No targets found across multiple indications")
        lines.append("- Consider expanding the number of generations or population size")
    lines.append("")

    return "\n".join(lines)


def generate_batch_analysis(
    run_dirs: list[str],
    output_path: str,
    top_n: int = 5,
    skip_competitive: bool = False,
) -> str:
    """Generate cross-indication analysis report from batch run directories.

    Args:
        run_dirs: List of run directory paths (each containing run.db and config.json).
        output_path: Where to save the markdown report.
        top_n: Number of top targets per indication.
        skip_competitive: If True, skip the OpenTargets competitive landscape lookup.

    Returns:
        The generated markdown report string.
    """
    summaries: list[IndicationSummary] = []
    for run_dir in run_dirs:
        db_path = os.path.join(run_dir, "run.db")
        if not os.path.exists(db_path):
            continue
        summaries.append(load_indication_summary(run_dir, top_n=top_n))

    if len(summaries) < 2:
        report = "# Cross-Indication Analysis\n\nInsufficient data (need 2+ runs).\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return report

    platform_targets = extract_platform_targets(summaries)

    # Competitive landscape for top 3 platform targets
    landscape: list[CompetitiveLandscape] | None = None
    if not skip_competitive and platform_targets:
        top_genes = [pt.gene for pt in platform_targets[:3]]
        try:
            landscape = asyncio.run(check_competitive_landscape(top_genes))
        except Exception:
            landscape = None

    report = _format_report(summaries, platform_targets, landscape)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report
