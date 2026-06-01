# BioCompute AI / Developer Usage Guide

This document is the operational guide for AI agents, maintainers, and developers. The public README is intentionally user-facing and focuses on who should use BioCompute, when, and why.

## Install

```bash
uv sync
```

## LLM backend

BioCompute uses LLMs for hypothesis generation, critique, and some evidence interpretation steps.

Default backend: local `claude` CLI.

```bash
# default behavior
export BIOCOMPUTE_LLM_BACKEND=claude
```

OpenAI-compatible backend:

```bash
export BIOCOMPUTE_LLM_BACKEND=openai
export OPENAI_API_KEY=your_key_here
```

Optional API keys for higher rate limits:

```bash
export NCBI_API_KEY=your_key_here
export S2_API_KEY=your_key_here
```

## CLI overview

```bash
uv run biocompute --help
```

Commands:

- `discover` — run therapeutic target hypothesis generation for one disease.
- `report` — display a report from a previous run.
- `verify` — verify top targets against literature evidence.
- `calibrate` — check the scoring function against the calibration set.
- `batch` — run discovery for multiple diseases from a config file.
- `compare` — compare results across multiple discovery runs.
- `export` — export archived results for downstream tools.
- `pipeline` — run discovery plus optional downstream handoff.

## Basic discovery run

```bash
uv run biocompute discover "Myofascial Pain Syndrome" \
  -d "Chronic pain from nerve hyperinnervation in fascial scar tissue" \
  -k scar -k hyperinnervation -k fascia \
  -g 10
```

Useful options:

```bash
uv run biocompute discover --help
```

- `-d, --description` — disease pathophysiology description. Required.
- `-k, --keywords` — search keywords. Can be repeated.
- `-g, --generations` — number of evolution generations.
- `-p, --population-size` — initial population size.
- `-o, --output-dir` — output directory.
- `--no-cache` — bypass API result cache.
- `--no-db-seed` — disable OpenTargets-seeded initial candidates.

## Report a previous run

```bash
uv run biocompute report archive/runs/<run_dir>/
```

## Verify top targets

```bash
uv run biocompute verify archive/runs/<run_dir>/ -n 3 -o archive/runs/<run_dir>/verification.md
```

## Calibration

Demo calibration:

```bash
uv run biocompute calibrate --demo
```

Live calibration against external APIs:

```bash
uv run biocompute calibrate --live
```

Live calibration can be slow because it intentionally respects public API rate limits.

## Batch runs

Example configs live under [`examples/`](../examples/):

```bash
uv run biocompute batch examples/batch-scar-indications.json
```

## Optional downstream handoff

BioCompute can export targets for downstream design pipelines. Keep this optional in public usage and treat downstream outputs as separate research artifacts.

```bash
uv run biocompute pipeline "Myofascial Pain Syndrome" \
  -d "Chronic pain from nerve hyperinnervation in fascial scar tissue" \
  -k scar -k hyperinnervation -k fascia \
  --top 1 --neuroregen-dir /path/to/neuroregen
```

Expected pass condition: the pipeline writes downstream design artifacts only when the external handoff succeeds.

## Test and build

Fast smoke check:

```bash
uv run biocompute --help
```

Core test slice used for public-readiness checks:

```bash
uv run pytest -q tests/test_cli.py tests/test_engine.py tests/test_export.py tests/test_models.py
```

Full test suite:

```bash
uv run pytest -q
```

Build package artifacts:

```bash
uv build
```

## Output artifacts

A normal discovery run should create a run directory under `archive/runs/` containing artifacts such as:

- `config.json`
- `run.db`
- `report.md`

`archive/runs/` and local databases are gitignored.

## Safety notes for agents

- Treat every output as a hypothesis, not a claim.
- Do not present rankings as validated biology.
- Do not use BioCompute outputs for patient care, treatment recommendations, regulatory claims, investment decisions, or IP conclusions.
- Preserve the research-only framing in public docs and generated reports.
- Do not commit `.env`, run databases, cache files, or local archive outputs.
