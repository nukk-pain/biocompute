# BioCompute

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#research-use-disclaimer)
[![Version](https://img.shields.io/badge/version-0.1.0-lightgrey.svg)](pyproject.toml)

BioCompute turns a disease description into a ranked research worklist of therapeutic target hypotheses.

Use it at the beginning of a target-discovery project, when you have a disease mechanism and want to quickly map candidate genes, pathways, modalities, evidence signals, and red flags before committing to a deeper literature or experimental review.

BioCompute is built for questions such as:

> "Given this disease biology, which targets are worth investigating next, and what should I check before taking them seriously?"

The output is not just a score. It is an archived report that shows candidate targets, supporting evidence, critiques, prior-knowledge maturity, and uncertainty signals that a researcher or AI agent can inspect, verify, compare, and export.

## Ask An AI Agent To Run It

Give an AI agent this repository and a task in this shape:

```text
Use BioCompute to run target discovery for:

Disease: Idiopathic Pulmonary Fibrosis
Description: Progressive lung fibrosis driven by epithelial injury, fibroblast activation, and extracellular matrix deposition.
Keywords: fibrosis, fibroblast, lung

Use the available LLM backend. Install dependencies if needed, run the discovery, find the generated run directory, summarize the report, and mention that outputs are research hypotheses only.
```

If the agent needs exact commands, tell it to follow the runbook below.

## Agent Runbook

### 1. Check prerequisites

Confirm these are available:

- Python 3.12 or newer
- `uv`
- one LLM backend: `claude`, `openai`, `openrouter`, or `codex`

```bash
python --version
uv --version
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure the LLM backend

Default local Claude CLI backend:

```bash
export BIOCOMPUTE_LLM_BACKEND=claude
```

Official OpenAI API backend:

```bash
export BIOCOMPUTE_LLM_BACKEND=openai
export OPENAI_API_KEY=your_key_here
```

OpenRouter backend:

```bash
export BIOCOMPUTE_LLM_BACKEND=openrouter
export OPENROUTER_API_KEY=your_key_here
```

Codex auth backend:

```bash
codex login
export BIOCOMPUTE_LLM_BACKEND=codex
```

Optional keys for higher public API rate limits:

```bash
export NCBI_API_KEY=your_key_here
export S2_API_KEY=your_key_here
```

### 4. Run a discovery

```bash
uv run biocompute discover "Idiopathic Pulmonary Fibrosis" \
  -d "Progressive lung fibrosis driven by epithelial injury, fibroblast activation, and extracellular matrix deposition" \
  -k fibrosis -k fibroblast -k lung \
  -g 10
```

The run writes artifacts under `archive/runs/`.

### 5. Find the latest run directory

```bash
ls -td archive/runs/* | head -1
```

### 6. Read the report

```bash
uv run biocompute report archive/runs/<run_dir>/
```

Expected run artifacts include:

- `config.json`
- `run.db`
- `report.md`

### 7. Verify top targets when deeper review is needed

```bash
uv run biocompute verify archive/runs/<run_dir>/ \
  -n 3 \
  -o archive/runs/<run_dir>/verification.md
```

### 8. Summarize results safely

When reporting results, say:

- the output is a computational hypothesis list;
- target rankings are not validated biology;
- evidence and critiques should be reviewed by domain experts;
- results must not be used for patient care, clinical decisions, regulatory claims, investment decisions, or IP conclusions.

## Command Reference

```bash
uv run biocompute --help
```

Commands:

- `discover` - run therapeutic target discovery for one disease.
- `report` - display a previous run report.
- `verify` - verify top targets against literature evidence.
- `calibrate` - check scoring behavior against calibration examples.
- `batch` - run multiple disease discoveries from a config file.
- `compare` - compare archived runs.
- `export` - export archived results for downstream tools.
- `pipeline` - run discovery plus optional downstream handoff.

Full machine-oriented usage details live in [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## What The Report Contains

A run usually produces:

- a ranked list of therapeutic target hypotheses;
- evidence scores across literature, expression, pathway, druggability, safety, and competitive context;
- critiques and red flags for high-ranking targets;
- prior-knowledge notes that distinguish novel hypotheses from crowded or clinically explored target areas;
- archived artifacts for review, comparison, verification, and export.

Illustrative abbreviated report entry:

```markdown
### #1: ITGB6 (fitness: 0.812)
- **Modality:** antibody / small molecule
- **Tissue:** lung epithelium
- **Scores:** literature=0.88, expression=0.74, druggability=0.69, safety=0.55
- **Evidence:**
  - [literature] PMID:xxxxxxxx: alpha-v beta-6 integrin can activate latent TGF-beta signaling in fibrotic lung disease.
  - [expression] GTEx: enriched in epithelial tissue contexts.
- **Critiques:**
  - TGF-beta pathway modulation can carry broad safety and tolerability risks.
- **Prior Knowledge:**
  - **Maturity:** CLINICALLY_EXPLORED
  - **Summary:** Integrin and TGF-beta pathway programs have been explored for fibrotic disease.
```

## Research-Use Disclaimer

BioCompute is a research prototype for hypothesis generation only.

It is not medical advice, clinical decision support, diagnostic software, regulatory advice, IP advice, or a substitute for expert biological and medical review. Outputs are unvalidated computational hypotheses. Do not use them to guide patient care, treatment decisions, clinical development, fundraising, investing, or IP strategy without independent expert validation.

## License

MIT. See [`LICENSE`](LICENSE).
