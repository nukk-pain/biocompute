# BioCompute

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#research-use-disclaimer)
[![Version](https://img.shields.io/badge/version-0.1.0-lightgrey.svg)](pyproject.toml)

BioCompute is a research prototype for generating and stress-testing therapeutic target hypotheses from a disease description.

It is built for the early question researchers often ask before committing weeks of literature review:

> "Given this disease biology, which targets are worth investigating next — and what evidence or red flags should I check first?"

BioCompute combines LLM-generated hypotheses with structured evidence checks across several biological and translational dimensions. The output is not an answer. It is a ranked research worklist with supporting context and uncertainty signals.

## Contents

- [Quick start](#quick-start)
- [Who should use this?](#who-should-use-this)
- [When should you use it?](#when-should-you-use-it)
- [What does it produce?](#what-does-it-produce)
- [How it works at a high level](#how-it-works-at-a-high-level)
- [Example use cases](#example-use-cases)
- [Research-use disclaimer](#research-use-disclaimer)
- [For AI agents and developers](#for-ai-agents-and-developers)
- [License](#license)

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Run a discovery for one disease
uv run biocompute discover "Myofascial Pain Syndrome" \
  -d "Chronic pain from nerve hyperinnervation in fascial scar tissue" \
  -k scar -k hyperinnervation -k fascia \
  -g 10
```

This writes a run directory under `archive/runs/`. To re-read it later:

```bash
uv run biocompute report archive/runs/<run_dir>/
```

By default BioCompute uses your local `claude` CLI as the LLM backend. To use an
OpenAI-compatible API instead, or for the full command set, see
[`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## Who should use this?

BioCompute is most useful for:

| User | When it helps |
| --- | --- |
| Translational researchers | You have a disease mechanism and want a first-pass target shortlist. |
| Biotech founders or scouts | You are comparing disease areas and need a fast hypothesis map before deeper diligence. |
| Computational biology builders | You want a compact example of an LLM + bio-data evidence loop. |
| Academic labs | You want to turn a disease brief into candidate targets for journal-club or grant-planning discussion. |
| AI agents | You need a CLI-driven target-discovery tool with archived outputs and verification steps. |

## When should you use it?

Use BioCompute when you have:

- a disease or phenotype of interest;
- a short description of the suspected biology;
- a few keywords that define the mechanism, tissue, pathway, or clinical context;
- a need to explore multiple target hypotheses quickly before manual review.

Do **not** use it as a clinical, regulatory, investment, or treatment-decision system.

## What does it produce?

A run typically produces:

- a ranked list of therapeutic target hypotheses;
- per-target evidence dimensions such as literature, expression, pathway, druggability, safety, and IP/competitive context;
- prior-knowledge notes that separate "novel hypothesis" from "already known / already failed / already clinically explored";
- archived run artifacts so results can be reported, compared, exported, or verified later.

The most important design choice is that BioCompute tries to show **why a candidate surfaced** and **what might make it weak**, not just a score.

A single candidate in a report looks roughly like this (illustrative and abridged):

```markdown
### #1: NGF (fitness: 0.812)
- **Modality:** small molecule / antibody
- **Tissue:** fascial scar tissue
- **Scores:** literature=0.88, expression=0.74, druggability=0.69, safety=0.55
- **Evidence:**
  - [literature] PMID:xxxxxxxx: NGF–TrkA signaling drives nociceptor sprouting.
  - [expression] GTEx: elevated in connective/fascial tissue.
- **Critiques:**
  - Anti-NGF programs carry known joint-safety liabilities.
- **Prior Knowledge:**
  - **Maturity:** CLINICALLY_EXPLORED
  - **Summary:** Anti-NGF antibodies reached late-stage trials for chronic pain.
```

The point is not the score on the first line — it is the evidence, critiques, and
prior-knowledge maturity that tell you whether a target is novel, crowded, or
already a known dead end.

## How it works at a high level

1. **Seed hypotheses** from a disease description and keywords.
2. **Score evidence** across multiple biological and translational dimensions.
3. **Select and mutate** candidates over several generations.
4. **Critique candidates** with a skeptical LLM pass.
5. **Enrich top results** with prior-knowledge assessment from literature and clinical context.
6. **Archive outputs** for review, reporting, verification, and downstream handoff.

## Example use cases

- "Find underexplored targets for chronic scar pain with hyperinnervation."
- "Compare target opportunities for fibrosis-like disease mechanisms."
- "Generate a first-pass target map before a wet-lab brainstorming session."
- "Identify whether a target is likely novel, crowded, or already clinically problematic."

## Research-use disclaimer

BioCompute is a research prototype for hypothesis generation only.

It is **not** medical advice, clinical decision support, diagnostic software, regulatory advice, IP advice, or a substitute for expert biological and medical review. Outputs are unvalidated computational hypotheses. Do not use them to guide patient care, treatment decisions, clinical development, fundraising, investing, or IP strategy without independent expert validation.

## For AI agents and developers

Beyond the Quick start above, full operational details — every CLI command, LLM
backend configuration, batch runs, calibration, tests, and live sanity workflows —
live in the machine-oriented usage guide.

See [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## License

MIT. See [`LICENSE`](LICENSE).
