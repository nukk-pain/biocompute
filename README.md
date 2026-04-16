# BioCompute

Disease-agnostic therapeutic target discovery engine. Uses evolutionary search
with LLM-driven mutation and critique to identify therapeutic opportunities and
gaps in biology from a disease description alone.

## Install

```bash
uv sync
```

### LLM Backend

BioCompute uses LLMs for hypothesis generation and critique. By default it calls
the local `claude` CLI. To use OpenAI instead, set the environment:

```bash
export BIOCOMPUTE_LLM_BACKEND=openai
export OPENAI_API_KEY=your_key_here
```

## Usage

```bash
# Discover targets for a disease
biocompute discover "Myofascial Pain Syndrome" \
    -d "Chronic pain from nerve hyperinnervation in fascial scar tissue" \
    -k scar -k hyperinnervation -k fascia \
    -g 10

# View results
biocompute report archive/runs/2026-04-10_myofascial-pain-syndro/

# Verify fitness function calibration
biocompute calibrate
```

## How It Works

1. **Seed**: LLM generates initial therapeutic hypotheses from disease description
2. **Evaluate**: 6-dimension fitness scoring (literature, expression, pathway, druggability, safety, IP)
3. **Select**: Keep top N + random diversity preservation
4. **Critique**: Skeptical LLM review of top candidates
5. **Mutate**: Generate new hypotheses via pathway neighbor / modality switch / lateral jump
6. **Repeat** until convergence or budget exhaustion
7. **Enrich**: Post-ranking prior-knowledge assessment for top candidates (informational only, does not affect scoring)

### Prior Knowledge Assessment

After ranking, BioCompute fetches PubMed abstracts for the top candidates and
assesses what is already known about each gene-disease pair. This produces an
**informational evidence layer** that helps interpret results:

- **Evidence Maturity** (L0-L5): From hypothesis-only to failed clinical trials
- **Known Facts**: Established biological associations
- **Attempted Approaches**: Prior therapeutic strategies
- **Gaps**: Remaining uncertainties and open questions

This assessment is **post-hoc and informational**. It does not modify the six-
dimension fitness scores or alter the ranking. When abstracts are unavailable,
the system falls back to a conservative L0 designation rather than fabricating
maturity claims.

## Test

```bash
uv run pytest -v
```

## Official Live Sanity Workflow

Use this when you need to prove the real pipeline still works beyond mocked tests.

```bash
# 1) Run a minimal real discovery
uv run biocompute discover "Myofascial Pain Syndrome" \
    -d "Chronic pain from nerve hyperinnervation in fascial scar tissue" \
    -k scar -k hyperinnervation -k fascia \
    -g 1 -p 5 --no-cache

# 2) Verify the top targets from the produced run directory
uv run biocompute verify <run_dir> -n 3 -o <run_dir>/verification.md

# 3) Run live calibration against real APIs
uv run biocompute calibrate --live

# 4) Optional: hand off top targets to an external mRNA design pipeline
uv run biocompute pipeline "Myofascial Pain Syndrome" \
    -d "Chronic pain from nerve hyperinnervation in fascial scar tissue" \
    -k scar -k hyperinnervation -k fascia \
    --top 1 --neuroregen-dir /path/to/neuroregen
```

Expected pass conditions:

- the discovery run directory contains `config.json`, `run.db`, and `report.md`
- `verify` writes `verification.md`
- `calibrate --live` prints scored/skipped counts and separation stats
- `pipeline` writes `mrna_designs.json` when the external handoff succeeds
