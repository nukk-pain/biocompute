# BioCompute Agent Runbook

This runbook is for Claude Code, Codex, OpenClaw, Hermes Agent, or any other
shell-capable AI agent.

## Prompt To Give An Agent

```text
Clone https://github.com/nukk-pain/biocompute.
Install it with uv sync.
Use Claude CLI or Codex OAuth as the LLM backend.
Run BioCompute for <DISEASE>.
Show me the top targets and the report path.
Do not commit .env files, caches, run databases, or archive outputs.
```

Example disease request:

```text
Run BioCompute for endometriosis:
Description: Chronic estrogen-dependent inflammatory disease where ectopic endometrial-like tissue drives pelvic pain, neuroangiogenesis, immune dysfunction, fibrosis, adhesions, and infertility.
Keywords: estrogen, inflammation, fibrosis, pain, angiogenesis.
```

## Backend Setup

Claude Code / Claude CLI:

```bash
claude login
export BIOCOMPUTE_LLM_BACKEND=claude
```

Codex:

```bash
codex login
export BIOCOMPUTE_LLM_BACKEND=codex
```

OpenClaw or Hermes Agent:

```bash
# Pick one backend available in the shell:
export BIOCOMPUTE_LLM_BACKEND=claude
# or
export BIOCOMPUTE_LLM_BACKEND=codex
```

Official OpenAI API key alternative:

```bash
export BIOCOMPUTE_LLM_BACKEND=openai
export OPENAI_API_KEY=your_key_here
```

OpenRouter alternative:

```bash
export BIOCOMPUTE_LLM_BACKEND=openrouter
export OPENROUTER_API_KEY=your_key_here
```

## Fast Smoke Run

Use this for a quick proof that the repo works:

```bash
uv sync

uv run biocompute discover "Endometriosis" \
  -d "Chronic estrogen-dependent inflammatory disease where ectopic endometrial-like tissue drives pelvic pain, neuroangiogenesis, immune dysfunction, fibrosis, adhesions, and infertility" \
  -k estrogen -k inflammation -k fibrosis -k pain -k angiogenesis \
  -g 1 -p 5

latest_run=$(ls -td archive/runs/* | head -1)
uv run biocompute report "$latest_run"
```

Expected artifacts:

```text
archive/runs/<timestamp>/config.json
archive/runs/<timestamp>/run.db
archive/runs/<timestamp>/report.md
```

## Deeper Run

Use more generations when quality matters more than speed:

```bash
uv run biocompute discover "Idiopathic Pulmonary Fibrosis" \
  -d "Progressive lung fibrosis driven by epithelial injury, fibroblast activation, and extracellular matrix deposition" \
  -k fibrosis -k fibroblast -k lung \
  -g 10
```

## Verification

```bash
latest_run=$(ls -td archive/runs/* | head -1)
test -f "$latest_run/report.md"
test -f "$latest_run/run.db"
test -f "$latest_run/config.json"
uv run biocompute report "$latest_run"
```

## Reporting Back

Keep the user-facing summary short:

```text
BioCompute completed.
Report: archive/runs/<timestamp>/report.md
Top targets: <GENE1>, <GENE2>, <GENE3>, <GENE4>, <GENE5>
Note: outputs are research hypotheses, not medical advice.
```

## Guardrails

Do not commit:

- `.env`
- API keys or tokens
- `archive/runs/`
- `run.db`
- cache files
- `__pycache__/`
- `.pytest_cache/`
- `.venv/`
- `dist/`

Treat all output as hypothesis generation for research review.
