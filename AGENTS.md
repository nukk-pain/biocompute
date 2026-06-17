# AGENTS.md

## What This Repo Does

BioCompute is a disease-agnostic therapeutic target discovery engine. It takes a
disease name, mechanism description, and biology keywords, then returns ranked
target hypotheses with scores, evidence, critiques, and a report.

## Agent Goal

When a user hands you this repo and asks you to run a disease, produce a real
BioCompute run. Success means:

- `uv sync` completes.
- One LLM backend is configured.
- `uv run biocompute discover ...` completes.
- A run directory exists under `archive/runs/<timestamp>/`.
- `report.md`, `run.db`, and `config.json` exist.
- You report the top targets and the report path.

## LLM Backend

Choose exactly one:

```bash
# Claude Code / Claude CLI
claude login
export BIOCOMPUTE_LLM_BACKEND=claude

# Codex / OpenAI OAuth
codex login
export BIOCOMPUTE_LLM_BACKEND=codex
```

Other options:

```bash
export BIOCOMPUTE_LLM_BACKEND=openai
export OPENAI_API_KEY=your_key_here

export BIOCOMPUTE_LLM_BACKEND=openrouter
export OPENROUTER_API_KEY=your_key_here
```

For OpenClaw, Hermes Agent, or another shell-capable agent, use the same commands
above. The agent framework does not matter; BioCompute only needs shell access,
Python 3.12+, `uv`, and one working LLM backend.

## Fast Smoke Run

```bash
uv sync

uv run biocompute discover "Endometriosis" \
  -d "Chronic estrogen-dependent inflammatory disease where ectopic endometrial-like tissue drives pelvic pain, neuroangiogenesis, immune dysfunction, fibrosis, adhesions, and infertility" \
  -k estrogen -k inflammation -k fibrosis -k pain -k angiogenesis \
  -g 1 -p 5

latest_run=$(ls -td archive/runs/* | head -1)
uv run biocompute report "$latest_run"
```

## Useful Commands

```bash
uv run biocompute --help
uv run biocompute discover --help
uv run biocompute calibrate --demo
uv run pytest -q tests/test_cli.py tests/test_llm.py tests/test_models.py tests/test_calibration.py
```

## Do Not Commit

- `.env`
- API keys or tokens
- `archive/runs/`
- `run.db`
- cache files
- `__pycache__/`
- `.pytest_cache/`
- `.venv/`
- `dist/`

## Safety

BioCompute outputs are research hypotheses, not medical advice, clinical
decision support, regulatory advice, IP advice, or investment advice. Always
preserve that framing when summarizing results.
