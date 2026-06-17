# BioCompute

BioCompute is an AI-assisted therapeutic target discovery engine.

Give it a disease description. It returns a ranked list of genes or biological
targets that may be worth investigating, with evidence, scores, critiques, and a
research report.

It is built to work well when handed to an AI coding agent such as Claude Code,
Codex, OpenClaw, or Hermes Agent.

Use it when you want to ask:

> For this disease mechanism, which therapeutic targets should we investigate first?

## Use With An AI Agent

Give your agent the repo URL and this task:

```text
Clone https://github.com/nukk-pain/biocompute.
Install it with uv sync.
Use either Claude CLI or Codex OAuth as the LLM backend.
Run BioCompute for endometriosis.
Show me the top targets and the report path.
Do not commit .env files, caches, run databases, or archive outputs.
```

For Claude Code:

```text
Use the local Claude CLI backend:
claude login
export BIOCOMPUTE_LLM_BACKEND=claude
```

For Codex:

```text
Use Codex / OpenAI OAuth:
codex login
export BIOCOMPUTE_LLM_BACKEND=codex
```

For OpenClaw or Hermes Agent, use the same repo task and choose whichever local
backend is available: `claude` if Claude CLI is logged in, or `codex` if Codex is
logged in.

## Run It Yourself

```bash
uv sync

# choose one
claude login
export BIOCOMPUTE_LLM_BACKEND=claude

# or
codex login
export BIOCOMPUTE_LLM_BACKEND=codex

uv run biocompute discover "Endometriosis" \
  -d "Chronic estrogen-dependent inflammatory disease where ectopic endometrial-like tissue drives pelvic pain, neuroangiogenesis, immune dysfunction, fibrosis, adhesions, and infertility" \
  -k estrogen -k inflammation -k fibrosis -k pain -k angiogenesis \
  -g 1 -p 5

latest_run=$(ls -td archive/runs/* | head -1)
uv run biocompute report "$latest_run"
```

Or use the agent smoke script:

```bash
bash examples/agent-endometriosis.sh
```

## What The Agent Should Return

Each run creates:

- `archive/runs/<timestamp>/report.md`
- `archive/runs/<timestamp>/run.db`
- `archive/runs/<timestamp>/config.json`

Ask the agent to report:

- the top ranked targets
- the report path
- any API errors or missing evidence
- the research-only caveat

See [`AGENTS.md`](AGENTS.md) and
[`docs/AGENT_RUNBOOK.md`](docs/AGENT_RUNBOOK.md) for agent-specific operating
instructions.

## Research Use Only

BioCompute generates research hypotheses. It is not medical advice, clinical
decision support, regulatory advice, IP advice, or an investment tool. Results
must be reviewed by qualified domain experts.

## License

MIT.
