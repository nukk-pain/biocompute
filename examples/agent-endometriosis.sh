#!/usr/bin/env bash
set -euo pipefail

uv sync

: "${BIOCOMPUTE_LLM_BACKEND:?Set BIOCOMPUTE_LLM_BACKEND to claude, codex, openai, or openrouter first.}"

uv run biocompute discover "Endometriosis" \
  -d "Chronic estrogen-dependent inflammatory disease where ectopic endometrial-like tissue drives pelvic pain, neuroangiogenesis, immune dysfunction, fibrosis, adhesions, and infertility" \
  -k estrogen -k inflammation -k fibrosis -k pain -k angiogenesis \
  -g "${BIOCOMPUTE_GENERATIONS:-1}" \
  -p "${BIOCOMPUTE_POPULATION_SIZE:-5}"

latest_run=$(ls -td archive/runs/* | head -1)
uv run biocompute report "$latest_run"

echo "Report: $latest_run/report.md"
