from __future__ import annotations

import json
import os
import re
import subprocess
from typing import cast

JSONValue = dict[str, object] | list[object]

# ---------------------------------------------------------------------------
# Backend selection via BIOCOMPUTE_LLM_BACKEND env var.
#   "claude"  (default) — Claude CLI subprocess
#   "openai"           — OpenAI Chat Completions API via httpx
# ---------------------------------------------------------------------------

_OPENAI_MODEL_MAP: dict[str, str] = {
    "haiku": "gpt-5.4-mini",
    "sonnet": "gpt-5.4",
}


def _get_backend() -> str:
    return os.environ.get("BIOCOMPUTE_LLM_BACKEND", "claude").lower()


def _query_claude(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
) -> str:
    cmd = [
        "claude",
        "--model",
        model,
        "--print",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    cmd.append(prompt)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI failed (exit {result.returncode}): {result.stderr}"
        )

    return result.stdout.strip()


def _load_codex_api_key() -> str | None:
    """Read API key or OAuth access_token from ~/.codex/auth.json."""
    auth_path = os.path.expanduser("~/.codex/auth.json")
    if not os.path.exists(auth_path):
        return None
    try:
        import json as _json

        with open(auth_path, encoding="utf-8") as f:
            data = _json.load(f)
        # Prefer explicit API key; fall back to OAuth access_token
        key = data.get("OPENAI_API_KEY")
        if key:
            return key
        tokens = data.get("tokens")
        if isinstance(tokens, dict):
            return tokens.get("access_token") or None
        return None
    except (OSError, ValueError):
        return None


def _query_openai(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
) -> str:
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        api_key = _load_codex_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env, export it, "
            "or login via `codex login` to populate ~/.codex/auth.json."
        )

    openai_model = _OPENAI_MODEL_MAP.get(model, model)
    is_oauth = not api_key.startswith("sk-")

    if is_oauth:
        return _query_openai_codex(
            prompt,
            api_key=api_key,
            model=openai_model,
            system_prompt=system_prompt,
        )

    # Standard API key path
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": openai_model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI API failed ({response.status_code}): {response.text[:500]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _query_openai_codex(
    prompt: str,
    *,
    api_key: str,
    model: str,
    system_prompt: str | None = None,
) -> str:
    """Call OpenAI via Codex OAuth (chatgpt.com/backend-api, Responses API streaming)."""
    import httpx

    input_messages: list[dict[str, str]] = [
        {"role": "user", "content": prompt},
    ]

    with httpx.stream(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "instructions": system_prompt or "You are a helpful assistant.",
            "input": input_messages,
            "store": False,
            "stream": True,
        },
        timeout=300,
    ) as response:
        if response.status_code != 200:
            body = response.read().decode()
            raise RuntimeError(
                f"OpenAI Codex API failed ({response.status_code}): {body[:500]}"
            )

        result_text = ""
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
                if chunk.get("type") == "response.output_text.done":
                    result_text = chunk.get("text", "")
            except json.JSONDecodeError:
                continue

    return result_text.strip()


_BACKENDS = {
    "claude": _query_claude,
    "openai": _query_openai,
}


def query_llm(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    backend = _get_backend()
    fn = _BACKENDS.get(backend)
    if fn is None:
        raise RuntimeError(
            f"Unknown LLM backend: {backend!r}. "
            f"Set BIOCOMPUTE_LLM_BACKEND to one of: {', '.join(_BACKENDS)}"
        )
    return fn(prompt, model=model, system_prompt=system_prompt)


def query_llm_json(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
) -> JSONValue | None:
    raw = query_llm(prompt, model=model, system_prompt=system_prompt)
    return parse_json_from_response(raw)


def parse_json_from_response(text: str) -> JSONValue | None:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return cast(JSONValue, json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass

    try:
        return cast(JSONValue, json.loads(text))
    except json.JSONDecodeError:
        pass

    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return cast(JSONValue, json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue

    return None
