from __future__ import annotations

import json
import os
import re
from typing import Protocol
from typing import cast

from biocompute.data.llm_backends import (
    query_claude,
    query_codex,
    query_openai,
    query_openrouter,
)

JSONValue = dict[str, object] | list[object]


class LLMBackend(Protocol):
    def __call__(
        self,
        prompt: str,
        *,
        model: str = "haiku",
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> str: ...


def _get_backend() -> str:
    return os.environ.get("BIOCOMPUTE_LLM_BACKEND", "claude").lower()


_BACKENDS: dict[str, LLMBackend] = {
    "claude": query_claude,
    "openai": query_openai,
    "openrouter": query_openrouter,
    "codex": query_codex,
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
            + f"Set BIOCOMPUTE_LLM_BACKEND to one of: {', '.join(_BACKENDS)}"
        )
    return fn(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


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
