from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from typing import Final, cast

OPENAI_CHAT_COMPLETIONS_URL: Final = "https://api.openai.com/v1/chat/completions"
OPENROUTER_CHAT_COMPLETIONS_URL: Final = "https://openrouter.ai/api/v1/chat/completions"
CODEX_RESPONSES_URL: Final = "https://chatgpt.com/backend-api/codex/responses"

OPENAI_MODEL_MAP: Final = {"haiku": "gpt-5.4-mini", "sonnet": "gpt-5.4"}
OPENROUTER_MODEL_MAP: Final = {"haiku": "openai/gpt-5.4-mini", "sonnet": "openai/gpt-5.4"}


def query_claude(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    _ = max_tokens
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


def load_codex_api_key() -> str | None:
    auth_path = os.path.expanduser("~/.codex/auth.json")
    if not os.path.exists(auth_path):
        return None
    try:
        with open(auth_path, encoding="utf-8") as f:
            loaded = cast(object, json.load(f))
        if not isinstance(loaded, Mapping):
            return None
        data = cast(Mapping[str, object], loaded)
        key = data.get("OPENAI_API_KEY")
        if isinstance(key, str) and key:
            return key
        tokens = data.get("tokens")
        if isinstance(tokens, Mapping):
            token_data = cast(Mapping[str, object], tokens)
            access_token = token_data.get("access_token")
            if isinstance(access_token, str) and access_token:
                return access_token
        return None
    except (OSError, ValueError):
        return None


def chat_messages(
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def post_chat_completion(
    *,
    provider_name: str,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    extra_headers: dict[str, str] | None = None,
    max_tokens: int = 4096,
) -> str:
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    response = httpx.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": chat_messages(prompt, system_prompt=system_prompt),
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"{provider_name} failed ({response.status_code}): {response.text[:500]}"
        )

    data = cast(object, response.json())
    return chat_completion_content(data)


def chat_completion_content(data: object) -> str:
    if not isinstance(data, Mapping):
        raise RuntimeError("Chat completion response was not an object.")
    response_data = cast(Mapping[str, object], data)
    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat completion response did not include choices.")
    choices_data = cast(list[object], choices)
    first_choice = choices_data[0]
    if not isinstance(first_choice, Mapping):
        raise RuntimeError("Chat completion choice was not an object.")
    choice_data = cast(Mapping[str, object], first_choice)
    message = choice_data.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("Chat completion choice did not include a message.")
    message_data = cast(Mapping[str, object], message)
    content = message_data.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Chat completion message content was not text.")
    return content.strip()


def query_openai(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env or export it. "
            + "For Codex OAuth, set BIOCOMPUTE_LLM_BACKEND=codex."
        )

    return post_chat_completion(
        provider_name="OpenAI API",
        url=OPENAI_CHAT_COMPLETIONS_URL,
        api_key=api_key,
        model=OPENAI_MODEL_MAP.get(model, model),
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


def query_openrouter(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to .env or export it."
        )

    extra_headers: dict[str, str] = {}
    site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
    app_name = os.environ.get("OPENROUTER_APP_NAME", "").strip()
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if app_name:
        extra_headers["X-Title"] = app_name

    return post_chat_completion(
        provider_name="OpenRouter API",
        url=OPENROUTER_CHAT_COMPLETIONS_URL,
        api_key=api_key,
        model=OPENROUTER_MODEL_MAP.get(model, model),
        prompt=prompt,
        system_prompt=system_prompt,
        extra_headers=extra_headers,
        max_tokens=max_tokens,
    )


def query_codex(
    prompt: str,
    *,
    model: str = "haiku",
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    api_key = load_codex_api_key()
    if not api_key:
        raise RuntimeError(
            "Codex auth not found. Run `codex login` to populate "
            + "~/.codex/auth.json, or use BIOCOMPUTE_LLM_BACKEND=openai "
            + "with OPENAI_API_KEY."
        )

    return query_openai_codex(
        prompt,
        api_key=api_key,
        model=OPENAI_MODEL_MAP.get(model, model),
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


def query_openai_codex(
    prompt: str,
    *,
    api_key: str,
    model: str,
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    import httpx

    input_messages: list[dict[str, str]] = [
        {"role": "user", "content": prompt},
    ]
    _ = max_tokens

    with httpx.stream(
        "POST",
        CODEX_RESPONSES_URL,
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
                chunk = cast(object, json.loads(line[6:]))
                if not isinstance(chunk, Mapping):
                    continue
                chunk_data = cast(Mapping[str, object], chunk)
                chunk_type = chunk_data.get("type")
                chunk_text = chunk_data.get("text")
                if chunk_type == "response.output_text.done" and isinstance(
                    chunk_text, str
                ):
                    result_text = chunk_text
            except json.JSONDecodeError:
                continue

    return result_text.strip()
