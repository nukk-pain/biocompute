# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAny=false

from unittest.mock import MagicMock, patch

import pytest

from biocompute.data.llm import parse_json_from_response, query_llm


def test_parse_json_from_response_extracts_json_block():
    response = """Here is the analysis:
```json
{"genes": ["CXCL12", "NGF"]}
```
That's my answer."""
    result = parse_json_from_response(response)
    assert result == {"genes": ["CXCL12", "NGF"]}


def test_parse_json_from_response_raw_json():
    response = '{"genes": ["CXCL12"]}'
    result = parse_json_from_response(response)
    assert result == {"genes": ["CXCL12"]}


def test_parse_json_from_response_no_json_returns_none():
    response = "No JSON here, just text."
    result = parse_json_from_response(response)
    assert result is None


def test_query_llm_calls_subprocess():
    mock_result = MagicMock()
    mock_result.stdout = '{"answer": "test"}'
    mock_result.returncode = 0

    with patch(
        "biocompute.data.llm_backends.subprocess.run", return_value=mock_result
    ) as mock_run:
        result = query_llm("What is biology?", model="haiku")
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "claude" in cmd[0]
        assert "--model" in cmd
        assert "haiku" in cmd
        assert result == '{"answer": "test"}'


def test_query_llm_with_system_prompt():
    mock_result = MagicMock()
    mock_result.stdout = "response"
    mock_result.returncode = 0

    with patch(
        "biocompute.data.llm_backends.subprocess.run", return_value=mock_result
    ) as mock_run:
        _ = query_llm(
            "question", model="sonnet", system_prompt="You are a biologist"
        )
        cmd = mock_run.call_args[0][0]
        assert "--system-prompt" in cmd


def _make_openai_mock(content: str = "ok") -> MagicMock:
    mock_httpx = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    mock_httpx.post.return_value = mock_response
    return mock_httpx


def test_query_llm_openai_backend():
    mock_httpx = _make_openai_mock('{"genes": ["SMAD3"]}')

    with (
        patch.dict(
            "os.environ",
            {"BIOCOMPUTE_LLM_BACKEND": "openai", "OPENAI_API_KEY": "sk-test"},
        ),
        patch.dict("sys.modules", {"httpx": mock_httpx}),
    ):
        result = query_llm(
            "Find targets", model="haiku", system_prompt="You are a biologist"
        )

    mock_httpx.post.assert_called_once()
    call_kwargs = mock_httpx.post.call_args
    assert "api.openai.com" in call_kwargs.args[0]
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-test"
    body = call_kwargs.kwargs["json"]
    assert body["model"] == "gpt-5.4-mini"
    assert body["messages"][0] == {"role": "system", "content": "You are a biologist"}
    assert body["messages"][1] == {"role": "user", "content": "Find targets"}
    assert result == '{"genes": ["SMAD3"]}'


def test_query_llm_openai_model_mapping():
    mock_httpx = _make_openai_mock()

    with (
        patch.dict(
            "os.environ",
            {"BIOCOMPUTE_LLM_BACKEND": "openai", "OPENAI_API_KEY": "sk-test"},
        ),
        patch.dict("sys.modules", {"httpx": mock_httpx}),
    ):
        _ = query_llm("test", model="sonnet")

    body = mock_httpx.post.call_args.kwargs["json"]
    assert body["model"] == "gpt-5.4"


def test_query_llm_openai_missing_key_raises():
    with (
        patch.dict(
            "os.environ",
            {"BIOCOMPUTE_LLM_BACKEND": "openai", "OPENAI_API_KEY": ""},
        ),
        patch(
            "biocompute.data.llm_backends.load_codex_api_key",
            return_value="codex-token",
        ) as mock_load_codex,
        pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
    ):
        _ = query_llm("test")

    mock_load_codex.assert_not_called()


def test_query_llm_openrouter_backend():
    mock_httpx = _make_openai_mock("router-ok")

    with (
        patch.dict(
            "os.environ",
            {
                "BIOCOMPUTE_LLM_BACKEND": "openrouter",
                "OPENROUTER_API_KEY": "or-test",
                "OPENROUTER_SITE_URL": "https://example.org",
                "OPENROUTER_APP_NAME": "BioCompute Test",
            },
        ),
        patch.dict("sys.modules", {"httpx": mock_httpx}),
    ):
        result = query_llm(
            "Find targets", model="sonnet", system_prompt="You are a biologist"
        )

    mock_httpx.post.assert_called_once()
    call_kwargs = mock_httpx.post.call_args
    assert call_kwargs.args[0] == "https://openrouter.ai/api/v1/chat/completions"
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer or-test"
    assert call_kwargs.kwargs["headers"]["HTTP-Referer"] == "https://example.org"
    assert call_kwargs.kwargs["headers"]["X-Title"] == "BioCompute Test"
    body = call_kwargs.kwargs["json"]
    assert body["model"] == "openai/gpt-5.4"
    assert body["messages"][0] == {"role": "system", "content": "You are a biologist"}
    assert body["messages"][1] == {"role": "user", "content": "Find targets"}
    assert result == "router-ok"


def test_query_llm_openrouter_missing_key_raises():
    with (
        patch.dict(
            "os.environ",
            {"BIOCOMPUTE_LLM_BACKEND": "openrouter", "OPENROUTER_API_KEY": ""},
        ),
        pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"),
    ):
        _ = query_llm("test")


def test_query_llm_codex_backend_uses_codex_auth():
    mock_httpx = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        'data: {"type":"response.output_text.done","text":"codex-ok"}',
        "data: [DONE]",
    ]
    mock_stream = MagicMock()
    mock_stream.__enter__.return_value = mock_response
    mock_stream.__exit__.return_value = None
    mock_httpx.stream.return_value = mock_stream

    with (
        patch.dict("os.environ", {"BIOCOMPUTE_LLM_BACKEND": "codex"}),
        patch(
            "biocompute.data.llm_backends.load_codex_api_key",
            return_value="codex-token",
        ),
        patch.dict("sys.modules", {"httpx": mock_httpx}),
    ):
        result = query_llm("Find targets", model="haiku")

    mock_httpx.stream.assert_called_once()
    call_kwargs = mock_httpx.stream.call_args
    assert call_kwargs.args[0] == "POST"
    assert call_kwargs.args[1] == "https://chatgpt.com/backend-api/codex/responses"
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer codex-token"
    assert "max_output_tokens" not in call_kwargs.kwargs["json"]
    assert result == "codex-ok"


def test_query_llm_codex_missing_auth_raises():
    with (
        patch.dict("os.environ", {"BIOCOMPUTE_LLM_BACKEND": "codex"}),
        patch("biocompute.data.llm_backends.load_codex_api_key", return_value=None),
        pytest.raises(RuntimeError, match="codex login"),
    ):
        _ = query_llm("test")
