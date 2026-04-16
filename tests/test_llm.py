# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAny=false

import sys
from unittest.mock import MagicMock, patch

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
        "biocompute.data.llm.subprocess.run", return_value=mock_result
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
        "biocompute.data.llm.subprocess.run", return_value=mock_result
    ) as mock_run:
        query_llm("question", model="sonnet", system_prompt="You are a biologist")
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
    """OpenAI backend sends correct request via httpx."""
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
    """OpenAI backend maps sonnet to gpt-4o."""
    mock_httpx = _make_openai_mock()

    with (
        patch.dict(
            "os.environ",
            {"BIOCOMPUTE_LLM_BACKEND": "openai", "OPENAI_API_KEY": "sk-test"},
        ),
        patch.dict("sys.modules", {"httpx": mock_httpx}),
    ):
        query_llm("test", model="sonnet")

    body = mock_httpx.post.call_args.kwargs["json"]
    assert body["model"] == "gpt-5.4"


def test_query_llm_openai_missing_key_raises():
    """OpenAI backend raises if OPENAI_API_KEY is not set and no codex auth."""
    with (
        patch.dict(
            "os.environ",
            {"BIOCOMPUTE_LLM_BACKEND": "openai", "OPENAI_API_KEY": ""},
        ),
        patch(
            "biocompute.data.llm._load_codex_api_key",
            return_value=None,
        ),
    ):
        try:
            query_llm("test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "OPENAI_API_KEY" in str(e)
