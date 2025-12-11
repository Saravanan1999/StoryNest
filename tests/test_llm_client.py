from unittest.mock import MagicMock, patch

import openai
import pytest

from storynest.llm_client import LLMClientError, call_model


def _make_fake_response(content: str):
    choice = MagicMock()
    choice.message = {"content": content}
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("storynest.llm_client.openai.ChatCompletion.create")
def test_call_model_success(mock_create: MagicMock) -> None:
    mock_create.return_value = _make_fake_response("hello world")

    result = call_model("test prompt", max_tokens=10, temperature=0.0)

    assert result == "hello world"
    mock_create.assert_called_once()


@patch("storynest.llm_client.openai.ChatCompletion.create")
def test_call_model_raises_llm_client_error(mock_create: MagicMock) -> None:
    # Simulate a generic error from the underlying client
    mock_create.side_effect = RuntimeError("boom")

    with pytest.raises(LLMClientError):
        call_model("test prompt", max_tokens=10, temperature=0.0)


@patch("storynest.llm_client.openai.ChatCompletion.create")
def test_call_model_supports_extra_args(mock_create: MagicMock) -> None:
    # Verify that extra_args are threaded through to the underlying client call.
    mock_create.return_value = _make_fake_response("ok")

    result = call_model("prompt", max_tokens=5, temperature=0.1, extra_args={"foo": "bar"})

    assert result == "ok"
    # Ensure extra arg was passed.
    assert mock_create.call_args is not None
    kwargs = mock_create.call_args.kwargs
    assert kwargs["foo"] == "bar"



