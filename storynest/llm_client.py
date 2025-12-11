import os
from typing import Any, Dict

from dotenv import load_dotenv
import openai


def _setup_openai() -> None:
    """
    Load environment variables and configure the OpenAI client.

    This keeps all OpenAI-related side effects in one place.
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


def call_model(
    prompt: str,
    max_tokens: int = 900,
    temperature: float = 0.3,
    model: str = "gpt-3.5-turbo",
    extra_args: Dict[str, Any] | None = None,
) -> str:
    """
    Thin wrapper around the OpenAI ChatCompletion API.

    Other modules should depend on this function instead of importing
    `openai` directly, to keep the boundary to external services clean.
    """
    _setup_openai()
    kwargs: Dict[str, Any] = {
        "model": model,  # required by the assignment to remain gpt-3.5-turbo
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_args:
        kwargs.update(extra_args)

    resp = openai.ChatCompletion.create(**kwargs)
    return resp.choices[0].message["content"]  # type: ignore[return-value]


__all__ = ["call_model"]


