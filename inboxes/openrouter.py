"""Small shared helper for calling chat-completions models on OpenRouter."""

from __future__ import annotations

import json
import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "Set OPEN_ROUTER_API_KEY (or OPENROUTER_API_KEY) in the environment or .env"
        )
    return key


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise OpenRouterError(f"model response did not contain JSON:\n{text}")
    return json.loads(text[start : end + 1])


def chat(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0,
    timeout: int = 120,
    max_tokens: int | None = None,
) -> str:
    """Call the OpenRouter chat-completions endpoint and return the reply text."""
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {response.text}")
    body = response.json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"unexpected OpenRouter response: {body}") from exc
