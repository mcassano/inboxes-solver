import pytest

from inboxes import openrouter


def test_api_key_reads_primary_env_var(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "primary-key")
    assert openrouter.api_key() == "primary-key"


def test_api_key_falls_back_to_alternate_env_var(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "alternate-key")
    assert openrouter.api_key() == "alternate-key"


def test_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(openrouter.OpenRouterError, match="Set OPEN_ROUTER_API_KEY"):
        openrouter.api_key()


def test_extract_json_plain():
    assert openrouter.extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence():
    text = '```json\n{"a": 1, "b": [1, 2]}\n```'
    assert openrouter.extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_with_surrounding_commentary():
    text = 'Sure, here is the JSON:\n{"a": 1}\nLet me know if you need more.'
    assert openrouter.extract_json(text) == {"a": 1}


def test_extract_json_no_braces_raises():
    with pytest.raises(openrouter.OpenRouterError, match="did not contain JSON"):
        openrouter.extract_json("no json here")


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_chat_returns_message_content(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == openrouter.OPENROUTER_URL
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "some/model"
        return _FakeResponse(200, {"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr(openrouter.requests, "post", fake_post)
    result = openrouter.chat("some/model", [{"role": "user", "content": "hi"}])
    assert result == "hello"


def test_chat_raises_on_non_200(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        openrouter.requests, "post", lambda *a, **k: _FakeResponse(500, text="boom")
    )
    with pytest.raises(openrouter.OpenRouterError, match="500"):
        openrouter.chat("some/model", [])


def test_chat_raises_on_malformed_body(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        openrouter.requests, "post", lambda *a, **k: _FakeResponse(200, {"unexpected": True})
    )
    with pytest.raises(openrouter.OpenRouterError, match="unexpected OpenRouter response"):
        openrouter.chat("some/model", [])
