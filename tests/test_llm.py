"""llm.py against a fake OpenAI client: no network, no spend."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from app import llm


def _reply(content, cost=0.001):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, cost=cost)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=usage, model_extra={})


def _status_error(code, headers=None):
    req = httpx.Request("POST", "http://gateway/chat/completions")
    resp = httpx.Response(code, headers=headers or {}, request=req)
    cls = openai.RateLimitError if code == 429 else openai.InternalServerError
    if code < 500 and code != 429:
        cls = openai.BadRequestError
    return cls("err", response=resp, body=None)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)  # items: Exception or (model -> reply) callables / replies
        self.models_called = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.models_called.append(params["model"])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    def install(script):
        client = FakeClient(script)
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        client.sleeps = sleeps
        return client
    return install


MSG = [{"role": "user", "content": "hi"}]


def test_retries_429_honouring_retry_after_then_succeeds(fake):
    client = fake([_status_error(429, {"retry-after": "7"}), _status_error(503), _reply('{"a": 1}')])
    usage = llm.Usage()
    assert llm.chat_json(MSG, usage=usage) == {"a": 1}
    assert client.sleeps[0] >= 7
    meta = usage.to_metadata()
    assert meta["model_calls"] == 3  # retries count
    assert meta["tokens_in"] == 100 and meta["cost_usd"] == 0.001


def test_falls_back_to_next_model(fake, monkeypatch):
    monkeypatch.setattr(llm, "MAX_ATTEMPTS", 2)
    client = fake([_status_error(500), _status_error(500), _reply('{"ok": true}')])
    usage = llm.Usage()
    assert llm.chat_json(MSG, usage=usage, fallbacks=["anthropic/claude-sonnet-5.5"]) == {"ok": True}
    assert client.models_called == ["google/gemini-3.8-flash"] * 2 + ["anthropic/claude-sonnet-5.5"]


def test_non_retryable_error_skips_straight_to_fallback(fake):
    client = fake([_status_error(400), _reply('{"ok": 1}')])
    llm.chat_json(MSG, usage=llm.Usage())
    assert client.models_called == ["google/gemini-3.8-flash", "google/gemini-3.1-pro-preview"]


def test_bad_json_is_retried_and_fences_are_tolerated(fake):
    fake([_reply("not json at all"), _reply('```json\n{"x": 2}\n```')])
    assert llm.chat_json(MSG, usage=llm.Usage()) == {"x": 2}


def test_all_models_failing_raises(fake, monkeypatch):
    monkeypatch.setattr(llm, "MAX_ATTEMPTS", 1)
    fake([_status_error(500), _status_error(500)])
    with pytest.raises(llm.LLMError):
        llm.chat_json(MSG, usage=llm.Usage())


def test_disallowed_model_is_rejected_before_any_call(fake):
    client = fake([])
    with pytest.raises(ValueError):
        llm.chat_json(MSG, usage=llm.Usage(), model="openai/gpt-6.1-sol")
    assert client.models_called == []


def test_second_identical_call_is_served_from_cache(fake):
    client = fake([_reply('{"a": 1}')])
    llm.chat_json(MSG, usage=llm.Usage())
    usage = llm.Usage()
    assert llm.chat_json(MSG, usage=usage) == {"a": 1}
    assert len(client.models_called) == 1
    assert usage.to_metadata()["cache_hits"] == 1


def test_retry_after_http_date():
    exc = _status_error(429, {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert llm._retry_after_seconds(exc) == 0.0
