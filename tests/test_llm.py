"""llm.py against a mocked OpenAI client: no network, no spend."""

import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import llm

FLASH, HAIKU, LUNA = "google/gemini-3.8-flash", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"
PRO, SONNET = "google/gemini-3.1-pro-preview", "anthropic/claude-sonnet-5.5"
MSG = [{"role": "user", "content": "hi"}]


def _reply(content, cost=0.001, finish="stop", t_in=100, t_out=20):
    usage = SimpleNamespace(prompt_tokens=t_in, completion_tokens=t_out, cost=cost,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage, model_extra={})


def _status_error(code, headers=None, message="err"):
    req = httpx.Request("POST", "http://gateway/chat/completions")
    resp = httpx.Response(code, headers=headers or {}, request=req)
    if code == 429:
        cls = openai.RateLimitError
    elif code >= 500:
        cls = openai.InternalServerError
    else:
        cls = openai.BadRequestError
    return cls(message, response=resp, body=None)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)  # Exception (raised) or reply, one per request
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def models_called(self):
        return [r["model"] for r in self.requests]

    def _create(self, **params):
        self.requests.append(params)
        if not self.script:
            raise AssertionError("unexpected extra request")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.delenv("LLM_FALLBACK_MODELS", raising=False)

    def install(script):
        client = FakeClient(script)
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        client.sleeps = sleeps
        return client
    return install


# --------------------------------------------------------------------------- retries


def test_retries_429_honouring_retry_after_then_succeeds(fake):
    client = fake([_status_error(429, {"retry-after": "7"}), _status_error(503), _reply('{"a": 1}')])
    usage = llm.Usage()
    assert llm.chat_json(MSG, usage=usage) == {"a": 1}
    assert client.sleeps[0] >= 7
    meta = usage.to_metadata()
    assert meta["model_calls"] == 3  # retries count
    assert meta["tokens_in"] == 100 and meta["cost_usd"] == 0.001


def test_at_most_four_requests_per_call_including_fallback(fake):
    client = fake([_status_error(500)] * 10)
    with pytest.raises(llm.LLMError):
        llm.chat_json(MSG, usage=llm.Usage())
    assert len(client.requests) == 4
    assert client.models_called == [FLASH, FLASH, FLASH, HAIKU]  # one kept for the fallback


def test_truncated_reply_is_not_retried(fake):
    client = fake([_reply('{"rows": [{"a"', finish="length", t_out=8000), _reply("{}")])
    with pytest.raises(llm.LLMError, match="truncated"):
        llm.chat_json(MSG, usage=llm.Usage(), max_tokens=8000)
    assert len(client.requests) == 1


def test_empty_replies_count_towards_the_cap(fake):
    client = fake([_reply(""), _reply(None), _reply('{"ok": 1}')])
    assert llm.chat_json(MSG, usage=llm.Usage()) == {"ok": 1}
    assert client.models_called == [FLASH, FLASH, HAIKU]


def test_bad_json_is_retried_and_fences_are_tolerated(fake):
    fake([_reply("not json at all"), _reply('```json\n{"x": 2}\n```')])
    assert llm.chat_json(MSG, usage=llm.Usage()) == {"x": 2}


def test_retry_after_http_date():
    exc = _status_error(429, {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert llm._retry_after_seconds(exc) == 0.0


# --------------------------------------------------------------------------- models


def test_default_fallback_is_a_cheap_model(fake):
    client = fake([_status_error(400), _reply('{"ok": 1}')])
    llm.chat_json(MSG, usage=llm.Usage())
    assert client.models_called == [FLASH, HAIKU]


def test_fallback_env_cannot_add_expensive_models(fake, monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODELS", f"{PRO},{LUNA},{SONNET}")
    assert llm.default_fallbacks(FLASH) == [LUNA]
    client = fake([_status_error(400), _status_error(400)])
    with pytest.raises(llm.LLMError):
        llm.chat_json(MSG, usage=llm.Usage())
    assert client.models_called == [FLASH, LUNA]
    assert not set(client.models_called) & set(llm.EXPENSIVE_MODELS)


@pytest.mark.parametrize("kwargs", [{"model": PRO}, {"fallbacks": [SONNET]}])
def test_expensive_models_need_explicit_opt_in(fake, kwargs):
    client = fake([])
    with pytest.raises(ValueError, match="expensive"):
        llm.chat_json(MSG, usage=llm.Usage(), **kwargs)
    assert client.requests == []


def test_explicit_expensive_model_gets_no_automatic_fallback(fake):
    client = fake([_status_error(400)])
    with pytest.raises(llm.LLMError):
        llm.chat_json(MSG, usage=llm.Usage(), model=PRO, allow_expensive=True)
    assert client.models_called == [PRO]


def test_disallowed_model_is_rejected_before_any_call(fake):
    client = fake([])
    with pytest.raises(ValueError):
        llm.chat_json(MSG, usage=llm.Usage(), model="openai/gpt-6.1-sol")
    assert client.requests == []


# --------------------------------------------------------------------------- request shape


def test_flash_requests_minimal_reasoning_and_a_max_tokens(fake):
    client = fake([_reply("{}")])
    llm.chat_json(MSG, usage=llm.Usage())
    req = client.requests[0]
    assert req["extra_body"]["reasoning"] == {"effort": "minimal"}
    assert req["max_tokens"] == llm.DEFAULT_MAX_TOKENS


def test_max_tokens_is_capped(fake):
    client = fake([_reply("{}")])
    llm.chat_json(MSG, usage=llm.Usage(), max_tokens=100_000)
    assert client.requests[0]["max_tokens"] == llm.MAX_TOKENS_CAP == 8192


def test_anthropic_fallback_is_not_sent_a_reasoning_effort(fake):
    client = fake([_status_error(400), _reply("{}")])
    llm.chat_json(MSG, usage=llm.Usage())
    assert "reasoning" in client.requests[0]["extra_body"]
    assert "extra_body" not in client.requests[1]  # an effort would turn thinking ON


def test_gateway_rejecting_reasoning_param_is_retried_without_it(fake):
    client = fake([_status_error(400, message="Unrecognized parameter: reasoning"),
                   _reply('{"ok": 1}')])
    assert llm.chat_json(MSG, usage=llm.Usage()) == {"ok": 1}
    assert client.models_called == [FLASH, FLASH]
    assert "reasoning" not in client.requests[1].get("extra_body", {})


# --------------------------------------------------------------------------- budget


def test_job_cost_cap_stops_before_sending(fake):
    client = fake([_reply('{"n": 1}', cost=0.3), _reply('{"n": 2}', cost=0.3), _reply("{}")])
    usage = llm.Usage(max_cost_usd=0.5)
    llm.chat_json([{"role": "user", "content": "1"}], usage=usage)
    llm.chat_json([{"role": "user", "content": "2"}], usage=usage)  # $0.30 < $0.50: allowed
    with pytest.raises(llm.BudgetExceeded):
        llm.chat_json([{"role": "user", "content": "3"}], usage=usage)
    assert len(client.requests) == 2


def test_job_call_cap_counts_retries(fake):
    client = fake([_status_error(500), _status_error(500), _status_error(500)])
    usage = llm.Usage(max_calls=2)
    with pytest.raises(llm.BudgetExceeded):
        llm.chat_json(MSG, usage=usage)
    assert len(client.requests) == 2 and usage.spent_calls == 2


def test_cost_cap_defaults_from_env(monkeypatch):
    monkeypatch.setattr(llm, "MAX_JOB_COST_USD", 0.25)
    assert llm.Usage().max_cost_usd == 0.25
    assert llm.Usage().max_calls == 150


def test_spend_callback_sees_running_total(fake):
    seen = []
    fake([_reply('{"a": 1}', cost=0.01), _reply('{"b": 1}', cost=0.02)])
    usage = llm.Usage(spent_calls=5, spent_cost_usd=0.1, on_spend=lambda c, x: seen.append((c, x)))
    llm.chat_json([{"role": "user", "content": "a"}], usage=usage)
    llm.chat_json([{"role": "user", "content": "b"}], usage=usage)
    assert seen[0] == (6, pytest.approx(0.11)) and seen[1] == (7, pytest.approx(0.13))


# --------------------------------------------------------------------------- cache + logging


def test_repeat_call_is_served_from_cache_without_a_request(fake, caplog):
    client = fake([_reply('{"a": 1}', cost=0.004)])
    caplog.set_level(logging.INFO, logger="app.llm")
    first = llm.Usage(label="job=one")
    assert llm.chat_json(MSG, usage=first, tag="rows 0") == {"a": 1}

    second = llm.Usage(label="job=two", max_calls=0)  # no budget at all: must not need one
    assert llm.chat_json(MSG, usage=second, tag="rows 0") == {"a": 1}
    assert len(client.requests) == 1
    assert second.cache_hits == 1 and second.spent_calls == 0 and second.spent_cost_usd == 0
    assert second.to_metadata()["cost_usd"] == 0.004  # reported at the original cost

    lines = [r.getMessage() for r in caplog.records if r.name == "app.llm"]
    miss = next(m for m in lines if "cache=miss" in m)
    hit = next(m for m in lines if "cache=hit" in m)
    for part in (f"model={FLASH}", "tokens_in=100", "tokens_out=20", "cost=$0.00400", "job=one"):
        assert part in miss
    assert "job=two" in hit and "cost=$0" in hit


def test_cache_key_differs_when_request_differs(fake):
    client = fake([_reply('{"a": 1}'), _reply('{"a": 2}')])
    llm.chat_json(MSG, usage=llm.Usage(), max_tokens=1000)
    llm.chat_json(MSG, usage=llm.Usage(), max_tokens=2000)
    assert len(client.requests) == 2


def test_failures_are_not_cached(fake):
    client = fake([_status_error(400), _status_error(400), _reply('{"ok": 1}')])
    with pytest.raises(llm.LLMError):
        llm.chat_json(MSG, usage=llm.Usage())
    assert llm.chat_json(MSG, usage=llm.Usage()) == {"ok": 1}
    assert len(client.requests) == 3
