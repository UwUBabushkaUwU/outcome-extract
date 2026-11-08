"""The only way the service talks to a model.

* OpenAI-compatible client on LLM_BASE_URL / LLM_API_KEY (read at call time, never hardcoded).
* Only models in ALLOWED_MODELS can be called. Expensive models (pro / sonnet) are refused
  unless the caller passes allow_expensive=True; they are never reached by fallback.
* Fallback goes from flash to another cheap model only (LLM_FALLBACK_MODELS, default haiku).
* Reasoning is set to the minimum on every call that supports it (LLM_REASONING_EFFORT), and
  every call has a max_tokens (default 2048, hard cap LLM_MAX_TOKENS_CAP=8192): hidden
  reasoning tokens are billed as output.
* At most LLM_MAX_ATTEMPTS (4) HTTP attempts per chat_json call, fallback included. Retries
  429/5xx/timeouts with backoff and jitter, honouring Retry-After. A truncated reply
  (finish_reason=length) is not retried: the same request would be truncated again.
* Per-job budget (Usage): at most MAX_JOB_LLM_CALLS (150) HTTP calls and MAX_JOB_COST_USD
  ($0.50). Checked before every request; when spent, BudgetExceeded is raised and no
  request goes out. Cache hits cost nothing and don't count.
* Every call is logged at INFO: model, cache hit/miss, tokens in/out (and reasoning), cost.
* Successful replies are cached on disk in CACHE_DIR, keyed by sha256 of the full request.
"""

import email.utils
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import httpx
import openai

from app import config

log = logging.getLogger("app.llm")

CHEAP_MODELS = (
    "google/gemini-3.8-flash",
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
)
EXPENSIVE_MODELS = (
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-5.5",
)
ALLOWED_MODELS = CHEAP_MODELS + EXPENSIVE_MODELS
DEFAULT_MODEL = "google/gemini-3.8-flash"
DEFAULT_FALLBACKS = ("anthropic/claude-haiku-5.5",)

MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", 4))  # per chat_json call, all models
DEFAULT_MAX_TOKENS = 2048
MAX_TOKENS_CAP = int(os.environ.get("LLM_MAX_TOKENS_CAP", 8192))
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "minimal")  # "off" = don't send
MAX_JOB_COST_USD = float(os.environ.get("MAX_JOB_COST_USD", 0.50))
MAX_JOB_LLM_CALLS = int(os.environ.get("MAX_JOB_LLM_CALLS", 150))
BACKOFF_BASE = float(os.environ.get("LLM_BACKOFF_BASE", 1.0))
BACKOFF_CAP = float(os.environ.get("LLM_BACKOFF_CAP", 30.0))
RETRY_AFTER_CAP = float(os.environ.get("LLM_RETRY_AFTER_CAP", 120.0))
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", 120.0))
CACHE_ENABLED = os.environ.get("LLM_CACHE", "1") != "0"


def is_cache_enabled() -> bool:
    if not CACHE_ENABLED:
        return False
    val = os.environ.get("LLM_CACHE", "1").strip().lower()
    return val not in ("0", "off", "false", "no", "disabled")

_RETRYABLE_STATUS = {408, 409, 425, 429} | set(range(500, 600))


class LLMError(Exception):
    """The call could not produce a usable reply."""


class BudgetExceeded(LLMError):
    """The job's call or cost cap is spent; no further requests are made."""


class Truncated(LLMError):
    """The reply stopped at max_tokens. Retrying the same request would be truncated again;
    the caller has to ask for less (e.g. split the input)."""


class _ModelFailed(Exception):
    """One model is exhausted; move on to the next."""

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


# --------------------------------------------------------------------------- usage


@dataclass
class ModelUsage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class Usage:
    """Per-job accounting and budget. Thread-safe: a pipeline fans calls out over threads.

    `models` is what the result metadata reports (cache replays included at their original
    cost). `spent_calls` / `spent_cost_usd` is real spend against the caps, carried over
    from earlier attempts of the same job (the worker seeds them from the job row), and
    `on_spend` lets the worker persist it after every request."""

    models: dict[str, ModelUsage] = field(default_factory=dict)
    cache_hits: int = 0
    label: str = ""
    max_calls: int = field(default_factory=lambda: MAX_JOB_LLM_CALLS)
    max_cost_usd: float = field(default_factory=lambda: MAX_JOB_COST_USD)
    spent_calls: int = 0
    spent_cost_usd: float = 0.0
    on_spend: Callable[[int, float], None] | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve_call(self, model: str) -> None:
        """Take one request from the budget, or raise BudgetExceeded without sending it."""
        with self._lock:
            if self.spent_calls >= self.max_calls:
                raise BudgetExceeded(
                    f"job call cap reached ({self.spent_calls}/{self.max_calls} calls)")
            if self.spent_cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"job cost cap reached (${self.spent_cost_usd:.4f} of "
                    f"${self.max_cost_usd:.2f})")
            self.spent_calls += 1

    def add_spend(self, cost_usd: float) -> None:
        with self._lock:
            self.spent_cost_usd += cost_usd
            calls, cost = self.spent_calls, self.spent_cost_usd
        if self.on_spend:
            try:
                self.on_spend(calls, cost)
            except Exception:  # noqa: BLE001 - accounting must never break a call
                log.exception("%s could not persist LLM spend", self.label)

    def record(self, model: str, tokens_in: int = 0, tokens_out: int = 0,
               cost_usd: float = 0.0, calls: int = 1) -> None:
        with self._lock:
            m = self.models.setdefault(model, ModelUsage())
            m.calls += calls
            m.tokens_in += tokens_in
            m.tokens_out += tokens_out
            m.cost_usd += cost_usd

    def record_cache_hit(self, model: str, cached: dict) -> None:
        # A cache hit reports what the call cost when it was made, so committed outputs
        # show the real per-document cost. It costs nothing now, so it isn't budgeted.
        with self._lock:
            self.cache_hits += 1
        self.record(model, cached.get("tokens_in", 0), cached.get("tokens_out", 0),
                    cached.get("cost_usd", 0.0), calls=cached.get("calls", 1))

    def to_metadata(self) -> dict[str, Any]:
        with self._lock:
            models = {k: asdict(v) for k, v in self.models.items()}
            hits = self.cache_hits
        for m in models.values():
            m["cost_usd"] = round(m["cost_usd"], 6)
        return {
            "model_calls": sum(m["calls"] for m in models.values()),
            "tokens_in": sum(m["tokens_in"] for m in models.values()),
            "tokens_out": sum(m["tokens_out"] for m in models.values()),
            "cost_usd": round(sum(m["cost_usd"] for m in models.values()), 6),
            "models": models,
            "cache_hits": hits,
        }


def _prices() -> dict[str, tuple[float, float]]:
    """Optional fallback price table, USD per million tokens:
    LLM_PRICES='{"google/gemini-3.8-flash": [0.3, 2.5]}'. Used only if the gateway
    doesn't return usage.cost."""
    raw = os.environ.get("LLM_PRICES")
    if not raw:
        return {}
    return {k: (float(v[0]), float(v[1])) for k, v in json.loads(raw).items()}


def _cost(model: str, usage_obj: Any, tokens_in: int, tokens_out: int) -> float:
    cost = getattr(usage_obj, "cost", None)
    if cost is None and getattr(usage_obj, "model_extra", None):
        cost = usage_obj.model_extra.get("cost")
    if cost is not None:
        return float(cost)
    price = _prices().get(model)
    if price:
        return tokens_in / 1e6 * price[0] + tokens_out / 1e6 * price[1]
    return 0.0


def _reasoning_tokens(usage_obj: Any) -> int | None:
    details = getattr(usage_obj, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", None) if details is not None else None


# --------------------------------------------------------------------------- client

_client: openai.OpenAI | None = None
_client_lock = threading.Lock()


def _get_client() -> openai.OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            base_url = os.environ.get("LLM_BASE_URL")
            api_key = os.environ.get("LLM_API_KEY")
            if not base_url or not api_key:
                raise LLMError("LLM_BASE_URL and LLM_API_KEY must both be set")
            _client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=0,  # retries are ours, so we can count, cap and log them
                timeout=httpx.Timeout(TIMEOUT, connect=15.0),
            )
        return _client


# --------------------------------------------------------------------------- cache


def cache_key(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key: str):
    return config.CACHE_DIR / key[:2] / f"{key}.json"


def _cache_get(key: str) -> dict | None:
    if not is_cache_enabled():
        return None
    try:
        return json.loads(_cache_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_put(key: str, entry: dict) -> None:
    if not is_cache_enabled():
        return
    path = _cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic, so concurrent workers never read half a file
    except OSError as e:
        log.warning("cache write failed for %s: %s", key[:12], e)


# --------------------------------------------------------------------------- retry


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = response.headers
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:  # HTTP-date form
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Exponential backoff with equal jitter. Retry-After, when given, is a floor."""
    exp = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
    delay = exp / 2 + random.uniform(0, exp / 2)
    if retry_after is not None:
        delay = max(delay, min(retry_after, RETRY_AFTER_CAP) + random.uniform(0, 1))
    return delay


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS
    return False


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_json(text: str) -> Any:
    """Parse a model's JSON reply, tolerating code fences and prose around one object."""
    text = text.strip()
    if m := _FENCE.match(text):
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def reasoning_param(model: str) -> dict | None:
    """Minimum reasoning for models that think by default (Gemini, OpenAI). Anthropic
    models don't think unless asked, and sending an effort would turn thinking on."""
    if REASONING_EFFORT == "off" or model.startswith("anthropic/"):
        return None
    return {"effort": REASONING_EFFORT}


def _call_model(params: dict, usage: Usage, max_attempts: int, tag: str) -> tuple[Any, dict]:
    """Call one model, at most max_attempts requests. Returns (parsed JSON, spend entry)."""
    model = params["model"]
    client = _get_client()
    bad_replies = 0
    spent = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    last_error = "no attempts made"
    who = " ".join(x for x in (usage.label, tag) if x)

    for attempt in range(max_attempts):
        usage.reserve_call(model)  # raises BudgetExceeded before anything is sent
        spent["calls"] += 1
        try:
            resp = client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001 - classified just below
            usage.record(model)
            usage.add_spend(0.0)
            status = getattr(exc, "status_code", None)
            if (status == 400 and "reasoning" in str(exc).lower()
                    and "reasoning" in params.get("extra_body", {})):
                # This gateway doesn't take the reasoning parameter: send it without.
                log.warning("llm %s model=%s rejected the reasoning parameter; retrying "
                            "without it", who, model)
                params = {**params, "extra_body": {
                    k: v for k, v in params["extra_body"].items() if k != "reasoning"}}
                continue
            if not _is_retryable(exc):
                raise _ModelFailed(f"{model}: {type(exc).__name__}: {exc}",
                                   spent["calls"]) from exc
            retry_after = _retry_after_seconds(exc)
            delay = backoff_delay(attempt, retry_after)
            last_error = f"{type(exc).__name__}: {status or ''} {exc}"
            log.warning("llm %s model=%s attempt %d/%d failed (%s)%s", who, model,
                        attempt + 1, max_attempts, last_error[:200],
                        f"; retry in {delay:.1f}s" if attempt + 1 < max_attempts else "")
            if attempt + 1 < max_attempts:
                time.sleep(delay)
            continue

        u = resp.usage
        t_in = (u.prompt_tokens or 0) if u else 0
        t_out = (u.completion_tokens or 0) if u else 0
        cost = _cost(model, u, t_in, t_out) if u else 0.0
        usage.record(model, t_in, t_out, cost)
        usage.add_spend(cost)
        spent["tokens_in"] += t_in
        spent["tokens_out"] += t_out
        spent["cost_usd"] += cost
        choice = resp.choices[0] if resp.choices else None
        finish = getattr(choice, "finish_reason", None)
        log.info("llm %s model=%s cache=miss attempt=%d tokens_in=%d tokens_out=%d "
                 "reasoning=%s cost=$%.5f finish=%s job_spent=$%.4f (%d/%d calls)",
                 who, model, attempt + 1, t_in, t_out, _reasoning_tokens(u), cost, finish,
                 usage.spent_cost_usd, usage.spent_calls, usage.max_calls)

        content = choice.message.content if choice else None
        if finish == "length":
            # Retrying would pay for the same truncated reply again.
            raise Truncated(f"{model}: reply truncated at max_tokens="
                           f"{params.get('max_tokens')} ({t_out} tokens out)")
        if not content:  # e.g. a 200 with an error body and no choices
            err = (getattr(resp, "model_extra", None) or {}).get("error")
            last_error = f"empty reply{f': {err}' if err else ''}"
            log.warning("llm %s model=%s attempt %d: %s", who, model, attempt + 1, last_error)
        else:
            try:
                return parse_json(content), spent
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON: {exc}"
                log.warning("llm %s model=%s invalid JSON: %.200r", who, model, content)
        bad_replies += 1
        if bad_replies >= 2:
            break
        if attempt + 1 < max_attempts:
            time.sleep(backoff_delay(attempt))

    raise _ModelFailed(f"{model}: gave up after {spent['calls']} attempts: {last_error}",
                       spent["calls"])


def default_fallbacks(model: str) -> list[str]:
    """Cheap models only. LLM_FALLBACK_MODELS (comma-separated) overrides the default;
    expensive models in it are dropped. An expensive primary model gets no fallback."""
    if model not in CHEAP_MODELS:
        return []
    raw = os.environ.get("LLM_FALLBACK_MODELS")
    chain = [m.strip() for m in raw.split(",") if m.strip()] if raw is not None \
        else list(DEFAULT_FALLBACKS)
    dropped = [m for m in chain if m not in CHEAP_MODELS]
    if dropped:
        log.warning("ignoring non-cheap fallback models %s", dropped)
    return [m for m in chain if m in CHEAP_MODELS and m != model]


def _params(model: str, messages: list[dict], temperature: float, max_tokens: int) -> dict:
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    if (reasoning := reasoning_param(model)) is not None:
        params["extra_body"] = {"reasoning": reasoning}
    return params


def chat_json(
    messages: list[dict],
    *,
    usage: Usage,
    model: str = DEFAULT_MODEL,
    fallbacks: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_cache: bool = True,
    allow_expensive: bool = False,
    tag: str = "",
) -> Any:
    """Send a chat request that must come back as JSON. Returns the parsed JSON.
    Raises LLMError (BudgetExceeded when the job's caps are spent) if no usable reply."""
    chain = [model, *(default_fallbacks(model) if fallbacks is None else fallbacks)]
    for m in chain:
        if m not in ALLOWED_MODELS:
            raise ValueError(f"model {m!r} is not in the allowed list")
        if m in EXPENSIVE_MODELS and not allow_expensive:
            raise ValueError(f"model {m!r} is expensive; pass allow_expensive=True to use it")
    max_tokens = max(1, min(int(max_tokens), MAX_TOKENS_CAP))

    errors = []
    attempts_left = MAX_ATTEMPTS
    who = " ".join(x for x in (usage.label, tag) if x)
    for i, m in enumerate(chain):
        params = _params(m, messages, temperature, max_tokens)
        key = cache_key(params)
        if use_cache and (hit := _cache_get(key)) is not None:
            usage.record_cache_hit(m, hit["usage"])
            u = hit["usage"]
            log.info("llm %s model=%s cache=hit tokens_in=%d tokens_out=%d cost=$0 "
                     "(originally $%.5f)", who, m, u.get("tokens_in", 0),
                     u.get("tokens_out", 0), u.get("cost_usd", 0.0))
            return hit["json"]
        # Leave at least one attempt for each model still to come.
        allowed = attempts_left - (len(chain) - i - 1)
        if allowed <= 0:
            break
        try:
            data, spent = _call_model(params, usage, allowed, tag)
        except _ModelFailed as exc:
            attempts_left -= exc.attempts
            errors.append(str(exc))
            if i + 1 < len(chain) and attempts_left > 0:
                log.error("llm %s model failed, falling back to %s: %s", who, chain[i + 1], exc)
            continue
        _cache_put(key, {"model": m, "json": data, "usage": spent})
        return data
    raise LLMError("all models failed: " + " | ".join(errors))
