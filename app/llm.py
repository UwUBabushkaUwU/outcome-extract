"""The only way the service talks to a model.

* OpenAI-compatible client on LLM_BASE_URL / LLM_API_KEY (read at call time, never hardcoded).
* Only models in ALLOWED_MODELS can be called; anything else raises before a request goes out.
* JSON output: response_format=json_object, and the reply is parsed (code fences tolerated).
* Retries 408/409/429/5xx, connection errors and timeouts with exponential backoff and jitter,
  honouring Retry-After / Retry-After-Ms. An unusable reply (no choices, bad JSON) is retried
  once. When a model is exhausted, the next model in the fallback chain is tried.
* Each HTTP attempt is counted in Usage (model_calls includes retries, as the contract asks).
  Tokens come from `usage`; cost from `usage.cost` (OpenRouter) or, failing that, LLM_PRICES.
* Successful replies are cached on disk in CACHE_DIR, keyed by sha256 of the full request
  (model + messages + generation params), so re-running the samples costs nothing.
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
from typing import Any

import httpx
import openai

from app import config

log = logging.getLogger("app.llm")

ALLOWED_MODELS = (
    "google/gemini-3.8-flash",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-5.5",
)
DEFAULT_MODEL = ALLOWED_MODELS[0]

MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", 8))  # per model
BACKOFF_BASE = float(os.environ.get("LLM_BACKOFF_BASE", 1.0))
BACKOFF_CAP = float(os.environ.get("LLM_BACKOFF_CAP", 30.0))
RETRY_AFTER_CAP = float(os.environ.get("LLM_RETRY_AFTER_CAP", 120.0))
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", 180.0))
CACHE_ENABLED = os.environ.get("LLM_CACHE", "1") != "0"

_RETRYABLE_STATUS = {408, 409, 425, 429} | set(range(500, 600))


class LLMError(Exception):
    """Every model in the chain failed."""


class _ModelFailed(Exception):
    """One model is exhausted; move on to the next."""


# --------------------------------------------------------------------------- usage


@dataclass
class ModelUsage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class Usage:
    """Per-document accounting. Thread-safe so a pipeline can fan calls out over threads."""

    models: dict[str, ModelUsage] = field(default_factory=dict)
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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
        # show the real per-document cost. cache_hits says how many were replayed.
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
                max_retries=0,  # retries are ours, so we can count and log them
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
    if not CACHE_ENABLED:
        return None
    try:
        return json.loads(_cache_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_put(key: str, entry: dict) -> None:
    if not CACHE_ENABLED:
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


def _call_model(params: dict, usage: Usage) -> tuple[Any, dict]:
    """Call one model with retries. Returns (parsed JSON, usage entry for the cache)."""
    model = params["model"]
    client = _get_client()
    bad_replies = 0
    spent = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    last_error = "no attempts made"

    for attempt in range(MAX_ATTEMPTS):
        spent["calls"] += 1
        try:
            resp = client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001 - classified just below
            usage.record(model)
            if not _is_retryable(exc):
                raise _ModelFailed(f"{model}: {type(exc).__name__}: {exc}") from exc
            retry_after = _retry_after_seconds(exc)
            delay = backoff_delay(attempt, retry_after)
            last_error = f"{type(exc).__name__}: {getattr(exc, 'status_code', '')} {exc}"
            log.warning("llm %s attempt %d/%d failed (%s); retry in %.1fs%s", model,
                        attempt + 1, MAX_ATTEMPTS, last_error[:200], delay,
                        f" (Retry-After {retry_after:.0f}s)" if retry_after is not None else "")
            time.sleep(delay)
            continue

        u = resp.usage
        t_in = (u.prompt_tokens or 0) if u else 0
        t_out = (u.completion_tokens or 0) if u else 0
        cost = _cost(model, u, t_in, t_out) if u else 0.0
        usage.record(model, t_in, t_out, cost)
        spent["tokens_in"] += t_in
        spent["tokens_out"] += t_out
        spent["cost_usd"] += cost

        # Gateways sometimes answer 200 with an error body and no choices.
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            err = (getattr(resp, "model_extra", None) or {}).get("error")
            last_error = f"empty reply{f': {err}' if err else ''}"
            log.warning("llm %s attempt %d/%d: %s", model, attempt + 1, MAX_ATTEMPTS, last_error)
            time.sleep(backoff_delay(attempt))
            continue
        try:
            return parse_json(content), spent
        except json.JSONDecodeError as exc:
            bad_replies += 1
            last_error = f"invalid JSON: {exc}"
            log.warning("llm %s returned invalid JSON (%d): %.200r", model, bad_replies, content)
            if bad_replies >= 2:
                break

    raise _ModelFailed(f"{model}: gave up after {spent['calls']} attempts: {last_error}")


def default_fallbacks(model: str) -> list[str]:
    """LLM_FALLBACK_MODELS (comma-separated) or, by default, the other Gemini model."""
    raw = os.environ.get("LLM_FALLBACK_MODELS")
    chain = [m.strip() for m in raw.split(",") if m.strip()] if raw is not None else [
        "google/gemini-3.1-pro-preview"]
    return [m for m in chain if m != model]


def chat_json(
    messages: list[dict],
    *,
    usage: Usage,
    model: str = DEFAULT_MODEL,
    fallbacks: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    use_cache: bool = True,
) -> Any:
    """Send a chat request that must come back as JSON. Returns the parsed JSON.
    Raises LLMError only when every model in the chain has failed."""
    chain = [model, *(default_fallbacks(model) if fallbacks is None else fallbacks)]
    for m in chain:
        if m not in ALLOWED_MODELS:
            raise ValueError(f"model {m!r} is not in the allowed list")

    errors = []
    for m in chain:
        params: dict[str, Any] = {
            "model": m,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        key = cache_key(params)
        if use_cache and (hit := _cache_get(key)) is not None:
            usage.record_cache_hit(m, hit["usage"])
            return hit["json"]
        try:
            data, spent = _call_model(params, usage)
        except _ModelFailed as exc:
            log.error("llm model failed, falling back: %s", exc)
            errors.append(str(exc))
            continue
        _cache_put(key, {"model": m, "json": data, "usage": spent})
        return data
    raise LLMError("all models failed: " + " | ".join(errors))
