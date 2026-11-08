# Design notes

## Spend: the $3.48 run and the guards it led to

**What happened.** On 2026-11-07 a single run of all 8 sample documents cost $3.48, about a third of the $10 budget. We first took it for one box-score job, but the job table shows otherwise. The box score had been submitted earlier, under the stub pipeline. Idempotency returned that stored all-null result, so the box score cost $0. The $3.48 is the other seven documents:

| Document | Calls | Cost |
|---|---|---|
| rentroll_001 | 59 | $1.49 |
| t12_001 | 8 | $0.94 (one call fell back to gemini-3.1-pro-preview: $0.27) |
| lease_003 | 24 | $0.33 |
| budget_001 | 9 | $0.30 |
| lease_001 | 27 | $0.28 |
| lease_002 | 2 | $0.10 |
| om_001 | 3 | $0.04 |

**Why it was expensive.** The cause was the run itself. No single call misbehaved; what was missing were guards on our side:

- Gemini flash reasoning was on. Billed output was 3.4× the visible JSON, so about 70% of output tokens were hidden reasoning.
- Output length had no real limit: no `max_tokens` on field prep or document calls, and 32k on row chunks. A reply truncated by reasoning was retried at full price.
- Fallback went to the pro model by default.
- There was no per-job budget. Up to 8 attempts per model per call were allowed.

**Guards now in place** (`app/llm.py`, commits `42e5620` and `a8c57d0`):

- **Fallback:** flash → haiku only. Pro and sonnet require an explicit `allow_expensive=True` and are never reached by fallback.
- **Reasoning:** effort is set to `minimal` on every Gemini/OpenAI call.
- **max_tokens:** prep 2k, rows 8k, document 6k, transcription 4k, with a hard cap of 8k on every call.
- **Retries:** at most 4 requests per call, fallback included. A reply truncated at max_tokens is not retried.
- **Per-job caps:** 150 requests and $0.50 (`MAX_JOB_COST_USD`), checked before each request. When a cap is hit, the job stops with PARTIAL / `BUDGET_EXCEEDED`. Spend is stored on the job row, so the caps hold across restarts.
- **Logging:** every call is logged with model, cache hit/miss, tokens, reasoning tokens and cost.

With the guards, the box score cost $0.028 (4 calls).

**Process lessons:**
- Run one document after any pipeline change, and check the per-call log, before running the full sample set.
- Idempotency returns results from older pipeline versions. After a pipeline change, use a fresh data directory, or add a pipeline version to the idempotency key.

**Baseline** (`0dc4383`): mean 0.923 on the sample set, with 0 hallucinations. Seven of the eight outputs predate the guards, so their cost per document is not representative yet.
