# Design

## Architecture

- **API.** FastAPI (`app/main.py`), on port 8080. `POST /jobs` streams the upload to `data/files/<sha256>.<type>`, records a job, and returns 202 at once. The file type comes from the file's bytes, not its name.
- **Job table.** SQLite in WAL mode (`app/db.py`).
  - **Idempotency:** the key is sha256(file) + sha256(parsed schema) + organization id, so resubmitting the same document and field list returns the existing job.
  - **Tenant isolation:** `GET /jobs/{id}` is scoped to the `X-Organization-Id` header, and another organization's job is a 404.
- **Workers.** `python -m app` starts the API plus `WORKERS` (default 4) worker processes, and a supervisor restarts any that die.
  - **Claiming:** a worker claims the oldest PENDING job under `BEGIN IMMEDIATE`, with a 60 s lease that a heartbeat renews.
  - **Terminal states:** they are written only `WHERE status='RUNNING' AND lease_owner=<me>`, so they never change, and a worker that lost its lease can't overwrite the worker that took over.
- **Crash recovery.** A job whose lease expires returns to PENDING. After 3 claims it is FAILED with `WORKER_LOST`. The job's LLM spend is stored on its row, so a recovered attempt continues the same budget.

## Pipeline (`app/pipeline.py`)

1. **Parse** (`app/parsers/`). PDF pages become text chunks, and a page with under 50 characters of text is rendered to PNG. DOCX is read in reading order. XLSX and CSV give one chunk per sheet, with A1 cell references, merged cells filled in, and empty cells dropped.
2. **Transcribe scanned pages.** Gemini Flash transcribes each page image, and the transcript replaces that page's text.
3. **Field prep.** Flash infers each field's type, scope (document or row), date format and absent-default, and sharpens its description. It runs in batches of 15 fields and is cached per schema.
4. **Document-level fields.** One call over the full text (capped at 150k characters, keeping the head and tail of each section).
5. **Row fields.** Only for schemas with a primary field. Table text is cut into chunks of about 60 lines, and only where a new record starts:
   - in spreadsheets, where the primary-key column takes a new value;
   - in PDFs, at a line starting with the document's dominant code shape (e.g. `6101-0000`).

   So a rent-roll unit and its charge lines are never split across calls. Each chunk carries the column headers and the previous 8 lines as context, and chunks run 4 at a time.
6. **Normalize** (`app/normalize.py`).
   - Coerce numbers ($ and commas stripped, parentheses mean negative), dates and lists.
   - Check each citation against the source.
   - Copy document-level fields into every row, and fill every schema key.

## Models

| Model | Role |
|---|---|
| `google/gemini-3.8-flash` | Every call: page transcription, field prep, extraction. Fast, cheap, multimodal. |
| `anthropic/claude-haiku-5.5` | Automatic fallback when Flash fails. `gpt-6-luna` is the other allowed fallback. |
| `gemini-3.1-pro-preview`, `claude-sonnet-5.5` | Never used automatically. They require `LLM_ALLOW_EXPENSIVE=1`. |

Every Flash call asks for minimal reasoning and has a `max_tokens` limit: 2k for prep, 8k for rows, 6k for document fields, 4k for transcription.

## Cost

From `cost_usd` and `model_calls` in each `outputs/*.json`:

| Document | Calls | Cost |
|---|---|---|
| boxscore_001 | 4 | $0.0281 |
| budget_001 | 9 | $0.3028 |
| lease_001 | 27 | $0.2751 |
| lease_002 | 2 | $0.0990 |
| lease_003 | 24 | $0.3252 |
| om_001 | 3 | $0.0393 |
| rentroll_001 | 59 | $1.4908 |
| t12_001 | 8 | $0.9440 |
| **Total** | **136** | **$3.50** |

About $3.48 of the $10 went on one full 8-document run on 2026-11-07, before per-job caps existed. The box score in that run cost $0, because idempotency returned an older stub result; it was re-run later for $0.03.

Three things made that run expensive:
- Flash reasoning was on, and billed output came to 3.4× the visible JSON.
- Output had no limit, apart from 32k on row calls.
- Fallback went to the pro model (one T12 call cost $0.27).

The guards added afterwards (commits `42e5620`, `a8c57d0`):
- **Calls:** cheap-only fallback, minimal reasoning, per-call-type `max_tokens`, and at most 4 requests per call. A truncated reply is not retried.
- **Per job:** at most 300 requests and $1.50 (`MAX_JOB_COST_USD`), checked before every request. Past either cap the job ends PARTIAL with `BUDGET_EXCEEDED`. The caps hold across restarts.
- **Logging:** every call is logged with model, cache hit or miss, tokens and cost.
- **Cache:** replies are cached on disk, so re-runs are free.

Seven of the eight outputs predate the guards, so their cost is not yet representative.

## Null discipline

A value survives only if its citation is found in the source.
- **Spreadsheets:** the cited cell must exist and hold the text.
- **Pages:** each snippet must appear in the text.

Anything else becomes null, with confidence 0.3. The one exception is a field's own absent-default (below). That is why the baseline has 0 hallucinations. The same rule causes most of the misses: in T12, 42 of 119 rows have `category` and `months_with_activity` nulled. Those values are derived, so nothing verbatim backs them.

## What broke and how it was handled

- **Cost overrun.** Described above.
- **Scanned PDFs.** lease_001 has 25 pages with no text layer and lease_003 has 21, so each page becomes one Flash transcription call. Citations are checked against the transcript.
- **The 325-unit rent roll.** About 3,050 lines, with one block per unit, and summary tables at the bottom that look like data. It went through 57 row-chunk calls cut on unit boundaries, and 319 of 325 rows were recovered.
- **429s and transient errors.** These are retried with exponential backoff and jitter, honouring `Retry-After`, up to 4 requests per call. If a call still fails, that step is lost and the job returns PARTIAL; it is never FAILED.

## Decisions where the brief was silent

- **Docker:** the dev machine has no Docker, so the image is built and health-checked in GitHub Actions on every push.
- **Duplicate primary keys:** the first row wins. The rent roll repeats some units in a later "Future Resident Details" section, and the sample answers keep the first occurrence.
- **Absent-defaults:** a description like "0 if none", "empty list if none" or "false otherwise" defines a value that has nothing to cite. Field prep extracts each field's default, and that value alone is kept without a citation.
- **File type:** taken from the bytes, so an upload with no extension or a wrong one still works.
- **Validation:** an unsupported type is a 400, per the contract. An empty file is also a 400, which was our choice.

## With another week

- **Per-field eval loop:** score every field across runs and fix the worst first.
- **Lease calculated fields:** a stronger model for those calls only, e.g. total base rent and rent commencement. The lease scores are 0.825–0.850.
- **Rent roll:** find and fix the 6 missing rows.
- **T12:** fix the misses by letting a derived value carry a verified citation to its inputs, instead of being nulled.
- **Confidence:** calibrate it against the sample answers.
