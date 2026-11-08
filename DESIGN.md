# Design

## Architecture

- **API.** FastAPI (`app/main.py`), port 8080. `POST /jobs` streams the upload to `data/files/<sha256>.<type>`, records a job, and returns 202. File type is detected from magic bytes.
- **Job table.** SQLite in WAL mode (`app/db.py`).
  - **Idempotency:** sha256(file) + sha256(schema) + organization id. Resubmissions return existing jobs.
  - **Tenant isolation:** scoped to `X-Organization-Id`; cross-tenant requests return 404.
- **Workers.** `python -m app` starts the API plus `WORKERS` worker processes. Workers claim the oldest PENDING job under `BEGIN IMMEDIATE` with a 60 s heartbeat-renewed lease. Terminal states are written conditionally so stale workers cannot overwrite results.
- **Crash recovery.** Expired leases return to PENDING. After 3 claims, jobs fail with `WORKER_LOST`. LLM spend persists across attempts to preserve budget caps.

## Pipeline (`app/pipeline.py`)

1. **Parse** (`app/parsers/`). PDF pages become text chunks (pages < 50 chars render to PNG at 150 dpi). DOCX preserves reading order. XLSX/CSV produce one chunk per sheet with A1 cell coordinates, merged cells filled, and empty cells dropped.
2. **Transcribe.** Flash transcribes scanned page images to markdown text.
3. **Field prep.** Flash infers field type, scope (document vs row), date format, and absent-defaults in batches of 15, memoized per schema.
4. **Document fields.** One call over the full text (capped at 150k characters).
5. **Row fields.** Tables are chunked along record boundaries (e.g. unit IDs or account codes) so records are never split across calls. Chunks carry header and context lines, executing concurrently.
6. **Normalize** (`app/normalize.py`). Coerce types, verify citations against source text/cells, fill document fields into rows, and assemble JSON entries.

## Models & Budget Guards

| Model | Role |
|---|---|
| `google/gemini-3.8-flash` | Primary model for transcription, prep, and extraction. Fast and cheap. |
| `anthropic/claude-haiku-5.5` | Cheap automatic fallback on failure. |
| `gemini-3.1-pro-preview`, `claude-sonnet-5.5` | Expensive models; require `LLM_ALLOW_EXPENSIVE=1`. |

Every call enforces minimal reasoning, per-task `max_tokens`, disk caching, and a per-job safety net of $1.50 (`MAX_JOB_COST_USD`) and 300 calls (`MAX_JOB_LLM_CALLS`).

## Iteration

The eval loop (`scripts/eval.py`) drove prompt and chunking improvements while spend guards cut run costs in half:

| Document | Baseline Score | Baseline Cost | Final Score | Final Cost |
|---|---|---|---|---|
| boxscore_001 | 1.000 | $0.028 | 1.000 | $0.028 |
| budget_001 | 0.979 | $0.303 | 0.982 | $0.184 |
| lease_001 | 0.850 | $0.275 | 0.850 | $0.157 |
| lease_002 | 0.825 | $0.099 | 0.825 | $0.041 |
| lease_003 | 0.850 | $0.325 | 0.850 | $0.169 |
| om_001 | 0.966 | $0.039 | 1.000 | $0.028 |
| rentroll_001 | 0.981 | $1.491 | 0.948 | $0.999 |
| t12_001 | 0.933 | $0.944 | 0.996 | $0.167 |
| **Mean / Total** | **0.923** | **$3.504** | **0.931** | **$1.775** |

- **Compact format regression:** An experiment removing cell tags and telling the model subtotals were not records dropped T12 to 0.625 (25 missing rows, 216 misses). The eval loop caught it immediately; reverting to explicit cell references and clarifying subtotal record rules achieved 0.996 (119/119 rows) while token reductions lowered cost from $0.94 to $0.17.
- **Per-job cap calibration:** The initial $0.50 budget cap halted the 325-unit rent roll midway (170 rows, 0.523 score). Clear error logging and skipped-chunk tracking made the root cause obvious; measuring the full workload showed ~$1.00 needed, so the runaway safety cap was raised to $1.50 / 300 calls.

## Null Discipline

Values survive only when backed by exact verbatim citations in the source text or matching spreadsheet cells; unverified values are nulled with confidence 0.3. Absent defaults (e.g. '0 if none') are populated without citation. This strict verification produced 0 hallucinations across all runs.

## Decisions Where the Brief Was Silent

- **Docker:** Local dev runs directly on Windows; Docker packaging is tested in GitHub Actions.
- **Deduplication:** When primary keys repeat (e.g. future resident entries in rent roll), the first occurrence is kept.
- **Empty & invalid uploads:** Handled cleanly with 400 Bad Request.

## With Another Week

- **Rent roll:** Recover the 17 missing rows (308/325 extracted) by refining boundary chunking around complex multi-unit blocks.
- **Lease calculated fields:** Escalate complex arithmetic (e.g. commencement date, total base rent) to Sonnet 3.5.
- **Derived citations:** Allow calculated fields to cite their arithmetic inputs instead of nulling.
- **Confidence calibration:** Align confidence scores against validation accuracy.
