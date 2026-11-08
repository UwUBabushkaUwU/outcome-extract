# Document extraction service

## Quick start

1. Set `LLM_BASE_URL` and `LLM_API_KEY` in `.env`.
2. Start the service: `docker compose up` (listens on `http://localhost:8080`).
3. Check health: `curl http://localhost:8080/health`.
4. Submit a job: `curl -F "file=@doc.pdf" -F "schema=<schema.json" http://localhost:8080/jobs`.
5. Poll result: `curl http://localhost:8080/jobs/<job_id>` until status is `SUCCESS`.

**Role: AI/ML Developer**

You give the service a commercial real estate document (PDF, DOCX, XLSX or CSV) and a field list (a JSON schema of field names and descriptions). It returns the extracted values as JSON, in the format of `ref/contract/output.schema.json`. `POST /jobs` accepts the upload and returns at once. Worker processes then do the work in the background. Each document is parsed to text, and scanned pages are transcribed by Gemini Flash. A table document is split into chunks of whole records. One model call extracts the document-level fields, and one call per chunk extracts the table rows. Every value must be backed by a citation that is found in the document; any value whose citation can't be found becomes null. You poll `GET /jobs/{id}` until the status is `SUCCESS`, `PARTIAL` or `FAILED`. See [DESIGN.md](DESIGN.md) for how it works and why.

## Run it

With Docker:

```bash
docker compose up
```

It listens on port 8080. It needs `LLM_BASE_URL` and `LLM_API_KEY`, set either in the environment or in a `.env` file in the repo root. `docker-compose.yml` reads `.env` if it exists. The job database and the LLM cache are mounted from `./data` and `./cache`.

Without Docker (Python 3.13):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
$env:LLM_BASE_URL = "https://openrouter.ai/api/v1"   # bash: export LLM_BASE_URL=...
$env:LLM_API_KEY  = "<your key>"
python -m app
```

`python -m app` starts the API on port 8080 and the worker processes. Some useful settings:

| Variable | Default | Meaning |
|---|---|---|
| `WORKERS` | 4 | worker processes |
| `MAX_JOB_COST_USD` | 0.50 | per-job spend cap; when reached, the job ends `PARTIAL` with `BUDGET_EXCEEDED` |
| `MAX_JOB_LLM_CALLS` | 150 | per-job request cap |
| `LLM_FALLBACK_MODELS` | `anthropic/claude-haiku-5.5` | cheap fallback models only |
| `LLM_ALLOW_EXPENSIVE` | 0 | set to 1 to allow gemini-3.1-pro-preview or claude-sonnet-5.5 as `LLM_MODEL` |

## Test it

With the service running (PowerShell; use `curl.exe`, not the `curl` alias):

```powershell
curl.exe -s http://localhost:8080/health

$r = curl.exe -s -X POST http://localhost:8080/jobs -H "X-Organization-Id: acme" `
  -F "file=@tests/fixtures/tiny.csv" -F "schema=<tests/fixtures/tiny_schema.json"
$id = ($r | ConvertFrom-Json).job_id

curl.exe -s http://localhost:8080/jobs/$id -H "X-Organization-Id: acme"
```

In `-F`, `@` uploads a file and `<` sends a file's contents as a text field. A job is visible only with the organization id it was created under. Any other id gets a 404.

Unit tests (no network, no LLM calls; a fixture blocks real clients):

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Reproduce the outputs and scores

These need the reference package in `ref/`, which is not committed, and a running service.

```bash
python scripts/run_samples.py            # all 8 samples -> outputs/<document>.json
python scripts/run_samples.py --only boxscore_001
python scripts/eval.py                   # runs ref/tools/validate.py, appends a row to eval/scores.md
```

A full run makes real model calls. Run `--only` on one document first, and check the `llm ... cost=` lines in the service log.

## Sample scores

The latest row of [eval/scores.md](eval/scores.md), git SHA `a8c57d0`, scored by `ref/tools/validate.py`. All 8 outputs are `VALID`.

| Document | Score |
|---|---|
| boxscore_001 | 1.000 |
| budget_001 | 0.979 |
| lease_001 | 0.850 |
| lease_002 | 0.825 |
| lease_003 | 0.850 |
| om_001 | 0.966 |
| rentroll_001 | 0.981 |
| t12_001 | 0.933 |
| **Mean** | **0.923** |

There were 0 hallucinations and 105 misses, for a total recorded cost of $3.50. Seven of the eight outputs come from a run before the cost guards existed; see DESIGN.md.

## CI

The dev machine has no Docker, so Docker is checked in GitHub Actions (`.github/workflows/docker.yml`) on every push. The workflow runs `docker compose up -d --build`, waits up to 60 s for `GET /health`, and checks that `POST /jobs` returns 202. It uses a dummy key, so it tests the build and the API, not extraction.
