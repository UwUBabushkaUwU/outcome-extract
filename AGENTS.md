# Project: document extraction service (Outcome take-home, AI/ML role)
Read ref/ASSIGNMENT.md and ref/contract/ before changing anything.

## Environment
- Python 3.13 (same locally and in the Docker image).
- Dev OS is Windows (PowerShell); service runs in Linux Docker. Use pathlib, no shell-specific scripts.
- No Docker on the dev machine. App must run with `python -m app`. Docker is verified only in GitHub Actions.
- Reference package (read-only, gitignored): ref/ASSIGNMENT.md, ref/contract/, ref/sample/, ref/tools/validate.py
- Validate: python ref/tools/validate.py outputs/ --schema-dir ref/sample/schemas --truth-dir ref/sample/sample-data-output-validated

## Stack
- FastAPI, SQLite (WAL) job table, worker processes. Port 8080.
- Must match ref/contract/openapi.yaml and ref/contract/output.schema.json exactly.

## LLM rules
- LLM calls ONLY via OpenAI-compatible client using env LLM_BASE_URL + LLM_API_KEY. Never hardcode keys or URLs.
- Allowed models ONLY: google/gemini-3.8-flash (default), google/gemini-3.1-pro-preview,
  anthropic/claude-sonnet-5.5 (escalation). No other providers or paid services.
- Budget is $10 total. All LLM calls go through app/llm.py with a disk cache (cache/).
  Never run all samples against a non-flash model without asking.

## Extraction rules
- Never hardcode schemas/field names: must work for unseen field lists.
- Null when a value isn't explicitly in the document. Never guess.

## Working rules
- Commit your own work after each finished task, in small logical commits with clear messages (feat:, fix:, test:, build:, docs:).
- Stage only your own files by explicit path. Never use `git add .` or `git add -A`.
- Before every commit, run `git status` and make sure ref/, .env, data/, cache/ and any sample documents or answer files are NOT staged.
- Never push unless asked. Never rewrite history.
- Do not touch files outside your assigned folders.
- Never read or print .env.

## Layout
- app/main.py (API), app/db.py (jobs), app/worker.py, app/llm.py, app/pipeline.py, app/normalize.py
- app/parsers/{pdf,docx,xlsx,csv}.py -> parse(path) -> Document (see app/parsers/base.py)
- scripts/run_samples.py, scripts/eval.py, tests/, outputs/, eval/scores.md