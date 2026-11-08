#!/usr/bin/env python3
"""Run extraction jobs on sample documents against a running service.

Usage:
    python scripts/run_samples.py [--only <document_id>] [--url http://localhost:8080]
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "ref" / "sample" / "manifest.csv"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "ref" / "sample" / "sample-data-documents"
DEFAULT_SCHEMAS_DIR = PROJECT_ROOT / "ref" / "sample" / "schemas"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def process_document(
    row: dict[str, str],
    base_url: str,
    docs_dir: Path,
    schemas_dir: Path,
    outputs_dir: Path,
    poll_interval: float = 2.0,
) -> bool:
    """Submit one document job and poll until terminal, writing output on completion."""
    doc_id = row["document_id"]
    file_name = row["file"]
    schema_name = row["schema_name"]

    doc_path = docs_dir / file_name
    schema_path = schemas_dir / f"{schema_name}.json"

    if not doc_path.is_file():
        print(f"[{doc_id}] Error: document file not found: {doc_path}", file=sys.stderr)
        return False
    if not schema_path.is_file():
        print(f"[{doc_id}] Error: schema file not found: {schema_path}", file=sys.stderr)
        return False

    schema_text = schema_path.read_text(encoding="utf-8")

    # POST to /jobs
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        try:
            with open(doc_path, "rb") as f:
                files = {"file": (file_name, f)}
                data = {"schema": schema_text}
                resp = client.post("/jobs", files=files, data=data)
        except Exception as exc:
            print(f"[{doc_id}] Error posting to {base_url}/jobs: {exc}", file=sys.stderr)
            return False

        if resp.status_code != 202:
            print(f"[{doc_id}] POST /jobs failed ({resp.status_code}): {resp.text}", file=sys.stderr)
            return False

        job_info = resp.json()
        job_id = job_info["job_id"]
        print(f"[{doc_id}] Submitted job {job_id} (schema={schema_name})")

        # Poll GET /jobs/{job_id} every 2 s until terminal
        while True:
            time.sleep(poll_interval)
            try:
                get_resp = client.get(f"/jobs/{job_id}")
            except Exception as exc:
                print(f"[{doc_id}] Polling error for job {job_id}: {exc}", file=sys.stderr)
                continue

            if get_resp.status_code != 200:
                print(f"[{doc_id}] GET /jobs/{job_id} returned {get_resp.status_code}", file=sys.stderr)
                continue

            body = get_resp.json()
            status = body.get("status")

            if status in ("SUCCESS", "PARTIAL"):
                result = body.get("result") or {}
                # Result object plus a top-level "schema": "<schema_name>" key
                output_data = {"schema": schema_name, **result}
                outputs_dir.mkdir(parents=True, exist_ok=True)
                out_file = outputs_dir / f"{doc_id}.json"
                out_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[{doc_id}] Finished with status={status} -> {out_file.name}")
                return True
            elif status == "FAILED":
                err = body.get("error")
                print(f"[{doc_id}] Job {job_id} FAILED: {err}", file=sys.stderr)
                return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit sample documents and collect extraction outputs.")
    parser.add_argument("--only", help="Process only this document_id (e.g. boxscore_001)")
    parser.add_argument("--url", default="http://localhost:8080", help="Service base URL (default: http://localhost:8080)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to manifest.csv")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR, help="Path to sample-data-documents")
    parser.add_argument("--schemas-dir", type=Path, default=DEFAULT_SCHEMAS_DIR, help="Path to schemas")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="Path to outputs directory")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent requests (default: 4)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds (default: 2.0)")

    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"Error: manifest file not found: {args.manifest}", file=sys.stderr)
        return 1

    # Check health
    try:
        with httpx.Client(base_url=args.url, timeout=5.0) as client:
            h = client.get("/health")
            if h.status_code != 200:
                print(f"Warning: /health returned {h.status_code}: {h.text}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: Could not connect to {args.url}/health ({exc}). Proceeding anyway...", file=sys.stderr)

    with args.manifest.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.only:
        target = args.only.strip()
        target_stem = Path(target).stem
        rows = [r for r in rows if r["document_id"] in (target, target_stem) or r["file"] == target]
        if not rows:
            print(f"Error: document '{args.only}' not found in manifest.", file=sys.stderr)
            return 1

    print(f"Processing {len(rows)} document(s) with concurrency={args.concurrency}...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_document,
                row,
                args.url,
                args.docs_dir,
                args.schemas_dir,
                args.outputs_dir,
                args.poll_interval,
            ): row["document_id"]
            for row in rows
        }

        for fut in as_completed(futures):
            doc_id = futures[fut]
            try:
                ok = fut.result()
                if ok:
                    success_count += 1
            except Exception as exc:
                print(f"[{doc_id}] Unhandled error: {exc}", file=sys.stderr)

    print(f"\nDone: {success_count}/{len(rows)} documents completed successfully.")
    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

