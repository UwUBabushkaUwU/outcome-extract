#!/usr/bin/env python3
"""Run ref/tools/validate.py on outputs/ and record run results in eval/scores.md.

Usage:
    python scripts/eval.py [--outputs-dir outputs/] [--schema-dir ref/sample/schemas] [--truth-dir ref/sample/sample-data-output-validated]
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "ref" / "sample" / "schemas"
DEFAULT_TRUTH_DIR = PROJECT_ROOT / "ref" / "sample" / "sample-data-output-validated"
DEFAULT_VALIDATE_PY = PROJECT_ROOT / "ref" / "tools" / "validate.py"
DEFAULT_SCORES_MD = PROJECT_ROOT / "eval" / "scores.md"

SAMPLE_DOC_IDS = [
    "boxscore_001",
    "budget_001",
    "lease_001",
    "lease_002",
    "lease_003",
    "om_001",
    "rentroll_001",
    "t12_001",
]


def load_validate_module(validate_path: Path):
    """Dynamically import validate.py."""
    spec = importlib.util.spec_from_file_location("validate", str(validate_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validate module from {validate_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_git_short_sha() -> str:
    """Get the current short git SHA."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate extraction outputs and append results to eval/scores.md.")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="Path to outputs directory")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR, help="Path to schema directory")
    parser.add_argument("--truth-dir", type=Path, default=DEFAULT_TRUTH_DIR, help="Path to ground truth directory")
    parser.add_argument("--validate-py", type=Path, default=DEFAULT_VALIDATE_PY, help="Path to validate.py script")
    parser.add_argument("--scores-file", type=Path, default=DEFAULT_SCORES_MD, help="Path to scores.md file")

    args = parser.parse_args()

    if not args.outputs_dir.is_dir():
        print(f"Error: outputs directory not found: {args.outputs_dir}", file=sys.stderr)
        return 1
    if not args.schema_dir.is_dir():
        print(f"Error: schema directory not found: {args.schema_dir}", file=sys.stderr)
        return 1
    if not args.validate_py.is_file():
        print(f"Error: validate.py not found: {args.validate_py}", file=sys.stderr)
        return 1

    output_files = sorted(args.outputs_dir.glob("*.json"))
    if not output_files:
        print(f"Error: No .json files found in {args.outputs_dir}", file=sys.stderr)
        return 1

    # 1. Run ref/tools/validate.py via subprocess
    cmd = [
        sys.executable,
        str(args.validate_py),
        str(args.outputs_dir),
        "--schema-dir",
        str(args.schema_dir),
        "--truth-dir",
        str(args.truth_dir),
    ]
    print(f"Executing: {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd)

    # 2. Extract metrics and scores
    validate_mod = load_validate_module(args.validate_py)

    doc_scores: dict[str, float] = {}
    total_hallucinations = 0
    total_misses = 0
    total_cost_usd = 0.0

    for out_path in output_files:
        doc_id = out_path.stem
        try:
            res = validate_mod.load_json(out_path)
        except Exception as exc:
            print(f"Warning: Failed to load {out_path}: {exc}", file=sys.stderr)
            continue

        cost = res.get("metadata", {}).get("cost_usd", 0.0)
        if isinstance(cost, (int, float)):
            total_cost_usd += float(cost)

        schema_name = res.get("schema")
        if not schema_name:
            continue

        schema_path = args.schema_dir / f"{schema_name}.json"
        truth_path = args.truth_dir / out_path.name

        if schema_path.is_file() and truth_path.is_file():
            try:
                schema_obj = validate_mod.load_json(schema_path)
                truth_obj = validate_mod.load_json(truth_path)
                rep = validate_mod.score(res, truth_obj, schema_obj)
                doc_scores[doc_id] = rep["score"]
                total_hallucinations += len(rep.get("hallucinations", []))
                total_misses += len(rep.get("misses", []))
            except Exception as exc:
                print(f"Warning: Scoring failed for {doc_id}: {exc}", file=sys.stderr)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_sha = get_git_short_sha()

    scored_values = list(doc_scores.values())
    mean_score = sum(scored_values) / len(scored_values) if scored_values else 0.0

    # 3. Append to eval/scores.md
    args.scores_file.parent.mkdir(parents=True, exist_ok=True)
    if not args.scores_file.is_file() or not args.scores_file.read_text(encoding="utf-8").strip():
        # Initialize header
        header = (
            "# Evaluation Scores\n\n"
            "| Timestamp | Git SHA | Mean Score | "
            + " | ".join(SAMPLE_DOC_IDS)
            + " | Hallucinations | Misses | Total Cost ($) |\n"
            "|---|---|---|"
            + "|".join(["---"] * len(SAMPLE_DOC_IDS))
            + "|---|---|---|\n"
        )
        args.scores_file.write_text(header, encoding="utf-8")

    doc_cells = [f"{doc_scores[d]:.3f}" if d in doc_scores else "-" for d in SAMPLE_DOC_IDS]
    row = (
        f"| {timestamp} | {git_sha} | {mean_score:.3f} | "
        + " | ".join(doc_cells)
        + f" | {total_hallucinations} | {total_misses} | ${total_cost_usd:.4f} |\n"
    )

    with args.scores_file.open("a", encoding="utf-8") as f:
        f.write(row)

    print(f"\nAppended score row to {args.scores_file.relative_to(PROJECT_ROOT)}:")
    print(row.strip())

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
