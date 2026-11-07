"""Extraction pipeline. STUB: parses the document, then returns one entry with every schema
field null.

Interface the worker relies on:
    run(ctx) -> PipelineOutput
The worker builds `result = {"entries": out.entries, "metadata": ...}` from ctx.usage,
so the pipeline only has to record its LLM calls through app.llm with usage=ctx.usage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app import parsers
from app.llm import Usage


@dataclass
class JobContext:
    job_id: str
    file_path: Path
    file_type: str  # pdf | docx | xlsx | csv
    filename: str | None
    schema: dict
    set_stage: Callable[[str], None]
    usage: Usage = field(default_factory=Usage)


@dataclass
class PipelineOutput:
    status: str  # SUCCESS or PARTIAL
    entries: list[dict[str, Any]]
    error: dict | None = None  # optional explanation for PARTIAL


def null_field() -> dict[str, Any]:
    return {"value": None, "citation": "", "confidence": 0.0}


def run(ctx: JobContext) -> PipelineOutput:
    ctx.set_stage("PARSING")
    # The type detected from the bytes at upload, never the user's filename.
    parsers.parse(ctx.file_path, ctx.file_type)
    ctx.set_stage("EXTRACTING")
    entry = {key: null_field() for key in ctx.schema["fields"]}
    return PipelineOutput(status="SUCCESS", entries=[entry])
