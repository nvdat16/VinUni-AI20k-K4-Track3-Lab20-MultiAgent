"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, TypeVar

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

F = TypeVar("F")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal local span context used alongside provider tracing."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def render_trace_json(state: ResearchState) -> str:
    """Serialize the local state trace for inspection or report artifacts."""

    payload = {
        "query": state.request.query,
        "route_history": state.route_history,
        "trace": state.trace,
        "errors": state.errors,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def configure_langsmith_tracing() -> bool:
    """Enable LangSmith tracing from settings when a LangSmith API key is configured."""

    settings = get_settings()
    if not settings.langsmith_api_key:
        return False

    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    return True


def langsmith_traceable(name: str, run_type: str = "chain") -> Any:
    """Return a LangSmith trace decorator that is disabled when not configured."""

    from langsmith import traceable

    settings = get_settings()
    enabled = configure_langsmith_tracing()
    return traceable(
        name=name,
        run_type=run_type,
        project_name=settings.langsmith_project,
        enabled=enabled,
        process_inputs=_serialize_langsmith_payload,
        process_outputs=_serialize_langsmith_payload,
    )


def _serialize_langsmith_payload(payload: Any) -> Any:
    """Convert Pydantic state objects into JSON-safe LangSmith payloads."""

    if isinstance(payload, ResearchState):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return {
            key: _serialize_langsmith_payload(value)
            for key, value in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        return [_serialize_langsmith_payload(value) for value in payload]
    return payload
