"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.state import ResearchState


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.
    """

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
