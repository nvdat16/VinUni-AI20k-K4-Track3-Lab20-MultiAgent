"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

ROUGH_COST_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return benchmark metrics for one runner/query pair."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:
        state = ResearchState(request=ResearchQuery(query=query), errors=[str(exc)])
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimate_cost_usd(state),
        quality_score=score_quality_proxy(state),
        citation_coverage=estimate_citation_coverage(state),
        failure_rate=1.0 if state.errors else 0.0,
        notes=build_metric_notes(state),
    )
    return state, metrics


def estimate_cost_usd(state: ResearchState) -> float | None:
    """Estimate provider cost from token metadata collected in traces/agent results."""

    input_tokens, output_tokens = collect_token_usage(state)
    if input_tokens == 0 and output_tokens == 0:
        return None

    model = get_settings().openai_model
    input_rate, output_rate = ROUGH_COST_PER_1M_TOKENS.get(model, (0.0, 0.0))
    if input_rate == 0.0 and output_rate == 0.0:
        return None
    return (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)


def collect_token_usage(state: ResearchState) -> tuple[int, int]:
    """Collect input/output tokens from agent metadata, falling back to trace events."""

    input_tokens = 0
    output_tokens = 0
    for result in state.agent_results:
        input_tokens += int(result.metadata.get("input_tokens") or 0)
        output_tokens += int(result.metadata.get("output_tokens") or 0)

    if input_tokens or output_tokens:
        return input_tokens, output_tokens

    for event in state.trace:
        payload = event.get("payload", {})
        input_tokens += int(payload.get("input_tokens") or 0)
        output_tokens += int(payload.get("output_tokens") or 0)
    return input_tokens, output_tokens


def estimate_citation_coverage(state: ResearchState) -> float | None:
    """Estimate claim citation coverage from bracketed source IDs in the final answer."""

    if not state.final_answer:
        return 0.0 if state.errors else None

    claim_lines = [
        line
        for line in state.final_answer.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and len(line.strip().split()) >= 6
    ]
    if not claim_lines:
        return None

    cited_lines = [line for line in claim_lines if re.search(r"\[[A-Za-z0-9_-]+\]", line)]
    return len(cited_lines) / len(claim_lines)


def score_quality_proxy(state: ResearchState) -> float:
    """Return a simple 0-10 proxy score for lab benchmarking.

    Peer review should replace this score for final grading.
    """

    if state.errors or not state.final_answer:
        return 0.0

    score = 4.0
    if len(state.final_answer.split()) >= 120:
        score += 1.5
    if state.sources:
        score += 1.5
    if estimate_citation_coverage(state) and estimate_citation_coverage(state) > 0:
        score += 1.0
    if state.analysis_notes:
        score += 1.0
    if any(result.metadata.get("llm_used") for result in state.agent_results):
        score += 1.0
    return min(score, 10.0)


def build_metric_notes(state: ResearchState) -> str:
    """Summarize trace details useful for a benchmark table."""

    input_tokens, output_tokens = collect_token_usage(state)
    parts = [
        f"routes={','.join(state.route_history) or 'n/a'}",
        f"sources={len(state.sources)}",
        f"tokens={input_tokens}/{output_tokens}",
    ]
    if state.errors:
        parts.append(f"errors={len(state.errors)}")
    return "; ".join(parts)
