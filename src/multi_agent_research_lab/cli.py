"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import render_trace_json
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline_state(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are a concise research assistant. Answer the user's research query for "
        f"{request.audience}. Use clear structure, mention uncertainty, and avoid inventing "
        "sources or citations."
    )
    response = llm.complete(system_prompt=system_prompt, user_prompt=request.query)
    state.final_answer = response.content
    state.add_trace_event(
        "baseline_llm_complete",
        {
            "model": get_settings().openai_model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    )
    return state


def _run_multi_agent_state(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    try:
        state = _run_baseline_state(_parse_query(query).query)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=2) from exc

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    token_event = state.trace[-1]["payload"] if state.trace else {}
    input_tokens = token_event.get("input_tokens")
    output_tokens = token_event.get("output_tokens")
    if input_tokens is not None or output_tokens is not None:
        console.print(
            f"Tokens: input={input_tokens or 0}, output={output_tokens or 0}"
        )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    queries: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Benchmark query. May be provided multiple times."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for report and trace artifacts."),
    ] = Path("reports"),
) -> None:
    """Run baseline and multi-agent benchmarks and write markdown/trace artifacts."""

    _init()
    benchmark_queries = queries or [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]
    store = LocalArtifactStore(output_dir)
    metrics = []
    for index, item in enumerate(benchmark_queries, start=1):
        _parse_query(item)
        for run_name, runner in (
            ("baseline", _run_baseline_state),
            ("multi-agent", _run_multi_agent_state),
        ):
            state, metric = run_benchmark(f"{run_name}-q{index}", item, runner)
            metrics.append(metric)
            store.write_text(f"traces/{run_name}-q{index}.json", render_trace_json(state))

    report = render_markdown_report(metrics)
    report_path = store.write_text("benchmark_report.md", report)
    console.print(Panel.fit(f"Wrote benchmark report to {report_path}", title="Benchmark"))


if __name__ == "__main__":
    app()
