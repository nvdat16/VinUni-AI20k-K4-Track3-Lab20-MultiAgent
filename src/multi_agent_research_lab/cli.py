"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

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


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are a concise research assistant. Answer the user's research query for "
        f"{request.audience}. Use clear structure, mention uncertainty, and avoid inventing "
        "sources or citations."
    )
    try:
        response = llm.complete(system_prompt=system_prompt, user_prompt=request.query)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=2) from exc

    state.final_answer = response.content
    state.add_trace_event(
        "baseline_llm_complete",
        {
            "model": get_settings().openai_model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    )
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    if response.input_tokens is not None or response.output_tokens is not None:
        console.print(
            f"Tokens: input={response.input_tokens or 0}, output={response.output_tokens or 0}"
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


if __name__ == "__main__":
    app()
