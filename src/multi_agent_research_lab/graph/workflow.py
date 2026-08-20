"""LangGraph workflow skeleton."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> dict[str, BaseAgent]:
        """Create the agent registry used by the workflow runner.

        This milestone keeps orchestration explicit and deterministic. Later milestones can
        replace this registry with a compiled LangGraph graph without changing agent internals.
        """

        return {
            "supervisor": SupervisorAgent(),
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
            "critic": CriticAgent(),
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute supervisor routing and worker nodes until the workflow stops."""

        agents = self.build()
        supervisor = agents["supervisor"]
        while True:
            state = supervisor.run(state)
            route = state.route_history[-1]
            if route == "done":
                return state
            worker = agents.get(route)
            if worker is None:
                raise AgentExecutionError(f"Supervisor returned unknown route: {route}")
            try:
                state = worker.run(state)
            except StudentTodoError:
                raise
            except AgentExecutionError as exc:
                state.errors.append(str(exc))
                state.add_trace_event("agent_error", {"agent": route, "error": str(exc)})
