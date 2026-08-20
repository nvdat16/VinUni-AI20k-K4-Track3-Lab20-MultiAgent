"""LangGraph workflow for the multi-agent research system."""

from typing import Any

from langgraph.graph import END, StateGraph

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
from multi_agent_research_lab.observability.tracing import langsmith_traceable


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.agents: dict[str, BaseAgent] = {
            "supervisor": SupervisorAgent(),
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
            "critic": CriticAgent(),
        }

    def build(self) -> Any:
        """Build and compile the LangGraph workflow."""

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._run_agent("supervisor"))
        graph.add_node("researcher", self._run_agent("researcher"))
        graph.add_node("analyst", self._run_agent("analyst"))
        graph.add_node("writer", self._run_agent("writer"))
        graph.add_node("critic", self._run_agent("critic"))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )
        for node_name in ("researcher", "analyst", "writer", "critic"):
            graph.add_edge(node_name, "supervisor")

        return graph.compile(name="multi-agent-research-workflow")

    def _run_agent(self, name: str) -> Any:
        """Create a LangGraph node that executes one agent and records recoverable errors."""

        agent = self.agents[name]

        def node(state: ResearchState) -> ResearchState:
            try:
                return agent.run(state)
            except StudentTodoError:
                raise
            except AgentExecutionError as exc:
                state.errors.append(str(exc))
                state.add_trace_event("agent_error", {"agent": name, "error": str(exc)})
                return state

        return node

    def _next_route(self, state: ResearchState) -> str:
        """Return the route chosen by SupervisorAgent for LangGraph conditional edges."""

        if not state.route_history:
            raise AgentExecutionError("Supervisor did not record a route.")

        route = state.route_history[-1]
        allowed_routes = {"researcher", "analyst", "writer", "critic", "done"}
        if route not in allowed_routes:
            raise AgentExecutionError(f"Supervisor returned unknown route: {route}")
        return route

    @langsmith_traceable(name="multi_agent_workflow", run_type="chain")
    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled LangGraph graph and return final state."""

        result = self.build().invoke(state)
        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
