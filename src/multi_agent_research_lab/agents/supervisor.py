"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routing policy:
        - Stop on recorded errors so the caller can inspect/fallback safely.
        - Stop when max iterations is reached.
        - Research first when sources and notes are missing.
        - Analyze once research notes or sources exist.
        - Write once analysis notes exist.
        - Critique once a final answer exists.
        - Stop after the critic has reviewed the answer.
        """

        route = self.decide_next_route(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor_route",
            {
                "route": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
                "has_critic_review": self.has_critic_review(state),
                "errors": len(state.errors),
            },
        )
        return state

    def decide_next_route(self, state: ResearchState) -> str:
        """Return the next workflow route and record stop errors when needed."""

        settings = get_settings()
        if state.errors:
            return "done"
        if state.iteration >= settings.max_iterations:
            state.errors.append(f"Max iterations reached: {settings.max_iterations}")
            return "done"
        if state.final_answer:
            if not self.has_critic_review(state):
                return "critic"
            return "done"
        if not state.sources and not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        return "writer"

    def has_critic_review(self, state: ResearchState) -> bool:
        """Return whether CriticAgent has already reviewed this state."""

        return any(result.agent == AgentName.CRITIC for result in state.agent_results)
