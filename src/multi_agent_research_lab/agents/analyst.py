"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.sources and not state.research_notes:
            raise AgentExecutionError("Analyst requires sources or research notes.")

        findings = []
        for source in state.sources[: state.request.max_sources]:
            citation_id = source.metadata.get("citation_id", "source")
            findings.append(f"- [{citation_id}] {source.snippet}")

        if not findings and state.research_notes:
            findings.append(f"- {state.research_notes}")

        state.analysis_notes = "\n".join(
            [
                "Key findings:",
                *findings,
                "",
                "Synthesis:",
                (
                    "- Use the cited evidence to separate architecture benefits from "
                    "coordination costs."
                ),
                "- Treat synthetic corpus evidence as benchmark material, not as real-world proof.",
                "- Preserve uncertainty when evidence is indirect or only thematically related.",
                "",
                "Evidence gaps:",
                (
                    "- Offline retrieval can support the lab task, but live/current claims need "
                    "external search."
                ),
            ]
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={"finding_count": len(findings)},
            )
        )
        state.add_trace_event("analyst_complete", {"finding_count": len(findings)})
        return state
