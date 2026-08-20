"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self.search_client.search(
            state.request.query,
            max_results=state.request.max_sources,
        )
        if not sources:
            raise AgentExecutionError("Researcher found no relevant sources in the offline corpus.")

        state.sources = sources
        note_lines = ["Research notes from offline corpus:"]
        for index, source in enumerate(sources, start=1):
            citation_id = source.metadata.get("citation_id", f"S{index}")
            note_lines.append(f"{index}. [{citation_id}] {source.title}: {source.snippet}")
        state.research_notes = "\n".join(note_lines)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"source_count": len(sources)},
            )
        )
        state.add_trace_event(
            "researcher_complete",
            {
                "source_count": len(sources),
                "citations": [source.metadata.get("citation_id") for source in sources],
            },
        )
        return state
