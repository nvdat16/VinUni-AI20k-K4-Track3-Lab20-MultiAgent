"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.analysis_notes:
            raise AgentExecutionError("Writer requires analysis notes.")

        citations = []
        for source in state.sources:
            citation_id = source.metadata.get("citation_id")
            if citation_id and citation_id not in citations:
                citations.append(str(citation_id))

        citation_text = ", ".join(f"[{citation}]" for citation in citations) or "offline corpus"
        state.final_answer = "\n\n".join(
            [
                f"Answer for: {state.request.query}",
                "Summary:",
                self._summary_from_analysis(state.analysis_notes),
                "Evidence-aware synthesis:",
                state.analysis_notes,
                "Conclusion:",
                (
                    "A multi-agent workflow is most useful when the task benefits from distinct "
                    "research, analysis, and synthesis steps. For simple questions, the "
                    "coordination cost may outweigh the quality gain."
                ),
                f"Sources used: {citation_text}",
            ]
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"citation_count": len(citations)},
            )
        )
        state.add_trace_event("writer_complete", {"citation_count": len(citations)})
        return state

    def _summary_from_analysis(self, analysis_notes: str) -> str:
        lines = [
            line.removeprefix("- ").strip()
            for line in analysis_notes.splitlines()
            if line.startswith("- ")
        ]
        selected = lines[:3]
        if not selected:
            return analysis_notes.splitlines()[0]
        return " ".join(selected)
