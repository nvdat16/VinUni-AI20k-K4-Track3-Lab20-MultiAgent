"""Analyst agent skeleton."""

from typing import Any

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.sources and not state.research_notes:
            raise AgentExecutionError("Analyst requires sources or research notes.")

        llm_used = True
        input_tokens: int | None = None
        output_tokens: int | None = None
        try:
            response = self.llm_client.complete(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(state),
            )
            analysis_notes = response.content
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            if not analysis_notes:
                raise AgentExecutionError("Analyst LLM returned an empty response.")
        except AgentExecutionError as exc:
            llm_used = False
            analysis_notes = self._fallback_analysis(state)
            state.add_trace_event("analyst_llm_fallback", {"error": str(exc)})

        state.analysis_notes = analysis_notes
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={
                    "source_count": len(state.sources),
                    "llm_used": llm_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
        )
        state.add_trace_event(
            "analyst_complete",
            {
                "source_count": len(state.sources),
                "llm_used": llm_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        return state

    def _system_prompt(self) -> str:
        return (
            "You are the Analyst agent in a multi-agent research workflow. Convert evidence "
            "into structured analysis for a Writer agent. Do not invent citations. Preserve "
            "citation IDs exactly as provided, such as [A01] or [T01-SYN-A]. Mark synthetic "
            "benchmark evidence as synthetic when relevant."
        )

    def _user_prompt(self, state: ResearchState) -> str:
        source_blocks = []
        for index, source in enumerate(state.sources, start=1):
            citation_id = source.metadata.get("citation_id", f"S{index}")
            synthetic = source.metadata.get("is_synthetic")
            source_blocks.append(
                "\n".join(
                    [
                        f"Source {index}: [{citation_id}] {source.title}",
                        f"Synthetic: {synthetic}",
                        f"Snippet: {source.snippet}",
                    ]
                )
            )
        return "\n\n".join(
            [
                f"Research question: {state.request.query}",
                f"Audience: {state.request.audience}",
                "Research notes:",
                state.research_notes or "",
                "Retrieved sources:",
                "\n\n".join(source_blocks),
                (
                    "Write analysis notes with sections: Key claims, Trade-offs, Evidence "
                    "quality, Risks or gaps, Recommended synthesis angle."
                ),
            ]
        )

    def _fallback_analysis(self, state: ResearchState) -> str:
        findings = []
        for source in state.sources[: state.request.max_sources]:
            citation_id = source.metadata.get("citation_id", "source")
            findings.append(f"- [{citation_id}] {source.snippet}")

        if not findings and state.research_notes:
            findings.append(f"- {state.research_notes}")

        return "\n".join(
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
