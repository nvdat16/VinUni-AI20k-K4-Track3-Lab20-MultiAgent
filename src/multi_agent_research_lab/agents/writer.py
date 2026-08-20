"""Writer agent skeleton."""

from typing import Any

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.analysis_notes:
            raise AgentExecutionError("Writer requires analysis notes.")

        llm_used = True
        input_tokens: int | None = None
        output_tokens: int | None = None
        try:
            response = self.llm_client.complete(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(state),
            )
            final_answer = response.content
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            if not final_answer:
                raise AgentExecutionError("Writer LLM returned an empty response.")
        except AgentExecutionError as exc:
            llm_used = False
            final_answer = self._fallback_answer(state)
            state.add_trace_event("writer_llm_fallback", {"error": str(exc)})

        state.final_answer = final_answer
        citations = self._citations(state)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={
                    "citation_count": len(citations),
                    "llm_used": llm_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
        )
        state.add_trace_event(
            "writer_complete",
            {
                "citation_count": len(citations),
                "llm_used": llm_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        return state

    def _system_prompt(self) -> str:
        return (
            "You are the Writer agent in a multi-agent research workflow. Write a clear, "
            "natural answer for technical learners using only the provided analysis and "
            "sources. Preserve citation IDs exactly, such as [T01-SYN-A] or [A01]. Every "
            "factual claim bullet or paragraph must include at least one citation ID beside "
            "the claim. Do not put citations only in a final sources section. Do not invent "
            "citation IDs. Do not claim synthetic benchmark evidence is a real publication."
        )

    def _user_prompt(self, state: ResearchState) -> str:
        sources = []
        for index, source in enumerate(state.sources, start=1):
            citation_id = source.metadata.get("citation_id", f"S{index}")
            sources.append(f"- [{citation_id}] {source.title}: {source.snippet}")
        return "\n\n".join(
            [
                f"Research question: {state.request.query}",
                f"Audience: {state.request.audience}",
                "Analysis notes:",
                state.analysis_notes,
                "Available sources:",
                "\n".join(sources),
                (
                    "Write the final answer with: direct answer, key trade-offs, when to use "
                    "single-agent, when to use multi-agent, risks, and a short conclusion. "
                    "Attach source IDs like [T01-SYN-A] to each factual claim."
                ),
            ]
        )

    def _fallback_answer(self, state: ResearchState) -> str:
        citations = self._citations(state)
        citation_text = ", ".join(f"[{citation}]" for citation in citations) or "offline corpus"
        return "\n\n".join(
            [
                f"Answer for: {state.request.query}",
                "Summary:",
                self._summary_from_analysis(state.analysis_notes or ""),
                "Evidence-aware synthesis:",
                state.analysis_notes or "",
                "Conclusion:",
                (
                    "A multi-agent workflow is most useful when the task benefits from distinct "
                    "research, analysis, and synthesis steps. For simple questions, the "
                    "coordination cost may outweigh the quality gain."
                ),
                f"Sources used: {citation_text}",
            ]
        )

    def _citations(self, state: ResearchState) -> list[str]:
        citations = []
        for source in state.sources:
            citation_id = source.metadata.get("citation_id")
            if citation_id and citation_id not in citations:
                citations.append(str(citation_id))
        return citations

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
