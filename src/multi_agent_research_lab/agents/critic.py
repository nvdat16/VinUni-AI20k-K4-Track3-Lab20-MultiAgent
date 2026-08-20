"""Optional critic agent skeleton for bonus work."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append review findings."""

        if not state.final_answer:
            raise AgentExecutionError("Critic requires a final answer.")

        known_citations = self._known_citations(state)
        cited_ids = self._cited_ids(state.final_answer)
        unsupported_citations = sorted(cited_ids - known_citations)
        citation_coverage = self._citation_coverage(state.final_answer)
        synthetic_ids = self._synthetic_citations(state)
        mentions_synthetic = "synthetic" in state.final_answer.lower()

        issues: list[str] = []
        if not cited_ids and known_citations:
            issues.append("Final answer does not cite any retrieved source IDs.")
        if unsupported_citations:
            issues.append(
                "Final answer cites unknown source IDs: "
                + ", ".join(f"[{citation}]" for citation in unsupported_citations)
            )
        if citation_coverage < 0.25 and known_citations:
            issues.append(f"Low citation coverage: {citation_coverage:.0%}.")
        if synthetic_ids and not mentions_synthetic:
            issues.append(
                "Final answer uses synthetic benchmark evidence but does not clearly label it."
            )

        passed = not issues
        findings = [
            "Critic review:",
            f"- Passed: {passed}",
            f"- Citation coverage: {citation_coverage:.0%}",
            f"- Known citations: {', '.join(sorted(known_citations)) or 'none'}",
            f"- Cited in answer: {', '.join(sorted(cited_ids)) or 'none'}",
        ]
        if synthetic_ids:
            findings.append(f"- Synthetic evidence IDs: {', '.join(sorted(synthetic_ids))}")
        if issues:
            findings.append("- Issues:")
            findings.extend(f"  - {issue}" for issue in issues)
        else:
            findings.append("- Issues: none")

        content = "\n".join(findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=content,
                metadata={
                    "passed": passed,
                    "issue_count": len(issues),
                    "citation_coverage": citation_coverage,
                    "unsupported_citations": unsupported_citations,
                },
            )
        )
        state.add_trace_event(
            "critic_complete",
            {
                "passed": passed,
                "issue_count": len(issues),
                "citation_coverage": citation_coverage,
            },
        )
        return state

    def _known_citations(self, state: ResearchState) -> set[str]:
        return {
            str(source.metadata["citation_id"])
            for source in state.sources
            if source.metadata.get("citation_id")
        }

    def _synthetic_citations(self, state: ResearchState) -> set[str]:
        return {
            str(source.metadata["citation_id"])
            for source in state.sources
            if source.metadata.get("citation_id") and source.metadata.get("is_synthetic")
        }

    def _cited_ids(self, text: str) -> set[str]:
        return set(re.findall(r"\[([A-Za-z0-9_-]+)\]", text))

    def _citation_coverage(self, text: str) -> float:
        claim_lines = [
            line
            for line in text.splitlines()
            if line.strip()
            and not line.lstrip().startswith("#")
            and len(line.strip().split()) >= 6
        ]
        if not claim_lines:
            return 0.0
        cited_lines = [line for line in claim_lines if self._cited_ids(line)]
        return len(cited_lines) / len(claim_lines)
