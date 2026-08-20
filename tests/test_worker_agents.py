from multi_agent_research_lab.agents import AnalystAgent, CriticAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse


class FakeLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(content=self.content, input_tokens=10, output_tokens=20)


def test_worker_agents_populate_research_analysis_and_final_answer() -> None:
    state = ResearchState(request=ResearchQuery(query="Compare single-agent and multi-agent"))

    state = ResearcherAgent().run(state)
    assert state.sources
    assert state.research_notes

    state = AnalystAgent().run(state)
    assert state.analysis_notes

    state = WriterAgent().run(state)
    assert state.final_answer
    assert state.agent_results[-1].agent == "writer"

    state = CriticAgent().run(state)
    assert state.agent_results[-1].agent == "critic"
    assert "Critic review" in state.agent_results[-1].content


def test_multi_agent_workflow_runs_to_done() -> None:
    state = ResearchState(request=ResearchQuery(query="Compare single-agent and multi-agent"))
    result = MultiAgentWorkflow().run(state)

    assert result.final_answer
    assert result.route_history == ["researcher", "analyst", "writer", "critic", "done"]


def test_analyst_and_writer_use_injected_llm_client() -> None:
    state = ResearchState(request=ResearchQuery(query="Compare single-agent and multi-agent"))
    state = ResearcherAgent().run(state)

    state = AnalystAgent(llm_client=FakeLLMClient("LLM analysis notes")).run(state)
    assert state.analysis_notes == "LLM analysis notes"
    assert state.agent_results[-1].metadata["llm_used"] is True

    state = WriterAgent(llm_client=FakeLLMClient("LLM final answer")).run(state)
    assert state.final_answer == "LLM final answer"
    assert state.agent_results[-1].metadata["llm_used"] is True


def test_writer_repairs_uncited_claim_lines() -> None:
    state = ResearchState(request=ResearchQuery(query="Compare single-agent and multi-agent"))
    state = ResearcherAgent().run(state)
    state.analysis_notes = "Analysis notes"

    writer = WriterAgent(
        llm_client=FakeLLMClient(
            "This factual claim has enough words but no citation.\n"
            "This claim already has a citation [T01-SYN-B]."
        )
    )
    state = writer.run(state)

    assert "This factual claim has enough words but no citation. [T01-SYN-A]" in (
        state.final_answer or ""
    )
    assert state.agent_results[-1].metadata["citation_repairs"] == 1
