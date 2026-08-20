from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report
    assert "Measurement Notes" in report


def test_benchmark_metrics_include_failure_and_citation_coverage() -> None:
    def runner(query: str) -> ResearchState:
        return ResearchState(
            request=ResearchQuery(query=query),
            final_answer=(
                "This supported claim has a citation [A01]\n"
                "This unsupported claim has no citation marker"
            ),
        )

    _, metrics = run_benchmark("demo", "Explain multi-agent systems", runner)

    assert metrics.failure_rate == 0.0
    assert metrics.citation_coverage == 0.5
