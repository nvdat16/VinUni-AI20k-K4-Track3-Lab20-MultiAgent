"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "## Measurement Notes",
            "",
            "- Latency is wall-clock runtime for each runner/query pair.",
            "- Cost is a rough token-based estimate when model pricing is configured in code.",
            "- Quality is a proxy score for lab smoke testing; replace it with peer review.",
            "- Citation coverage counts answer lines that include bracketed source IDs.",
            "- Failure rate is 100% when a run records errors, otherwise 0%.",
            "",
            "## Follow-up Review",
            "",
            "- Check whether citations actually support the claims they appear beside.",
            "- Compare answer usefulness manually with the peer-review rubric.",
            "- Inspect trace events for retries, fallbacks, and route history.",
        ]
    )
    return "\n".join(lines) + "\n"
