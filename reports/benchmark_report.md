# Benchmark Report

## Summary

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline-q1 | 7.50 | 0.0002 | 5.5 | 0% | 0% | routes=n/a; sources=0; tokens=49/388 |
| multi-agent-q1 | 16.23 | 0.0012 | 10.0 | 100% | 0% | routes=researcher,analyst,writer,critic,done; sources=5; tokens=2891/1248 |

## Measurement Notes

- Latency is wall-clock runtime for each runner/query pair.
- Cost is a rough token-based estimate when model pricing is configured in code.
- Quality is a proxy score for lab smoke testing; replace it with peer review.
- Citation coverage counts answer lines that include bracketed source IDs.
- Failure rate is 100% when a run records errors, otherwise 0%.

## Follow-up Review

- Check whether citations actually support the claims they appear beside.
- Compare answer usefulness manually with the peer-review rubric.
- Inspect trace events for retries, fallbacks, and route history.


## Failure Mode and Fix

- Failure mode: The Writer agent may produce fluent claims without source IDs, causing low citation coverage and making it hard to verify whether each claim is grounded in retrieved evidence.

- Fix: The Writer prompt now requires every factual claim to include citation IDs, and the Writer post-processes claim lines to append a known citation when the LLM omits one. The Critic agent then checks citation coverage, unknown citations, and whether synthetic evidence is clearly labeled.