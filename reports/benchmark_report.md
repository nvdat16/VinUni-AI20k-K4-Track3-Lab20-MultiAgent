# Benchmark Report

## Summary

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline-q1 | 7.46 | 0.0002 | 5.5 | 0% | 0% | routes=n/a; sources=0; tokens=49/372 |
| multi-agent-q1 | 14.89 | 0.0012 | 10.0 | 81% | 0% | routes=researcher,analyst,writer,critic,done; sources=5; tokens=2917/1217 |

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
