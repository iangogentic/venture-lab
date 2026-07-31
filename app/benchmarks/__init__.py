"""Benchmarks: fixed questions, and the metrics that say whether a run answered them.

The problem this package exists for: after changing a prompt or a model, the output
is different. Nothing about it being different says it is better. So a benchmark
pins a question and the shape of a healthy answer to it, and scoring turns a
finished run into numbers that two runs can be compared on.

Two halves:

* `spec` — what a benchmark is and how it is read off disk (`benchmarks/<name>/
  benchmark.json`).
* `scoring` — `score_run`, for whether one run is any good, and `compare_runs`, for
  whether two runs stayed stable.

Neither half decides anything. There is no pass mark here on purpose: a score is a
reading to go and look at, and the moment it became a gate the interesting cases —
an honestly unsized market, a search that ran and found nothing — would start being
optimised away.
"""

from app.benchmarks.scoring import (
    OVERLAP_THRESHOLD,
    SCORED_KINDS,
    STOPWORDS,
    GroundingScore,
    RunComparison,
    RunScore,
    StageScore,
    ThemeScore,
    VerdictPair,
    compare_runs,
    content_tokens,
    jaccard,
    normalise_text,
    score_run,
    similarity,
)
from app.benchmarks.spec import (
    BENCHMARK_FILENAME,
    BENCHMARKS_DIRNAME,
    Benchmark,
    Expectation,
    available,
    benchmark_dir,
    benchmark_path,
    benchmarks_root,
    load_all,
    load_benchmark,
)

__all__ = [
    "BENCHMARKS_DIRNAME",
    "BENCHMARK_FILENAME",
    "OVERLAP_THRESHOLD",
    "SCORED_KINDS",
    "STOPWORDS",
    "Benchmark",
    "Expectation",
    "GroundingScore",
    "RunComparison",
    "RunScore",
    "StageScore",
    "ThemeScore",
    "VerdictPair",
    "available",
    "benchmark_dir",
    "benchmark_path",
    "benchmarks_root",
    "compare_runs",
    "content_tokens",
    "jaccard",
    "load_all",
    "load_benchmark",
    "normalise_text",
    "score_run",
    "similarity",
]
