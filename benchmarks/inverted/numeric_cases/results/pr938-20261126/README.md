# PR #938 benchmark reports

These reports were generated from the already executed benchmark JSON using the
existing numeric benchmark validation, aggregation, and rendering functions in
`run_benchmarks.py`, `benchmark_helpers.py`, and `report.py`. The example report is
unchanged.

| Version | Measured revision | 100K documents | 10M documents |
| --- | --- | --- | --- |
| Baseline | `8d978b06e1a6f16e0179e5d13bb0653598272113` | [72 workloads](baseline-100k/report.md) | [8 workloads](baseline-10m/report.md) |
| PR | `bdd20e0c88b21094780be14f70091cd2dd280db8` | [72 workloads](pr-100k/report.md) | [8 workloads](pr-10m/report.md) |

The 10M selection is W01, W06, W12, W18, W44, W50, W57, and W65 in the original
72-profile Cartesian-product order. Within each generated report, workloads are
numbered in selection order, as in the stock renderer; each adjacent `run.json`
records the original workload IDs and all source revisions.

Each workload has **one fresh build and three independent query processes** per
version, with 100 query iterations per case. Both versions read identical input
files generated with seed 19991230. Every query result was verified against the
original records, and match counts agree across versions and repetitions.

The stock `summarize_metrics` function uses arithmetic means. The generated
reports therefore show the mean of the three per-process query means; the PR
description's earlier comparison used per-case medians and separately measured
hot-query timings. The two aggregations have distinct meanings. Additional
open, first-query, hot-query, and RSS samples remain in the original raw result
archive.

The existing generation chain is:

1. `run_benchmarks.py` runs `prepare`, `build`, `verify`, and `query` and reads the
   resulting JSON.
2. Its validation functions check the dataset, build metrics, query coverage,
   operands, and hit-count shape.
3. `summarize_metrics` and `report_details` construct the report input.
4. `report.py:render_report()` generates
   `build/results/inverted/numeric/<run-id>/report.md`.

For this run, the already executed JSON was imported at step 2, preserving the
actual one-build/three-query sample counts. The Markdown files here are copies
of the renderer output, without a separate handwritten report template. The
measured production sources, benchmark runner, report generator, and example
report have no changes in this results update.
