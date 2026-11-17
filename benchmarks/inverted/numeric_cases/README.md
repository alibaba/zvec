# Numeric benchmarks

Benchmarks for inverted indexes over INT32 and INT64 data.

`profile.h/.cc` defines workload configuration; `dataset.h/.cc` owns dataset creation,
metadata, and batch reading; `numeric_cases.h/.cc` implements build, verify, and query.

Follow the [shared build setup](../../README.md), then run from the repository root:

```bash
cmake --build build --parallel --target inverted_numeric_bench

python3 benchmarks/inverted/numeric_cases/run_benchmarks.py --list

python3 benchmarks/inverted/numeric_cases/run_benchmarks.py \
  --type int64 --num-docs 100000

python3 benchmarks/inverted/numeric_cases/run_benchmarks.py \
  --type int32,int64 --num-docs 100000 --profile uniform
```

The script requires only `inverted_numeric_bench`. Runs require an explicit `--type` and
`--num-docs`; `--type all` selects both INT32 and INT64. `--type` and `--profile` accept
comma-separated names. `--list` previews workload combinations without running the executable;
omitting these filters includes both types and all distributions. Query cases are determined by
C++ when creating each dataset and recorded in `dataset.json`.

## Workload

One `NumericProfile` describes each workload. Python passes the selected configuration to C++,
which validates it and records the resolved configuration in the reports.

| Flag | Values | Default |
| --- | --- | --- |
| `--type` | `int32`, `int64`, or `all` | Required for runs |
| `--num-docs` | `100000` or `10000000` | Required for runs |
| `--profile` | `uniform`, `hotspot_moderate`, `hotspot_extreme` | All distributions |
| `--value-range` | `0_to_30000`, `0_to_type_max` | `0_to_30000` |
| `--value-order` | `ascending`, `shuffled` | `ascending` |
| `--null-ratio` | `0`, `0.1`, `0.5` | `0` |

The Python script translates `--profile` to the executable's `--distribution`.

Uniform samples every candidate value with equal probability. Hotspot selects distinct hot
values across the domain; remaining documents sample cold values outside that set. Each group
samples uniformly. `hotspot_extreme` concentrates documents on hot values more strongly than
`hotspot_moderate`; both use fixed presets identified by name in reports.

`0_to_30000` means `[0, 30000]`; `0_to_type_max` means `[0, INT32_MAX]` or `[0, INT64_MAX]`.
Both bounds are inclusive. Sampling and statistics do not allocate the full value domain.
Large domains can produce many one-document postings.

`shuffled` distributes values randomly across document IDs. `ascending` assigns non-NULL
values in nondecreasing order. Both use the same values and NULL positions. NULL ratios produce
zero, `num_docs / 10`, or `num_docs / 2` NULLs, with deterministic, randomly distributed positions.

Query operands are derived from the generated data. Reports retain compact cardinality,
frequency, and value-bound summaries.
Range targets refer to non-NULL documents and may be approximate; ordinary match ratios use all
documents. Query reports retain actual hit counts.

## Execution

Each repetition builds one dataset, verifies all applicable queries, then measures all queries.
These three phases run in separate processes. The query process reopens the index and warms up
for each case. Seed `20261115`, at least 0.2 seconds of warmup, and at least 1 second of measurement
per case are fixed in C++; each interval completes whole query-pool passes.
Use `--repetitions` to change the default of three repetitions.

Build creates the dataset and releases generation arrays before opening the index.
It then reads up to 32,768 records per batch, inserts them sequentially on one
thread, and seals the index. Verify scans this same file for each query operand and compares
every matching document ID with the index result. It does not regenerate data. Query uses
the verified counts and index without reading `data.bin`.

`data.bin` stores records in document ID order: one NULL byte followed by 4-byte INT32 or
8-byte INT64 native values. NULL payloads are zero-filled; document IDs are implicit. The
file has no header; `dataset.json` holds its metadata. This is a local run format.

Both dataset entry points require an existing directory; the CLI creates `--run_dir` in build mode.
`NumericDataset::Create(profile, directory)` writes both dataset files.
`NumericDataset::Load(directory)` loads metadata; the first `read_next_batch()` opens `data.bin`.
`rewind()` resets the document cursor, and `query_cases()` returns the cached query definitions.

The executable accepts a single `--run_dir` containing:

| File | Contents |
| --- | --- |
| `data.bin` | Original records in document ID order |
| `dataset.json` | `profile`, `seed`, `stats`, and `queries` (each with `name` and `inputs`) |
| `index/` | Built inverted index |
| `build.json` | Build timing, disk usage, memory, and environment |
| `verify.json` | Verification status, `match_counts`, and match ratios per query case |
| `query.json` | Query timing, `match_counts`, memory, and environment per query case |

Verification and query reports identify each case by `name`.
The C++ benchmark writes the phase reports; the dataset class writes the two dataset files.
To run the phases directly:

```bash
build/bin/inverted_numeric_bench --mode=build --data_type=int64 \
  --num_docs=100000 --distribution=hotspot_moderate \
  --value_range=0_to_type_max --value_order=shuffled --null_ratio=0.1 \
  --run_dir=build/results/inverted/numeric-example

build/bin/inverted_numeric_bench --mode=verify \
  --run_dir=build/results/inverted/numeric-example

build/bin/inverted_numeric_bench --mode=query \
  --run_dir=build/results/inverted/numeric-example
```

Build requires `--data_type` and defaults to 100,000 documents and `uniform`.
Verify and query restore the workload from `dataset.json`. The executable supports only these
three modes; run it without arguments or with `--help` to see its parameters.

Script results default to `build/results/inverted/numeric/<run-id>/`.
See [result metrics and output controls](../README.md#results).
Numeric query cases share a process: later cases may retain allocator memory, and peak RSS
accumulates across all cases in that query run.
