# FTS benchmark

Benchmark full-text indexing, search latency, throughput, and recall on BEIR data.
Follow the [shared build setup](../README.md), then build from the repository root:

```bash
cmake --build build-bench --target fts_bench
```

## Inputs

- `--corpus`: BEIR `corpus.jsonl` with `_id`, `text`, and optional `title`.
- `--query`: BEIR `queries.jsonl` with `_id` and `text`.
- `--qrels`: directory containing `test.tsv`, or `train.tsv` as a fallback.
  The TSV needs a header and columns `query-id`, `corpus-id`, `score`, in that
  order; rows with integer scores greater than zero count as relevant.

Every command requires `--index`; build also requires `--corpus`, and search
requires both `--query` and `--qrels`. Build removes any existing index directory.

## Raw index

`--mode raw` uses the FTS indexer and RocksDB directly.

```bash
build-bench/bin/fts_bench --mode raw --cmd build \
  --index /tmp/zvec-fts-raw --corpus /path/to/beir/corpus.jsonl
build-bench/bin/fts_bench --mode raw --cmd search \
  --index /tmp/zvec-fts-raw --query /path/to/beir/queries.jsonl \
  --qrels /path/to/beir/qrels --threads 1 --topk 10 --default_operator or
```

## Collection API

`--mode db` uses the Collection API and requires its own index directory.

```bash
build-bench/bin/fts_bench --mode db --cmd build \
  --index /tmp/zvec-fts-db --corpus /path/to/beir/corpus.jsonl
build-bench/bin/fts_bench --mode db --cmd search \
  --index /tmp/zvec-fts-db --query /path/to/beir/queries.jsonl \
  --qrels /path/to/beir/qrels --threads 1 --topk 10
```

`--threads` controls raw build and both search modes (default: 16); `--topk`
defaults to 10. `--default_operator or|and` applies only to raw search (default: `or`).

## Raw index statistics

```bash
build-bench/bin/fts_bench --mode raw --cmd stats --index /tmp/zvec-fts-raw
```

Stats supports raw mode only. Its current posting scan skips iteration, so
vocabulary and posting counts report zero; these figures are not usable yet.
