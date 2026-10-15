# Continuous benchmark datasets

The benchmark keeps the existing Cohere 1M, Cohere 10M and OpenAI 500K cases,
and adds GIST-1M (L2) and Text2Image-1M (IP). Each runs with int8, int4, fp16
and fp32: 20 invocations in total. The workflow remains disabled (`if: false`).
Running the shell script manually still runs benchmarks and publishes metrics;
use `--dry-run` to inspect commands without installing, downloading, or publishing.

## Prepare data on the runner

From the zvec checkout:

```bash
python3 -m venv .venv-datasets
source .venv-datasets/bin/activate
pip install -r .github/workflows/scripts/requirements-datasets.txt
export DATASET_LOCAL_DIR=/opt/bench-datasets
python .github/workflows/scripts/prepare_datasets.py --root "$DATASET_LOCAL_DIR"
```

The directory must be writable by the runner. The default is
`/tmp/vectordb_bench/dataset`, matching VectorDBBench. Prefer persistent storage
for repeated runs. The script supports `--dataset gist` or `--dataset text2image`,
`--raw-dir`, `--threads`, `--force` (rebuild outputs from verified cached sources),
and `--recompute-gt` (also regenerate GIST GT with exact L2 search).

Sources and fixed selections:

| Dataset | Download | Base / queries | GT |
| --- | --- | --- | --- |
| GIST | [ANN-Benchmarks HDF5](https://ann-benchmarks.com/gist-960-euclidean.hdf5) | 1M x 960 float32; all 1,000 published test queries | Published first 100 neighbors, or exact `IndexFlatL2` with `--recompute-gt` |
| Text2Image | [Yandex base](https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin), [public queries](https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin) | First 1M x 200 float32 base vectors; first 10,000 public queries | Exact `IndexFlatIP`, top 100, recomputed against this 1M subset |

GIST's HDF5 query selection differs from the larger query file in the original
TexMex archive; its embedded neighbors match the selected HDF5 test set.
Text2Image vectors are kept unchanged: no normalization, random sampling or
shuffling. GT from the 10M/1B base is never reused. Both datasets assign base IDs
by original row order and query IDs by selected query order, starting at zero.

The downloader requests only 800,000,008 bytes of the Text2Image base and
8,000,008 bytes of the query file, including fbin headers. If a server ignores
HTTP Range, the reader still stops after the requested prefix and closes the
connection; it never downloads the full 800 GB base. It checks response ranges,
lengths and fbin dimensions, retries failed transfers up to three times, and
publishes downloads only after completion. Failed transfers restart from zero.

The scripts write:

```text
$DATASET_LOCAL_DIR/
  _sources/                         # Raw downloads and metadata
  gist/gist_1m_v1/
    train.parquet
    test.parquet
    neighbors.parquet
    manifest.json
  text2image/text2image_1m_v1/
    train.parquet
    test.parquet
    neighbors.parquet
    manifest.json
```

`train` and `test` have `id: int64, emb: fixed_size_list<float32>`;
`neighbors` has `id: int64, neighbors_id: list<int64>`. Neighbor IDs are ordered
by ascending L2 squared distance or descending inner product. Query IDs and
row order match exactly between `test` and `neighbors`. No scalar-label file is
needed for these unfiltered cases.

Manifests record selections, source URLs, ETags, source/output SHA-256 hashes,
GT method and Faiss version. Validated output hashes allow offline reuse;
corrupt or incomplete outputs are rebuilt in a staging directory. The manifest
is published last, and an exclusive preparation lock prevents concurrent writers.
Hashes detect local corruption; they are not publisher-provided checksums.
To adopt changed upstream contents, use a new raw directory and `--force`, and
bump the dataset version before comparing with earlier runs.

Allow approximately 15–20 GB of free disk for sources, prepared files and rebuild
staging, plus benchmark indexes/results (kept for each run). Exact Text2Image GT
holds an approximately 0.8 GB flat index and streams query batches. Recomputing
GIST GT additionally needs approximately 3.84 GB for the flat index. GT generation
can take substantial CPU time but runs once, before timed benchmarking. The
default Faiss thread count is capped at 16 and can be changed with `--threads`.

Source attribution: GIST is from [TexMex](http://corpus-texmex.irisa.fr/), mirrored
by [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks). Text2Image is
Yandex Text-to-Image, released under CC BY 4.0; see the
[Big ANN dataset page](https://big-ann-benchmarks.com/neurips21.html#benchmark-datasets).

## Continuous benchmark configuration

```bash
# Prints all 20 commands without side effects; works outside GitHub Actions.
bash .github/workflows/scripts/run_vdb.sh --dry-run
```

The actual run installs VectorDBBench from `VDBBENCH_ROOT` (default
`/opt/VectorDBBench`), prepares/verifies the two datasets, then executes the
original and new cases. The installed VectorDBBench must expose
`PerformanceCustomDataset` and the `--custom-dataset-*` options visible in the
dry-run output. No VectorDBBench source modification is required.

The new cases use `m=32`, `ef-search=200`, `k=100`, and the existing concurrency
list `12,14,16,18,20`. These are initial baselines, not tuned recall guarantees.
The original cases retain their m/ef/refiner settings. As before, fp32 omits
`--quantize-type`; other formats pass it explicitly.

`GIST1ML2` and `Text2Image1MIP` are monitoring identities, while both invoke the
CLI case `PerformanceCustomDataset`. Their index paths, logs, result directories,
Prometheus labels and Pushgateway grouping keys are separate. Existing monitoring
group keys remain unchanged. Pipeline failures propagate through `tee`.

Environment overrides:

- `DATASET_LOCAL_DIR`: persistent data root; applies to old and new cases.
- `VDBBENCH_ROOT`: installed VectorDBBench checkout.
- `BENCH_RUN_ROOT`: logs, results and indexes; defaults to
  `zvec/bench-runs/<branch-date-sha>/`.
- `PUSHGATEWAY_URL`: metrics destination; defaults to the existing endpoint.

## Offline tests

```bash
pip install pytest
python -m pytest -o addopts='' -o filterwarnings='' \
  .github/workflows/scripts/test_prepare_datasets.py
bash -n .github/workflows/scripts/run_vdb.sh
```

Tests use small local fixtures and mocked HTTP responses. They compare generated
L2/IP GT with NumPy brute force, exercise bounded downloads, corrupt caches and
Parquet schemas, and check preservation of the original command matrix and the
disabled workflow. They do not run the full million-vector benchmarks.
