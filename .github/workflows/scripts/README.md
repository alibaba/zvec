# Continuous benchmark datasets

## Download the existing Cohere cases locally

`download_cohere.py` downloads the original VectorDBBench Cohere 1M and 10M
datasets (768 dimensions, cosine metric) into the same dataset root:

```bash
export DATASET_LOCAL_DIR="$PWD/datasets"
python zvec/.github/workflows/scripts/download_cohere.py \
  --root "$DATASET_LOCAL_DIR" --workers 8 --parallel-files 4
```

This requires PyArrow from `requirements-datasets.txt` and `curl`. It defaults
to VectorDBBench's AliyunOSS mirror; `--source S3` selects its AWS endpoint.
Use `--dataset 1m` or `--dataset 10m` to download one size. Downloads resume
checked byte ranges after interruptions and verify cached files using SHA-256.
`--workers` controls range requests per file; `--parallel-files` controls the
number of files downloaded at once.
Allow approximately 50.4 GB (47 GiB) for both sizes.

The native loader layout is preserved:

```text
datasets/cohere/cohere_medium_1m/
  shuffle_train.parquet
  test.parquet
  neighbors.parquet
  scalar_labels.parquet
  manifest.json
datasets/cohere/cohere_large_10m/
  shuffle_train-00-of-10.parquet ... shuffle_train-09-of-10.parquet
  test.parquet
  neighbors.parquet
  scalar_labels.parquet
  manifest.json
```

No conversion, normalization or GT generation is necessary. The original IDs,
shuffled insertion order, query vectors and size-specific GT are retained.
`scalar_labels.parquet` is included because the current built-in loader requests
it even for non-filtered cases. Filter-specific GT files are not downloaded.

With VectorDBBench and zvec installed, run the existing cases directly:

```bash
export USE_SHUFFLED_DATA=true
export DATASET_SOURCE=AliyunOSS
vectordbbench zvec --case-type Performance768D1M \
  --path "$PWD/bench-index/cohere1m" --m 15 --ef-search 180
vectordbbench zvec --case-type Performance768D10M \
  --path "$PWD/bench-index/cohere10m" --m 50 --ef-search 118 --is-using-refiner
```

The commands use unquantized vectors; add the desired `--quantize-type` when
comparing quantized indexes. The built-in reader may still issue remote size
checks for existing files. Downloading does not run benchmarks or enable CI.

## Prepare locally supplied TexMex / Big ANN files

The additional `download_text2image.py` and `prepare_local_datasets.py` scripts
support a local `gist.tar.gz` and complete public Text2Image evaluation sets.
They do not change the original cases or enable CI. From the workspace containing
`gist.tar.gz`, `zvec`, and `big-ann-benchmarks`:

```bash
export DATASET_LOCAL_DIR="$PWD/datasets"
SCRIPTS="$PWD/zvec/.github/workflows/scripts"
# Use a Python environment with requirements-datasets.txt installed, plus curl.
mkdir -p "$DATASET_LOCAL_DIR/_sources"
tar -xzf gist.tar.gz -C "$DATASET_LOCAL_DIR/_sources" \
  gist/gist_base.fvecs gist/gist_query.fvecs gist/gist_groundtruth.ivecs
python "$SCRIPTS/download_text2image.py" \
  --root "$DATASET_LOCAL_DIR/_sources/text2image1B"
python "$SCRIPTS/prepare_local_datasets.py" --root "$DATASET_LOCAL_DIR"
```

The downloader uses the first 10M rows of `base.1B.fbin`, checks HTTP ranges and
ETags, and derives the first 1M locally. It never downloads the full 1B base.
Do not substitute the separately hosted `base.10M.fbin`: its contents differ
from this prefix. It downloads the public 100K queries and official 10M top-100
GT, without training or heldout queries. Interrupted downloads retain a checked
chunk journal; rerun the same command to resume. Each final file has SHA-256 metadata.

The converter uses the archive's 1,000 GIST queries and matching published GT.
It computes exact `IndexFlatIP` top-100 GT for all 100K queries against the 1M
Text2Image base, with checkpointed batches. The 10M variant uses its official GT.
All vectors and original row ordering are preserved. The outputs deliberately
use distinct versioned directories to record the TexMex source and the full
Text2Image public-query selection:

| Custom dataset name | Custom dataset dir | Base rows | Queries | Dimension | Metric |
| --- | --- | ---: | ---: | ---: | --- |
| gist | gist_texmex_1m_v1 | 1,000,000 | 1,000 | 960 | L2 |
| text2image | text2image_1m_public100k_v1 | 1,000,000 | 100,000 | 200 | IP |
| text2image | text2image_10m_public100k_v1 | 10,000,000 | 100,000 | 200 | IP |

Each `$DATASET_LOCAL_DIR/<name>/<dir>` contains `train.parquet`, `test.parquet`,
`neighbors.parquet` and `manifest.json`. Sources remain under `_sources`;
`text2image-1M` is the generated GT in Big ANN binary format (header, IDs, scores).
For the original Big ANN loader, set `Text2Image1B(10).basedir` to the raw
`_sources/text2image1B` directory. For `Text2Image1B(1)`, also set
`gt_fn = "text2image-1M"`: upstream's 1M registration has no published GT path.
Both cropped base files have their actual row counts written into their headers.
Allow about 25 GB beyond the existing GIST archive for sources and Parquet files,
plus space for benchmark indexes. `--dataset gist` and `--dataset text2image`
can prepare independently; `--threads` controls Faiss CPU threads.

The one-shot pipeline also accepts the archive and runs extraction, download,
GT generation and conversion in sequence:

```bash
PYTHON_BIN=/path/to/venv/bin/python bash "$SCRIPTS/prepare_benchmark_data.sh" \
  "$DATASET_LOCAL_DIR" "$PWD/gist.tar.gz"
```

`DOWNLOAD_WORKERS` controls download concurrency (default 12). Completed files
are checked and reused on reruns. The final `ALL THREE DATASETS PREPARED` log line
is emitted only after every conversion succeeds. It does not run a benchmark,
publish metrics or change CI.

For example, run a prepared 10M dataset with an installed VectorDBBench and zvec:

```bash
vectordbbench zvec --case-type PerformanceCustomDataset \
  --custom-case-name Text2Image10MIPPublic100K \
  --custom-dataset-name text2image \
  --custom-dataset-dir text2image_10m_public100k_v1 \
  --custom-dataset-size 10000000 --custom-dataset-dim 200 \
  --custom-dataset-metric-type IP --custom-dataset-file-count 1 \
  --skip-custom-dataset-use-shuffled --custom-dataset-with-gt \
  --k 100 --m 32 --ef-search 200 --path "$PWD/bench-index/text2image10m"
```

Use the matching name/directory/size/dimension/metric from the table for other
datasets. The existing `run_vdb.sh` still uses its original HDF5/10K-query dataset
versions; use the explicit custom-dataset arguments for these local editions.

## Original continuous benchmark preparation

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

GIST's HDF5 edition uses its embedded queries and matching neighbors. The local
TexMex edition above uses the archive's query file and its paired GT file.
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
