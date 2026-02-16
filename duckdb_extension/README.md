# zvec DuckDB Extension

A DuckDB extension that enables vector similarity search using zvec collections directly from SQL.

## Prerequisites

- zvec built from source (the parent directory) with all dependencies
- DuckDB source (added as a git submodule or placed in `duckdb/`)
- CMake 3.13+, C++17 compiler

## Build

First, build zvec from the repository root:

```bash
cd /path/to/zvec
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Then build the extension:

```bash
cd duckdb_extension
# Add DuckDB as a submodule (one-time setup)
git submodule add https://github.com/duckdb/duckdb.git duckdb
cd duckdb && git checkout v1.2.0 && cd ..

make
```

## Usage

```sql
-- Load the extension
LOAD 'build/release/zvec.duckdb_extension';

-- Create a collection
SELECT zvec_create('/tmp/my_col', '{
  "name": "articles",
  "fields": [
    {"name": "title", "type": "STRING"},
    {"name": "embedding", "type": "VECTOR_FP32", "dimension": 128,
     "index": {"type": "HNSW", "metric": "COSINE"}}
  ]
}');

-- Insert documents
SELECT zvec_insert('/tmp/my_col', 'doc1',
  '{"title": "hello world", "embedding": [0.1, 0.2, ...]}');

-- Vector similarity search (returns pk, score, and all forward fields)
SELECT * FROM zvec_search('/tmp/my_col', 'embedding',
  [0.1, 0.2, ...]::FLOAT[], 10);

-- Fetch a document by primary key
SELECT * FROM zvec_fetch('/tmp/my_col', 'doc1');
```

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `zvec_create(path, schema_json)` | Scalar | Create a new collection from JSON schema |
| `zvec_insert(path, pk, doc_json)` | Scalar | Insert a document with JSON field values |
| `zvec_search(path, field, vector, topk)` | Table | Vector similarity search returning top-k results |
| `zvec_fetch(path, pk)` | Table | Fetch a document by primary key |

## Schema JSON Format

```json
{
  "name": "collection_name",
  "fields": [
    {"name": "field_name", "type": "STRING"},
    {"name": "vec_field", "type": "VECTOR_FP32", "dimension": 128,
     "index": {"type": "HNSW", "metric": "COSINE", "m": 16, "ef_construction": 200}}
  ]
}
```

**Supported field types**: `BOOL`, `INT32`, `INT64`, `UINT32`, `UINT64`, `FLOAT`, `DOUBLE`, `STRING`, `BINARY`, `VECTOR_FP32`, `VECTOR_FP64`, `VECTOR_FP16`, `VECTOR_INT8`, `VECTOR_INT16`, `SPARSE_VECTOR_FP32`, `SPARSE_VECTOR_FP16`, and array variants.

**Supported index types**: `HNSW`, `FLAT`, `IVF`, `INVERT`

**Supported metrics**: `L2`, `IP`, `COSINE`
