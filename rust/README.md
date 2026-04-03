# zvec Rust SDK

Safe, idiomatic Rust bindings for the [zvec](https://github.com/amap-ai/zvec) vector database.

## Architecture

The SDK is organized as a Cargo workspace with two crates:

- **`zvec-sys`** — Low-level FFI bindings to the zvec C-API (`libzvec_c_api`)
- **`zvec`** — Safe, high-level Rust wrapper with RAII, Builder patterns, and type safety

## Prerequisites

1. **Build zvec from source** (the C/C++ library):

```bash
# From the project root
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_C_API=ON
make -j$(nproc)
```

2. **Set environment variables** pointing to the built library:

```bash
export ZVEC_LIB_DIR=/path/to/zvec/build/src/binding/c
export ZVEC_INCLUDE_DIR=/path/to/zvec/src/include
```

## Build

```bash
cd rust
cargo build
```

## Testing

### Unit Tests

Pure logic tests for enum conversions, error handling, etc. (no C library required):

```bash
cd rust
cargo test --lib
```

### Integration Tests

Full end-to-end tests requiring the zvec C library:

```bash
cd rust
cargo test --test integration_test
```

### Run All Tests

```bash
cd rust
cargo test
```

## Benchmark

Performance benchmarks using [Criterion.rs](https://github.com/bheisler/criterion.rs):

```bash
cd rust
cargo bench
```

Benchmark results with HTML reports are generated in `target/criterion/`.

Benchmarked operations include:
- **Document creation** — empty, with fields, with vectors (32d–1024d)
- **Field access** — get_pk, get_string, get_i64, get_vector_f32
- **Insert** — batch sizes 1, 10, 100
- **Vector query** — topk 1, 10, 50 on 1000 docs
- **Schema creation** — simple and complex schemas
- **Type conversions** — pure logic enum conversions

## Code Coverage

Generate code coverage reports using [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov):

```bash
# Install (one-time)
cargo install cargo-llvm-cov

# HTML report
./scripts/coverage.sh --html

# Text summary
./scripts/coverage.sh --text

# LCOV format (for CI integration)
./scripts/coverage.sh --lcov
```

## Quick Start

```rust
use zvec::*;

fn main() -> zvec::Result<()> {
    initialize(None)?;

    // Define schema with a vector field
    let schema = CollectionSchema::builder("my_collection")
        .add_field(FieldSchema::new("id", DataType::String, false, 0))
        .add_vector_field("embedding", DataType::VectorFp32, 128,
            IndexParams::hnsw(MetricType::Cosine, 16, 200))
        .build()?;

    // Create and open collection
    let collection = Collection::create_and_open("./data", &schema, None)?;

    // Insert a document
    let mut doc = Doc::new()?;
    doc.set_pk("doc1");
    doc.add_string("id", "doc1")?;
    doc.add_vector_f32("embedding", &vec![0.1; 128])?;
    collection.insert(&[&doc])?;

    // Vector similarity search
    let query = VectorQuery::new("embedding", &vec![0.2; 128], 10)?;
    let results = collection.query(&query)?;
    for result in &results {
        println!("pk={}, score={:.4}", result.get_pk().unwrap_or(""), result.get_score());
    }

    shutdown()?;
    Ok(())
}
```

## API Overview

### Initialization

| Function | Description |
|---|---|
| `initialize(config)` | Initialize the library (call once before use) |
| `shutdown()` | Shut down and release all resources |
| `version()` | Get the library version string |

### Schema Definition

```rust
// Builder pattern
let schema = CollectionSchema::builder("name")
    .add_field(FieldSchema::new("field", DataType::String, false, 0))
    .add_vector_field("vec", DataType::VectorFp32, 128,
        IndexParams::hnsw(MetricType::Cosine, 16, 200))
    .build()?;
```

### Collection Operations

| Method | Description |
|---|---|
| `Collection::create_and_open()` | Create a new collection |
| `Collection::open()` | Open an existing collection |
| `collection.insert(&docs)` | Insert documents |
| `collection.update(&docs)` | Update documents |
| `collection.upsert(&docs)` | Insert or update documents |
| `collection.delete(&pks)` | Delete by primary keys |
| `collection.query(&query)` | Vector similarity search |
| `collection.fetch(&pks)` | Fetch documents by primary keys |

### Document Operations

```rust
let mut doc = Doc::new()?;
doc.set_pk("my_pk");
doc.add_string("name", "value")?;
doc.add_i64("count", 42)?;
doc.add_vector_f32("embedding", &[0.1, 0.2, 0.3])?;

// Read fields
let name = doc.get_string("name")?;
let count = doc.get_i64("count")?;
```

### Vector Query

```rust
// Simple query
let query = VectorQuery::new("embedding", &query_vec, 10)?;

// Builder pattern with filters
let query = VectorQuery::builder()
    .field_name("embedding")
    .vector(&query_vec)
    .topk(10)
    .filter("category == 'tech'")
    .include_vector(false)
    .output_fields(&["id", "name"])
    .build()?;
```

## Supported Types

| Category | Types |
|---|---|
| **Scalar** | `Bool`, `Int32`, `Int64`, `Uint32`, `Uint64`, `Float`, `Double`, `String`, `Binary` |
| **Vector** | `VectorFp16`, `VectorFp32`, `VectorFp64`, `VectorInt4`, `VectorInt8`, `VectorInt16`, `VectorBinary32`, `VectorBinary64` |
| **Sparse** | `SparseVectorFp16`, `SparseVectorFp32` |
| **Array** | `ArrayBool`, `ArrayInt32`, `ArrayInt64`, `ArrayFloat`, `ArrayDouble`, `ArrayString`, etc. |

## Index Types

| Type | Description |
|---|---|
| `IndexParams::hnsw()` | HNSW graph index (recommended for most use cases) |
| `IndexParams::ivf()` | IVF inverted file index |
| `IndexParams::flat()` | Brute-force flat index |
| `IndexParams::invert()` | Inverted index for scalar fields |

## License

Same as the zvec project — see [LICENSE](../LICENSE).
