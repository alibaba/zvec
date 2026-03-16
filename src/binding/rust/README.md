# zvec — Rust Bindings

Safe Rust bindings for [zvec](https://zvec.org), an open-source in-process vector database built on Proxima.

## Crates

| Crate      | Description |
|------------|-------------|
| `zvec-sys` | Raw FFI declarations (`extern "C"`) for the zvec C API |
| `zvec`     | Safe, idiomatic Rust wrapper (Schema, Doc, Collection, Query) |

## Quick Start

```rust
use zvec::{Schema, Doc, Collection, DataType};

// Create schema
let mut schema = Schema::new("example");
schema.add_field("embedding", DataType::VectorFp32, 4).unwrap();

// Create collection
let mut col = Collection::create_and_open("./data", &schema).unwrap();

// Insert documents
let mut doc = Doc::new();
doc.set_pk("doc_1");
doc.set_vector("embedding", &[0.1, 0.2, 0.3, 0.4]).unwrap();
col.insert(&[&doc]).unwrap();

// Search
let results = col.query("embedding", &[0.4, 0.3, 0.2, 0.1], 10).unwrap();
for r in &results {
    println!("{}: {:.4}", r.pk, r.score);
}
```

## Building

Requires a pre-built zvec C library. Set `ZVEC_LIB_DIR` and `ZVEC_EXT_LIB_DIR` environment variables, or build zvec from source first:

```bash
# Build zvec C library (from repo root)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Then build Rust bindings
cd src/binding/rust
cargo build
cargo test
```

## License

Apache-2.0
