//! Benchmark tests for the zvec Rust SDK.
//!
//! These benchmarks require the zvec C library (`libzvec_c_api`) to be available.
//! Set `ZVEC_LIB_DIR` and `ZVEC_INCLUDE_DIR` environment variables before running.
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use std::sync::Once;
use zvec::*;

static INIT: Once = Once::new();

fn ensure_initialized() {
    INIT.call_once(|| {
        initialize(None).expect("failed to initialize zvec");
    });
}

fn create_bench_collection(parent_dir: &std::path::Path, dim: u32) -> Collection {
    let dir = parent_dir.join("zvec_data");
    let schema = CollectionSchema::builder("bench_collection")
        .add_field(FieldSchema::new("id", DataType::String, false, 0))
        .add_field(FieldSchema::new("category", DataType::String, true, 0))
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            dim,
            IndexParams::hnsw(MetricType::Cosine, 16, 200),
        )
        .build()
        .expect("failed to build schema");

    Collection::create_and_open(dir.to_str().unwrap(), &schema, None)
        .expect("failed to create collection")
}

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

// =============================================================================
// Document Creation Benchmarks
// =============================================================================

fn bench_doc_creation(criterion: &mut Criterion) {
    ensure_initialized();

    criterion.bench_function("doc_create_empty", |bencher| {
        bencher.iter(|| {
            let doc = Doc::new().unwrap();
            black_box(doc);
        });
    });

    criterion.bench_function("doc_create_with_fields", |bencher| {
        bencher.iter(|| {
            let mut doc = Doc::new().unwrap();
            doc.set_pk("bench_pk");
            doc.add_string("id", "bench_pk").unwrap();
            doc.add_string("category", "test").unwrap();
            doc.add_i64("count", 42).unwrap();
            doc.add_f32("score", 0.95).unwrap();
            black_box(doc);
        });
    });

    let mut group = criterion.benchmark_group("doc_create_with_vector");
    for dim in [32, 128, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, &dim| {
            let vector = random_vector(dim);
            bencher.iter(|| {
                let mut doc = Doc::new().unwrap();
                doc.set_pk("bench_pk");
                doc.add_vector_f32("embedding", &vector).unwrap();
                black_box(doc);
            });
        });
    }
    group.finish();
}

// =============================================================================
// Document Field Access Benchmarks
// =============================================================================

fn bench_doc_field_access(criterion: &mut Criterion) {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    doc.set_pk("bench_pk");
    doc.add_string("name", "hello world").unwrap();
    doc.add_i64("count", 42).unwrap();
    doc.add_f32("score", 0.95).unwrap();
    let vector = random_vector(128);
    doc.add_vector_f32("embedding", &vector).unwrap();

    criterion.bench_function("doc_get_pk", |bencher| {
        bencher.iter(|| {
            black_box(doc.get_pk());
        });
    });

    criterion.bench_function("doc_get_string", |bencher| {
        bencher.iter(|| {
            black_box(doc.get_string("name").unwrap());
        });
    });

    criterion.bench_function("doc_get_i64", |bencher| {
        bencher.iter(|| {
            black_box(doc.get_i64("count").unwrap());
        });
    });

    criterion.bench_function("doc_get_f32", |bencher| {
        bencher.iter(|| {
            black_box(doc.get_f32("score").unwrap());
        });
    });

    criterion.bench_function("doc_get_vector_f32_128d", |bencher| {
        bencher.iter(|| {
            black_box(doc.get_vector_f32("embedding").unwrap());
        });
    });

    criterion.bench_function("doc_has_field", |bencher| {
        bencher.iter(|| {
            black_box(doc.has_field("name"));
        });
    });
}

// =============================================================================
// Insert Benchmarks
// =============================================================================

fn bench_insert(criterion: &mut Criterion) {
    ensure_initialized();

    let mut group = criterion.benchmark_group("insert");
    group.sample_size(10);

    for batch_size in [1, 10, 100] {
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |bencher, &batch_size| {
                bencher.iter_with_setup(
                    || {
                        let tmp_dir = tempfile::tempdir().unwrap();
                        let collection = create_bench_collection(tmp_dir.path(), 128);
                        let mut docs = Vec::with_capacity(batch_size);
                        for i in 0..batch_size {
                            let mut doc = Doc::new().unwrap();
                            let pk = format!("pk_{}", i);
                            doc.set_pk(&pk);
                            doc.add_string("id", &pk).unwrap();
                            doc.add_string("category", "bench").unwrap();
                            doc.add_vector_f32("embedding", &random_vector(128)).unwrap();
                            docs.push(doc);
                        }
                        (tmp_dir, collection, docs)
                    },
                    |(_tmp_dir, collection, docs)| {
                        let doc_refs: Vec<&Doc> = docs.iter().collect();
                        black_box(collection.insert(&doc_refs).unwrap());
                    },
                );
            },
        );
    }
    group.finish();
}

// =============================================================================
// Query Benchmarks
// =============================================================================

fn bench_query(criterion: &mut Criterion) {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_bench_collection(tmp_dir.path(), 128);

    // Pre-populate with 1000 documents
    let doc_count = 1000;
    let batch_size = 100;
    for batch_start in (0..doc_count).step_by(batch_size) {
        let mut docs = Vec::with_capacity(batch_size);
        for i in batch_start..batch_start + batch_size {
            let mut doc = Doc::new().unwrap();
            let pk = format!("pk_{}", i);
            doc.set_pk(&pk);
            doc.add_string("id", &pk).unwrap();
            doc.add_string("category", if i % 2 == 0 { "even" } else { "odd" })
                .unwrap();
            doc.add_vector_f32("embedding", &random_vector(128)).unwrap();
            docs.push(doc);
        }
        let doc_refs: Vec<&Doc> = docs.iter().collect();
        collection.insert(&doc_refs).unwrap();
    }

    let mut group = criterion.benchmark_group("query");

    for topk in [1, 10, 50] {
        group.bench_with_input(
            BenchmarkId::new("topk", topk),
            &topk,
            |bencher, &topk| {
                let query_vec = random_vector(128);
                bencher.iter(|| {
                    let query = VectorQuery::new("embedding", &query_vec, topk).unwrap();
                    black_box(collection.query(&query).unwrap());
                });
            },
        );
    }
    group.finish();

    // Fetch benchmark
    criterion.bench_function("fetch_single", |bencher| {
        bencher.iter(|| {
            black_box(collection.fetch(&["pk_0"]).unwrap());
        });
    });

    criterion.bench_function("fetch_batch_10", |bencher| {
        let pks: Vec<String> = (0..10).map(|i| format!("pk_{}", i)).collect();
        let pk_refs: Vec<&str> = pks.iter().map(|s| s.as_str()).collect();
        bencher.iter(|| {
            black_box(collection.fetch(&pk_refs).unwrap());
        });
    });
}

// =============================================================================
// Schema Creation Benchmarks
// =============================================================================

fn bench_schema_creation(criterion: &mut Criterion) {
    ensure_initialized();

    criterion.bench_function("schema_builder_simple", |bencher| {
        bencher.iter(|| {
            let schema = CollectionSchema::builder("bench")
                .add_field(FieldSchema::new("id", DataType::String, false, 0))
                .add_vector_field(
                    "embedding",
                    DataType::VectorFp32,
                    128,
                    IndexParams::hnsw(MetricType::Cosine, 16, 200),
                )
                .build()
                .unwrap();
            black_box(schema);
        });
    });

    criterion.bench_function("schema_builder_complex", |bencher| {
        bencher.iter(|| {
            let schema = CollectionSchema::builder("bench")
                .add_field(FieldSchema::new("id", DataType::String, false, 0))
                .add_field(FieldSchema::new("name", DataType::String, true, 0))
                .add_field(FieldSchema::new("score", DataType::Float, true, 0))
                .add_field(FieldSchema::new("count", DataType::Int64, true, 0))
                .add_field(FieldSchema::new("flag", DataType::Bool, true, 0))
                .add_vector_field(
                    "embedding",
                    DataType::VectorFp32,
                    128,
                    IndexParams::hnsw(MetricType::Cosine, 16, 200),
                )
                .add_vector_field(
                    "embedding2",
                    DataType::VectorFp32,
                    64,
                    IndexParams::flat(MetricType::L2),
                )
                .build()
                .unwrap();
            black_box(schema);
        });
    });
}

// =============================================================================
// Type Conversion Benchmarks (pure logic, no C library)
// =============================================================================

fn bench_type_conversions(criterion: &mut Criterion) {
    criterion.bench_function("data_type_from_u32", |bencher| {
        bencher.iter(|| {
            for i in 0..50u32 {
                black_box(DataType::from(i));
            }
        });
    });

    criterion.bench_function("error_code_from_u32", |bencher| {
        bencher.iter(|| {
            for i in 0..12u32 {
                black_box(ErrorCode::from(i));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_doc_creation,
    bench_doc_field_access,
    bench_insert,
    bench_query,
    bench_schema_creation,
    bench_type_conversions,
);
criterion_main!(benches);
