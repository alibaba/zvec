//! Integration tests for the zvec Rust SDK.
//!
//! These tests require the zvec C library (`libzvec_c_api`) to be available.
//! Set `ZVEC_LIB_DIR` and `ZVEC_INCLUDE_DIR` environment variables before running.
//!
//! Run with: `cargo test --test integration_test`

use std::sync::Once;
use zvec::*;

static INIT: Once = Once::new();

fn ensure_initialized() {
    INIT.call_once(|| {
        initialize(None).expect("failed to initialize zvec");
    });
}

fn create_test_collection(parent_dir: &std::path::Path) -> Collection {
    let dir = parent_dir.join("zvec_data");
    let schema = CollectionSchema::builder("test_collection")
        .add_field(FieldSchema::new("id", DataType::String, false, 0))
        .add_field(FieldSchema::new("category", DataType::String, true, 0))
        .add_field(FieldSchema::new("score", DataType::Float, true, 0))
        .add_field(FieldSchema::new("count", DataType::Int64, true, 0))
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            4,
            IndexParams::hnsw(MetricType::Cosine, 16, 200),
        )
        .build()
        .expect("failed to build schema");

    Collection::create_and_open(dir.to_str().unwrap(), &schema, None)
        .expect("failed to create collection")
}

fn insert_test_docs(collection: &Collection, count: usize) -> Vec<String> {
    let mut pks = Vec::with_capacity(count);
    let mut docs = Vec::with_capacity(count);

    for i in 0..count {
        let mut doc = Doc::new().unwrap();
        let pk = format!("pk_{}", i);
        doc.set_pk(&pk);
        doc.add_string("id", &pk).unwrap();
        doc.add_string("category", if i % 2 == 0 { "even" } else { "odd" })
            .unwrap();
        doc.add_f32("score", i as f32 * 0.1).unwrap();
        doc.add_i64("count", i as i64).unwrap();
        let vector = [
            (i as f32 + 1.0) * 0.1,
            (i as f32 + 2.0) * 0.1,
            (i as f32 + 3.0) * 0.1,
            (i as f32 + 4.0) * 0.1,
        ];
        doc.add_vector_f32("embedding", &vector).unwrap();
        pks.push(pk);
        docs.push(doc);
    }

    let doc_refs: Vec<&Doc> = docs.iter().collect();
    let result = collection.insert(&doc_refs).unwrap();
    assert_eq!(result.success_count, count as u64);
    assert_eq!(result.error_count, 0);

    pks
}

// =============================================================================
// Version & Initialization Tests
// =============================================================================

#[test]
fn test_version_string_not_empty() {
    let ver = version();
    assert!(!ver.is_empty(), "version string should not be empty");
}

#[test]
fn test_initialize_and_is_initialized() {
    ensure_initialized();
    assert!(is_initialized());
}

// =============================================================================
// Schema Tests
// =============================================================================

#[test]
fn test_field_schema_creation() {
    ensure_initialized();

    let field = FieldSchema::new("test_field", DataType::String, false, 0);
    assert_eq!(field.name(), "test_field");
    assert_eq!(field.data_type(), DataType::String);
    assert!(!field.is_nullable());
    assert_eq!(field.dimension(), 0);
    assert!(!field.is_vector_field());
}

#[test]
fn test_field_schema_vector_field() {
    ensure_initialized();

    let field = FieldSchema::new("vec_field", DataType::VectorFp32, false, 128);
    assert_eq!(field.name(), "vec_field");
    assert_eq!(field.data_type(), DataType::VectorFp32);
    assert_eq!(field.dimension(), 128);
    assert!(field.is_vector_field());
    assert!(field.is_dense_vector());
    assert!(!field.is_sparse_vector());
}

#[test]
fn test_field_schema_nullable() {
    ensure_initialized();

    let field = FieldSchema::new("nullable_field", DataType::Int64, true, 0);
    assert!(field.is_nullable());
}

#[test]
fn test_collection_schema_builder() {
    ensure_initialized();

    let schema = CollectionSchema::builder("my_collection")
        .add_field(FieldSchema::new("id", DataType::String, false, 0))
        .add_field(FieldSchema::new("name", DataType::String, true, 0))
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            64,
            IndexParams::hnsw(MetricType::L2, 16, 200),
        )
        .build()
        .unwrap();

    assert_eq!(schema.name(), "my_collection");
    assert!(schema.has_field("id"));
    assert!(schema.has_field("name"));
    assert!(schema.has_field("embedding"));
    assert!(!schema.has_field("nonexistent"));
}

#[test]
fn test_collection_schema_max_doc_count() {
    ensure_initialized();

    let schema = CollectionSchema::builder("test")
        .add_field(FieldSchema::new("id", DataType::String, false, 0))
        .max_doc_count_per_segment(50000)
        .build()
        .unwrap();

    assert_eq!(schema.max_doc_count_per_segment(), 50000);
}

#[test]
fn test_index_params_hnsw() {
    ensure_initialized();

    let params = IndexParams::hnsw(MetricType::Cosine, 32, 400);
    assert_eq!(params.index_type(), IndexType::Hnsw);
    assert_eq!(params.metric_type(), MetricType::Cosine);
}

#[test]
fn test_index_params_ivf() {
    ensure_initialized();

    let params = IndexParams::ivf(MetricType::L2, 128, 10, false);
    assert_eq!(params.index_type(), IndexType::Ivf);
    assert_eq!(params.metric_type(), MetricType::L2);
}

#[test]
fn test_index_params_flat() {
    ensure_initialized();

    let params = IndexParams::flat(MetricType::Ip);
    assert_eq!(params.index_type(), IndexType::Flat);
    assert_eq!(params.metric_type(), MetricType::Ip);
}

#[test]
fn test_index_params_invert() {
    ensure_initialized();

    let params = IndexParams::invert(true, false);
    assert_eq!(params.index_type(), IndexType::Invert);
}

#[test]
fn test_index_params_hnsw_with_quantize() {
    ensure_initialized();

    let params =
        IndexParams::hnsw_with_quantize(MetricType::Cosine, 16, 200, QuantizeType::Int8);
    assert_eq!(params.index_type(), IndexType::Hnsw);
    assert_eq!(params.metric_type(), MetricType::Cosine);
    assert_eq!(params.quantize_type(), QuantizeType::Int8);
}

// =============================================================================
// Document Tests
// =============================================================================

#[test]
fn test_doc_creation_and_pk() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    assert!(doc.is_empty());

    // A newly created doc may have an empty PK (not necessarily None)
    let initial_pk = doc.get_pk();
    assert!(initial_pk.is_none() || initial_pk == Some(""));

    doc.set_pk("my_pk");
    assert_eq!(doc.get_pk(), Some("my_pk"));
}

#[test]
fn test_doc_string_field() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    doc.add_string("name", "hello world").unwrap();
    assert!(doc.has_field("name"));
    assert_eq!(doc.get_string("name").unwrap(), "hello world");
    assert_eq!(doc.field_count(), 1);
}

#[test]
fn test_doc_numeric_fields() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    doc.add_i32("int32_val", 42).unwrap();
    doc.add_i64("int64_val", 123456789).unwrap();
    doc.add_f32("float_val", 3.14).unwrap();
    doc.add_f64("double_val", 2.718281828).unwrap();

    assert_eq!(doc.get_i32("int32_val").unwrap(), 42);
    assert_eq!(doc.get_i64("int64_val").unwrap(), 123456789);
    assert!((doc.get_f32("float_val").unwrap() - 3.14).abs() < 0.001);
    assert!((doc.get_f64("double_val").unwrap() - 2.718281828).abs() < 1e-9);
}

#[test]
fn test_doc_bool_field() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    doc.add_bool("flag", true).unwrap();
    assert!(doc.get_bool("flag").unwrap());
}

#[test]
fn test_doc_vector_f32_field() {
    ensure_initialized();

    let vector = vec![0.1, 0.2, 0.3, 0.4];
    let mut doc = Doc::new().unwrap();
    doc.add_vector_f32("embedding", &vector).unwrap();

    let retrieved = doc.get_vector_f32("embedding").unwrap();
    assert_eq!(retrieved.len(), 4);
    for (a, b) in retrieved.iter().zip(vector.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_doc_null_field() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    doc.add_string("name", "test").unwrap();
    assert!(!doc.is_field_null("name"));

    doc.set_field_null("name").unwrap();
    assert!(doc.is_field_null("name"));
}

#[test]
fn test_doc_remove_field() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    doc.add_string("name", "test").unwrap();
    assert!(doc.has_field("name"));

    doc.remove_field("name").unwrap();
    assert!(!doc.has_field("name"));
}

#[test]
fn test_doc_clear() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    doc.add_string("a", "1").unwrap();
    doc.add_i32("b", 2).unwrap();
    assert!(!doc.is_empty());

    doc.clear();
    assert!(doc.is_empty());
    assert_eq!(doc.field_count(), 0);
}

#[test]
fn test_doc_field_count() {
    ensure_initialized();

    let mut doc = Doc::new().unwrap();
    assert_eq!(doc.field_count(), 0);

    doc.add_string("a", "1").unwrap();
    assert_eq!(doc.field_count(), 1);

    doc.add_i32("b", 2).unwrap();
    assert_eq!(doc.field_count(), 2);

    doc.add_f32("c", 3.0).unwrap();
    assert_eq!(doc.field_count(), 3);
}

// =============================================================================
// Collection CRUD Tests
// =============================================================================

#[test]
fn test_collection_create_and_open() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());

    let schema = collection.schema().unwrap();
    assert_eq!(schema.name(), "test_collection");
    assert!(schema.has_field("id"));
    assert!(schema.has_field("embedding"));
}

#[test]
fn test_collection_insert_and_fetch() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    let pks = insert_test_docs(&collection, 5);

    let fetched = collection.fetch(&[pks[0].as_str(), pks[2].as_str()]).unwrap();
    assert_eq!(fetched.len(), 2);

    let fetched_pks: Vec<&str> = fetched.iter().filter_map(|d| d.get_pk()).collect();
    assert!(fetched_pks.contains(&pks[0].as_str()));
    assert!(fetched_pks.contains(&pks[2].as_str()));
}

#[test]
fn test_collection_vector_query() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 10);

    let query_vec = [0.2, 0.3, 0.4, 0.5];
    let query = VectorQuery::new("embedding", &query_vec, 3).unwrap();
    let results = collection.query(&query).unwrap();

    assert!(results.len() <= 3);
    assert!(!results.is_empty());

    for doc in &results {
        assert!(doc.get_pk().is_some());
    }
}

#[test]
fn test_collection_vector_query_builder() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 10);

    let query_vec = [0.2, 0.3, 0.4, 0.5];
    let query = VectorQuery::builder()
        .field_name("embedding")
        .vector(&query_vec)
        .topk(5)
        .build()
        .unwrap();

    let results = collection.query(&query).unwrap();
    assert!(results.len() <= 5);
}

#[test]
fn test_collection_upsert() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 3);

    // Upsert existing doc with updated score
    let mut doc = Doc::new().unwrap();
    doc.set_pk("pk_0");
    doc.add_string("id", "pk_0").unwrap();
    doc.add_string("category", "updated").unwrap();
    doc.add_f32("score", 99.9).unwrap();
    doc.add_i64("count", 999).unwrap();
    doc.add_vector_f32("embedding", &[0.9, 0.9, 0.9, 0.9]).unwrap();

    let result = collection.upsert(&[&doc]).unwrap();
    assert_eq!(result.success_count, 1);

    let fetched = collection.fetch(&["pk_0"]).unwrap();
    assert_eq!(fetched.len(), 1);
}

#[test]
fn test_collection_delete() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 5);

    let result = collection.delete(&["pk_0", "pk_1"]).unwrap();
    assert_eq!(result.success_count, 2);
    assert_eq!(result.error_count, 0);

    let fetched = collection.fetch(&["pk_0"]).unwrap();
    assert!(fetched.is_empty());
}

#[test]
fn test_collection_stats() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 10);

    let stats = collection.stats().unwrap();
    assert_eq!(stats.doc_count, 10);
}

#[test]
fn test_collection_flush() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 3);

    collection.flush().unwrap();
}

#[test]
fn test_collection_reopen() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let data_path = tmp_dir.path().join("zvec_data");
    let path_str = data_path.to_str().unwrap().to_string();

    {
        let collection = create_test_collection(tmp_dir.path());
        insert_test_docs(&collection, 5);
        collection.flush().unwrap();
    }

    let collection = Collection::open(&path_str, None).unwrap();
    let stats = collection.stats().unwrap();
    assert_eq!(stats.doc_count, 5);
}

// =============================================================================
// Collection Options Tests
// =============================================================================

#[test]
fn test_collection_options() {
    ensure_initialized();

    let mut opts = CollectionOptions::new().unwrap();
    opts.set_enable_mmap(true).unwrap();
    assert!(opts.enable_mmap());

    opts.set_read_only(false).unwrap();
    assert!(!opts.read_only());

    opts.set_max_buffer_size(1024 * 1024).unwrap();
    assert_eq!(opts.max_buffer_size(), 1024 * 1024);
}

// =============================================================================
// Query Parameters Tests
// =============================================================================

#[test]
fn test_hnsw_query_params() {
    ensure_initialized();

    let mut params = HnswQueryParams::new(100, 0.0, false, false);
    assert_eq!(params.ef(), 100);

    params.set_ef(200).unwrap();
    assert_eq!(params.ef(), 200);
}

#[test]
fn test_ivf_query_params() {
    ensure_initialized();

    let mut params = IvfQueryParams::new(16, false, 1.0);
    assert_eq!(params.nprobe(), 16);

    params.set_nprobe(32).unwrap();
    assert_eq!(params.nprobe(), 32);
}

#[test]
fn test_vector_query_with_hnsw_params() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 10);

    let query_vec = [0.2, 0.3, 0.4, 0.5];
    let hnsw_params = HnswQueryParams::new(50, 0.0, false, false);
    let mut query = VectorQuery::new("embedding", &query_vec, 5).unwrap();
    query.set_hnsw_params(hnsw_params).unwrap();

    let results = collection.query(&query).unwrap();
    assert!(!results.is_empty());
}

// =============================================================================
// DDL Tests
// =============================================================================

#[test]
fn test_collection_add_and_drop_column() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 3);

    // add_column only supports basic numeric types (int32, int64, uint32, uint64, float, double)
    let new_field = FieldSchema::new("new_col", DataType::Int64, true, 0);
    collection.add_column(&new_field, None).unwrap();

    let schema = collection.schema().unwrap();
    assert!(schema.has_field("new_col"));

    collection.drop_column("new_col").unwrap();
    let schema = collection.schema().unwrap();
    assert!(!schema.has_field("new_col"));
}

// =============================================================================
// Edge Cases & Error Handling Tests
// =============================================================================

#[test]
fn test_fetch_nonexistent_pk() {
    ensure_initialized();

    let tmp_dir = tempfile::tempdir().unwrap();
    let collection = create_test_collection(tmp_dir.path());
    insert_test_docs(&collection, 3);

    let fetched = collection.fetch(&["nonexistent_pk"]).unwrap();
    assert!(fetched.is_empty());
}

#[test]
fn test_doc_has_field_nonexistent() {
    ensure_initialized();

    let doc = Doc::new().unwrap();
    assert!(!doc.has_field("nonexistent"));
}

#[test]
fn test_vector_query_builder_missing_field_name() {
    ensure_initialized();

    let result = VectorQuery::builder()
        .vector(&[0.1, 0.2, 0.3])
        .topk(10)
        .build();

    match result {
        Err(err) => assert!(err.is_invalid_argument()),
        Ok(_) => panic!("expected error for missing field_name"),
    }
}

#[test]
fn test_vector_query_builder_missing_vector() {
    ensure_initialized();

    let result = VectorQuery::builder()
        .field_name("embedding")
        .topk(10)
        .build();

    match result {
        Err(err) => assert!(err.is_invalid_argument()),
        Ok(_) => panic!("expected error for missing vector"),
    }
}
