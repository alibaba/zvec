// Copyright 2025-present the zvec project
use zvec::*;
use zvec_sys::zvec_data_type_t::*;

fn main() -> Result<()> {
    // 1. Create Schema
    let mut schema = Schema::new("test_collection");
    schema.add_field("vector", ZVEC_TYPE_VECTOR_FP32, 128)?;
    schema.add_field("age", ZVEC_TYPE_INT32, 0)?;

    // 2. Create and Open Collection
    let mut collection = Collection::create_and_open("/tmp/zvec_rust_example", &schema)?;

    // 3. Upsert Documents
    let mut doc1 = Doc::new();
    doc1.set_pk("user_1");
    doc1.set_vector("vector", &vec![0.1; 128])?;
    doc1.set_int32("age", 30)?;

    collection.upsert(&[&doc1])?;
    collection.flush()?;

    // 4. Get Stats
    let count = collection.stats()?;
    println!("Total docs: {}", count);

    // 5. Fetch Document
    let docs = collection.fetch(&["user_1"])?;
    for doc in docs {
        println!("Fetched PK: {}", doc.pk());
    }

    Ok(())
}
