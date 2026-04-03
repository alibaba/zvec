use zvec::*;

fn main() -> zvec::Result<()> {
    // Initialize the zvec library
    println!("zvec version: {}", version());
    initialize(None)?;

    let data_path = "./zvec_rust_example_data";

    // Define collection schema using the builder pattern
    let schema = CollectionSchema::builder("example_collection")
        .add_field(FieldSchema::new("id", DataType::String, false, 0))
        .add_field(FieldSchema::new("category", DataType::String, true, 0))
        .add_field(FieldSchema::new("score", DataType::Float, true, 0))
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            4,
            IndexParams::hnsw(MetricType::Cosine, 16, 200),
        )
        .build()?;

    // Create and open the collection
    let collection = Collection::create_and_open(data_path, &schema, None)?;
    println!("Collection created successfully!");

    // Insert documents
    let vectors: Vec<[f32; 4]> = vec![
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6, 0.7],
        [0.8, 0.9, 0.1, 0.2],
    ];

    let categories = ["sports", "tech", "sports", "tech", "science"];
    let scores: [f32; 5] = [0.95, 0.87, 0.76, 0.92, 0.88];

    let mut docs = Vec::new();
    for i in 0..5 {
        let mut doc = Doc::new()?;
        let pk = format!("doc_{}", i);
        doc.set_pk(&pk);
        doc.add_string("id", &pk)?;
        doc.add_string("category", categories[i])?;
        doc.add_f32("score", scores[i])?;
        doc.add_vector_f32("embedding", &vectors[i])?;
        docs.push(doc);
    }

    let doc_refs: Vec<&Doc> = docs.iter().collect();
    let write_result = collection.insert(&doc_refs)?;
    println!(
        "Inserted {} docs (errors: {})",
        write_result.success_count, write_result.error_count
    );

    // Vector similarity search
    let query_vector = [0.5, 0.6, 0.7, 0.8];
    let query = VectorQuery::new("embedding", &query_vector, 3)?;
    let results = collection.query(&query)?;

    println!("\nVector search results (top 3):");
    for (i, result) in results.iter().enumerate() {
        let pk = result.get_pk().unwrap_or("<unknown>");
        let result_score = result.get_score();
        println!("  #{}: pk={}, similarity={:.4}", i + 1, pk, result_score);
    }

    // Fetch by primary key
    let fetched = collection.fetch(&["doc_0", "doc_2"])?;
    println!("\nFetched {} documents by PK:", fetched.len());
    for doc in &fetched {
        let pk = doc.get_pk().unwrap_or("<unknown>");
        println!("  pk={}", pk);
    }

    // Collection statistics
    let stats = collection.stats()?;
    println!("\nCollection stats:");
    println!("  doc_count: {}", stats.doc_count);
    for (name, completeness) in stats.index_names.iter().zip(&stats.index_completeness) {
        println!("  index '{}': {:.1}% complete", name, completeness * 100.0);
    }

    // Delete documents
    let delete_result = collection.delete(&["doc_0"])?;
    println!(
        "\nDeleted {} docs (errors: {})",
        delete_result.success_count, delete_result.error_count
    );

    // Clean up
    collection.close()?;
    shutdown()?;

    // Remove example data directory
    let _ = std::fs::remove_dir_all(data_path);
    println!("\nDone!");

    Ok(())
}
