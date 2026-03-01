// Copyright 2025-present the zvec project
import ZVec
import Foundation

@main
struct OneMinuteExample {
    static func main() {
        do {
            // 1. Create Schema
            let schema = Schema(name: "test_collection")
            try schema.addField(name: "vector", type: ZVEC_TYPE_VECTOR_FP32, dimension: 128)
            try schema.addField(name: "age", type: ZVEC_TYPE_INT32)

            // 2. Create and Open Collection
            let collection = try Collection(path: "/tmp/zvec_swift_example", schema: schema)

            // 3. Upsert Documents
            let doc1 = Doc()
            doc1.setPK("user_1")
            try doc1.setVector(field: "vector", vector: Array(repeating: 0.1, count: 128))
            try doc1.setInt32(field: "age", value: 30)

            try collection.upsert(docs: [doc1])
            try collection.flush()

            // 4. Get Stats
            let count = try collection.stats()
            print("Total docs: \(count)")

            // 5. Fetch Document
            let docs = try collection.fetch(pks: ["user_1"])
            for doc in docs {
                print("Fetched PK: \(doc.pk())")
            }
        } catch {
            print("Error: \(error)")
        }
    }
}
