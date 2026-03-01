// Copyright 2025-present the zvec project
import XCTest
@testable import ZVec

final class ZVecTests: XCTestCase {

    func testDir(_ name: String) -> String {
        let dir = "/tmp/zvec_swift_test_\(name)"
        try? FileManager.default.removeItem(atPath: dir)
        return dir
    }

    func testSchemaCreate() throws {
        let schema = Schema(name: "test_schema")
        try schema.addField(name: "vec", type: ZVEC_TYPE_VECTOR_FP32, dimension: 4)
        try schema.addField(name: "name", type: ZVEC_TYPE_STRING)
        try schema.addField(name: "age", type: ZVEC_TYPE_INT32)
    }

    func testDocLifecycle() throws {
        let doc = Doc()
        doc.setPK("pk_001")
        XCTAssertEqual(doc.pk(), "pk_001")

        try doc.setString(field: "name", value: "Alice")
        try doc.setInt32(field: "age", value: 30)
        try doc.setVector(field: "vec", vector: [0.1, 0.2, 0.3, 0.4])
    }

    func testCollectionCRUD() throws {
        let dir = testDir("crud")
        let schema = Schema(name: "crud_col")
        try schema.addField(name: "vec", type: ZVEC_TYPE_VECTOR_FP32, dimension: 4)
        try schema.addField(name: "name", type: ZVEC_TYPE_STRING)

        let col = try Collection(path: dir, schema: schema)

        // Upsert
        let doc = Doc()
        doc.setPK("u1")
        try doc.setVector(field: "vec", vector: [1.0, 2.0, 3.0, 4.0])
        try doc.setString(field: "name", value: "Bob")
        try col.upsert(docs: [doc])
        try col.flush()

        // Stats
        let count = try col.stats()
        XCTAssertEqual(count, 1)

        // Fetch
        let results = try col.fetch(pks: ["u1"])
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].pk(), "u1")
    }

    func testInvalidOpen() {
        XCTAssertThrowsError(try Collection(path: "/tmp/zvec_swift_nonexistent_abc123"))
    }

    func testBatchUpsert() throws {
        let dir = testDir("batch")
        let schema = Schema(name: "batch_col")
        try schema.addField(name: "vec", type: ZVEC_TYPE_VECTOR_FP32, dimension: 4)

        let col = try Collection(path: dir, schema: schema)

        var docs: [Doc] = []
        for i in 0..<100 {
            let d = Doc()
            d.setPK("batch_\(i)")
            try d.setVector(field: "vec", vector: [Float(i), Float(i), Float(i), Float(i)])
            docs.append(d)
        }
        try col.upsert(docs: docs)
        try col.flush()

        let count = try col.stats()
        XCTAssertEqual(count, 100)
    }

    func testFetchNonexistentPK() throws {
        let dir = testDir("fetchnone")
        let schema = Schema(name: "fetchnone_col")
        try schema.addField(name: "vec", type: ZVEC_TYPE_VECTOR_FP32, dimension: 4)

        let col = try Collection(path: dir, schema: schema)

        let results = try col.fetch(pks: ["does_not_exist"])
        XCTAssertEqual(results.count, 0)
    }
}
