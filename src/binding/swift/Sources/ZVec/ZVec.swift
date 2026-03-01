// Copyright 2025-present the zvec project
@_exported import CZVec
import Foundation

public enum ZVecError: Error {
    case notFound
    case alreadyExists
    case invalidArgument
    case permissionDenied
    case internalError
    case unknown(String)
}

func mapError(_ status: OpaquePointer?) throws {
    guard let status = status else { return }
    defer { zvec_status_destroy(status) }

    let code = zvec_status_code(status)
    let msg = String(cString: zvec_status_message(status))

    switch code {
    case ZVEC_NOT_FOUND: throw ZVecError.notFound
    case ZVEC_ALREADY_EXISTS: throw ZVecError.alreadyExists
    case ZVEC_INVALID_ARGUMENT: throw ZVecError.invalidArgument
    case ZVEC_PERMISSION_DENIED: throw ZVecError.permissionDenied
    case ZVEC_INTERNAL_ERROR: throw ZVecError.internalError
    default: throw ZVecError.unknown(msg)
    }
}

public class Schema {
    var ptr: OpaquePointer

    public init(name: String) {
        self.ptr = zvec_schema_create(name)!
    }

    deinit {
        zvec_schema_destroy(ptr)
    }

    public func addField(name: String, type: zvec_data_type_t, dimension: UInt32 = 0) throws {
        let status = zvec_schema_add_field(ptr, name, type, dimension)
        try mapError(status)
    }
}

public class Doc {
    var ptr: OpaquePointer

    public init() {
        self.ptr = zvec_doc_create()!
    }

    internal init(ptr: OpaquePointer) {
        self.ptr = ptr
    }

    deinit {
        zvec_doc_destroy(ptr)
    }

    public func setPK(_ pk: String) {
        zvec_doc_set_pk(ptr, pk)
    }

    public func pk() -> String {
        return String(cString: zvec_doc_pk(ptr)!)
    }

    public func setString(field: String, value: String) throws {
        let status = zvec_doc_set_string(ptr, field, value)
        try mapError(status)
    }

    public func setInt32(field: String, value: Int32) throws {
        let status = zvec_doc_set_int32(ptr, field, value)
        try mapError(status)
    }

    public func setVector(field: String, vector: [Float]) throws {
        let status = zvec_doc_set_float_vector(ptr, field, vector, UInt32(vector.count))
        try mapError(status)
    }
}

public class Collection {
    var ptr: OpaquePointer

    public init(path: String, schema: Schema) throws {
        var outPtr: OpaquePointer?
        let status = zvec_collection_create_and_open(path, schema.ptr, &outPtr)
        try mapError(status)
        self.ptr = outPtr!
    }

    public init(path: String) throws {
        var outPtr: OpaquePointer?
        let status = zvec_collection_open(path, &outPtr)
        try mapError(status)
        self.ptr = outPtr!
    }

    deinit {
        zvec_collection_destroy(ptr)
    }

    public func upsert(docs: [Doc]) throws {
        var ptrs = docs.map { Optional($0.ptr) }
        let status = zvec_collection_upsert(ptr, &ptrs, ptrs.count)
        try mapError(status)
    }

    public func fetch(pks: [String]) throws -> [Doc] {
        var pkPtrs = pks.map { ($0 as NSString).utf8String }
        var listPtr: OpaquePointer?
        let status = zvec_collection_fetch(ptr, &pkPtrs, pkPtrs.count, &listPtr)
        try mapError(status)

        let size = zvec_doc_list_size(listPtr)
        var results: [Doc] = []
        for i in 0..<size {
            let docPtr = zvec_doc_list_get(listPtr, i)
            results.append(Doc(ptr: docPtr!))
        }
        zvec_doc_list_destroy(listPtr)
        return results
    }

    public func flush() throws {
        try mapError(zvec_collection_flush(ptr))
    }

    public func stats() throws -> UInt64 {
        var statsPtr: OpaquePointer?
        try mapError(zvec_collection_get_stats(ptr, &statsPtr))
        let count = zvec_stats_total_docs(statsPtr)
        zvec_stats_destroy(statsPtr)
        return count
    }
}
