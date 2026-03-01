// Copyright 2025-present the zvec project
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_float, c_int, c_void};

pub type zvec_status_t = c_void;
pub type zvec_collection_t = c_void;
pub type zvec_schema_t = c_void;
pub type zvec_doc_t = c_void;
pub type zvec_doc_list_t = c_void;
pub type zvec_stats_t = c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum zvec_status_code_t {
    ZVEC_OK = 0,
    ZVEC_NOT_FOUND = 1,
    ZVEC_ALREADY_EXISTS = 2,
    ZVEC_INVALID_ARGUMENT = 3,
    ZVEC_PERMISSION_DENIED = 4,
    ZVEC_FAILED_PRECONDITION = 5,
    ZVEC_RESOURCE_EXHAUSTED = 6,
    ZVEC_UNAVAILABLE = 7,
    ZVEC_INTERNAL_ERROR = 8,
    ZVEC_NOT_SUPPORTED = 9,
    ZVEC_UNKNOWN = 10,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum zvec_data_type_t {
    ZVEC_TYPE_UNDEFINED = 0,
    ZVEC_TYPE_BINARY = 1,
    ZVEC_TYPE_STRING = 2,
    ZVEC_TYPE_BOOL = 3,
    ZVEC_TYPE_INT32 = 4,
    ZVEC_TYPE_INT64 = 5,
    ZVEC_TYPE_UINT32 = 6,
    ZVEC_TYPE_UINT64 = 7,
    ZVEC_TYPE_FLOAT = 8,
    ZVEC_TYPE_DOUBLE = 9,
    ZVEC_TYPE_VECTOR_BINARY32 = 20,
    ZVEC_TYPE_VECTOR_FP16 = 22,
    ZVEC_TYPE_VECTOR_FP32 = 23,
    ZVEC_TYPE_VECTOR_FP64 = 24,
    ZVEC_TYPE_VECTOR_INT8 = 26,
    ZVEC_TYPE_SPARSE_VECTOR_FP16 = 30,
    ZVEC_TYPE_SPARSE_VECTOR_FP32 = 31,
    ZVEC_TYPE_ARRAY_STRING = 41,
    ZVEC_TYPE_ARRAY_INT32 = 43,
    ZVEC_TYPE_ARRAY_FLOAT = 47,
}

extern "C" {
    // Status API
    pub fn zvec_status_code(status: *mut zvec_status_t) -> zvec_status_code_t;
    pub fn zvec_status_message(status: *mut zvec_status_t) -> *const c_char;
    pub fn zvec_status_destroy(status: *mut zvec_status_t);

    // Schema API
    pub fn zvec_schema_create(name: *const c_char) -> *mut zvec_schema_t;
    pub fn zvec_schema_destroy(schema: *mut zvec_schema_t);
    pub fn zvec_schema_add_field(
        schema: *mut zvec_schema_t,
        name: *const c_char,
        data_type: zvec_data_type_t,
        dimension: u32,
    ) -> *mut zvec_status_t;

    // Collection API
    pub fn zvec_collection_create_and_open(
        path: *const c_char,
        schema: *mut zvec_schema_t,
        out_collection: *mut *mut zvec_collection_t,
    ) -> *mut zvec_status_t;
    pub fn zvec_collection_open(
        path: *const c_char,
        out_collection: *mut *mut zvec_collection_t,
    ) -> *mut zvec_status_t;
    pub fn zvec_collection_destroy(collection: *mut zvec_collection_t);
    pub fn zvec_collection_flush(collection: *mut zvec_collection_t) -> *mut zvec_status_t;
    pub fn zvec_collection_destroy_physical(collection: *mut zvec_collection_t) -> *mut zvec_status_t;
    pub fn zvec_collection_get_stats(
        collection: *mut zvec_collection_t,
        out_stats: *mut *mut zvec_stats_t,
    ) -> *mut zvec_status_t;

    // DML API
    pub fn zvec_collection_insert(
        collection: *mut zvec_collection_t,
        docs: *mut *mut zvec_doc_t,
        count: usize,
    ) -> *mut zvec_status_t;
    pub fn zvec_collection_upsert(
        collection: *mut zvec_collection_t,
        docs: *mut *mut zvec_doc_t,
        count: usize,
    ) -> *mut zvec_status_t;
    pub fn zvec_collection_update(
        collection: *mut zvec_collection_t,
        docs: *mut *mut zvec_doc_t,
        count: usize,
    ) -> *mut zvec_status_t;
    pub fn zvec_collection_delete(
        collection: *mut zvec_collection_t,
        pks: *mut *const c_char,
        count: usize,
    ) -> *mut zvec_status_t;

    // DQL API
    pub fn zvec_collection_query(
        collection: *mut zvec_collection_t,
        field_name: *const c_char,
        vector: *const c_float,
        count: u32,
        topk: c_int,
        out_results: *mut *mut zvec_doc_list_t,
    ) -> *mut zvec_status_t;
    pub fn zvec_collection_fetch(
        collection: *mut zvec_collection_t,
        pks: *mut *const c_char,
        count: usize,
        out_results: *mut *mut zvec_doc_list_t,
    ) -> *mut zvec_status_t;

    // Doc API
    pub fn zvec_doc_create() -> *mut zvec_doc_t;
    pub fn zvec_doc_destroy(doc: *mut zvec_doc_t);
    pub fn zvec_doc_set_pk(doc: *mut zvec_doc_t, pk: *const c_char);
    pub fn zvec_doc_pk(doc: *mut zvec_doc_t) -> *const c_char;
    pub fn zvec_doc_set_score(doc: *mut zvec_doc_t, score: c_float);
    pub fn zvec_doc_score(doc: *mut zvec_doc_t) -> c_float;
    pub fn zvec_doc_set_string(
        doc: *mut zvec_doc_t,
        field: *const c_char,
        value: *const c_char,
    ) -> *mut zvec_status_t;
    pub fn zvec_doc_set_int32(
        doc: *mut zvec_doc_t,
        field: *const c_char,
        value: i32,
    ) -> *mut zvec_status_t;
    pub fn zvec_doc_set_float(
        doc: *mut zvec_doc_t,
        field: *const c_char,
        value: f32,
    ) -> *mut zvec_status_t;
    pub fn zvec_doc_set_float_vector(
        doc: *mut zvec_doc_t,
        field: *const c_char,
        data: *const f32,
        count: u32,
    ) -> *mut zvec_status_t;

    // Doc List API
    pub fn zvec_doc_list_size(list: *mut zvec_doc_list_t) -> usize;
    pub fn zvec_doc_list_get(list: *mut zvec_doc_list_t, index: usize) -> *mut zvec_doc_t;
    pub fn zvec_doc_list_destroy(list: *mut zvec_doc_list_t);

    // Stats API
    pub fn zvec_stats_total_docs(stats: *mut zvec_stats_t) -> u64;
    pub fn zvec_stats_destroy(stats: *mut zvec_stats_t);
}
