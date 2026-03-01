// Copyright 2025-present the zvec project
use std::ffi::{CStr, CString};
use std::ptr;
use zvec_sys::*;

#[derive(Debug)]
pub enum ZVecError {
    NotFound(String),
    AlreadyExists(String),
    InvalidArgument(String),
    PermissionDenied(String),
    InternalError(String),
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, ZVecError>;

fn map_error(status: *mut zvec_status_t) -> ZVecError {
    let code = unsafe { zvec_status_code(status) };
    let msg = unsafe {
        let ptr = zvec_status_message(status);
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    unsafe { zvec_status_destroy(status) };

    match code {
        zvec_status_code_t::ZVEC_NOT_FOUND => ZVecError::NotFound(msg),
        zvec_status_code_t::ZVEC_ALREADY_EXISTS => ZVecError::AlreadyExists(msg),
        zvec_status_code_t::ZVEC_INVALID_ARGUMENT => ZVecError::InvalidArgument(msg),
        zvec_status_code_t::ZVEC_PERMISSION_DENIED => ZVecError::PermissionDenied(msg),
        zvec_status_code_t::ZVEC_INTERNAL_ERROR => ZVecError::InternalError(msg),
        _ => ZVecError::Unknown(msg),
    }
}

pub struct Schema {
    ptr: *mut zvec_schema_t,
}

impl Schema {
    pub fn new(name: &str) -> Self {
        let c_name = CString::new(name).unwrap();
        let ptr = unsafe { zvec_schema_create(c_name.as_ptr()) };
        Self { ptr }
    }

    pub fn add_field(&mut self, name: &str, data_type: zvec_data_type_t, dimension: u32) -> Result<()> {
        let c_name = CString::new(name).unwrap();
        let status = unsafe { zvec_schema_add_field(self.ptr, c_name.as_ptr(), data_type, dimension) };
        if status.is_null() {
            Ok(())
        } else {
            Err(map_error(status))
        }
    }
}

impl Drop for Schema {
    fn drop(&mut self) {
        unsafe { zvec_schema_destroy(self.ptr) };
    }
}

pub struct Doc {
    ptr: *mut zvec_doc_t,
}

impl Doc {
    pub fn new() -> Self {
        Self {
            ptr: unsafe { zvec_doc_create() },
        }
    }

    pub fn set_pk(&mut self, pk: &str) {
        let c_pk = CString::new(pk).unwrap();
        unsafe { zvec_doc_set_pk(self.ptr, c_pk.as_ptr()) };
    }

    pub fn pk(&self) -> String {
        let ptr = unsafe { zvec_doc_pk(self.ptr) };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
        }
    }

    pub fn set_string(&mut self, field: &str, value: &str) -> Result<()> {
        let c_field = CString::new(field).unwrap();
        let c_value = CString::new(value).unwrap();
        let status = unsafe { zvec_doc_set_string(self.ptr, c_field.as_ptr(), c_value.as_ptr()) };
        if status.is_null() {
            Ok(())
        } else {
            Err(map_error(status))
        }
    }

    pub fn set_int32(&mut self, field: &str, value: i32) -> Result<()> {
        let c_field = CString::new(field).unwrap();
        let status = unsafe { zvec_doc_set_int32(self.ptr, c_field.as_ptr(), value) };
        if status.is_null() {
            Ok(())
        } else {
            Err(map_error(status))
        }
    }

    pub fn set_vector(&mut self, field: &str, vector: &[f32]) -> Result<()> {
        let c_field = CString::new(field).unwrap();
        let status = unsafe {
            zvec_doc_set_float_vector(self.ptr, c_field.as_ptr(), vector.as_ptr(), vector.len() as u32)
        };
        if status.is_null() {
            Ok(())
        } else {
            Err(map_error(status))
        }
    }
}

impl Drop for Doc {
    fn drop(&mut self) {
        unsafe { zvec_doc_destroy(self.ptr) };
    }
}

pub struct Collection {
    ptr: *mut zvec_collection_t,
}

impl Collection {
    pub fn create_and_open(path: &str, schema: &Schema) -> Result<Self> {
        let c_path = CString::new(path).unwrap();
        let mut ptr = ptr::null_mut();
        let status =
            unsafe { zvec_collection_create_and_open(c_path.as_ptr(), schema.ptr, &mut ptr) };
        if status.is_null() {
            Ok(Self { ptr })
        } else {
            Err(map_error(status))
        }
    }

    pub fn open(path: &str) -> Result<Self> {
        let c_path = CString::new(path).unwrap();
        let mut ptr = ptr::null_mut();
        let status = unsafe { zvec_collection_open(c_path.as_ptr(), &mut ptr) };
        if status.is_null() {
            Ok(Self { ptr })
        } else {
            Err(map_error(status))
        }
    }

    pub fn upsert(&mut self, docs: &[&Doc]) -> Result<()> {
        let mut ptrs: Vec<*mut zvec_doc_t> = docs.iter().map(|d| d.ptr).collect();
        let status = unsafe { zvec_collection_upsert(self.ptr, ptrs.as_mut_ptr(), ptrs.len()) };
        if status.is_null() {
            Ok(())
        } else {
            Err(map_error(status))
        }
    }

    pub fn fetch(&self, pks: &[&str]) -> Result<Vec<Doc>> {
        let c_pks: Vec<CString> = pks.iter().map(|s| CString::new(*s).unwrap()).collect();
        let mut pk_ptrs: Vec<*const i8> = c_pks.iter().map(|s| s.as_ptr()).collect();
        let mut list_ptr = ptr::null_mut();
        let status = unsafe {
            zvec_collection_fetch(self.ptr, pk_ptrs.as_mut_ptr(), pk_ptrs.len(), &mut list_ptr)
        };
        if !status.is_null() {
            return Err(map_error(status));
        }

        let size = unsafe { zvec_doc_list_size(list_ptr) };
        let mut results = Vec::with_capacity(size);
        for i in 0..size {
            let doc_ptr = unsafe { zvec_doc_list_get(list_ptr, i) };
            results.push(Doc { ptr: doc_ptr });
        }
        unsafe { zvec_doc_list_destroy(list_ptr) };
        Ok(results)
    }

    pub fn flush(&mut self) -> Result<()> {
        let status = unsafe { zvec_collection_flush(self.ptr) };
        if status.is_null() {
            Ok(())
        } else {
            Err(map_error(status))
        }
    }

    pub fn stats(&self) -> Result<u64> {
        let mut stats_ptr = ptr::null_mut();
        let status = unsafe { zvec_collection_get_stats(self.ptr, &mut stats_ptr) };
        if status.is_null() {
            let count = unsafe { zvec_stats_total_docs(stats_ptr) };
            unsafe { zvec_stats_destroy(stats_ptr) };
            Ok(count)
        } else {
            Err(map_error(status))
        }
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        unsafe { zvec_collection_destroy(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> String {
        let dir = format!("/tmp/zvec_rust_test_{}", name);
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_schema_create() {
        let mut schema = Schema::new("test_schema");
        schema.add_field("vec", zvec_data_type_t::ZVEC_TYPE_VECTOR_FP32, 4).unwrap();
        schema.add_field("name", zvec_data_type_t::ZVEC_TYPE_STRING, 0).unwrap();
        schema.add_field("age", zvec_data_type_t::ZVEC_TYPE_INT32, 0).unwrap();
    }

    #[test]
    fn test_doc_lifecycle() {
        let mut doc = Doc::new();
        doc.set_pk("pk_001");
        assert_eq!(doc.pk(), "pk_001");

        doc.set_string("name", "Alice").unwrap();
        doc.set_int32("age", 30).unwrap();
        doc.set_vector("vec", &[0.1, 0.2, 0.3, 0.4]).unwrap();
    }

    #[test]
    fn test_collection_crud() {
        let dir = test_dir("crud");
        let mut schema = Schema::new("crud_col");
        schema.add_field("vec", zvec_data_type_t::ZVEC_TYPE_VECTOR_FP32, 4).unwrap();
        schema.add_field("name", zvec_data_type_t::ZVEC_TYPE_STRING, 0).unwrap();

        let mut col = Collection::create_and_open(&dir, &schema).unwrap();

        // Upsert
        let mut doc = Doc::new();
        doc.set_pk("u1");
        doc.set_vector("vec", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        doc.set_string("name", "Bob").unwrap();
        col.upsert(&[&doc]).unwrap();
        col.flush().unwrap();

        // Stats
        assert_eq!(col.stats().unwrap(), 1);

        // Fetch
        let results = col.fetch(&["u1"]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk(), "u1");
    }

    #[test]
    fn test_invalid_open() {
        let result = Collection::open("/tmp/zvec_nonexistent_path_abc123");
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_create() {
        let dir = test_dir("dup");
        let mut schema = Schema::new("dup_col");
        schema.add_field("vec", zvec_data_type_t::ZVEC_TYPE_VECTOR_FP32, 4).unwrap();

        let _col1 = Collection::create_and_open(&dir, &schema).unwrap();
        // Second create on the same path should fail
        let result = Collection::create_and_open(&dir, &schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_upsert() {
        let dir = test_dir("batch");
        let mut schema = Schema::new("batch_col");
        schema.add_field("vec", zvec_data_type_t::ZVEC_TYPE_VECTOR_FP32, 4).unwrap();

        let mut col = Collection::create_and_open(&dir, &schema).unwrap();

        let docs: Vec<Doc> = (0..100)
            .map(|i| {
                let mut d = Doc::new();
                d.set_pk(&format!("batch_{}", i));
                d.set_vector("vec", &[i as f32; 4]).unwrap();
                d
            })
            .collect();

        let refs: Vec<&Doc> = docs.iter().collect();
        col.upsert(&refs).unwrap();
        col.flush().unwrap();

        assert_eq!(col.stats().unwrap(), 100);
    }

    #[test]
    fn test_fetch_nonexistent_pk() {
        let dir = test_dir("fetchnone");
        let mut schema = Schema::new("fetchnone_col");
        schema.add_field("vec", zvec_data_type_t::ZVEC_TYPE_VECTOR_FP32, 4).unwrap();

        let col = Collection::create_and_open(&dir, &schema).unwrap();

        let results = col.fetch(&["does_not_exist"]).unwrap();
        assert_eq!(results.len(), 0);
    }
}
