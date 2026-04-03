use std::ffi::CStr;
use std::os::raw::c_void;

use crate::error::{check_error, to_cstring, Error, ErrorCode, Result};
use crate::types::DataType;

/// A document in a zvec collection.
///
/// Documents contain typed fields and are used for both writing data to
/// and reading data from collections.
pub struct Doc {
    pub(crate) handle: *mut zvec_sys::zvec_doc_t,
    owned: bool,
}

impl Doc {
    /// Creates a new empty document.
    pub fn new() -> Result<Self> {
        let handle = unsafe { zvec_sys::zvec_doc_create() };
        if handle.is_null() {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: "failed to create document".into(),
            });
        }
        Ok(Doc {
            handle,
            owned: true,
        })
    }

    /// Creates a non-owning wrapper around an existing handle.
    pub(crate) fn from_borrowed(handle: *mut zvec_sys::zvec_doc_t) -> Self {
        Doc {
            handle,
            owned: false,
        }
    }

    /// Sets the primary key.
    pub fn set_pk(&mut self, pk: &str) {
        let c_pk = to_cstring(pk).expect("pk must not contain null bytes");
        unsafe { zvec_sys::zvec_doc_set_pk(self.handle, c_pk.as_ptr()) };
    }

    /// Returns the primary key, or `None` if not set.
    pub fn get_pk(&self) -> Option<&str> {
        unsafe {
            let ptr = zvec_sys::zvec_doc_get_pk_pointer(self.handle);
            if ptr.is_null() {
                None
            } else {
                CStr::from_ptr(ptr).to_str().ok()
            }
        }
    }

    /// Returns the document score (set by query results).
    pub fn get_score(&self) -> f32 {
        unsafe { zvec_sys::zvec_doc_get_score(self.handle) }
    }

    /// Returns the document ID.
    pub fn get_doc_id(&self) -> u64 {
        unsafe { zvec_sys::zvec_doc_get_doc_id(self.handle) }
    }

    /// Returns the number of fields in the document.
    pub fn field_count(&self) -> usize {
        unsafe { zvec_sys::zvec_doc_get_field_count(self.handle) }
    }

    /// Returns whether the document is empty.
    pub fn is_empty(&self) -> bool {
        unsafe { zvec_sys::zvec_doc_is_empty(self.handle) }
    }

    /// Returns whether the document contains the specified field.
    pub fn has_field(&self, name: &str) -> bool {
        let c_name = match to_cstring(name) {
            Ok(s) => s,
            Err(_) => return false,
        };
        unsafe { zvec_sys::zvec_doc_has_field(self.handle, c_name.as_ptr()) }
    }

    /// Returns whether the specified field is null.
    pub fn is_field_null(&self, name: &str) -> bool {
        let c_name = match to_cstring(name) {
            Ok(s) => s,
            Err(_) => return false,
        };
        unsafe { zvec_sys::zvec_doc_is_field_null(self.handle, c_name.as_ptr()) }
    }

    // =========================================================================
    // Field setters
    // =========================================================================

    /// Adds a string field.
    pub fn add_string(&mut self, name: &str, value: &str) -> Result<()> {
        let c_name = to_cstring(name)?;
        let c_value = to_cstring(value)?;
        let bytes = c_value.as_bytes_with_nul();
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::String as u32,
                bytes.as_ptr() as *const c_void,
                bytes.len(),
            )
        })
    }

    /// Adds a boolean field.
    pub fn add_bool(&mut self, name: &str, value: bool) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::Bool as u32,
                &value as *const bool as *const c_void,
                std::mem::size_of::<bool>(),
            )
        })
    }

    /// Adds an i32 field.
    pub fn add_i32(&mut self, name: &str, value: i32) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::Int32 as u32,
                &value as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            )
        })
    }

    /// Adds an i64 field.
    pub fn add_i64(&mut self, name: &str, value: i64) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::Int64 as u32,
                &value as *const i64 as *const c_void,
                std::mem::size_of::<i64>(),
            )
        })
    }

    /// Adds a u32 field.
    pub fn add_u32(&mut self, name: &str, value: u32) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::Uint32 as u32,
                &value as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            )
        })
    }

    /// Adds a u64 field.
    pub fn add_u64(&mut self, name: &str, value: u64) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::Uint64 as u32,
                &value as *const u64 as *const c_void,
                std::mem::size_of::<u64>(),
            )
        })
    }

    /// Adds an f32 field.
    pub fn add_f32(&mut self, name: &str, value: f32) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::Float as u32,
                &value as *const f32 as *const c_void,
                std::mem::size_of::<f32>(),
            )
        })
    }

    /// Adds an f64 field.
    pub fn add_f64(&mut self, name: &str, value: f64) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::Double as u32,
                &value as *const f64 as *const c_void,
                std::mem::size_of::<f64>(),
            )
        })
    }

    /// Adds a dense FP32 vector field.
    pub fn add_vector_f32(&mut self, name: &str, vector: &[f32]) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::VectorFp32 as u32,
                vector.as_ptr() as *const c_void,
                vector.len() * std::mem::size_of::<f32>(),
            )
        })
    }

    /// Adds a dense FP64 vector field.
    pub fn add_vector_f64(&mut self, name: &str, vector: &[f64]) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_doc_add_field_by_value(
                self.handle,
                c_name.as_ptr(),
                DataType::VectorFp64 as u32,
                vector.as_ptr() as *const c_void,
                vector.len() * std::mem::size_of::<f64>(),
            )
        })
    }

    /// Sets a field to null.
    pub fn set_field_null(&mut self, name: &str) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe { zvec_sys::zvec_doc_set_field_null(self.handle, c_name.as_ptr()) })
    }

    /// Removes a field from the document.
    pub fn remove_field(&mut self, name: &str) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe { zvec_sys::zvec_doc_remove_field(self.handle, c_name.as_ptr()) })
    }

    // =========================================================================
    // Field getters
    // =========================================================================

    /// Gets a string field value.
    pub fn get_string(&self, name: &str) -> Result<String> {
        let c_name = to_cstring(name)?;
        let mut value_ptr: *const c_void = std::ptr::null();
        let mut value_size: usize = 0;
        check_error(unsafe {
            zvec_sys::zvec_doc_get_field_value_pointer(
                self.handle,
                c_name.as_ptr(),
                DataType::String as u32,
                &mut value_ptr,
                &mut value_size,
            )
        })?;
        if value_ptr.is_null() {
            return Ok(String::new());
        }
        unsafe {
            let cstr = CStr::from_ptr(value_ptr as *const std::os::raw::c_char);
            Ok(cstr.to_string_lossy().into_owned())
        }
    }

    /// Gets a boolean field value.
    pub fn get_bool(&self, name: &str) -> Result<bool> {
        let c_name = to_cstring(name)?;
        let mut value: bool = false;
        check_error(unsafe {
            zvec_sys::zvec_doc_get_field_value_basic(
                self.handle,
                c_name.as_ptr(),
                DataType::Bool as u32,
                &mut value as *mut bool as *mut c_void,
                std::mem::size_of::<bool>(),
            )
        })?;
        Ok(value)
    }

    /// Gets an i32 field value.
    pub fn get_i32(&self, name: &str) -> Result<i32> {
        let c_name = to_cstring(name)?;
        let mut value: i32 = 0;
        check_error(unsafe {
            zvec_sys::zvec_doc_get_field_value_basic(
                self.handle,
                c_name.as_ptr(),
                DataType::Int32 as u32,
                &mut value as *mut i32 as *mut c_void,
                std::mem::size_of::<i32>(),
            )
        })?;
        Ok(value)
    }

    /// Gets an i64 field value.
    pub fn get_i64(&self, name: &str) -> Result<i64> {
        let c_name = to_cstring(name)?;
        let mut value: i64 = 0;
        check_error(unsafe {
            zvec_sys::zvec_doc_get_field_value_basic(
                self.handle,
                c_name.as_ptr(),
                DataType::Int64 as u32,
                &mut value as *mut i64 as *mut c_void,
                std::mem::size_of::<i64>(),
            )
        })?;
        Ok(value)
    }

    /// Gets an f32 field value.
    pub fn get_f32(&self, name: &str) -> Result<f32> {
        let c_name = to_cstring(name)?;
        let mut value: f32 = 0.0;
        check_error(unsafe {
            zvec_sys::zvec_doc_get_field_value_basic(
                self.handle,
                c_name.as_ptr(),
                DataType::Float as u32,
                &mut value as *mut f32 as *mut c_void,
                std::mem::size_of::<f32>(),
            )
        })?;
        Ok(value)
    }

    /// Gets an f64 field value.
    pub fn get_f64(&self, name: &str) -> Result<f64> {
        let c_name = to_cstring(name)?;
        let mut value: f64 = 0.0;
        check_error(unsafe {
            zvec_sys::zvec_doc_get_field_value_basic(
                self.handle,
                c_name.as_ptr(),
                DataType::Double as u32,
                &mut value as *mut f64 as *mut c_void,
                std::mem::size_of::<f64>(),
            )
        })?;
        Ok(value)
    }

    /// Gets a dense FP32 vector field value.
    pub fn get_vector_f32(&self, name: &str) -> Result<Vec<f32>> {
        let c_name = to_cstring(name)?;
        let mut value_ptr: *const c_void = std::ptr::null();
        let mut value_size: usize = 0;
        check_error(unsafe {
            zvec_sys::zvec_doc_get_field_value_pointer(
                self.handle,
                c_name.as_ptr(),
                DataType::VectorFp32 as u32,
                &mut value_ptr,
                &mut value_size,
            )
        })?;
        if value_ptr.is_null() || value_size == 0 {
            return Ok(Vec::new());
        }
        let count = value_size / std::mem::size_of::<f32>();
        let slice = unsafe { std::slice::from_raw_parts(value_ptr as *const f32, count) };
        Ok(slice.to_vec())
    }

    /// Clears all fields from the document.
    pub fn clear(&mut self) {
        unsafe { zvec_sys::zvec_doc_clear(self.handle) };
    }
}

impl Drop for Doc {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            unsafe { zvec_sys::zvec_doc_destroy(self.handle) };
        }
    }
}

/// Frees a vector of documents returned by query/fetch operations.
pub fn free_docs(docs: Vec<Doc>) {
    // Documents are freed individually via their Drop implementations
    drop(docs);
}
