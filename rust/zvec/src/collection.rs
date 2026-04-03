use std::ffi::CStr;
use std::ptr;

use crate::doc::Doc;
use crate::error::{check_error, to_cstring, Error, ErrorCode, Result};
use crate::query::VectorQuery;
use crate::schema::{CollectionSchema, FieldSchema, IndexParams};

/// Options for creating or opening a collection.
pub struct CollectionOptions {
    pub(crate) handle: *mut zvec_sys::zvec_collection_options_t,
}

impl CollectionOptions {
    /// Creates new collection options with default values.
    pub fn new() -> Result<Self> {
        let handle = unsafe { zvec_sys::zvec_collection_options_create() };
        if handle.is_null() {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: "failed to create collection options".into(),
            });
        }
        Ok(CollectionOptions { handle })
    }

    /// Sets whether to enable memory mapping.
    pub fn set_enable_mmap(&mut self, enable: bool) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_collection_options_set_enable_mmap(self.handle, enable)
        })
    }

    /// Returns whether memory mapping is enabled.
    pub fn enable_mmap(&self) -> bool {
        unsafe { zvec_sys::zvec_collection_options_get_enable_mmap(self.handle) }
    }

    /// Sets the maximum buffer size in bytes.
    pub fn set_max_buffer_size(&mut self, size: u64) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_collection_options_set_max_buffer_size(self.handle, size as usize)
        })
    }

    /// Returns the maximum buffer size in bytes.
    pub fn max_buffer_size(&self) -> u64 {
        unsafe { zvec_sys::zvec_collection_options_get_max_buffer_size(self.handle) as u64 }
    }

    /// Sets whether the collection is read-only.
    pub fn set_read_only(&mut self, read_only: bool) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_collection_options_set_read_only(self.handle, read_only)
        })
    }

    /// Returns whether the collection is read-only.
    pub fn read_only(&self) -> bool {
        unsafe { zvec_sys::zvec_collection_options_get_read_only(self.handle) }
    }
}

impl Drop for CollectionOptions {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { zvec_sys::zvec_collection_options_destroy(self.handle) };
        }
    }
}

/// Statistics about a collection.
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub doc_count: u64,
    pub index_names: Vec<String>,
    pub index_completeness: Vec<f32>,
}

/// Result of a write operation (insert/update/upsert/delete).
#[derive(Debug, Clone)]
pub struct WriteResult {
    pub success_count: u64,
    pub error_count: u64,
}

/// A zvec collection for storing and querying vector data.
///
/// Collections are the primary data container in zvec. They hold documents
/// with typed fields and support vector similarity search.
///
/// The collection is automatically closed when dropped.
pub struct Collection {
    handle: *mut zvec_sys::zvec_collection_t,
}

impl Collection {
    /// Creates a new collection and opens it.
    pub fn create_and_open(
        path: &str,
        schema: &CollectionSchema,
        options: Option<&CollectionOptions>,
    ) -> Result<Self> {
        let c_path = to_cstring(path)?;
        let c_options = options
            .map(|o| o.handle as *const _)
            .unwrap_or(ptr::null());

        let mut handle: *mut zvec_sys::zvec_collection_t = ptr::null_mut();
        check_error(unsafe {
            zvec_sys::zvec_collection_create_and_open(
                c_path.as_ptr(),
                schema.handle,
                c_options,
                &mut handle,
            )
        })?;

        Ok(Collection { handle })
    }

    /// Opens an existing collection.
    pub fn open(path: &str, options: Option<&CollectionOptions>) -> Result<Self> {
        let c_path = to_cstring(path)?;
        let c_options = options
            .map(|o| o.handle as *const _)
            .unwrap_or(ptr::null());

        let mut handle: *mut zvec_sys::zvec_collection_t = ptr::null_mut();
        check_error(unsafe {
            zvec_sys::zvec_collection_open(c_path.as_ptr(), c_options, &mut handle)
        })?;

        Ok(Collection { handle })
    }

    /// Flushes collection data to disk.
    pub fn flush(&self) -> Result<()> {
        check_error(unsafe { zvec_sys::zvec_collection_flush(self.handle) })
    }

    /// Returns the collection schema.
    pub fn schema(&self) -> Result<CollectionSchema> {
        let mut schema_handle: *mut zvec_sys::zvec_collection_schema_t = ptr::null_mut();
        check_error(unsafe {
            zvec_sys::zvec_collection_get_schema(self.handle, &mut schema_handle)
        })?;
        Ok(CollectionSchema::from_owned(schema_handle))
    }

    /// Returns collection statistics.
    pub fn stats(&self) -> Result<CollectionStats> {
        let mut stats_handle: *mut zvec_sys::zvec_collection_stats_t = ptr::null_mut();
        check_error(unsafe {
            zvec_sys::zvec_collection_get_stats(self.handle, &mut stats_handle)
        })?;

        let doc_count =
            unsafe { zvec_sys::zvec_collection_stats_get_doc_count(stats_handle) };
        let index_count =
            unsafe { zvec_sys::zvec_collection_stats_get_index_count(stats_handle) };

        let mut index_names = Vec::with_capacity(index_count);
        let mut index_completeness = Vec::with_capacity(index_count);

        for i in 0..index_count {
            let name_ptr =
                unsafe { zvec_sys::zvec_collection_stats_get_index_name(stats_handle, i) };
            let name = if name_ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() }
            };
            index_names.push(name);

            let completeness = unsafe {
                zvec_sys::zvec_collection_stats_get_index_completeness(stats_handle, i)
            };
            index_completeness.push(completeness);
        }

        unsafe { zvec_sys::zvec_collection_stats_destroy(stats_handle) };

        Ok(CollectionStats {
            doc_count,
            index_names,
            index_completeness,
        })
    }

    // =========================================================================
    // DML Operations
    // =========================================================================

    /// Inserts documents into the collection.
    pub fn insert(&self, docs: &[&Doc]) -> Result<WriteResult> {
        let ptrs: Vec<*const zvec_sys::zvec_doc_t> =
            docs.iter().map(|d| d.handle as *const _).collect();
        let mut success_count: usize = 0;
        let mut error_count: usize = 0;

        check_error(unsafe {
            zvec_sys::zvec_collection_insert(
                self.handle,
                ptrs.as_ptr(),
                ptrs.len(),
                &mut success_count,
                &mut error_count,
            )
        })?;

        Ok(WriteResult {
            success_count: success_count as u64,
            error_count: error_count as u64,
        })
    }

    /// Updates documents in the collection.
    pub fn update(&self, docs: &[&Doc]) -> Result<WriteResult> {
        let ptrs: Vec<*const zvec_sys::zvec_doc_t> =
            docs.iter().map(|d| d.handle as *const _).collect();
        let mut success_count: usize = 0;
        let mut error_count: usize = 0;

        check_error(unsafe {
            zvec_sys::zvec_collection_update(
                self.handle,
                ptrs.as_ptr(),
                ptrs.len(),
                &mut success_count,
                &mut error_count,
            )
        })?;

        Ok(WriteResult {
            success_count: success_count as u64,
            error_count: error_count as u64,
        })
    }

    /// Inserts or updates documents (upsert).
    pub fn upsert(&self, docs: &[&Doc]) -> Result<WriteResult> {
        let ptrs: Vec<*const zvec_sys::zvec_doc_t> =
            docs.iter().map(|d| d.handle as *const _).collect();
        let mut success_count: usize = 0;
        let mut error_count: usize = 0;

        check_error(unsafe {
            zvec_sys::zvec_collection_upsert(
                self.handle,
                ptrs.as_ptr(),
                ptrs.len(),
                &mut success_count,
                &mut error_count,
            )
        })?;

        Ok(WriteResult {
            success_count: success_count as u64,
            error_count: error_count as u64,
        })
    }

    /// Deletes documents by primary keys.
    pub fn delete(&self, pks: &[&str]) -> Result<WriteResult> {
        let c_pks: Vec<_> = pks
            .iter()
            .map(|pk| to_cstring(pk))
            .collect::<Result<Vec<_>>>()?;
        let c_ptrs: Vec<_> = c_pks.iter().map(|pk| pk.as_ptr()).collect();
        let mut success_count: usize = 0;
        let mut error_count: usize = 0;

        check_error(unsafe {
            zvec_sys::zvec_collection_delete(
                self.handle,
                c_ptrs.as_ptr(),
                c_ptrs.len(),
                &mut success_count,
                &mut error_count,
            )
        })?;

        Ok(WriteResult {
            success_count: success_count as u64,
            error_count: error_count as u64,
        })
    }

    /// Deletes documents matching a filter expression.
    pub fn delete_by_filter(&self, filter: &str) -> Result<()> {
        let c_filter = to_cstring(filter)?;
        check_error(unsafe {
            zvec_sys::zvec_collection_delete_by_filter(self.handle, c_filter.as_ptr())
        })
    }

    // =========================================================================
    // DQL Operations
    // =========================================================================

    /// Performs a vector similarity search.
    pub fn query(&self, query: &VectorQuery) -> Result<Vec<Doc>> {
        let mut results: *mut *mut zvec_sys::zvec_doc_t = ptr::null_mut();
        let mut result_count: usize = 0;

        check_error(unsafe {
            zvec_sys::zvec_collection_query(
                self.handle,
                query.handle,
                &mut results,
                &mut result_count,
            )
        })?;

        let docs = unsafe { collect_docs(results, result_count) };
        Ok(docs)
    }

    /// Fetches documents by primary keys.
    pub fn fetch(&self, pks: &[&str]) -> Result<Vec<Doc>> {
        let c_pks: Vec<_> = pks
            .iter()
            .map(|pk| to_cstring(pk))
            .collect::<Result<Vec<_>>>()?;
        let c_ptrs: Vec<_> = c_pks.iter().map(|pk| pk.as_ptr()).collect();

        let mut documents: *mut *mut zvec_sys::zvec_doc_t = ptr::null_mut();
        let mut found_count: usize = 0;

        check_error(unsafe {
            zvec_sys::zvec_collection_fetch(
                self.handle,
                c_ptrs.as_ptr(),
                c_ptrs.len(),
                &mut documents,
                &mut found_count,
            )
        })?;

        let docs = unsafe { collect_docs(documents, found_count) };
        Ok(docs)
    }

    // =========================================================================
    // Index Management
    // =========================================================================

    /// Creates an index on a field.
    pub fn create_index(&self, field_name: &str, params: &IndexParams) -> Result<()> {
        let c_name = to_cstring(field_name)?;
        check_error(unsafe {
            zvec_sys::zvec_collection_create_index(self.handle, c_name.as_ptr(), params.handle)
        })
    }

    /// Drops an index from a field.
    pub fn drop_index(&self, field_name: &str) -> Result<()> {
        let c_name = to_cstring(field_name)?;
        check_error(unsafe {
            zvec_sys::zvec_collection_drop_index(self.handle, c_name.as_ptr())
        })
    }

    /// Optimizes the collection (rebuild indexes, merge segments, etc.).
    pub fn optimize(&self) -> Result<()> {
        check_error(unsafe { zvec_sys::zvec_collection_optimize(self.handle) })
    }

    // =========================================================================
    // DDL Operations
    // =========================================================================

    /// Adds a column to the collection.
    pub fn add_column(
        &self,
        field_schema: &FieldSchema,
        default_expr: Option<&str>,
    ) -> Result<()> {
        let c_expr_owned = default_expr.map(|e| to_cstring(e)).transpose()?;
        let c_expr_ptr = c_expr_owned
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null());
        check_error(unsafe {
            zvec_sys::zvec_collection_add_column(self.handle, field_schema.handle, c_expr_ptr)
        })
    }

    /// Drops a column from the collection.
    pub fn drop_column(&self, name: &str) -> Result<()> {
        let c_name = to_cstring(name)?;
        check_error(unsafe {
            zvec_sys::zvec_collection_drop_column(self.handle, c_name.as_ptr())
        })
    }

    /// Closes the collection explicitly.
    pub fn close(self) -> Result<()> {
        // Drop will handle the close
        drop(self);
        Ok(())
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Safety: handle was created by zvec_collection_create_and_open or zvec_collection_open
            unsafe { zvec_sys::zvec_collection_close(self.handle) };
        }
    }
}

// Safety: The C-API is internally thread-safe for collection operations
unsafe impl Send for Collection {}
unsafe impl Sync for Collection {}

/// Collects document pointers from a C array into a Vec<Doc>.
///
/// # Safety
/// `results` must point to a valid array of `count` document pointers
/// allocated by the zvec C library.
unsafe fn collect_docs(
    results: *mut *mut zvec_sys::zvec_doc_t,
    count: usize,
) -> Vec<Doc> {
    if results.is_null() || count == 0 {
        return Vec::new();
    }

    let mut docs = Vec::with_capacity(count);
    for i in 0..count {
        let doc_ptr = *results.add(i);
        if !doc_ptr.is_null() {
            docs.push(Doc::from_borrowed(doc_ptr));
        }
    }

    // Free the array itself (but not the individual docs, which are now owned by Doc wrappers)
    // Note: zvec_docs_free would free both the array and docs, but we want to manage docs ourselves.
    // We use zvec_free to free just the pointer array.
    zvec_sys::zvec_free(results as *mut std::os::raw::c_void);

    docs
}
