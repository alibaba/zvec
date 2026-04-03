use std::os::raw::c_void;

use crate::error::{check_error, to_cstring, Error, ErrorCode, Result};

/// HNSW-specific query parameters.
pub struct HnswQueryParams {
    pub(crate) handle: *mut zvec_sys::zvec_hnsw_query_params_t,
}

impl HnswQueryParams {
    /// Creates new HNSW query parameters.
    pub fn new(ef: i32, radius: f32, is_linear: bool, is_using_refiner: bool) -> Self {
        let handle = unsafe {
            zvec_sys::zvec_query_params_hnsw_create(ef, radius, is_linear, is_using_refiner)
        };
        HnswQueryParams { handle }
    }

    /// Sets the exploration factor.
    pub fn set_ef(&mut self, ef: i32) -> Result<()> {
        check_error(unsafe { zvec_sys::zvec_query_params_hnsw_set_ef(self.handle, ef) })
    }

    /// Returns the exploration factor.
    pub fn ef(&self) -> i32 {
        unsafe { zvec_sys::zvec_query_params_hnsw_get_ef(self.handle) }
    }
}

impl Drop for HnswQueryParams {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { zvec_sys::zvec_query_params_hnsw_destroy(self.handle) };
        }
    }
}

/// IVF-specific query parameters.
pub struct IvfQueryParams {
    pub(crate) handle: *mut zvec_sys::zvec_ivf_query_params_t,
}

impl IvfQueryParams {
    /// Creates new IVF query parameters.
    pub fn new(nprobe: i32, is_using_refiner: bool, scale_factor: f32) -> Self {
        let handle = unsafe {
            zvec_sys::zvec_query_params_ivf_create(nprobe, is_using_refiner, scale_factor)
        };
        IvfQueryParams { handle }
    }

    /// Sets the number of probe clusters.
    pub fn set_nprobe(&mut self, nprobe: i32) -> Result<()> {
        check_error(unsafe { zvec_sys::zvec_query_params_ivf_set_nprobe(self.handle, nprobe) })
    }

    /// Returns the number of probe clusters.
    pub fn nprobe(&self) -> i32 {
        unsafe { zvec_sys::zvec_query_params_ivf_get_nprobe(self.handle) }
    }
}

impl Drop for IvfQueryParams {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { zvec_sys::zvec_query_params_ivf_destroy(self.handle) };
        }
    }
}

/// Flat-specific query parameters.
pub struct FlatQueryParams {
    pub(crate) handle: *mut zvec_sys::zvec_flat_query_params_t,
}

impl FlatQueryParams {
    /// Creates new Flat query parameters.
    pub fn new(is_using_refiner: bool, scale_factor: f32) -> Self {
        let handle =
            unsafe { zvec_sys::zvec_query_params_flat_create(is_using_refiner, scale_factor) };
        FlatQueryParams { handle }
    }
}

impl Drop for FlatQueryParams {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { zvec_sys::zvec_query_params_flat_destroy(self.handle) };
        }
    }
}

/// A vector similarity search query.
pub struct VectorQuery {
    pub(crate) handle: *mut zvec_sys::zvec_vector_query_t,
}

impl VectorQuery {
    /// Creates a new vector query with the given field name, query vector, and topk.
    pub fn new(field_name: &str, vector: &[f32], topk: i32) -> Result<Self> {
        let handle = unsafe { zvec_sys::zvec_vector_query_create() };
        if handle.is_null() {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: "failed to create vector query".into(),
            });
        }

        let c_field = to_cstring(field_name)?;
        let query = VectorQuery { handle };

        check_error(unsafe {
            zvec_sys::zvec_vector_query_set_field_name(query.handle, c_field.as_ptr())
        })?;
        check_error(unsafe {
            zvec_sys::zvec_vector_query_set_query_vector(
                query.handle,
                vector.as_ptr() as *const c_void,
                vector.len() * std::mem::size_of::<f32>(),
            )
        })?;
        check_error(unsafe { zvec_sys::zvec_vector_query_set_topk(query.handle, topk) })?;

        Ok(query)
    }

    /// Returns a builder for constructing a vector query.
    pub fn builder() -> VectorQueryBuilder {
        VectorQueryBuilder::new()
    }

    /// Sets the filter expression.
    pub fn set_filter(&mut self, filter: &str) -> Result<()> {
        let c_filter = to_cstring(filter)?;
        check_error(unsafe {
            zvec_sys::zvec_vector_query_set_filter(self.handle, c_filter.as_ptr())
        })
    }

    /// Sets whether to include vector data in results.
    pub fn set_include_vector(&mut self, include: bool) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_vector_query_set_include_vector(self.handle, include)
        })
    }

    /// Sets whether to include doc ID in results.
    pub fn set_include_doc_id(&mut self, include: bool) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_vector_query_set_include_doc_id(self.handle, include)
        })
    }

    /// Sets the output fields to include in results.
    pub fn set_output_fields(&mut self, fields: &[&str]) -> Result<()> {
        let c_fields: Vec<_> = fields
            .iter()
            .map(|f| to_cstring(f))
            .collect::<Result<Vec<_>>>()?;
        let c_ptrs: Vec<_> = c_fields.iter().map(|f| f.as_ptr()).collect();
        check_error(unsafe {
            zvec_sys::zvec_vector_query_set_output_fields(
                self.handle,
                c_ptrs.as_ptr(),
                c_ptrs.len(),
            )
        })
    }

    /// Sets HNSW query parameters (takes ownership).
    pub fn set_hnsw_params(&mut self, mut params: HnswQueryParams) -> Result<()> {
        let result = check_error(unsafe {
            zvec_sys::zvec_vector_query_set_hnsw_params(self.handle, params.handle)
        });
        // Ownership transferred to query; prevent double-free
        params.handle = std::ptr::null_mut();
        result
    }

    /// Sets IVF query parameters (takes ownership).
    pub fn set_ivf_params(&mut self, mut params: IvfQueryParams) -> Result<()> {
        let result = check_error(unsafe {
            zvec_sys::zvec_vector_query_set_ivf_params(self.handle, params.handle)
        });
        params.handle = std::ptr::null_mut();
        result
    }

    /// Sets Flat query parameters (takes ownership).
    pub fn set_flat_params(&mut self, mut params: FlatQueryParams) -> Result<()> {
        let result = check_error(unsafe {
            zvec_sys::zvec_vector_query_set_flat_params(self.handle, params.handle)
        });
        params.handle = std::ptr::null_mut();
        result
    }
}

impl Drop for VectorQuery {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { zvec_sys::zvec_vector_query_destroy(self.handle) };
        }
    }
}

/// Builder for constructing a [`VectorQuery`].
pub struct VectorQueryBuilder {
    field_name: Option<String>,
    vector: Option<Vec<f32>>,
    topk: i32,
    filter: Option<String>,
    include_vector: Option<bool>,
    include_doc_id: Option<bool>,
    output_fields: Option<Vec<String>>,
}

impl VectorQueryBuilder {
    fn new() -> Self {
        VectorQueryBuilder {
            field_name: None,
            vector: None,
            topk: 10,
            filter: None,
            include_vector: None,
            include_doc_id: None,
            output_fields: None,
        }
    }

    /// Sets the field name to query.
    pub fn field_name(mut self, name: &str) -> Self {
        self.field_name = Some(name.to_string());
        self
    }

    /// Sets the query vector.
    pub fn vector(mut self, vector: &[f32]) -> Self {
        self.vector = Some(vector.to_vec());
        self
    }

    /// Sets the number of results to return.
    pub fn topk(mut self, topk: i32) -> Self {
        self.topk = topk;
        self
    }

    /// Sets the filter expression.
    pub fn filter(mut self, filter: &str) -> Self {
        self.filter = Some(filter.to_string());
        self
    }

    /// Sets whether to include vector data in results.
    pub fn include_vector(mut self, include: bool) -> Self {
        self.include_vector = Some(include);
        self
    }

    /// Sets whether to include doc ID in results.
    pub fn include_doc_id(mut self, include: bool) -> Self {
        self.include_doc_id = Some(include);
        self
    }

    /// Sets the output fields.
    pub fn output_fields(mut self, fields: &[&str]) -> Self {
        self.output_fields = Some(fields.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Builds the vector query.
    pub fn build(self) -> Result<VectorQuery> {
        let field_name = self.field_name.ok_or_else(|| Error {
            code: ErrorCode::InvalidArgument,
            message: "field_name is required".into(),
        })?;
        let vector = self.vector.ok_or_else(|| Error {
            code: ErrorCode::InvalidArgument,
            message: "vector is required".into(),
        })?;

        let mut query = VectorQuery::new(&field_name, &vector, self.topk)?;

        if let Some(filter) = &self.filter {
            query.set_filter(filter)?;
        }
        if let Some(include) = self.include_vector {
            query.set_include_vector(include)?;
        }
        if let Some(include) = self.include_doc_id {
            query.set_include_doc_id(include)?;
        }
        if let Some(fields) = &self.output_fields {
            let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
            query.set_output_fields(&field_refs)?;
        }

        Ok(query)
    }
}

/// A grouped vector similarity search query.
pub struct GroupByVectorQuery {
    pub(crate) handle: *mut zvec_sys::zvec_group_by_vector_query_t,
}

impl GroupByVectorQuery {
    /// Creates a new group-by vector query.
    pub fn new(
        field_name: &str,
        group_by_field: &str,
        vector: &[f32],
        group_count: u32,
        group_topk: u32,
    ) -> Result<Self> {
        let handle = unsafe { zvec_sys::zvec_group_by_vector_query_create() };
        if handle.is_null() {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: "failed to create group by vector query".into(),
            });
        }

        let c_field = to_cstring(field_name)?;
        let c_group_field = to_cstring(group_by_field)?;

        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_field_name(handle, c_field.as_ptr())
        })?;
        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_group_by_field_name(
                handle,
                c_group_field.as_ptr(),
            )
        })?;
        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_query_vector(
                handle,
                vector.as_ptr() as *const c_void,
                vector.len() * std::mem::size_of::<f32>(),
            )
        })?;
        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_group_count(handle, group_count)
        })?;
        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_group_topk(handle, group_topk)
        })?;

        Ok(GroupByVectorQuery { handle })
    }

    /// Sets the filter expression.
    pub fn set_filter(&mut self, filter: &str) -> Result<()> {
        let c_filter = to_cstring(filter)?;
        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_filter(self.handle, c_filter.as_ptr())
        })
    }

    /// Sets whether to include vector data in results.
    pub fn set_include_vector(&mut self, include: bool) -> Result<()> {
        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_include_vector(self.handle, include)
        })
    }

    /// Sets the output fields.
    pub fn set_output_fields(&mut self, fields: &[&str]) -> Result<()> {
        let c_fields: Vec<_> = fields
            .iter()
            .map(|f| to_cstring(f))
            .collect::<Result<Vec<_>>>()?;
        let c_ptrs: Vec<_> = c_fields.iter().map(|f| f.as_ptr()).collect();
        check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_output_fields(
                self.handle,
                c_ptrs.as_ptr(),
                c_ptrs.len(),
            )
        })
    }

    /// Sets HNSW query parameters (takes ownership).
    pub fn set_hnsw_params(&mut self, mut params: HnswQueryParams) -> Result<()> {
        let result = check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_hnsw_params(self.handle, params.handle)
        });
        params.handle = std::ptr::null_mut();
        result
    }

    /// Sets IVF query parameters (takes ownership).
    pub fn set_ivf_params(&mut self, mut params: IvfQueryParams) -> Result<()> {
        let result = check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_ivf_params(self.handle, params.handle)
        });
        params.handle = std::ptr::null_mut();
        result
    }

    /// Sets Flat query parameters (takes ownership).
    pub fn set_flat_params(&mut self, mut params: FlatQueryParams) -> Result<()> {
        let result = check_error(unsafe {
            zvec_sys::zvec_group_by_vector_query_set_flat_params(self.handle, params.handle)
        });
        params.handle = std::ptr::null_mut();
        result
    }
}

impl Drop for GroupByVectorQuery {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { zvec_sys::zvec_group_by_vector_query_destroy(self.handle) };
        }
    }
}
