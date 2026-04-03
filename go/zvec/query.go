package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// HNSWQueryParams represents query parameters for HNSW index.
type HNSWQueryParams struct {
	handle *C.zvec_hnsw_query_params_t
}

// NewHNSWQueryParams creates a new HNSW query parameters instance.
func NewHNSWQueryParams(ef int, radius float32, isLinear, isUsingRefiner bool) *HNSWQueryParams {
	handle := C.zvec_query_params_hnsw_create(
		C.int(ef),
		C.float(radius),
		C.bool(isLinear),
		C.bool(isUsingRefiner),
	)
	if handle == nil {
		return nil
	}
	return &HNSWQueryParams{handle: handle}
}

// Destroy releases the HNSW query parameters resources.
func (p *HNSWQueryParams) Destroy() {
	if p.handle != nil {
		C.zvec_query_params_hnsw_destroy(p.handle)
		p.handle = nil
	}
}

// SetEf sets the ef parameter for HNSW query.
func (p *HNSWQueryParams) SetEf(ef int) error {
	return toError(C.zvec_query_params_hnsw_set_ef(p.handle, C.int(ef)))
}

// GetEf returns the ef parameter.
func (p *HNSWQueryParams) GetEf() int {
	return int(C.zvec_query_params_hnsw_get_ef(p.handle))
}

// IVFQueryParams represents query parameters for IVF index.
type IVFQueryParams struct {
	handle *C.zvec_ivf_query_params_t
}

// NewIVFQueryParams creates a new IVF query parameters instance.
func NewIVFQueryParams(nprobe int, isUsingRefiner bool, scaleFactor float32) *IVFQueryParams {
	handle := C.zvec_query_params_ivf_create(
		C.int(nprobe),
		C.bool(isUsingRefiner),
		C.float(scaleFactor),
	)
	if handle == nil {
		return nil
	}
	return &IVFQueryParams{handle: handle}
}

// Destroy releases the IVF query parameters resources.
func (p *IVFQueryParams) Destroy() {
	if p.handle != nil {
		C.zvec_query_params_ivf_destroy(p.handle)
		p.handle = nil
	}
}

// SetNprobe sets the nprobe parameter for IVF query.
func (p *IVFQueryParams) SetNprobe(nprobe int) error {
	return toError(C.zvec_query_params_ivf_set_nprobe(p.handle, C.int(nprobe)))
}

// FlatQueryParams represents query parameters for Flat index.
type FlatQueryParams struct {
	handle *C.zvec_flat_query_params_t
}

// NewFlatQueryParams creates a new Flat query parameters instance.
func NewFlatQueryParams(isUsingRefiner bool, scaleFactor float32) *FlatQueryParams {
	handle := C.zvec_query_params_flat_create(
		C.bool(isUsingRefiner),
		C.float(scaleFactor),
	)
	if handle == nil {
		return nil
	}
	return &FlatQueryParams{handle: handle}
}

// Destroy releases the Flat query parameters resources.
func (p *FlatQueryParams) Destroy() {
	if p.handle != nil {
		C.zvec_query_params_flat_destroy(p.handle)
		p.handle = nil
	}
}

// VectorQuery represents a vector query operation.
type VectorQuery struct {
	handle *C.zvec_vector_query_t
}

// NewVectorQuery creates a new vector query instance.
func NewVectorQuery() *VectorQuery {
	handle := C.zvec_vector_query_create()
	if handle == nil {
		return nil
	}
	return &VectorQuery{handle: handle}
}

// Destroy releases the vector query resources.
func (q *VectorQuery) Destroy() {
	if q.handle != nil {
		C.zvec_vector_query_destroy(q.handle)
		q.handle = nil
	}
}

// SetFieldName sets the field name for the vector query.
func (q *VectorQuery) SetFieldName(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_vector_query_set_field_name(q.handle, cName))
}

// GetFieldName returns the field name.
func (q *VectorQuery) GetFieldName() string {
	return C.GoString(C.zvec_vector_query_get_field_name(q.handle))
}

// SetTopK sets the top-k parameter for the query.
func (q *VectorQuery) SetTopK(topk int) error {
	return toError(C.zvec_vector_query_set_topk(q.handle, C.int(topk)))
}

// GetTopK returns the top-k parameter.
func (q *VectorQuery) GetTopK() int {
	return int(C.zvec_vector_query_get_topk(q.handle))
}

// SetQueryVector sets the query vector data.
func (q *VectorQuery) SetQueryVector(data []float32) error {
	if len(data) == 0 {
		return &Error{Code: ErrInvalidArgument, Message: "query vector cannot be empty"}
	}
	return toError(C.zvec_vector_query_set_query_vector(
		q.handle,
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)*4),
	))
}

// SetFilter sets the filter expression for the query.
func (q *VectorQuery) SetFilter(filter string) error {
	cFilter := C.CString(filter)
	defer C.free(unsafe.Pointer(cFilter))
	return toError(C.zvec_vector_query_set_filter(q.handle, cFilter))
}

// GetFilter returns the filter expression.
func (q *VectorQuery) GetFilter() string {
	return C.GoString(C.zvec_vector_query_get_filter(q.handle))
}

// SetIncludeVector sets whether to include vector data in results.
func (q *VectorQuery) SetIncludeVector(include bool) error {
	return toError(C.zvec_vector_query_set_include_vector(q.handle, C.bool(include)))
}

// GetIncludeVector returns whether vector data is included in results.
func (q *VectorQuery) GetIncludeVector() bool {
	return bool(C.zvec_vector_query_get_include_vector(q.handle))
}

// SetIncludeDocID sets whether to include document ID in results.
func (q *VectorQuery) SetIncludeDocID(include bool) error {
	return toError(C.zvec_vector_query_set_include_doc_id(q.handle, C.bool(include)))
}

// GetIncludeDocID returns whether document ID is included in results.
func (q *VectorQuery) GetIncludeDocID() bool {
	return bool(C.zvec_vector_query_get_include_doc_id(q.handle))
}

// SetOutputFields sets the output fields for the query.
func (q *VectorQuery) SetOutputFields(fields []string) error {
	if len(fields) == 0 {
		return nil
	}
	cFields := make([]*C.char, len(fields))
	for i, f := range fields {
		cFields[i] = C.CString(f)
	}
	defer func() {
		for _, cf := range cFields {
			C.free(unsafe.Pointer(cf))
		}
	}()
	return toError(C.zvec_vector_query_set_output_fields(
		q.handle,
		(**C.char)(unsafe.Pointer(&cFields[0])),
		C.size_t(len(fields)),
	))
}

// SetHNSWParams sets the HNSW query parameters.
// Note: ownership of params is transferred to the query.
func (q *VectorQuery) SetHNSWParams(params *HNSWQueryParams) error {
	err := toError(C.zvec_vector_query_set_hnsw_params(q.handle, params.handle))
	if err == nil {
		params.handle = nil // ownership transferred
	}
	return err
}

// SetIVFParams sets the IVF query parameters.
func (q *VectorQuery) SetIVFParams(params *IVFQueryParams) error {
	err := toError(C.zvec_vector_query_set_ivf_params(q.handle, params.handle))
	if err == nil {
		params.handle = nil // ownership transferred
	}
	return err
}

// SetFlatParams sets the Flat query parameters.
func (q *VectorQuery) SetFlatParams(params *FlatQueryParams) error {
	err := toError(C.zvec_vector_query_set_flat_params(q.handle, params.handle))
	if err == nil {
		params.handle = nil // ownership transferred
	}
	return err
}

// GroupByVectorQuery represents a group-by vector query operation.
type GroupByVectorQuery struct {
	handle *C.zvec_group_by_vector_query_t
}

// NewGroupByVectorQuery creates a new group-by vector query instance.
func NewGroupByVectorQuery() *GroupByVectorQuery {
	handle := C.zvec_group_by_vector_query_create()
	if handle == nil {
		return nil
	}
	return &GroupByVectorQuery{handle: handle}
}

// Destroy releases the group-by vector query resources.
func (q *GroupByVectorQuery) Destroy() {
	if q.handle != nil {
		C.zvec_group_by_vector_query_destroy(q.handle)
		q.handle = nil
	}
}

// SetFieldName sets the field name for the vector query.
func (q *GroupByVectorQuery) SetFieldName(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_group_by_vector_query_set_field_name(q.handle, cName))
}

// SetGroupByFieldName sets the group-by field name.
func (q *GroupByVectorQuery) SetGroupByFieldName(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_group_by_vector_query_set_group_by_field_name(q.handle, cName))
}

// SetGroupCount sets the group count parameter.
func (q *GroupByVectorQuery) SetGroupCount(count uint32) error {
	return toError(C.zvec_group_by_vector_query_set_group_count(q.handle, C.uint32_t(count)))
}

// SetGroupTopK sets the group top-k parameter.
func (q *GroupByVectorQuery) SetGroupTopK(topk uint32) error {
	return toError(C.zvec_group_by_vector_query_set_group_topk(q.handle, C.uint32_t(topk)))
}

// SetQueryVector sets the query vector data.
func (q *GroupByVectorQuery) SetQueryVector(data []float32) error {
	if len(data) == 0 {
		return &Error{Code: ErrInvalidArgument, Message: "query vector cannot be empty"}
	}
	return toError(C.zvec_group_by_vector_query_set_query_vector(
		q.handle,
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)*4),
	))
}

// SetFilter sets the filter expression for the query.
func (q *GroupByVectorQuery) SetFilter(filter string) error {
	cFilter := C.CString(filter)
	defer C.free(unsafe.Pointer(cFilter))
	return toError(C.zvec_group_by_vector_query_set_filter(q.handle, cFilter))
}

// SetIncludeVector sets whether to include vector data in results.
func (q *GroupByVectorQuery) SetIncludeVector(include bool) error {
	return toError(C.zvec_group_by_vector_query_set_include_vector(q.handle, C.bool(include)))
}

// SetOutputFields sets the output fields for the query.
func (q *GroupByVectorQuery) SetOutputFields(fields []string) error {
	if len(fields) == 0 {
		return nil
	}
	cFields := make([]*C.char, len(fields))
	for i, f := range fields {
		cFields[i] = C.CString(f)
	}
	defer func() {
		for _, cf := range cFields {
			C.free(unsafe.Pointer(cf))
		}
	}()
	return toError(C.zvec_group_by_vector_query_set_output_fields(
		q.handle,
		(**C.char)(unsafe.Pointer(&cFields[0])),
		C.size_t(len(fields)),
	))
}

// SetHNSWParams sets the HNSW query parameters.
func (q *GroupByVectorQuery) SetHNSWParams(params *HNSWQueryParams) error {
	err := toError(C.zvec_group_by_vector_query_set_hnsw_params(q.handle, params.handle))
	if err == nil {
		params.handle = nil // ownership transferred
	}
	return err
}

// SetIVFParams sets the IVF query parameters.
func (q *GroupByVectorQuery) SetIVFParams(params *IVFQueryParams) error {
	err := toError(C.zvec_group_by_vector_query_set_ivf_params(q.handle, params.handle))
	if err == nil {
		params.handle = nil // ownership transferred
	}
	return err
}

// SetFlatParams sets the Flat query parameters.
func (q *GroupByVectorQuery) SetFlatParams(params *FlatQueryParams) error {
	err := toError(C.zvec_group_by_vector_query_set_flat_params(q.handle, params.handle))
	if err == nil {
		params.handle = nil // ownership transferred
	}
	return err
}
