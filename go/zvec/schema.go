package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"
)

// IndexParams wraps zvec_index_params_t (opaque pointer).
type IndexParams struct {
	handle *C.zvec_index_params_t
}

// NewIndexParams creates index parameters with the specified index type.
func NewIndexParams(indexType IndexType) *IndexParams {
	handle := C.zvec_index_params_create(C.zvec_index_type_t(indexType))
	if handle == nil {
		return nil
	}
	return &IndexParams{handle: handle}
}

// NewHNSWIndexParams creates HNSW index parameters with the specified metric type and parameters.
func NewHNSWIndexParams(metric MetricType, m, efConstruction int) *IndexParams {
	params := NewIndexParams(IndexTypeHNSW)
	if params == nil {
		return nil
	}
	_ = params.SetMetricType(metric)
	_ = params.SetHNSWParams(m, efConstruction)
	return params
}

// NewInvertIndexParams creates invert index parameters with the specified options.
func NewInvertIndexParams(enableRangeOpt, enableWildcard bool) *IndexParams {
	params := NewIndexParams(IndexTypeInvert)
	if params == nil {
		return nil
	}
	_ = params.SetInvertParams(enableRangeOpt, enableWildcard)
	return params
}

// NewIVFIndexParams creates IVF index parameters with the specified metric type and parameters.
func NewIVFIndexParams(metric MetricType, nList, nIters int, useSoar bool) *IndexParams {
	params := NewIndexParams(IndexTypeIVF)
	if params == nil {
		return nil
	}
	_ = params.SetMetricType(metric)
	_ = params.SetIVFParams(nList, nIters, useSoar)
	return params
}

// NewFlatIndexParams creates Flat index parameters with the specified metric type.
func NewFlatIndexParams(metric MetricType) *IndexParams {
	params := NewIndexParams(IndexTypeFlat)
	if params == nil {
		return nil
	}
	_ = params.SetMetricType(metric)
	return params
}

// Destroy releases the index parameters resources.
func (p *IndexParams) Destroy() {
	if p.handle != nil {
		C.zvec_index_params_destroy(p.handle)
		p.handle = nil
	}
}

// GetType returns the index type.
func (p *IndexParams) GetType() IndexType {
	return IndexType(C.zvec_index_params_get_type(p.handle))
}

// SetMetricType sets the metric type for vector indexes.
func (p *IndexParams) SetMetricType(metric MetricType) error {
	return toError(C.zvec_index_params_set_metric_type(p.handle, C.zvec_metric_type_t(metric)))
}

// GetMetricType returns the metric type.
func (p *IndexParams) GetMetricType() MetricType {
	return MetricType(C.zvec_index_params_get_metric_type(p.handle))
}

// SetQuantizeType sets the quantize type for vector indexes.
func (p *IndexParams) SetQuantizeType(quantize QuantizeType) error {
	return toError(C.zvec_index_params_set_quantize_type(p.handle, C.zvec_quantize_type_t(quantize)))
}

// GetQuantizeType returns the quantize type.
func (p *IndexParams) GetQuantizeType() QuantizeType {
	return QuantizeType(C.zvec_index_params_get_quantize_type(p.handle))
}

// SetHNSWParams sets HNSW specific parameters.
func (p *IndexParams) SetHNSWParams(m, efConstruction int) error {
	return toError(C.zvec_index_params_set_hnsw_params(p.handle, C.int(m), C.int(efConstruction)))
}

// GetHNSWM returns the HNSW m parameter.
func (p *IndexParams) GetHNSWM() int {
	return int(C.zvec_index_params_get_hnsw_m(p.handle))
}

// GetHNSWEfConstruction returns the HNSW ef_construction parameter.
func (p *IndexParams) GetHNSWEfConstruction() int {
	return int(C.zvec_index_params_get_hnsw_ef_construction(p.handle))
}

// SetIVFParams sets IVF specific parameters.
func (p *IndexParams) SetIVFParams(nList, nIters int, useSoar bool) error {
	return toError(C.zvec_index_params_set_ivf_params(p.handle, C.int(nList), C.int(nIters), C.bool(useSoar)))
}

// SetInvertParams sets invert index specific parameters.
func (p *IndexParams) SetInvertParams(enableRangeOpt, enableWildcard bool) error {
	return toError(C.zvec_index_params_set_invert_params(p.handle, C.bool(enableRangeOpt), C.bool(enableWildcard)))
}

// FieldSchema wraps zvec_field_schema_t (opaque pointer).
type FieldSchema struct {
	handle *C.zvec_field_schema_t
	owned  bool
}

// NewFieldSchema creates a new field schema with the specified parameters.
func NewFieldSchema(name string, dataType DataType, nullable bool, dimension uint32) *FieldSchema {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	handle := C.zvec_field_schema_create(cName, C.zvec_data_type_t(dataType), C.bool(nullable), C.uint32_t(dimension))
	if handle == nil {
		return nil
	}
	return &FieldSchema{handle: handle, owned: true}
}

// Destroy releases the field schema resources if owned.
func (f *FieldSchema) Destroy() {
	if f.handle != nil && f.owned {
		C.zvec_field_schema_destroy(f.handle)
		f.handle = nil
	}
}

// GetName returns the field name.
func (f *FieldSchema) GetName() string {
	return C.GoString(C.zvec_field_schema_get_name(f.handle))
}

// SetName sets the field name.
func (f *FieldSchema) SetName(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_field_schema_set_name(f.handle, cName))
}

// GetDataType returns the field data type.
func (f *FieldSchema) GetDataType() DataType {
	return DataType(C.zvec_field_schema_get_data_type(f.handle))
}

// SetDataType sets the field data type.
func (f *FieldSchema) SetDataType(dataType DataType) error {
	return toError(C.zvec_field_schema_set_data_type(f.handle, C.zvec_data_type_t(dataType)))
}

// IsNullable returns whether the field is nullable.
func (f *FieldSchema) IsNullable() bool {
	return bool(C.zvec_field_schema_is_nullable(f.handle))
}

// SetNullable sets whether the field is nullable.
func (f *FieldSchema) SetNullable(nullable bool) error {
	return toError(C.zvec_field_schema_set_nullable(f.handle, C.bool(nullable)))
}

// GetDimension returns the field dimension (for vector fields).
func (f *FieldSchema) GetDimension() uint32 {
	return uint32(C.zvec_field_schema_get_dimension(f.handle))
}

// SetDimension sets the field dimension (for vector fields).
func (f *FieldSchema) SetDimension(dimension uint32) error {
	return toError(C.zvec_field_schema_set_dimension(f.handle, C.uint32_t(dimension)))
}

// IsVectorField returns whether the field is a vector field (dense or sparse).
func (f *FieldSchema) IsVectorField() bool {
	return bool(C.zvec_field_schema_is_vector_field(f.handle))
}

// IsDenseVector returns whether the field is a dense vector field.
func (f *FieldSchema) IsDenseVector() bool {
	return bool(C.zvec_field_schema_is_dense_vector(f.handle))
}

// IsSparseVector returns whether the field is a sparse vector field.
func (f *FieldSchema) IsSparseVector() bool {
	return bool(C.zvec_field_schema_is_sparse_vector(f.handle))
}

// HasIndex returns whether the field has an index.
func (f *FieldSchema) HasIndex() bool {
	return bool(C.zvec_field_schema_has_index(f.handle))
}

// GetIndexType returns the index type of the field.
func (f *FieldSchema) GetIndexType() IndexType {
	return IndexType(C.zvec_field_schema_get_index_type(f.handle))
}

// SetIndexParams sets the index parameters for the field.
func (f *FieldSchema) SetIndexParams(params *IndexParams) error {
	return toError(C.zvec_field_schema_set_index_params(f.handle, params.handle))
}

// CollectionSchema wraps zvec_collection_schema_t (opaque pointer).
type CollectionSchema struct {
	handle *C.zvec_collection_schema_t
}

// NewCollectionSchema creates a new collection schema with the specified name.
func NewCollectionSchema(name string) *CollectionSchema {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	handle := C.zvec_collection_schema_create(cName)
	if handle == nil {
		return nil
	}
	return &CollectionSchema{handle: handle}
}

// Destroy releases the collection schema resources.
func (s *CollectionSchema) Destroy() {
	if s.handle != nil {
		C.zvec_collection_schema_destroy(s.handle)
		s.handle = nil
	}
}

// GetName returns the collection name.
func (s *CollectionSchema) GetName() string {
	return C.GoString(C.zvec_collection_schema_get_name(s.handle))
}

// SetName sets the collection name.
func (s *CollectionSchema) SetName(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_collection_schema_set_name(s.handle, cName))
}

// AddField adds a field to the collection schema.
func (s *CollectionSchema) AddField(field *FieldSchema) error {
	return toError(C.zvec_collection_schema_add_field(s.handle, field.handle))
}

// HasField returns whether the collection has a field with the specified name.
func (s *CollectionSchema) HasField(name string) bool {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return bool(C.zvec_collection_schema_has_field(s.handle, cName))
}

// GetField returns the field with the specified name (non-owning).
func (s *CollectionSchema) GetField(name string) *FieldSchema {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	handle := C.zvec_collection_schema_get_field(s.handle, cName)
	if handle == nil {
		return nil
	}
	return &FieldSchema{handle: handle, owned: false}
}

// DropField drops a field from the collection schema.
func (s *CollectionSchema) DropField(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_collection_schema_drop_field(s.handle, cName))
}

// AddIndex adds an index to a field.
func (s *CollectionSchema) AddIndex(fieldName string, params *IndexParams) error {
	cFieldName := C.CString(fieldName)
	defer C.free(unsafe.Pointer(cFieldName))
	return toError(C.zvec_collection_schema_add_index(s.handle, cFieldName, params.handle))
}

// DropIndex drops an index from a field.
func (s *CollectionSchema) DropIndex(fieldName string) error {
	cFieldName := C.CString(fieldName)
	defer C.free(unsafe.Pointer(cFieldName))
	return toError(C.zvec_collection_schema_drop_index(s.handle, cFieldName))
}

// HasIndex returns whether a field has an index.
func (s *CollectionSchema) HasIndex(fieldName string) bool {
	cFieldName := C.CString(fieldName)
	defer C.free(unsafe.Pointer(cFieldName))
	return bool(C.zvec_collection_schema_has_index(s.handle, cFieldName))
}

// SetMaxDocCountPerSegment sets the maximum document count per segment.
func (s *CollectionSchema) SetMaxDocCountPerSegment(count uint64) error {
	return toError(C.zvec_collection_schema_set_max_doc_count_per_segment(s.handle, C.uint64_t(count)))
}

// GetMaxDocCountPerSegment returns the maximum document count per segment.
func (s *CollectionSchema) GetMaxDocCountPerSegment() uint64 {
	return uint64(C.zvec_collection_schema_get_max_doc_count_per_segment(s.handle))
}
