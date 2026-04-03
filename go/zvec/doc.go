package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import "unsafe"

// Doc represents a document in the zvec vector database.
// It wraps the C zvec_doc_t handle and manages ownership.
type Doc struct {
	handle *C.zvec_doc_t
	owned  bool
}

// NewDoc creates a new document with owned=true.
// The caller is responsible for calling Destroy() when done.
func NewDoc() *Doc {
	handle := C.zvec_doc_create()
	if handle == nil {
		return nil
	}
	return &Doc{handle: handle, owned: true}
}

// Destroy releases the document resources.
// Only effective when owned=true.
func (d *Doc) Destroy() {
	if d.handle != nil && d.owned {
		C.zvec_doc_destroy(d.handle)
		d.handle = nil
	}
}

// Clear clears all fields and metadata from the document.
func (d *Doc) Clear() {
	C.zvec_doc_clear(d.handle)
}

// SetPK sets the primary key of the document.
func (d *Doc) SetPK(pk string) {
	cPK := C.CString(pk)
	defer C.free(unsafe.Pointer(cPK))
	C.zvec_doc_set_pk(d.handle, cPK)
}

// SetDocID sets the document ID.
func (d *Doc) SetDocID(docID uint64) {
	C.zvec_doc_set_doc_id(d.handle, C.uint64_t(docID))
}

// SetScore sets the document score.
func (d *Doc) SetScore(score float32) {
	C.zvec_doc_set_score(d.handle, C.float(score))
}

// SetOperator sets the document operator.
func (d *Doc) SetOperator(op DocOperator) {
	C.zvec_doc_set_operator(d.handle, C.zvec_doc_operator_t(op))
}

// AddStringField adds a string field to the document.
func (d *Doc) AddStringField(name, value string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_STRING,
		unsafe.Pointer(cValue), C.size_t(len(value)),
	))
}

// AddBoolField adds a boolean field to the document.
func (d *Doc) AddBoolField(name string, value bool) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var cValue C.bool = C.bool(value)
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_BOOL,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddInt32Field adds an int32 field to the document.
func (d *Doc) AddInt32Field(name string, value int32) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cValue := C.int32_t(value)
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT32,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddInt64Field adds an int64 field to the document.
func (d *Doc) AddInt64Field(name string, value int64) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cValue := C.int64_t(value)
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT64,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddUint32Field adds a uint32 field to the document.
func (d *Doc) AddUint32Field(name string, value uint32) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cValue := C.uint32_t(value)
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT32,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddUint64Field adds a uint64 field to the document.
func (d *Doc) AddUint64Field(name string, value uint64) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cValue := C.uint64_t(value)
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT64,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddFloatField adds a float32 field to the document.
func (d *Doc) AddFloatField(name string, value float32) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cValue := C.float(value)
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_FLOAT,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddDoubleField adds a float64 field to the document.
func (d *Doc) AddDoubleField(name string, value float64) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cValue := C.double(value)
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_DOUBLE,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddVectorFP32Field adds a float32 vector field to the document.
func (d *Doc) AddVectorFP32Field(name string, vector []float32) error {
	if len(vector) == 0 {
		return &Error{Code: ErrInvalidArgument, Message: "vector cannot be empty"}
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_VECTOR_FP32,
		unsafe.Pointer(&vector[0]), C.size_t(len(vector)*4),
	))
}

// AddBinaryField adds a binary field to the document.
// The data must not be empty; use SetFieldNull for null values.
func (d *Doc) AddBinaryField(name string, data []byte) error {
	if len(data) == 0 {
		return &Error{Code: ErrInvalidArgument, Message: "binary data cannot be empty"}
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_doc_add_field_by_value(
		d.handle, cName, C.ZVEC_DATA_TYPE_BINARY,
		unsafe.Pointer(&data[0]), C.size_t(len(data)),
	))
}

// SetFieldNull sets a field to null.
func (d *Doc) SetFieldNull(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_doc_set_field_null(d.handle, cName))
}

// RemoveField removes a field from the document.
func (d *Doc) RemoveField(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return toError(C.zvec_doc_remove_field(d.handle, cName))
}

// GetPK returns the primary key of the document.
func (d *Doc) GetPK() string {
	cPK := C.zvec_doc_get_pk_copy(d.handle)
	if cPK == nil {
		return ""
	}
	defer C.zvec_free(unsafe.Pointer(cPK))
	return C.GoString(cPK)
}

// GetDocID returns the document ID.
func (d *Doc) GetDocID() uint64 {
	return uint64(C.zvec_doc_get_doc_id(d.handle))
}

// GetScore returns the document score.
func (d *Doc) GetScore() float32 {
	return float32(C.zvec_doc_get_score(d.handle))
}

// GetOperator returns the document operator.
func (d *Doc) GetOperator() DocOperator {
	return DocOperator(C.zvec_doc_get_operator(d.handle))
}

// GetFieldCount returns the number of fields in the document.
func (d *Doc) GetFieldCount() int {
	return int(C.zvec_doc_get_field_count(d.handle))
}

// IsEmpty returns true if the document has no fields.
func (d *Doc) IsEmpty() bool {
	return bool(C.zvec_doc_is_empty(d.handle))
}

// GetStringField returns the string value of a field.
func (d *Doc) GetStringField(name string) (string, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var cValue unsafe.Pointer
	var valueSize C.size_t
	err := toError(C.zvec_doc_get_field_value_pointer(
		d.handle, cName, C.ZVEC_DATA_TYPE_STRING,
		(*unsafe.Pointer)(unsafe.Pointer(&cValue)), &valueSize,
	))
	if err != nil {
		return "", err
	}
	return C.GoStringN((*C.char)(cValue), C.int(valueSize)), nil
}

// GetBoolField returns the boolean value of a field.
func (d *Doc) GetBoolField(name string) (bool, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var value C.bool
	err := toError(C.zvec_doc_get_field_value_basic(
		d.handle, cName, C.ZVEC_DATA_TYPE_BOOL,
		unsafe.Pointer(&value), C.size_t(unsafe.Sizeof(value)),
	))
	if err != nil {
		return false, err
	}
	return bool(value), nil
}

// GetInt32Field returns the int32 value of a field.
func (d *Doc) GetInt32Field(name string) (int32, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var value C.int32_t
	err := toError(C.zvec_doc_get_field_value_basic(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT32,
		unsafe.Pointer(&value), C.size_t(unsafe.Sizeof(value)),
	))
	if err != nil {
		return 0, err
	}
	return int32(value), nil
}

// GetInt64Field returns the int64 value of a field.
func (d *Doc) GetInt64Field(name string) (int64, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var value C.int64_t
	err := toError(C.zvec_doc_get_field_value_basic(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT64,
		unsafe.Pointer(&value), C.size_t(unsafe.Sizeof(value)),
	))
	if err != nil {
		return 0, err
	}
	return int64(value), nil
}

// GetUint32Field returns the uint32 value of a field.
func (d *Doc) GetUint32Field(name string) (uint32, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var value C.uint32_t
	err := toError(C.zvec_doc_get_field_value_basic(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT32,
		unsafe.Pointer(&value), C.size_t(unsafe.Sizeof(value)),
	))
	if err != nil {
		return 0, err
	}
	return uint32(value), nil
}

// GetUint64Field returns the uint64 value of a field.
func (d *Doc) GetUint64Field(name string) (uint64, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var value C.uint64_t
	err := toError(C.zvec_doc_get_field_value_basic(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT64,
		unsafe.Pointer(&value), C.size_t(unsafe.Sizeof(value)),
	))
	if err != nil {
		return 0, err
	}
	return uint64(value), nil
}

// GetFloatField returns the float32 value of a field.
func (d *Doc) GetFloatField(name string) (float32, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var value C.float
	err := toError(C.zvec_doc_get_field_value_basic(
		d.handle, cName, C.ZVEC_DATA_TYPE_FLOAT,
		unsafe.Pointer(&value), C.size_t(unsafe.Sizeof(value)),
	))
	if err != nil {
		return 0, err
	}
	return float32(value), nil
}

// GetDoubleField returns the float64 value of a field.
func (d *Doc) GetDoubleField(name string) (float64, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var value C.double
	err := toError(C.zvec_doc_get_field_value_basic(
		d.handle, cName, C.ZVEC_DATA_TYPE_DOUBLE,
		unsafe.Pointer(&value), C.size_t(unsafe.Sizeof(value)),
	))
	if err != nil {
		return 0, err
	}
	return float64(value), nil
}

// GetVectorFP32Field returns the float32 vector value of a field.
func (d *Doc) GetVectorFP32Field(name string) ([]float32, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var cValue unsafe.Pointer
	var valueSize C.size_t
	err := toError(C.zvec_doc_get_field_value_pointer(
		d.handle, cName, C.ZVEC_DATA_TYPE_VECTOR_FP32,
		(*unsafe.Pointer)(unsafe.Pointer(&cValue)), &valueSize,
	))
	if err != nil {
		return nil, err
	}
	count := int(valueSize) / 4
	cSlice := unsafe.Slice((*float32)(cValue), count)
	result := make([]float32, count)
	copy(result, cSlice)
	return result, nil
}

// HasField returns true if the document has a field with the given name.
func (d *Doc) HasField(name string) bool {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return bool(C.zvec_doc_has_field(d.handle, cName))
}

// HasFieldValue returns true if the document has a field with the given name and a non-null value.
func (d *Doc) HasFieldValue(name string) bool {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return bool(C.zvec_doc_has_field_value(d.handle, cName))
}

// IsFieldNull returns true if the field with the given name is null.
func (d *Doc) IsFieldNull(name string) bool {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return bool(C.zvec_doc_is_field_null(d.handle, cName))
}

// GetFieldNames returns a list of all field names in the document.
func (d *Doc) GetFieldNames() ([]string, error) {
	var cNames **C.char
	var count C.size_t
	err := toError(C.zvec_doc_get_field_names(d.handle, &cNames, &count))
	if err != nil {
		return nil, err
	}
	defer C.zvec_free_str_array(cNames, count)
	nameSlice := unsafe.Slice(cNames, int(count))
	names := make([]string, int(count))
	for i := 0; i < int(count); i++ {
		names[i] = C.GoString(nameSlice[i])
	}
	return names, nil
}
