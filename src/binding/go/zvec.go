// Copyright 2025-present the zvec project
package zvec

/*
#cgo CFLAGS: -I${SRCDIR}/../c/include
#cgo LDFLAGS: -L${SRCDIR}/../../../build/lib -L${SRCDIR}/../../../build/external/usr/local/lib -L${SRCDIR}/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/release -L${SRCDIR}/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/re2_ep-prefix/src/re2_ep-build -L${SRCDIR}/../../../build/thirdparty/arrow/arrow/src/ARROW.BUILD-build/utf8proc_ep-prefix/src/utf8proc_ep-build
#cgo darwin LDFLAGS: -Wl,-force_load,${SRCDIR}/../../../build/lib/libzvec_c.a -Wl,-force_load,${SRCDIR}/../../../build/lib/libzvec_db.a -Wl,-force_load,${SRCDIR}/../../../build/lib/libzvec_core.a -Wl,-force_load,${SRCDIR}/../../../build/lib/libzvec_ailego.a
#cgo !darwin LDFLAGS: -lzvec_c -lzvec_db -lzvec_core -lzvec_ailego
#cgo LDFLAGS: -lzvec_proto -lrocksdb -lroaring -lglog -lprotobuf -larrow -lparquet -larrow_dataset -larrow_acero -larrow_compute -larrow_bundled_dependencies -lre2 -lutf8proc -llz4 -lantlr4-runtime -lgflags_nothreads -lz -lstdc++
#cgo darwin LDFLAGS: -framework CoreFoundation -framework Security
#include <stdlib.h>
#include "zvec/zvec.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

const (
	TypeUndefined        uint32 = 0
	TypeBinary           uint32 = 1
	TypeString           uint32 = 2
	TypeBool             uint32 = 3
	TypeInt32            uint32 = 4
	TypeInt64            uint32 = 5
	TypeUint32           uint32 = 6
	TypeUint64           uint32 = 7
	TypeFloat            uint32 = 8
	TypeDouble           uint32 = 9
	TypeVectorBinary32   uint32 = 20
	TypeVectorFp16       uint32 = 22
	TypeVectorFp32       uint32 = 23
	TypeVectorFp64       uint32 = 24
	TypeVectorInt8       uint32 = 26
	TypeSparseVectorFp16 uint32 = 30
	TypeSparseVectorFp32 uint32 = 31
	TypeArrayString      uint32 = 41
	TypeArrayInt32       uint32 = 43
	TypeArrayFloat       uint32 = 47
)

type ZVecError struct {
	Code    int
	Message string
}

func (e *ZVecError) Error() string {
	return fmt.Sprintf("zvec error %d: %s", e.Code, e.Message)
}

func mapError(status *C.zvec_status_t) error {
	if status == nil {
		return nil
	}
	defer C.zvec_status_destroy(status)
	code := int(C.zvec_status_code(status))
	msg := C.GoString(C.zvec_status_message(status))
	return &ZVecError{Code: code, Message: msg}
}

type Schema struct {
	ptr *C.zvec_schema_t
}

func NewSchema(name string) *Schema {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	s := &Schema{ptr: C.zvec_schema_create(cName)}
	runtime.SetFinalizer(s, func(s *Schema) {
		C.zvec_schema_destroy(s.ptr)
	})
	return s
}

func (s *Schema) AddField(name string, dataType uint32, dimension uint32) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	status := C.zvec_schema_add_field(s.ptr, cName, C.zvec_data_type_t(dataType), C.uint32_t(dimension))
	return mapError(status)
}

type Doc struct {
	ptr *C.zvec_doc_t
}

func NewDoc() *Doc {
	d := &Doc{ptr: C.zvec_doc_create()}
	runtime.SetFinalizer(d, func(d *Doc) {
		C.zvec_doc_destroy(d.ptr)
	})
	return d
}

func (d *Doc) SetPK(pk string) {
	cPK := C.CString(pk)
	defer C.free(unsafe.Pointer(cPK))
	C.zvec_doc_set_pk(d.ptr, cPK)
}

func (d *Doc) PK() string {
	return C.GoString(C.zvec_doc_pk(d.ptr))
}

func (d *Doc) SetVector(field string, vector []float32) error {
	cField := C.CString(field)
	defer C.free(unsafe.Pointer(cField))
	status := C.zvec_doc_set_float_vector(d.ptr, cField, (*C.float)(&vector[0]), C.uint32_t(len(vector)))
	return mapError(status)
}

func (d *Doc) SetInt32(field string, value int32) error {
	cField := C.CString(field)
	defer C.free(unsafe.Pointer(cField))
	status := C.zvec_doc_set_int32(d.ptr, cField, C.int32_t(value))
	return mapError(status)
}

type Collection struct {
	ptr *C.zvec_collection_t
}

func CreateAndOpenCollection(path string, schema *Schema) (*Collection, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	var ptr *C.zvec_collection_t
	status := C.zvec_collection_create_and_open(cPath, schema.ptr, &ptr)
	if err := mapError(status); err != nil {
		return nil, err
	}
	c := &Collection{ptr: ptr}
	runtime.SetFinalizer(c, func(c *Collection) {
		C.zvec_collection_destroy(c.ptr)
	})
	return c, nil
}

func (c *Collection) Upsert(docs []*Doc) error {
	ptrs := make([]*C.zvec_doc_t, len(docs))
	for i, d := range docs {
		ptrs[i] = d.ptr
	}
	status := C.zvec_collection_upsert(c.ptr, (**C.zvec_doc_t)(&ptrs[0]), C.size_t(len(docs)))
	return mapError(status)
}

func (c *Collection) Fetch(pks []string) ([]*Doc, error) {
	cPks := make([]*C.char, len(pks))
	for i, s := range pks {
		cPks[i] = C.CString(s)
		defer C.free(unsafe.Pointer(cPks[i]))
	}
	var listPtr *C.zvec_doc_list_t
	status := C.zvec_collection_fetch(c.ptr, (**C.char)(&cPks[0]), C.size_t(len(pks)), &listPtr)
	if err := mapError(status); err != nil {
		return nil, err
	}
	defer C.zvec_doc_list_destroy(listPtr)

	size := int(C.zvec_doc_list_size(listPtr))
	results := make([]*Doc, size)
	for i := 0; i < size; i++ {
		docPtr := C.zvec_doc_list_get(listPtr, C.size_t(i))
		d := &Doc{ptr: docPtr}
		runtime.SetFinalizer(d, func(d *Doc) {
			C.zvec_doc_destroy(d.ptr)
		})
		results[i] = d
	}
	return results, nil
}

func (c *Collection) Flush() error {
	return mapError(C.zvec_collection_flush(c.ptr))
}

func (c *Collection) Stats() (uint64, error) {
	var statsPtr *C.zvec_stats_t
	status := C.zvec_collection_get_stats(c.ptr, &statsPtr)
	if err := mapError(status); err != nil {
		return 0, err
	}
	defer C.zvec_stats_destroy(statsPtr)
	return uint64(C.zvec_stats_total_docs(statsPtr)), nil
}

func (d *Doc) SetString(field string, value string) error {
	cField := C.CString(field)
	defer C.free(unsafe.Pointer(cField))
	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))
	status := C.zvec_doc_set_string(d.ptr, cField, cValue)
	return mapError(status)
}

func OpenCollection(path string) (*Collection, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	var ptr *C.zvec_collection_t
	status := C.zvec_collection_open(cPath, &ptr)
	if err := mapError(status); err != nil {
		return nil, err
	}
	c := &Collection{ptr: ptr}
	runtime.SetFinalizer(c, func(c *Collection) {
		C.zvec_collection_destroy(c.ptr)
	})
	return c, nil
}
