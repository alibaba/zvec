package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// CollectionOptions represents options for creating or opening a collection.
type CollectionOptions struct {
	handle *C.zvec_collection_options_t
}

// NewCollectionOptions creates a new collection options instance.
func NewCollectionOptions() *CollectionOptions {
	handle := C.zvec_collection_options_create()
	if handle == nil {
		return nil
	}
	return &CollectionOptions{handle: handle}
}

// Destroy releases the collection options resources.
func (o *CollectionOptions) Destroy() {
	if o.handle != nil {
		C.zvec_collection_options_destroy(o.handle)
		o.handle = nil
	}
}

// SetEnableMmap sets whether to enable memory mapping.
func (o *CollectionOptions) SetEnableMmap(enable bool) error {
	return toError(C.zvec_collection_options_set_enable_mmap(o.handle, C.bool(enable)))
}

// GetEnableMmap returns whether memory mapping is enabled.
func (o *CollectionOptions) GetEnableMmap() bool {
	return bool(C.zvec_collection_options_get_enable_mmap(o.handle))
}

// SetMaxBufferSize sets the maximum buffer size in bytes.
func (o *CollectionOptions) SetMaxBufferSize(size uint64) error {
	return toError(C.zvec_collection_options_set_max_buffer_size(o.handle, C.size_t(size)))
}

// GetMaxBufferSize returns the maximum buffer size in bytes.
func (o *CollectionOptions) GetMaxBufferSize() uint64 {
	return uint64(C.zvec_collection_options_get_max_buffer_size(o.handle))
}

// SetReadOnly sets whether the collection is read-only.
func (o *CollectionOptions) SetReadOnly(readOnly bool) error {
	return toError(C.zvec_collection_options_set_read_only(o.handle, C.bool(readOnly)))
}

// GetReadOnly returns whether the collection is read-only.
func (o *CollectionOptions) GetReadOnly() bool {
	return bool(C.zvec_collection_options_get_read_only(o.handle))
}

// CollectionStats holds statistics about a collection.
type CollectionStats struct {
	DocCount          uint64
	IndexCount        int
	IndexNames        []string
	IndexCompleteness []float32
}

// WriteResult holds the result of a write operation (insert/update/upsert/delete).
type WriteResult struct {
	SuccessCount uint64
	ErrorCount   uint64
}

// Collection represents a zvec collection.
type Collection struct {
	handle *C.zvec_collection_t
}

// CreateAndOpen creates a new collection and opens it.
// The caller is responsible for calling Close() when done.
func CreateAndOpen(path string, schema *CollectionSchema, options *CollectionOptions) (*Collection, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cOptions *C.zvec_collection_options_t
	if options != nil {
		cOptions = options.handle
	}

	var cCollection *C.zvec_collection_t
	err := toError(C.zvec_collection_create_and_open(cPath, schema.handle, cOptions, &cCollection))
	if err != nil {
		return nil, err
	}
	return &Collection{handle: cCollection}, nil
}

// Open opens an existing collection.
// The caller is responsible for calling Close() when done.
func Open(path string, options *CollectionOptions) (*Collection, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cOptions *C.zvec_collection_options_t
	if options != nil {
		cOptions = options.handle
	}

	var cCollection *C.zvec_collection_t
	err := toError(C.zvec_collection_open(cPath, cOptions, &cCollection))
	if err != nil {
		return nil, err
	}
	return &Collection{handle: cCollection}, nil
}

// Close closes the collection and releases the handle.
// The collection data on disk is preserved and can be reopened with Open().
func (c *Collection) Close() error {
	if c.handle == nil {
		return nil
	}
	err := toError(C.zvec_collection_close(c.handle))
	c.handle = nil
	return err
}

// Destroy destroys the collection data on disk and releases the handle.
// After calling Destroy, the collection data is permanently deleted.
// Note: zvec_collection_destroy deletes data but does not free the handle;
// zvec_collection_close frees the handle (deletes the shared_ptr).
func (c *Collection) Destroy() error {
	if c.handle == nil {
		return nil
	}
	destroyErr := toError(C.zvec_collection_destroy(c.handle))
	// close frees the handle memory regardless of destroy result
	C.zvec_collection_close(c.handle)
	c.handle = nil
	return destroyErr
}

// Flush flushes collection data to disk.
func (c *Collection) Flush() error {
	return toError(C.zvec_collection_flush(c.handle))
}

// GetSchema returns the collection schema.
// The caller is responsible for calling Destroy() on the returned schema.
func (c *Collection) GetSchema() (*CollectionSchema, error) {
	var cSchema *C.zvec_collection_schema_t
	err := toError(C.zvec_collection_get_schema(c.handle, &cSchema))
	if err != nil {
		return nil, err
	}
	return &CollectionSchema{handle: cSchema}, nil
}

// GetOptions returns the collection options.
// The caller is responsible for calling Destroy() on the returned options.
func (c *Collection) GetOptions() (*CollectionOptions, error) {
	var cOptions *C.zvec_collection_options_t
	err := toError(C.zvec_collection_get_options(c.handle, &cOptions))
	if err != nil {
		return nil, err
	}
	return &CollectionOptions{handle: cOptions}, nil
}

// GetStats returns collection statistics.
func (c *Collection) GetStats() (*CollectionStats, error) {
	var cStats *C.zvec_collection_stats_t
	err := toError(C.zvec_collection_get_stats(c.handle, &cStats))
	if err != nil {
		return nil, err
	}
	defer C.zvec_collection_stats_destroy(cStats)

	indexCount := int(C.zvec_collection_stats_get_index_count(cStats))
	indexNames := make([]string, indexCount)
	indexCompleteness := make([]float32, indexCount)
	for i := 0; i < indexCount; i++ {
		cName := C.zvec_collection_stats_get_index_name(cStats, C.size_t(i))
		if cName != nil {
			indexNames[i] = C.GoString(cName)
		}
		indexCompleteness[i] = float32(C.zvec_collection_stats_get_index_completeness(cStats, C.size_t(i)))
	}

	return &CollectionStats{
		DocCount:          uint64(C.zvec_collection_stats_get_doc_count(cStats)),
		IndexCount:        indexCount,
		IndexNames:        indexNames,
		IndexCompleteness: indexCompleteness,
	}, nil
}

// Optimize optimizes the collection (rebuild indexes, merge segments, etc.).
func (c *Collection) Optimize() error {
	return toError(C.zvec_collection_optimize(c.handle))
}

// CreateIndex creates an index for a collection field.
func (c *Collection) CreateIndex(fieldName string, params *IndexParams) error {
	cFieldName := C.CString(fieldName)
	defer C.free(unsafe.Pointer(cFieldName))
	return toError(C.zvec_collection_create_index(c.handle, cFieldName, params.handle))
}

// DropIndex drops an index from a collection field.
func (c *Collection) DropIndex(fieldName string) error {
	cFieldName := C.CString(fieldName)
	defer C.free(unsafe.Pointer(cFieldName))
	return toError(C.zvec_collection_drop_index(c.handle, cFieldName))
}

// AddColumn adds a new column to the collection.
func (c *Collection) AddColumn(fieldSchema *FieldSchema, defaultExpr string) error {
	var cExpr *C.char
	if defaultExpr != "" {
		cExpr = C.CString(defaultExpr)
		defer C.free(unsafe.Pointer(cExpr))
	}
	return toError(C.zvec_collection_add_column(c.handle, fieldSchema.handle, cExpr))
}

// DropColumn drops a column from the collection.
func (c *Collection) DropColumn(columnName string) error {
	cColumnName := C.CString(columnName)
	defer C.free(unsafe.Pointer(cColumnName))
	return toError(C.zvec_collection_drop_column(c.handle, cColumnName))
}

// AlterColumn alters a column in the collection.
// Pass empty string for newName to skip renaming.
// Pass nil for newSchema to skip schema modification.
func (c *Collection) AlterColumn(columnName, newName string, newSchema *FieldSchema) error {
	cColumnName := C.CString(columnName)
	defer C.free(unsafe.Pointer(cColumnName))

	var cNewName *C.char
	if newName != "" {
		cNewName = C.CString(newName)
		defer C.free(unsafe.Pointer(cNewName))
	}

	var cNewSchema *C.zvec_field_schema_t
	if newSchema != nil {
		cNewSchema = newSchema.handle
	}

	return toError(C.zvec_collection_alter_column(c.handle, cColumnName, cNewName, cNewSchema))
}

// Insert inserts documents into the collection.
func (c *Collection) Insert(docs []*Doc) (*WriteResult, error) {
	if len(docs) == 0 {
		return &WriteResult{}, nil
	}
	cDocs := make([]*C.zvec_doc_t, len(docs))
	for i, doc := range docs {
		cDocs[i] = doc.handle
	}
	var successCount, errorCount C.size_t
	err := toError(C.zvec_collection_insert(
		c.handle,
		(**C.zvec_doc_t)(unsafe.Pointer(&cDocs[0])),
		C.size_t(len(docs)),
		&successCount,
		&errorCount,
	))
	if err != nil {
		return nil, err
	}
	return &WriteResult{
		SuccessCount: uint64(successCount),
		ErrorCount:   uint64(errorCount),
	}, nil
}

// Update updates documents in the collection.
func (c *Collection) Update(docs []*Doc) (*WriteResult, error) {
	if len(docs) == 0 {
		return &WriteResult{}, nil
	}
	cDocs := make([]*C.zvec_doc_t, len(docs))
	for i, doc := range docs {
		cDocs[i] = doc.handle
	}
	var successCount, errorCount C.size_t
	err := toError(C.zvec_collection_update(
		c.handle,
		(**C.zvec_doc_t)(unsafe.Pointer(&cDocs[0])),
		C.size_t(len(docs)),
		&successCount,
		&errorCount,
	))
	if err != nil {
		return nil, err
	}
	return &WriteResult{
		SuccessCount: uint64(successCount),
		ErrorCount:   uint64(errorCount),
	}, nil
}

// Upsert inserts or updates documents in the collection.
func (c *Collection) Upsert(docs []*Doc) (*WriteResult, error) {
	if len(docs) == 0 {
		return &WriteResult{}, nil
	}
	cDocs := make([]*C.zvec_doc_t, len(docs))
	for i, doc := range docs {
		cDocs[i] = doc.handle
	}
	var successCount, errorCount C.size_t
	err := toError(C.zvec_collection_upsert(
		c.handle,
		(**C.zvec_doc_t)(unsafe.Pointer(&cDocs[0])),
		C.size_t(len(docs)),
		&successCount,
		&errorCount,
	))
	if err != nil {
		return nil, err
	}
	return &WriteResult{
		SuccessCount: uint64(successCount),
		ErrorCount:   uint64(errorCount),
	}, nil
}

// Delete deletes documents by primary keys.
func (c *Collection) Delete(pks []string) (*WriteResult, error) {
	if len(pks) == 0 {
		return &WriteResult{}, nil
	}
	cPKs := make([]*C.char, len(pks))
	for i, pk := range pks {
		cPKs[i] = C.CString(pk)
	}
	defer func() {
		for _, cPK := range cPKs {
			C.free(unsafe.Pointer(cPK))
		}
	}()

	var successCount, errorCount C.size_t
	err := toError(C.zvec_collection_delete(
		c.handle,
		(**C.char)(unsafe.Pointer(&cPKs[0])),
		C.size_t(len(pks)),
		&successCount,
		&errorCount,
	))
	if err != nil {
		return nil, err
	}
	return &WriteResult{
		SuccessCount: uint64(successCount),
		ErrorCount:   uint64(errorCount),
	}, nil
}

// DeleteByFilter deletes documents matching the filter expression.
func (c *Collection) DeleteByFilter(filter string) error {
	cFilter := C.CString(filter)
	defer C.free(unsafe.Pointer(cFilter))
	return toError(C.zvec_collection_delete_by_filter(c.handle, cFilter))
}

// Query performs a vector similarity search.
// The caller is responsible for calling Destroy() on each returned Doc,
// or using FreeDocs() to free all at once.
func (c *Collection) Query(query *VectorQuery) ([]*Doc, error) {
	var cResults **C.zvec_doc_t
	var resultCount C.size_t
	err := toError(C.zvec_collection_query(c.handle, query.handle, &cResults, &resultCount))
	if err != nil {
		return nil, err
	}

	count := int(resultCount)
	if count == 0 {
		return nil, nil
	}

	resultSlice := unsafe.Slice(cResults, count)
	docs := make([]*Doc, count)
	for i := 0; i < count; i++ {
		docs[i] = &Doc{handle: resultSlice[i], owned: true}
	}
	return docs, nil
}

// Fetch retrieves documents by primary keys.
// The caller is responsible for calling Destroy() on each returned Doc,
// or using FreeDocs() to free all at once.
func (c *Collection) Fetch(primaryKeys []string) ([]*Doc, error) {
	if len(primaryKeys) == 0 {
		return nil, nil
	}
	cPKs := make([]*C.char, len(primaryKeys))
	for i, pk := range primaryKeys {
		cPKs[i] = C.CString(pk)
	}
	defer func() {
		for _, cPK := range cPKs {
			C.free(unsafe.Pointer(cPK))
		}
	}()

	var cDocs **C.zvec_doc_t
	var foundCount C.size_t
	err := toError(C.zvec_collection_fetch(
		c.handle,
		(**C.char)(unsafe.Pointer(&cPKs[0])),
		C.size_t(len(primaryKeys)),
		&cDocs,
		&foundCount,
	))
	if err != nil {
		return nil, err
	}

	count := int(foundCount)
	if count == 0 {
		return nil, nil
	}

	resultSlice := unsafe.Slice(cDocs, count)
	docs := make([]*Doc, count)
	for i := 0; i < count; i++ {
		docs[i] = &Doc{handle: resultSlice[i], owned: true}
	}
	return docs, nil
}

// FreeDocs is a convenience function to destroy multiple documents at once.
func FreeDocs(docs []*Doc) {
	for _, doc := range docs {
		if doc != nil {
			doc.Destroy()
		}
	}
}
