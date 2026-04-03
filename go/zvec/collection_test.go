//go:build integration

package zvec

import (
	"testing"
)

// Helper function to create a test schema
func createTestSchema() *CollectionSchema {
	schema := NewCollectionSchema("test_collection")

	invertParams := NewInvertIndexParams(true, false)
	hnswParams := NewHNSWIndexParams(MetricTypeCosine, 16, 200)

	idField := NewFieldSchema("id", DataTypeString, false, 0)
	idField.SetIndexParams(invertParams)
	schema.AddField(idField)

	textField := NewFieldSchema("text", DataTypeString, true, 0)
	schema.AddField(textField)

	embField := NewFieldSchema("embedding", DataTypeVectorFP32, false, 4)
	embField.SetIndexParams(hnswParams)
	schema.AddField(embField)

	return schema
}

// Helper function to create a test document
func createTestDoc(pk string, text string, vector []float32) *Doc {
	doc := NewDoc()
	doc.SetPK(pk)
	doc.AddStringField("id", pk)
	doc.AddStringField("text", text)
	doc.AddVectorFP32Field("embedding", vector)
	return doc
}

func TestCollectionOptions(t *testing.T) {
	options := NewCollectionOptions()
	if options == nil {
		t.Fatal("NewCollectionOptions() returned nil")
	}
	defer options.Destroy()

	// Test SetEnableMmap/GetEnableMmap
	testMmap := true
	if err := options.SetEnableMmap(testMmap); err != nil {
		t.Errorf("SetEnableMmap(%v) failed: %v", testMmap, err)
	}
	if got := options.GetEnableMmap(); got != testMmap {
		t.Errorf("GetEnableMmap() = %v, want %v", got, testMmap)
	}

	testMmap = false
	if err := options.SetEnableMmap(testMmap); err != nil {
		t.Errorf("SetEnableMmap(%v) failed: %v", testMmap, err)
	}
	if got := options.GetEnableMmap(); got != testMmap {
		t.Errorf("GetEnableMmap() = %v, want %v", got, testMmap)
	}

	// Test SetMaxBufferSize/GetMaxBufferSize
	testSize := uint64(1024 * 1024)
	if err := options.SetMaxBufferSize(testSize); err != nil {
		t.Errorf("SetMaxBufferSize(%d) failed: %v", testSize, err)
	}
	if got := options.GetMaxBufferSize(); got != testSize {
		t.Errorf("GetMaxBufferSize() = %d, want %d", got, testSize)
	}

	// Test SetReadOnly/GetReadOnly
	testReadOnly := true
	if err := options.SetReadOnly(testReadOnly); err != nil {
		t.Errorf("SetReadOnly(%v) failed: %v", testReadOnly, err)
	}
	if got := options.GetReadOnly(); got != testReadOnly {
		t.Errorf("GetReadOnly() = %v, want %v", got, testReadOnly)
	}

	testReadOnly = false
	if err := options.SetReadOnly(testReadOnly); err != nil {
		t.Errorf("SetReadOnly(%v) failed: %v", testReadOnly, err)
	}
	if got := options.GetReadOnly(); got != testReadOnly {
		t.Errorf("GetReadOnly() = %v, want %v", got, testReadOnly)
	}
}

func TestCollectionOptionsDestroy(t *testing.T) {
	options := NewCollectionOptions()
	if options == nil {
		t.Fatal("NewCollectionOptions() returned nil")
	}

	// Should not panic
	options.Destroy()

	// Second destroy should also not panic
	options.Destroy()
}

func TestCreateAndOpenCollection(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	if collection == nil {
		t.Fatal("CreateAndOpen() returned nil collection")
	}

	if err := collection.Close(); err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}

func TestCreateAndOpenCollectionWithOptions(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	options := NewCollectionOptions()
	defer options.Destroy()

	if err := options.SetEnableMmap(false); err != nil {
		t.Fatalf("SetEnableMmap() failed: %v", err)
	}

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, options)
	if err != nil {
		t.Fatalf("CreateAndOpen() with options failed: %v", err)
	}
	if collection == nil {
		t.Fatal("CreateAndOpen() with options returned nil collection")
	}

	if err := collection.Close(); err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}

func TestOpenCollection(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	// Create the collection and flush to persist data before closing
	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	if err := collection.Flush(); err != nil {
		t.Fatalf("Flush() failed: %v", err)
	}
	if err := collection.Close(); err != nil {
		t.Fatalf("Close() failed: %v", err)
	}

	// Open the existing collection
	collection, err = Open(path, nil)
	if err != nil {
		t.Fatalf("Open() failed: %v", err)
	}
	if collection == nil {
		t.Fatal("Open() returned nil collection")
	}

	if err := collection.Close(); err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}

func TestCollectionGetSchema(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	retrievedSchema, err := collection.GetSchema()
	if err != nil {
		t.Fatalf("GetSchema() failed: %v", err)
	}
	if retrievedSchema == nil {
		t.Fatal("GetSchema() returned nil")
	}
	defer retrievedSchema.Destroy()

	if name := retrievedSchema.GetName(); name != "test_collection" {
		t.Errorf("GetSchema().GetName() = %s, want test_collection", name)
	}
}

func TestCollectionGetOptions(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	options, err := collection.GetOptions()
	if err != nil {
		t.Fatalf("GetOptions() failed: %v", err)
	}
	if options == nil {
		t.Fatal("GetOptions() returned nil")
	}
	defer options.Destroy()
}

func TestCollectionGetStats(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	stats, err := collection.GetStats()
	if err != nil {
		t.Fatalf("GetStats() failed: %v", err)
	}
	if stats == nil {
		t.Fatal("GetStats() returned nil")
	}

	if stats.DocCount != 0 {
		t.Errorf("GetStats().DocCount = %d, want 0", stats.DocCount)
	}
}

func TestCollectionInsert(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
		createTestDoc("doc2", "foo bar", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(docs)

	result, err := collection.Insert(docs)
	if err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}
	if result == nil {
		t.Fatal("Insert() returned nil result")
	}

	if result.SuccessCount != 2 {
		t.Errorf("Insert().SuccessCount = %d, want 2", result.SuccessCount)
	}
	if result.ErrorCount != 0 {
		t.Errorf("Insert().ErrorCount = %d, want 0", result.ErrorCount)
	}
}

func TestCollectionInsertEmpty(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	result, err := collection.Insert([]*Doc{})
	if err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}
	if result == nil {
		t.Fatal("Insert() returned nil result")
	}

	if result.SuccessCount != 0 {
		t.Errorf("Insert().SuccessCount = %d, want 0", result.SuccessCount)
	}
	if result.ErrorCount != 0 {
		t.Errorf("Insert().ErrorCount = %d, want 0", result.ErrorCount)
	}
}

func TestCollectionInsertAndFlush(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	if err := collection.Flush(); err != nil {
		t.Errorf("Flush() failed: %v", err)
	}
}

func TestCollectionInsertAndGetStats(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
		createTestDoc("doc2", "foo bar", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	stats, err := collection.GetStats()
	if err != nil {
		t.Fatalf("GetStats() failed: %v", err)
	}
	if stats.DocCount != 2 {
		t.Errorf("GetStats().DocCount = %d, want 2", stats.DocCount)
	}
}

func TestCollectionUpdate(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Insert a document first
	doc := createTestDoc("doc1", "original text", []float32{0.1, 0.2, 0.3, 0.4})
	defer doc.Destroy()

	if _, err := collection.Insert([]*Doc{doc}); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	// Update the document
	updateDoc := createTestDoc("doc1", "updated text", []float32{0.9, 0.8, 0.7, 0.6})
	defer updateDoc.Destroy()

	result, err := collection.Update([]*Doc{updateDoc})
	if err != nil {
		t.Fatalf("Update() failed: %v", err)
	}
	if result == nil {
		t.Fatal("Update() returned nil result")
	}

	if result.SuccessCount != 1 {
		t.Errorf("Update().SuccessCount = %d, want 1", result.SuccessCount)
	}
}

func TestCollectionUpsert(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
	}
	defer FreeDocs(docs)

	result, err := collection.Upsert(docs)
	if err != nil {
		t.Fatalf("Upsert() failed: %v", err)
	}
	if result == nil {
		t.Fatal("Upsert() returned nil result")
	}

	if result.SuccessCount != 1 {
		t.Errorf("Upsert().SuccessCount = %d, want 1", result.SuccessCount)
	}

	// Upsert again with same PK (should update)
	updateDocs := []*Doc{
		createTestDoc("doc1", "updated text", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(updateDocs)

	result, err = collection.Upsert(updateDocs)
	if err != nil {
		t.Fatalf("Upsert() update failed: %v", err)
	}

	if result.SuccessCount != 1 {
		t.Errorf("Upsert() update SuccessCount = %d, want 1", result.SuccessCount)
	}
}

func TestCollectionDelete(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Insert documents first
	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
		createTestDoc("doc2", "foo bar", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	// Delete one document
	result, err := collection.Delete([]string{"doc1"})
	if err != nil {
		t.Fatalf("Delete() failed: %v", err)
	}
	if result == nil {
		t.Fatal("Delete() returned nil result")
	}

	if result.SuccessCount != 1 {
		t.Errorf("Delete().SuccessCount = %d, want 1", result.SuccessCount)
	}

	// Verify stats
	stats, err := collection.GetStats()
	if err != nil {
		t.Fatalf("GetStats() failed: %v", err)
	}
	if stats.DocCount != 1 {
		t.Errorf("GetStats().DocCount after delete = %d, want 1", stats.DocCount)
	}
}

func TestCollectionDeleteEmpty(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	result, err := collection.Delete([]string{})
	if err != nil {
		t.Fatalf("Delete() failed: %v", err)
	}
	if result == nil {
		t.Fatal("Delete() returned nil result")
	}

	if result.SuccessCount != 0 {
		t.Errorf("Delete().SuccessCount = %d, want 0", result.SuccessCount)
	}
	if result.ErrorCount != 0 {
		t.Errorf("Delete().ErrorCount = %d, want 0", result.ErrorCount)
	}
}

func TestCollectionDeleteByFilter(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Insert documents first
	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
		createTestDoc("doc2", "foo bar", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	// Delete by filter
	if err := collection.DeleteByFilter("id = 'doc1'"); err != nil {
		t.Fatalf("DeleteByFilter() failed: %v", err)
	}

	// Verify stats
	stats, err := collection.GetStats()
	if err != nil {
		t.Fatalf("GetStats() failed: %v", err)
	}
	if stats.DocCount != 1 {
		t.Errorf("GetStats().DocCount after DeleteByFilter = %d, want 1", stats.DocCount)
	}
}

func TestCollectionQuery(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Insert documents
	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
		createTestDoc("doc2", "foo bar", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	// Create a query
	query := NewVectorQuery()
	defer query.Destroy()

	query.SetFieldName("embedding")
	query.SetQueryVector([]float32{0.1, 0.2, 0.3, 0.4})
	query.SetTopK(10)

	// Execute query
	results, err := collection.Query(query)
	if err != nil {
		t.Fatalf("Query() failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Query() returned no results")
	}
	defer FreeDocs(results)

	if len(results) > 2 {
		t.Errorf("Query() returned %d results, want at most 2", len(results))
	}

	// Verify first result
	firstDoc := results[0]
	if firstDoc == nil {
		t.Fatal("First result is nil")
	}

	pk := firstDoc.GetPK()
	if pk == "" {
		t.Error("First result has empty primary key")
	}
}

func TestCollectionFetch(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Insert documents
	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
		createTestDoc("doc2", "foo bar", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	// Fetch documents
	fetchedDocs, err := collection.Fetch([]string{"doc1", "doc2"})
	if err != nil {
		t.Fatalf("Fetch() failed: %v", err)
	}

	if fetchedDocs == nil {
		t.Fatal("Fetch() returned nil")
	}
	defer FreeDocs(fetchedDocs)

	if len(fetchedDocs) != 2 {
		t.Errorf("Fetch() returned %d documents, want 2", len(fetchedDocs))
	}

	// Verify first document
	doc1 := fetchedDocs[0]
	if doc1 == nil {
		t.Fatal("First fetched document is nil")
	}

	pk := doc1.GetPK()
	if pk != "doc1" && pk != "doc2" {
		t.Errorf("First fetched document has unexpected PK: %s", pk)
	}
}

func TestCollectionFetchEmpty(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	result, err := collection.Fetch([]string{})
	if err != nil {
		t.Fatalf("Fetch() failed: %v", err)
	}

	if result != nil {
		t.Errorf("Fetch() returned %v, want nil", result)
	}
}

func TestCollectionOptimize(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Insert some data
	docs := []*Doc{
		createTestDoc("doc1", "hello world", []float32{0.1, 0.2, 0.3, 0.4}),
		createTestDoc("doc2", "foo bar", []float32{0.5, 0.6, 0.7, 0.8}),
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}

	// Optimize should not error
	if err := collection.Optimize(); err != nil {
		t.Errorf("Optimize() failed: %v", err)
	}
}

func TestFreeDocs(t *testing.T) {
	docs := []*Doc{
		NewDoc(),
		NewDoc(),
		NewDoc(),
	}

	// Should not panic
	FreeDocs(docs)
}

func TestFreeDocsWithNil(t *testing.T) {
	docs := []*Doc{
		NewDoc(),
		nil,
		NewDoc(),
		nil,
	}

	// Should not panic even with nil elements
	FreeDocs(docs)
}

func TestCollectionCreateIndex(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Create index on text field
	indexParams := NewInvertIndexParams(true, false)
	defer indexParams.Destroy()

	if err := collection.CreateIndex("text", indexParams); err != nil {
		t.Errorf("CreateIndex() failed: %v", err)
	}
}

func TestCollectionDropIndex(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Create index first
	indexParams := NewInvertIndexParams(true, false)
	defer indexParams.Destroy()

	if err := collection.CreateIndex("text", indexParams); err != nil {
		t.Fatalf("CreateIndex() failed: %v", err)
	}

	// Drop the index
	if err := collection.DropIndex("text"); err != nil {
		t.Errorf("DropIndex() failed: %v", err)
	}
}

func TestCollectionAddColumn(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Add a new column
	newField := NewFieldSchema("new_field", DataTypeInt32, true, 0)
	defer newField.Destroy()

	if err := collection.AddColumn(newField, ""); err != nil {
		t.Errorf("AddColumn() failed: %v", err)
	}
}

func TestCollectionAddColumnWithDefault(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Add a new column with default value
	newField := NewFieldSchema("score", DataTypeFloat, true, 0)
	defer newField.Destroy()

	if err := collection.AddColumn(newField, "0.5"); err != nil {
		t.Errorf("AddColumn() with default failed: %v", err)
	}
}

func TestCollectionDropColumn(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	// Add a numeric column that can be dropped
	numField := NewFieldSchema("score", DataTypeFloat, true, 0)
	schema.AddField(numField)

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Drop the numeric column (zvec only supports dropping numeric types)
	if err := collection.DropColumn("score"); err != nil {
		t.Errorf("DropColumn() failed: %v", err)
	}
}

func TestCollectionAlterColumn(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	// Add a numeric column that can be altered
	numField := NewFieldSchema("score", DataTypeFloat, true, 0)
	schema.AddField(numField)

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Alter column: rename numeric column (zvec only supports altering numeric types)
	if err := collection.AlterColumn("score", "new_score", nil); err != nil {
		t.Errorf("AlterColumn() rename failed: %v", err)
	}
}

func TestCollectionAlterColumnWithSchema(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	// Add a numeric column that can be altered
	numField := NewFieldSchema("score", DataTypeFloat, true, 0)
	schema.AddField(numField)

	tmpDir := t.TempDir()
	path := tmpDir + "/test_collection"

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer collection.Close()

	// Alter column: change schema of numeric column
	newSchema := NewFieldSchema("score", DataTypeDouble, true, 0)
	defer newSchema.Destroy()

	if err := collection.AlterColumn("score", "", newSchema); err != nil {
		t.Errorf("AlterColumn() with schema failed: %v", err)
	}
}
