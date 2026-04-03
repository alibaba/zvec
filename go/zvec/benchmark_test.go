//go:build integration

package zvec

import (
	"fmt"
	"testing"
)

// Helper function to create a benchmark schema
func benchmarkCreateSchema(dimension uint32) *CollectionSchema {
	schema := NewCollectionSchema("bench_collection")

	invertParams := NewInvertIndexParams(true, false)
	hnswParams := NewHNSWIndexParams(MetricTypeCosine, 16, 200)

	idField := NewFieldSchema("id", DataTypeString, false, 0)
	idField.SetIndexParams(invertParams)
	schema.AddField(idField)

	embField := NewFieldSchema("embedding", DataTypeVectorFP32, false, dimension)
	embField.SetIndexParams(hnswParams)
	schema.AddField(embField)

	return schema
}

// Helper function to generate a random vector
func generateRandomVector(dimension int) []float32 {
	vec := make([]float32, dimension)
	for i := range vec {
		vec[i] = float32(i) * 0.01
	}
	return vec
}

// Schema-related benchmarks

func BenchmarkNewCollectionSchema(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		schema := NewCollectionSchema("bench_collection")
		schema.Destroy()
	}
}

func BenchmarkNewFieldSchema(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		field := NewFieldSchema("test_field", DataTypeString, false, 0)
		field.Destroy()
	}
}

func BenchmarkNewHNSWIndexParams(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
		params.Destroy()
	}
}

// Doc-related benchmarks

func BenchmarkDocCreateDestroy(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		doc := NewDoc()
		doc.Destroy()
	}
}

func BenchmarkDocSetPK(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pk := fmt.Sprintf("doc_%d", i)
		doc.SetPK(pk)
	}
}

func BenchmarkDocAddStringField(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fieldName := fmt.Sprintf("field_%d", i%10)
		doc.AddStringField(fieldName, "test value")
	}
}

func BenchmarkDocAddVectorFP32Field_4D(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	vector := generateRandomVector(4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fieldName := fmt.Sprintf("vector_%d", i%10)
		doc.AddVectorFP32Field(fieldName, vector)
	}
}

func BenchmarkDocAddVectorFP32Field_128D(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	vector := generateRandomVector(128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fieldName := fmt.Sprintf("vector_%d", i%10)
		doc.AddVectorFP32Field(fieldName, vector)
	}
}

func BenchmarkDocAddVectorFP32Field_768D(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	vector := generateRandomVector(768)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fieldName := fmt.Sprintf("vector_%d", i%10)
		doc.AddVectorFP32Field(fieldName, vector)
	}
}

func BenchmarkDocGetStringField(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	// Pre-populate with fields
	for i := 0; i < 100; i++ {
		fieldName := fmt.Sprintf("field_%d", i%10)
		doc.AddStringField(fieldName, "test value")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fieldName := fmt.Sprintf("field_%d", i%10)
		doc.GetStringField(fieldName)
	}
}

func BenchmarkDocGetVectorFP32Field_128D(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	// Pre-populate with vector field
	vector := generateRandomVector(128)
	doc.AddVectorFP32Field("embedding", vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc.GetVectorFP32Field("embedding")
	}
}

// Query-related benchmarks

func BenchmarkNewVectorQuery(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		query := NewVectorQuery()
		query.Destroy()
	}
}

func BenchmarkVectorQuerySetup(b *testing.B) {
	b.ReportAllocs()
	query := NewVectorQuery()
	defer query.Destroy()

	queryVector := generateRandomVector(128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query.SetFieldName("embedding")
		query.SetTopK(10)
		query.SetQueryVector(queryVector)
		query.SetFilter("id > 0")
	}
}

// Collection-related benchmarks

func BenchmarkCollectionInsert(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer collection.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc := NewDoc()
		pk := fmt.Sprintf("doc_%d", i)
		doc.SetPK(pk)
		doc.AddStringField("id", pk)
		vector := generateRandomVector(128)
		doc.AddVectorFP32Field("embedding", vector)

		_, err := collection.Insert([]*Doc{doc})
		if err != nil {
			b.Fatalf("Failed to insert document: %v", err)
		}
		doc.Destroy()
	}
}

func BenchmarkCollectionInsertBatch10(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer collection.Close()

	batchSize := 10

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		docs := make([]*Doc, batchSize)
		for j := 0; j < batchSize; j++ {
			doc := NewDoc()
			pk := fmt.Sprintf("doc_%d_%d", i, j)
			doc.SetPK(pk)
			doc.AddStringField("id", pk)
			vector := generateRandomVector(128)
			doc.AddVectorFP32Field("embedding", vector)
			docs[j] = doc
		}

		_, err := collection.Insert(docs)
		if err != nil {
			b.Fatalf("Failed to insert documents: %v", err)
		}

		for _, doc := range docs {
			doc.Destroy()
		}
	}
}

func BenchmarkCollectionInsertBatch100(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer collection.Close()

	batchSize := 100

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		docs := make([]*Doc, batchSize)
		for j := 0; j < batchSize; j++ {
			doc := NewDoc()
			pk := fmt.Sprintf("doc_%d_%d", i, j)
			doc.SetPK(pk)
			doc.AddStringField("id", pk)
			vector := generateRandomVector(128)
			doc.AddVectorFP32Field("embedding", vector)
			docs[j] = doc
		}

		_, err := collection.Insert(docs)
		if err != nil {
			b.Fatalf("Failed to insert documents: %v", err)
		}

		for _, doc := range docs {
			doc.Destroy()
		}
	}
}

func BenchmarkCollectionQuery(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer collection.Close()

	// Pre-insert 1000 documents
	b.StopTimer()
	for i := 0; i < 1000; i++ {
		doc := NewDoc()
		pk := fmt.Sprintf("doc_%d", i)
		doc.SetPK(pk)
		doc.AddStringField("id", pk)
		vector := generateRandomVector(128)
		doc.AddVectorFP32Field("embedding", vector)

		_, err := collection.Insert([]*Doc{doc})
		if err != nil {
			b.Fatalf("Failed to insert document: %v", err)
		}
		doc.Destroy()
	}

	if err := collection.Flush(); err != nil {
		b.Fatalf("Failed to flush collection: %v", err)
	}
	b.StartTimer()

	// Benchmark query
	query := NewVectorQuery()
	defer query.Destroy()
	query.SetFieldName("embedding")
	query.SetTopK(10)
	queryVector := generateRandomVector(128)
	query.SetQueryVector(queryVector)

	for i := 0; i < b.N; i++ {
		_, err := collection.Query(query)
		if err != nil {
			b.Fatalf("Failed to query collection: %v", err)
		}
	}
}

func BenchmarkCollectionFetch(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer collection.Close()

	// Pre-insert 100 documents
	b.StopTimer()
	for i := 0; i < 100; i++ {
		doc := NewDoc()
		pk := fmt.Sprintf("doc_%d", i)
		doc.SetPK(pk)
		doc.AddStringField("id", pk)
		vector := generateRandomVector(128)
		doc.AddVectorFP32Field("embedding", vector)

		_, err := collection.Insert([]*Doc{doc})
		if err != nil {
			b.Fatalf("Failed to insert document: %v", err)
		}
		doc.Destroy()
	}

	if err := collection.Flush(); err != nil {
		b.Fatalf("Failed to flush collection: %v", err)
	}
	b.StartTimer()

	// Benchmark fetch
	for i := 0; i < b.N; i++ {
		pk := fmt.Sprintf("doc_%d", i%100)
		_, err := collection.Fetch([]string{pk})
		if err != nil {
			b.Fatalf("Failed to fetch document: %v", err)
		}
	}
}
