// Package main demonstrates basic usage of the zvec Go SDK.
//
// Before running this example, you need to:
// 1. Build the zvec C-API library (libzvec_c_api)
// 2. Set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS) to include the library path
//
// Example:
//
//	cd /path/to/zvec
//	mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j
//	cd /path/to/zvec/go/examples/basic
//	DYLD_LIBRARY_PATH=/path/to/zvec/build/src/binding/c go run main.go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/alibaba/zvec/go/zvec"
)

func main() {
	fmt.Println("=== ZVec Go SDK Basic Example ===")
	fmt.Println()

	// Print version
	fmt.Printf("ZVec version: %s\n", zvec.GetVersion())
	fmt.Printf("Version: %d.%d.%d\n", zvec.GetVersionMajor(), zvec.GetVersionMinor(), zvec.GetVersionPatch())
	fmt.Println()

	// Initialize the library
	if err := zvec.Initialize(nil); err != nil {
		log.Fatalf("Failed to initialize zvec: %v", err)
	}
	defer func() {
		if err := zvec.Shutdown(); err != nil {
			log.Printf("Warning: failed to shutdown zvec: %v", err)
		}
	}()

	// Create collection schema
	schema := zvec.NewCollectionSchema("test_collection")
	defer schema.Destroy()

	// Create index parameters
	invertParams := zvec.NewInvertIndexParams(true, false)
	defer invertParams.Destroy()

	hnswParams := zvec.NewHNSWIndexParams(zvec.MetricTypeCosine, 16, 200)
	defer hnswParams.Destroy()

	// Add ID field (primary key with inverted index)
	idField := zvec.NewFieldSchema("id", zvec.DataTypeString, false, 0)
	defer idField.Destroy()
	if err := idField.SetIndexParams(invertParams); err != nil {
		log.Fatalf("Failed to set index params for id field: %v", err)
	}
	if err := schema.AddField(idField); err != nil {
		log.Fatalf("Failed to add id field: %v", err)
	}

	// Add text field (with inverted index)
	textField := zvec.NewFieldSchema("text", zvec.DataTypeString, true, 0)
	defer textField.Destroy()
	if err := textField.SetIndexParams(invertParams); err != nil {
		log.Fatalf("Failed to set index params for text field: %v", err)
	}
	if err := schema.AddField(textField); err != nil {
		log.Fatalf("Failed to add text field: %v", err)
	}

	// Add embedding field (HNSW vector index, 3 dimensions)
	embeddingField := zvec.NewFieldSchema("embedding", zvec.DataTypeVectorFP32, false, 3)
	defer embeddingField.Destroy()
	if err := embeddingField.SetIndexParams(hnswParams); err != nil {
		log.Fatalf("Failed to set index params for embedding field: %v", err)
	}
	if err := schema.AddField(embeddingField); err != nil {
		log.Fatalf("Failed to add embedding field: %v", err)
	}

	// Create and open collection
	collectionPath := "./test_go_collection"
	defer os.RemoveAll(collectionPath)

	collection, err := zvec.CreateAndOpen(collectionPath, schema, nil)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	defer collection.Close()
	fmt.Println("✓ Collection created successfully")

	// Prepare documents
	doc1 := zvec.NewDoc()
	defer doc1.Destroy()
	doc1.SetPK("doc1")
	doc1.AddStringField("id", "doc1")
	doc1.AddStringField("text", "First document")
	doc1.AddVectorFP32Field("embedding", []float32{0.1, 0.2, 0.3})

	doc2 := zvec.NewDoc()
	defer doc2.Destroy()
	doc2.SetPK("doc2")
	doc2.AddStringField("id", "doc2")
	doc2.AddStringField("text", "Second document")
	doc2.AddVectorFP32Field("embedding", []float32{0.4, 0.5, 0.6})

	// Insert documents
	result, err := collection.Insert([]*zvec.Doc{doc1, doc2})
	if err != nil {
		log.Fatalf("Failed to insert documents: %v", err)
	}
	fmt.Printf("✓ Documents inserted — Success: %d, Failed: %d\n", result.SuccessCount, result.ErrorCount)

	// Flush collection
	if err := collection.Flush(); err != nil {
		log.Fatalf("Failed to flush collection: %v", err)
	}
	fmt.Println("✓ Collection flushed successfully")

	// Get collection statistics
	stats, err := collection.GetStats()
	if err != nil {
		log.Fatalf("Failed to get stats: %v", err)
	}
	fmt.Printf("✓ Collection stats — Document count: %d\n", stats.DocCount)

	// Query documents
	query := zvec.NewVectorQuery()
	defer query.Destroy()
	query.SetFieldName("embedding")
	query.SetQueryVector([]float32{0.1, 0.2, 0.3})
	query.SetTopK(10)
	query.SetFilter("")
	query.SetIncludeVector(true)
	query.SetIncludeDocID(true)

	results, err := collection.Query(query)
	if err != nil {
		log.Fatalf("Failed to query: %v", err)
	}
	defer zvec.FreeDocs(results)

	fmt.Printf("✓ Query successful — Returned %d results\n", len(results))
	for i, doc := range results {
		pk := doc.GetPK()
		fmt.Printf("  Result %d: PK=%s, DocID=%d, Score=%.4f\n",
			i+1, pk, doc.GetDocID(), doc.GetScore())
	}

	fmt.Println()
	fmt.Println("✓ Example completed successfully!")
}
