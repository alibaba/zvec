// Copyright 2025-present the zvec project
package main

import (
	"fmt"
	"zvec"
)

func main() {
	// 1. Create Schema
	schema := zvec.NewSchema("test_collection")
	schema.AddField("vector", zvec.TypeVectorFp32, 128)
	schema.AddField("age", zvec.TypeInt32, 0)

	// 2. Create and Open Collection
	collection, err := zvec.CreateAndOpenCollection("/tmp/zvec_go_example", schema)
	if err != nil {
		panic(err)
	}

	// 3. Upsert Documents
	doc1 := zvec.NewDoc()
	doc1.SetPK("user_1")
	doc1.SetVector("vector", make([]float32, 128))
	doc1.SetInt32("age", 30)

	err = collection.Upsert([]*zvec.Doc{doc1})
	if err != nil {
		panic(err)
	}
	collection.Flush()

	// 4. Get Stats
	count, _ := collection.Stats()
	fmt.Printf("Total docs: %d\n", count)

	// 5. Fetch Document
	docs, err := collection.Fetch([]string{"user_1"})
	if err != nil {
		panic(err)
	}
	for _, d := range docs {
		fmt.Printf("Fetched PK: %s\n", d.PK())
	}
}
