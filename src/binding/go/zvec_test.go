// Copyright 2025-present the zvec project
package zvec

import (
	"fmt"
	"os"
	"testing"
)

func testDir(t *testing.T, name string) string {
	dir := fmt.Sprintf("/tmp/zvec_go_test_%s", name)
	os.RemoveAll(dir)
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

func TestSchemaCreate(t *testing.T) {
	schema := NewSchema("test_schema")
	if err := schema.AddField("vec", TypeVectorFp32, 4); err != nil {
		t.Fatalf("AddField vec: %v", err)
	}
	if err := schema.AddField("name", TypeString, 0); err != nil {
		t.Fatalf("AddField name: %v", err)
	}
	if err := schema.AddField("age", TypeInt32, 0); err != nil {
		t.Fatalf("AddField age: %v", err)
	}
}

func TestDocLifecycle(t *testing.T) {
	doc := NewDoc()
	doc.SetPK("pk_001")
	if pk := doc.PK(); pk != "pk_001" {
		t.Fatalf("expected pk_001, got %s", pk)
	}
	if err := doc.SetString("name", "Alice"); err != nil {
		t.Fatalf("SetString: %v", err)
	}
	if err := doc.SetInt32("age", 30); err != nil {
		t.Fatalf("SetInt32: %v", err)
	}
	if err := doc.SetVector("vec", []float32{0.1, 0.2, 0.3, 0.4}); err != nil {
		t.Fatalf("SetVector: %v", err)
	}
}

func TestCollectionCRUD(t *testing.T) {
	dir := testDir(t, "crud")
	schema := NewSchema("crud_col")
	schema.AddField("vec", TypeVectorFp32, 4)
	schema.AddField("name", TypeString, 0)

	col, err := CreateAndOpenCollection(dir, schema)
	if err != nil {
		t.Fatalf("CreateAndOpen: %v", err)
	}

	// Upsert
	doc := NewDoc()
	doc.SetPK("u1")
	doc.SetVector("vec", []float32{1.0, 2.0, 3.0, 4.0})
	doc.SetString("name", "Bob")
	if err := col.Upsert([]*Doc{doc}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := col.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// Stats
	count, err := col.Stats()
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected 1 doc, got %d", count)
	}

	// Fetch
	results, err := col.Fetch([]string{"u1"})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if pk := results[0].PK(); pk != "u1" {
		t.Fatalf("expected pk u1, got %s", pk)
	}
}

func TestInvalidOpen(t *testing.T) {
	_, err := OpenCollection("/tmp/zvec_go_nonexistent_abc123")
	if err == nil {
		t.Fatal("expected error for non-existent path")
	}
}

func TestBatchUpsert(t *testing.T) {
	dir := testDir(t, "batch")
	schema := NewSchema("batch_col")
	schema.AddField("vec", TypeVectorFp32, 4)

	col, err := CreateAndOpenCollection(dir, schema)
	if err != nil {
		t.Fatalf("CreateAndOpen: %v", err)
	}

	docs := make([]*Doc, 100)
	for i := 0; i < 100; i++ {
		d := NewDoc()
		d.SetPK(fmt.Sprintf("batch_%d", i))
		d.SetVector("vec", []float32{float32(i), float32(i), float32(i), float32(i)})
		docs[i] = d
	}

	if err := col.Upsert(docs); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := col.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	count, err := col.Stats()
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if count != 100 {
		t.Fatalf("expected 100 docs, got %d", count)
	}
}

func TestFetchNonexistentPK(t *testing.T) {
	dir := testDir(t, "fetchnone")
	schema := NewSchema("fetchnone_col")
	schema.AddField("vec", TypeVectorFp32, 4)

	col, err := CreateAndOpenCollection(dir, schema)
	if err != nil {
		t.Fatalf("CreateAndOpen: %v", err)
	}

	results, err := col.Fetch([]string{"does_not_exist"})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}
