//go:build integration

package zvec

import (
	"testing"
)

// IndexParams Tests

func TestNewIndexParamsHNSW(t *testing.T) {
	params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
	if params == nil {
		t.Fatal("NewHNSWIndexParams() returned nil")
	}
	defer params.Destroy()

	if got := params.GetType(); got != IndexTypeHNSW {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeHNSW)
	}

	if got := params.GetMetricType(); got != MetricTypeCosine {
		t.Errorf("GetMetricType() = %v, want %v", got, MetricTypeCosine)
	}

	if got := params.GetHNSWM(); got != 16 {
		t.Errorf("GetHNSWM() = %d, want %d", got, 16)
	}

	if got := params.GetHNSWEfConstruction(); got != 200 {
		t.Errorf("GetHNSWEfConstruction() = %d, want %d", got, 200)
	}
}

func TestNewIndexParamsInvert(t *testing.T) {
	params := NewInvertIndexParams(true, false)
	if params == nil {
		t.Fatal("NewInvertIndexParams() returned nil")
	}
	defer params.Destroy()

	if got := params.GetType(); got != IndexTypeInvert {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeInvert)
	}
}

func TestNewIndexParamsIVF(t *testing.T) {
	params := NewIVFIndexParams(MetricTypeL2, 100, 10, false)
	if params == nil {
		t.Fatal("NewIVFIndexParams() returned nil")
	}
	defer params.Destroy()

	if got := params.GetType(); got != IndexTypeIVF {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeIVF)
	}

	if got := params.GetMetricType(); got != MetricTypeL2 {
		t.Errorf("GetMetricType() = %v, want %v", got, MetricTypeL2)
	}
}

func TestNewIndexParamsFlat(t *testing.T) {
	params := NewFlatIndexParams(MetricTypeIP)
	if params == nil {
		t.Fatal("NewFlatIndexParams() returned nil")
	}
	defer params.Destroy()

	if got := params.GetType(); got != IndexTypeFlat {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeFlat)
	}

	if got := params.GetMetricType(); got != MetricTypeIP {
		t.Errorf("GetMetricType() = %v, want %v", got, MetricTypeIP)
	}
}

func TestIndexParamsSetQuantizeType(t *testing.T) {
	params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
	if params == nil {
		t.Fatal("NewHNSWIndexParams() returned nil")
	}
	defer params.Destroy()

	// Test round-trip for all quantize types
	testTypes := []QuantizeType{
		QuantizeTypeFP16,
		QuantizeTypeInt8,
		QuantizeTypeInt4,
	}

	for _, testType := range testTypes {
		if err := params.SetQuantizeType(testType); err != nil {
			t.Errorf("SetQuantizeType(%v) failed: %v", testType, err)
			continue
		}

		if got := params.GetQuantizeType(); got != testType {
			t.Errorf("GetQuantizeType() = %v, want %v", got, testType)
		}
	}
}

func TestIndexParamsDestroy(t *testing.T) {
	params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
	if params == nil {
		t.Fatal("NewHNSWIndexParams() returned nil")
	}

	// First Destroy should not panic
	params.Destroy()

	// Second Destroy should also not panic
	params.Destroy()
}

func TestIndexParamsSetMetricType(t *testing.T) {
	params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
	if params == nil {
		t.Fatal("NewHNSWIndexParams() returned nil")
	}
	defer params.Destroy()

	// Test round-trip for all metric types
	testTypes := []MetricType{
		MetricTypeL2,
		MetricTypeIP,
		MetricTypeCosine,
		MetricTypeMIPSL2,
	}

	for _, testType := range testTypes {
		if err := params.SetMetricType(testType); err != nil {
			t.Errorf("SetMetricType(%v) failed: %v", testType, err)
			continue
		}

		if got := params.GetMetricType(); got != testType {
			t.Errorf("GetMetricType() = %v, want %v", got, testType)
		}
	}
}

// FieldSchema Tests

func TestNewFieldSchema(t *testing.T) {
	field := NewFieldSchema("test_field", DataTypeString, false, 0)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	if got := field.GetName(); got != "test_field" {
		t.Errorf("GetName() = %s, want %s", got, "test_field")
	}

	if got := field.GetDataType(); got != DataTypeString {
		t.Errorf("GetDataType() = %v, want %v", got, DataTypeString)
	}

	if got := field.IsNullable(); got != false {
		t.Errorf("IsNullable() = %v, want %v", got, false)
	}

	if got := field.GetDimension(); got != uint32(0) {
		t.Errorf("GetDimension() = %d, want %d", got, 0)
	}
}

func TestFieldSchemaSetters(t *testing.T) {
	field := NewFieldSchema("original_name", DataTypeInt32, true, 0)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	// Test SetName round-trip
	newName := "updated_name"
	if err := field.SetName(newName); err != nil {
		t.Errorf("SetName(%s) failed: %v", newName, err)
	}
	if got := field.GetName(); got != newName {
		t.Errorf("GetName() = %s, want %s", got, newName)
	}

	// Test SetDataType round-trip
	newType := DataTypeInt64
	if err := field.SetDataType(newType); err != nil {
		t.Errorf("SetDataType(%v) failed: %v", newType, err)
	}
	if got := field.GetDataType(); got != newType {
		t.Errorf("GetDataType() = %v, want %v", got, newType)
	}

	// Test SetNullable round-trip
	newNullable := false
	if err := field.SetNullable(newNullable); err != nil {
		t.Errorf("SetNullable(%v) failed: %v", newNullable, err)
	}
	if got := field.IsNullable(); got != newNullable {
		t.Errorf("IsNullable() = %v, want %v", got, newNullable)
	}

	// Test SetDimension round-trip
	newDimension := uint32(128)
	if err := field.SetDimension(newDimension); err != nil {
		t.Errorf("SetDimension(%d) failed: %v", newDimension, err)
	}
	if got := field.GetDimension(); got != newDimension {
		t.Errorf("GetDimension() = %d, want %d", got, newDimension)
	}
}

func TestFieldSchemaVectorDetection(t *testing.T) {
	field := NewFieldSchema("embedding", DataTypeVectorFP32, false, 128)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	if got := field.IsVectorField(); got != true {
		t.Errorf("IsVectorField() = %v, want %v", got, true)
	}

	if got := field.IsDenseVector(); got != true {
		t.Errorf("IsDenseVector() = %v, want %v", got, true)
	}

	if got := field.IsSparseVector(); got != false {
		t.Errorf("IsSparseVector() = %v, want %v", got, false)
	}
}

func TestFieldSchemaSparseVectorDetection(t *testing.T) {
	field := NewFieldSchema("sparse_embedding", DataTypeSparseVectorFP32, false, 0)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	if got := field.IsVectorField(); got != true {
		t.Errorf("IsVectorField() = %v, want %v", got, true)
	}

	if got := field.IsDenseVector(); got != false {
		t.Errorf("IsDenseVector() = %v, want %v", got, false)
	}

	if got := field.IsSparseVector(); got != true {
		t.Errorf("IsSparseVector() = %v, want %v", got, true)
	}
}

func TestFieldSchemaIndex(t *testing.T) {
	field := NewFieldSchema("indexed_field", DataTypeVectorFP32, false, 128)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	// Initially should not have index
	if got := field.HasIndex(); got != false {
		t.Errorf("HasIndex() = %v (before SetIndexParams), want %v", got, false)
	}

	// Set index params
	params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
	if params == nil {
		t.Fatal("NewHNSWIndexParams() returned nil")
	}
	defer params.Destroy()

	if err := field.SetIndexParams(params); err != nil {
		t.Errorf("SetIndexParams() failed: %v", err)
	}

	// Now should have index
	if got := field.HasIndex(); got != true {
		t.Errorf("HasIndex() = %v (after SetIndexParams), want %v", got, true)
	}

	if got := field.GetIndexType(); got != IndexTypeHNSW {
		t.Errorf("GetIndexType() = %v, want %v", got, IndexTypeHNSW)
	}
}

func TestFieldSchemaDestroy(t *testing.T) {
	field := NewFieldSchema("test_field", DataTypeString, false, 0)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}

	// First Destroy should not panic
	field.Destroy()

	// Second Destroy should also not panic
	field.Destroy()
}

// CollectionSchema Tests

func TestNewCollectionSchema(t *testing.T) {
	schema := NewCollectionSchema("test_collection")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	if got := schema.GetName(); got != "test_collection" {
		t.Errorf("GetName() = %s, want %s", got, "test_collection")
	}
}

func TestCollectionSchemaSetName(t *testing.T) {
	schema := NewCollectionSchema("original_name")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	newName := "updated_name"
	if err := schema.SetName(newName); err != nil {
		t.Errorf("SetName(%s) failed: %v", newName, err)
	}

	if got := schema.GetName(); got != newName {
		t.Errorf("GetName() = %s, want %s", got, newName)
	}
}

func TestCollectionSchemaAddField(t *testing.T) {
	schema := NewCollectionSchema("test_collection")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	field := NewFieldSchema("test_field", DataTypeString, false, 0)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	if err := schema.AddField(field); err != nil {
		t.Errorf("AddField() failed: %v", err)
	}

	if got := schema.HasField("test_field"); got != true {
		t.Errorf("HasField() = %v, want %v", got, true)
	}

	retrievedField := schema.GetField("test_field")
	if retrievedField == nil {
		t.Error("GetField() returned nil")
	} else {
		if got := retrievedField.GetName(); got != "test_field" {
			t.Errorf("Retrieved field name = %s, want %s", got, "test_field")
		}
	}
}

func TestCollectionSchemaDropField(t *testing.T) {
	schema := NewCollectionSchema("test_collection")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	field := NewFieldSchema("test_field", DataTypeString, false, 0)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	if err := schema.AddField(field); err != nil {
		t.Errorf("AddField() failed: %v", err)
	}

	if got := schema.HasField("test_field"); got != true {
		t.Errorf("HasField() (before DropField) = %v, want %v", got, true)
	}

	if err := schema.DropField("test_field"); err != nil {
		t.Errorf("DropField() failed: %v", err)
	}

	if got := schema.HasField("test_field"); got != false {
		t.Errorf("HasField() (after DropField) = %v, want %v", got, false)
	}
}

func TestCollectionSchemaIndex(t *testing.T) {
	schema := NewCollectionSchema("test_collection")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	field := NewFieldSchema("indexed_field", DataTypeVectorFP32, false, 128)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	if err := schema.AddField(field); err != nil {
		t.Errorf("AddField() failed: %v", err)
	}

	// Initially should not have index
	if got := schema.HasIndex("indexed_field"); got != false {
		t.Errorf("HasIndex() (before AddIndex) = %v, want %v", got, false)
	}

	// Add index
	params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
	if params == nil {
		t.Fatal("NewHNSWIndexParams() returned nil")
	}
	defer params.Destroy()

	if err := schema.AddIndex("indexed_field", params); err != nil {
		t.Errorf("AddIndex() failed: %v", err)
	}

	// Now should have index
	if got := schema.HasIndex("indexed_field"); got != true {
		t.Errorf("HasIndex() (after AddIndex) = %v, want %v", got, true)
	}

	// Drop index
	if err := schema.DropIndex("indexed_field"); err != nil {
		t.Errorf("DropIndex() failed: %v", err)
	}

	// Should not have index anymore
	if got := schema.HasIndex("indexed_field"); got != false {
		t.Errorf("HasIndex() (after DropIndex) = %v, want %v", got, false)
	}
}

func TestCollectionSchemaMaxDocCount(t *testing.T) {
	schema := NewCollectionSchema("test_collection")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	testCount := uint64(1000)
	if err := schema.SetMaxDocCountPerSegment(testCount); err != nil {
		t.Errorf("SetMaxDocCountPerSegment(%d) failed: %v", testCount, err)
	}

	if got := schema.GetMaxDocCountPerSegment(); got != testCount {
		t.Errorf("GetMaxDocCountPerSegment() = %d, want %d", got, testCount)
	}
}

func TestCollectionSchemaGetFieldOwnership(t *testing.T) {
	schema := NewCollectionSchema("test_collection")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	field := NewFieldSchema("test_field", DataTypeString, false, 0)
	if field == nil {
		t.Fatal("NewFieldSchema() returned nil")
	}
	defer field.Destroy()

	if err := schema.AddField(field); err != nil {
		t.Errorf("AddField() failed: %v", err)
	}

	retrievedField := schema.GetField("test_field")
	if retrievedField == nil {
		t.Fatal("GetField() returned nil")
	}

	// Verify that retrieved field is not owned (owned == false)
	if retrievedField.owned != false {
		t.Errorf("GetField() returned field with owned=%v, want %v", retrievedField.owned, false)
	}

	// Calling Destroy on non-owned field should not panic
	retrievedField.Destroy()
	retrievedField.Destroy()
}
