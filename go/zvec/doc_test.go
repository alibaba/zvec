//go:build integration

package zvec

import (
	"math"
	"testing"
)

// TestNewDoc tests creating a new document.
func TestNewDoc(t *testing.T) {
	doc := NewDoc()
	if doc == nil {
		t.Fatal("NewDoc() returned nil")
	}
	defer doc.Destroy()

	if !doc.IsEmpty() {
		t.Error("Expected empty document")
	}

	if doc.GetFieldCount() != 0 {
		t.Errorf("Expected 0 fields, got %d", doc.GetFieldCount())
	}
}

// TestDocPK tests SetPK and GetPK round-trip.
func TestDocPK(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	pk := "test-primary-key"
	doc.SetPK(pk)

	if got := doc.GetPK(); got != pk {
		t.Errorf("GetPK() = %q, want %q", got, pk)
	}
}

// TestDocDocID tests SetDocID and GetDocID round-trip.
func TestDocDocID(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	docID := uint64(123456789)
	doc.SetDocID(docID)

	if got := doc.GetDocID(); got != docID {
		t.Errorf("GetDocID() = %d, want %d", got, docID)
	}
}

// TestDocScore tests SetScore and GetScore round-trip.
func TestDocScore(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	score := float32(0.95)
	doc.SetScore(score)

	got := doc.GetScore()
	if math.Abs(float64(got-score)) > 1e-6 {
		t.Errorf("GetScore() = %f, want %f", got, score)
	}
}

// TestDocOperator tests SetOperator and GetOperator round-trip.
func TestDocOperator(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	op := DocOpUpsert
	doc.SetOperator(op)

	if got := doc.GetOperator(); got != op {
		t.Errorf("GetOperator() = %v, want %v", got, op)
	}
}

// TestDocStringField tests AddStringField and GetStringField round-trip.
func TestDocStringField(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value string
	}{
		{"empty", ""},
		{"chinese", "你好世界"},
		{"normal", "test string value"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "str_field_" + tc.name
			if err := doc.AddStringField(fieldName, tc.value); err != nil {
				t.Fatalf("AddStringField failed: %v", err)
			}

			got, err := doc.GetStringField(fieldName)
			if err != nil {
				t.Fatalf("GetStringField failed: %v", err)
			}

			if got != tc.value {
				t.Errorf("GetStringField() = %q, want %q", got, tc.value)
			}
		})
	}
}

// TestDocBoolField tests AddBoolField and GetBoolField round-trip.
func TestDocBoolField(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value bool
	}{
		{"true", true},
		{"false", false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "bool_field_" + tc.name
			if err := doc.AddBoolField(fieldName, tc.value); err != nil {
				t.Fatalf("AddBoolField failed: %v", err)
			}

			got, err := doc.GetBoolField(fieldName)
			if err != nil {
				t.Fatalf("GetBoolField failed: %v", err)
			}

			if got != tc.value {
				t.Errorf("GetBoolField() = %v, want %v", got, tc.value)
			}
		})
	}
}

// TestDocInt32Field tests AddInt32Field and GetInt32Field round-trip.
func TestDocInt32Field(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value int32
	}{
		{"positive", 42},
		{"negative", -42},
		{"zero", 0},
		{"max", math.MaxInt32},
		{"min", math.MinInt32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "int32_field_" + tc.name
			if err := doc.AddInt32Field(fieldName, tc.value); err != nil {
				t.Fatalf("AddInt32Field failed: %v", err)
			}

			got, err := doc.GetInt32Field(fieldName)
			if err != nil {
				t.Fatalf("GetInt32Field failed: %v", err)
			}

			if got != tc.value {
				t.Errorf("GetInt32Field() = %d, want %d", got, tc.value)
			}
		})
	}
}

// TestDocInt64Field tests AddInt64Field and GetInt64Field round-trip.
func TestDocInt64Field(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value int64
	}{
		{"positive", 1234567890},
		{"negative", -1234567890},
		{"zero", 0},
		{"max", math.MaxInt64},
		{"min", math.MinInt64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "int64_field_" + tc.name
			if err := doc.AddInt64Field(fieldName, tc.value); err != nil {
				t.Fatalf("AddInt64Field failed: %v", err)
			}

			got, err := doc.GetInt64Field(fieldName)
			if err != nil {
				t.Fatalf("GetInt64Field failed: %v", err)
			}

			if got != tc.value {
				t.Errorf("GetInt64Field() = %d, want %d", got, tc.value)
			}
		})
	}
}

// TestDocUint32Field tests AddUint32Field and GetUint32Field round-trip.
func TestDocUint32Field(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value uint32
	}{
		{"positive", 42},
		{"zero", 0},
		{"max", math.MaxUint32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "uint32_field_" + tc.name
			if err := doc.AddUint32Field(fieldName, tc.value); err != nil {
				t.Fatalf("AddUint32Field failed: %v", err)
			}

			got, err := doc.GetUint32Field(fieldName)
			if err != nil {
				t.Fatalf("GetUint32Field failed: %v", err)
			}

			if got != tc.value {
				t.Errorf("GetUint32Field() = %d, want %d", got, tc.value)
			}
		})
	}
}

// TestDocUint64Field tests AddUint64Field and GetUint64Field round-trip.
func TestDocUint64Field(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value uint64
	}{
		{"positive", 1234567890},
		{"zero", 0},
		{"max", math.MaxUint64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "uint64_field_" + tc.name
			if err := doc.AddUint64Field(fieldName, tc.value); err != nil {
				t.Fatalf("AddUint64Field failed: %v", err)
			}

			got, err := doc.GetUint64Field(fieldName)
			if err != nil {
				t.Fatalf("GetUint64Field failed: %v", err)
			}

			if got != tc.value {
				t.Errorf("GetUint64Field() = %d, want %d", got, tc.value)
			}
		})
	}
}

// TestDocFloatField tests AddFloatField and GetFloatField round-trip.
func TestDocFloatField(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value float32
	}{
		{"positive", 3.14},
		{"negative", -3.14},
		{"zero", 0},
		{"max", math.MaxFloat32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "float_field_" + tc.name
			if err := doc.AddFloatField(fieldName, tc.value); err != nil {
				t.Fatalf("AddFloatField failed: %v", err)
			}

			got, err := doc.GetFloatField(fieldName)
			if err != nil {
				t.Fatalf("GetFloatField failed: %v", err)
			}

			if math.Abs(float64(got-tc.value)) > 1e-6 {
				t.Errorf("GetFloatField() = %f, want %f", got, tc.value)
			}
		})
	}
}

// TestDocDoubleField tests AddDoubleField and GetDoubleField round-trip.
func TestDocDoubleField(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name  string
		value float64
	}{
		{"positive", 3.1415926535},
		{"negative", -3.1415926535},
		{"zero", 0},
		{"max", math.MaxFloat64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "double_field_" + tc.name
			if err := doc.AddDoubleField(fieldName, tc.value); err != nil {
				t.Fatalf("AddDoubleField failed: %v", err)
			}

			got, err := doc.GetDoubleField(fieldName)
			if err != nil {
				t.Fatalf("GetDoubleField failed: %v", err)
			}

			if math.Abs(got-tc.value) > 1e-10 {
				t.Errorf("GetDoubleField() = %f, want %f", got, tc.value)
			}
		})
	}
}

// TestDocVectorFP32Field tests AddVectorFP32Field and GetVectorFP32Field round-trip.
func TestDocVectorFP32Field(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	vector := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	fieldName := "vector_field"

	if err := doc.AddVectorFP32Field(fieldName, vector); err != nil {
		t.Fatalf("AddVectorFP32Field failed: %v", err)
	}

	got, err := doc.GetVectorFP32Field(fieldName)
	if err != nil {
		t.Fatalf("GetVectorFP32Field failed: %v", err)
	}

	if len(got) != len(vector) {
		t.Fatalf("GetVectorFP32Field() length = %d, want %d", len(got), len(vector))
	}

	for i := range vector {
		if math.Abs(float64(got[i]-vector[i])) > 1e-6 {
			t.Errorf("GetVectorFP32Field()[%d] = %f, want %f", i, got[i], vector[i])
		}
	}
}

// TestDocVectorFP32FieldEmpty tests that AddVectorFP32Field with empty slice returns error.
func TestDocVectorFP32FieldEmpty(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	err := doc.AddVectorFP32Field("empty_vector", []float32{})
	if err == nil {
		t.Error("AddVectorFP32Field with empty slice should return error")
	}
}

// TestDocBinaryField tests AddBinaryField.
func TestDocBinaryField(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	testCases := []struct {
		name string
		data []byte
	}{
		{"single_byte", []byte{0x00}},
		{"normal", []byte{0x01, 0x02, 0x03, 0x04}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fieldName := "binary_field_" + tc.name
			if err := doc.AddBinaryField(fieldName, tc.data); err != nil {
				t.Fatalf("AddBinaryField failed: %v", err)
			}
		})
	}
}

// TestDocSetFieldNull tests SetFieldNull and IsFieldNull.
func TestDocSetFieldNull(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	fieldName := "test_field"
	if err := doc.AddStringField(fieldName, "test value"); err != nil {
		t.Fatalf("AddStringField failed: %v", err)
	}

	if err := doc.SetFieldNull(fieldName); err != nil {
		t.Fatalf("SetFieldNull failed: %v", err)
	}

	if !doc.IsFieldNull(fieldName) {
		t.Error("IsFieldNull() should return true after SetFieldNull")
	}
}

// TestDocRemoveField tests RemoveField.
func TestDocRemoveField(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	fieldName := "test_field"
	if err := doc.AddStringField(fieldName, "test value"); err != nil {
		t.Fatalf("AddStringField failed: %v", err)
	}

	if err := doc.RemoveField(fieldName); err != nil {
		t.Fatalf("RemoveField failed: %v", err)
	}

	if doc.HasField(fieldName) {
		t.Error("HasField() should return false after RemoveField")
	}
}

// TestDocHasField tests HasField and HasFieldValue.
func TestDocHasField(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	fieldName := "test_field"
	if err := doc.AddStringField(fieldName, "test value"); err != nil {
		t.Fatalf("AddStringField failed: %v", err)
	}

	if !doc.HasField(fieldName) {
		t.Error("HasField() should return true for existing field")
	}

	if !doc.HasFieldValue(fieldName) {
		t.Error("HasFieldValue() should return true for field with value")
	}
}

// TestDocGetFieldCount tests GetFieldCount.
func TestDocGetFieldCount(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	expectedCount := 5
	for i := 0; i < expectedCount; i++ {
		if err := doc.AddStringField("field"+string(rune('0'+i)), "value"); err != nil {
			t.Fatalf("AddStringField failed: %v", err)
		}
	}

	if got := doc.GetFieldCount(); got != expectedCount {
		t.Errorf("GetFieldCount() = %d, want %d", got, expectedCount)
	}
}

// TestDocGetFieldNames tests GetFieldNames.
func TestDocGetFieldNames(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	expectedNames := []string{"field1", "field2", "field3"}
	for _, name := range expectedNames {
		if err := doc.AddStringField(name, "value"); err != nil {
			t.Fatalf("AddStringField failed: %v", err)
		}
	}

	names, err := doc.GetFieldNames()
	if err != nil {
		t.Fatalf("GetFieldNames failed: %v", err)
	}

	if len(names) != len(expectedNames) {
		t.Errorf("GetFieldNames() length = %d, want %d", len(names), len(expectedNames))
	}

	// Create a map for easier comparison
	nameMap := make(map[string]bool)
	for _, name := range names {
		nameMap[name] = true
	}

	for _, expected := range expectedNames {
		if !nameMap[expected] {
			t.Errorf("GetFieldNames() missing expected field: %s", expected)
		}
	}
}

// TestDocClear tests Clear.
func TestDocClear(t *testing.T) {
	doc := NewDoc()
	defer doc.Destroy()

	// Add some fields
	if err := doc.AddStringField("field1", "value1"); err != nil {
		t.Fatalf("AddStringField failed: %v", err)
	}
	if err := doc.AddInt32Field("field2", 42); err != nil {
		t.Fatalf("AddInt32Field failed: %v", err)
	}

	doc.Clear()

	if !doc.IsEmpty() {
		t.Error("IsEmpty() should return true after Clear")
	}

	if doc.GetFieldCount() != 0 {
		t.Errorf("GetFieldCount() should be 0 after Clear, got %d", doc.GetFieldCount())
	}
}

// TestDocDestroy tests Destroy.
func TestDocDestroy(t *testing.T) {
	doc := NewDoc()

	// First destroy should not panic
	doc.Destroy()

	// Second destroy should not panic
	doc.Destroy()
}

// BenchmarkDocCreate benchmarks creating and destroying documents.
func BenchmarkDocCreate(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		doc := NewDoc()
		doc.Destroy()
	}
}
