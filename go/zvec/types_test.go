package zvec

import "testing"

func TestDataTypeConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      DataType
		expected uint32
	}{
		{"Undefined", DataTypeUndefined, 0},
		{"Binary", DataTypeBinary, 1},
		{"String", DataTypeString, 2},
		{"Bool", DataTypeBool, 3},
		{"Int32", DataTypeInt32, 4},
		{"Int64", DataTypeInt64, 5},
		{"Uint32", DataTypeUint32, 6},
		{"Uint64", DataTypeUint64, 7},
		{"Float", DataTypeFloat, 8},
		{"Double", DataTypeDouble, 9},
		{"VectorBinary32", DataTypeVectorBinary32, 20},
		{"VectorBinary64", DataTypeVectorBinary64, 21},
		{"VectorFP16", DataTypeVectorFP16, 22},
		{"VectorFP32", DataTypeVectorFP32, 23},
		{"VectorFP64", DataTypeVectorFP64, 24},
		{"VectorInt4", DataTypeVectorInt4, 25},
		{"VectorInt8", DataTypeVectorInt8, 26},
		{"VectorInt16", DataTypeVectorInt16, 27},
		{"SparseVectorFP16", DataTypeSparseVectorFP16, 30},
		{"SparseVectorFP32", DataTypeSparseVectorFP32, 31},
		{"ArrayBinary", DataTypeArrayBinary, 40},
		{"ArrayString", DataTypeArrayString, 41},
		{"ArrayBool", DataTypeArrayBool, 42},
		{"ArrayInt32", DataTypeArrayInt32, 43},
		{"ArrayInt64", DataTypeArrayInt64, 44},
		{"ArrayUint32", DataTypeArrayUint32, 45},
		{"ArrayUint64", DataTypeArrayUint64, 46},
		{"ArrayFloat", DataTypeArrayFloat, 47},
		{"ArrayDouble", DataTypeArrayDouble, 48},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if uint32(tt.got) != tt.expected {
				t.Errorf("DataType %s = %d, want %d", tt.name, tt.got, tt.expected)
			}
		})
	}
}

func TestIndexTypeConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      IndexType
		expected uint32
	}{
		{"Undefined", IndexTypeUndefined, 0},
		{"HNSW", IndexTypeHNSW, 1},
		{"IVF", IndexTypeIVF, 2},
		{"Flat", IndexTypeFlat, 3},
		{"Invert", IndexTypeInvert, 10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if uint32(tt.got) != tt.expected {
				t.Errorf("IndexType %s = %d, want %d", tt.name, tt.got, tt.expected)
			}
		})
	}
}

func TestMetricTypeConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      MetricType
		expected uint32
	}{
		{"Undefined", MetricTypeUndefined, 0},
		{"L2", MetricTypeL2, 1},
		{"IP", MetricTypeIP, 2},
		{"Cosine", MetricTypeCosine, 3},
		{"MIPSL2", MetricTypeMIPSL2, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if uint32(tt.got) != tt.expected {
				t.Errorf("MetricType %s = %d, want %d", tt.name, tt.got, tt.expected)
			}
		})
	}
}

func TestQuantizeTypeConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      QuantizeType
		expected uint32
	}{
		{"Undefined", QuantizeTypeUndefined, 0},
		{"FP16", QuantizeTypeFP16, 1},
		{"Int8", QuantizeTypeInt8, 2},
		{"Int4", QuantizeTypeInt4, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if uint32(tt.got) != tt.expected {
				t.Errorf("QuantizeType %s = %d, want %d", tt.name, tt.got, tt.expected)
			}
		})
	}
}

func TestLogLevelConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      LogLevel
		expected int
	}{
		{"Debug", LogLevelDebug, 0},
		{"Info", LogLevelInfo, 1},
		{"Warn", LogLevelWarn, 2},
		{"Error", LogLevelError, 3},
		{"Fatal", LogLevelFatal, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int(tt.got) != tt.expected {
				t.Errorf("LogLevel %s = %d, want %d", tt.name, tt.got, tt.expected)
			}
		})
	}
}

func TestDocOperatorConstants(t *testing.T) {
	tests := []struct {
		name     string
		got      DocOperator
		expected int
	}{
		{"Insert", DocOpInsert, 0},
		{"Update", DocOpUpdate, 1},
		{"Upsert", DocOpUpsert, 2},
		{"Delete", DocOpDelete, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int(tt.got) != tt.expected {
				t.Errorf("DocOperator %s = %d, want %d", tt.name, tt.got, tt.expected)
			}
		})
	}
}
