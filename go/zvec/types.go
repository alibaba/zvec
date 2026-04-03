package zvec

// DataType represents the data type of a field.
type DataType uint32

const (
	DataTypeUndefined        DataType = 0
	DataTypeBinary           DataType = 1
	DataTypeString           DataType = 2
	DataTypeBool             DataType = 3
	DataTypeInt32            DataType = 4
	DataTypeInt64            DataType = 5
	DataTypeUint32           DataType = 6
	DataTypeUint64           DataType = 7
	DataTypeFloat            DataType = 8
	DataTypeDouble           DataType = 9
	DataTypeVectorBinary32   DataType = 20
	DataTypeVectorBinary64   DataType = 21
	DataTypeVectorFP16       DataType = 22
	DataTypeVectorFP32       DataType = 23
	DataTypeVectorFP64       DataType = 24
	DataTypeVectorInt4       DataType = 25
	DataTypeVectorInt8       DataType = 26
	DataTypeVectorInt16      DataType = 27
	DataTypeSparseVectorFP16 DataType = 30
	DataTypeSparseVectorFP32 DataType = 31
	DataTypeArrayBinary      DataType = 40
	DataTypeArrayString      DataType = 41
	DataTypeArrayBool        DataType = 42
	DataTypeArrayInt32       DataType = 43
	DataTypeArrayInt64       DataType = 44
	DataTypeArrayUint32      DataType = 45
	DataTypeArrayUint64      DataType = 46
	DataTypeArrayFloat       DataType = 47
	DataTypeArrayDouble      DataType = 48
)

// IndexType represents the type of index.
type IndexType uint32

const (
	IndexTypeUndefined IndexType = 0
	IndexTypeHNSW      IndexType = 1
	IndexTypeIVF       IndexType = 2
	IndexTypeFlat      IndexType = 3
	IndexTypeInvert    IndexType = 10
)

// MetricType represents the distance metric type.
type MetricType uint32

const (
	MetricTypeUndefined MetricType = 0
	MetricTypeL2        MetricType = 1
	MetricTypeIP        MetricType = 2
	MetricTypeCosine    MetricType = 3
	MetricTypeMIPSL2    MetricType = 4
)

// QuantizeType represents the quantization type.
type QuantizeType uint32

const (
	QuantizeTypeUndefined QuantizeType = 0
	QuantizeTypeFP16      QuantizeType = 1
	QuantizeTypeInt8      QuantizeType = 2
	QuantizeTypeInt4      QuantizeType = 3
)

// LogLevel represents the log level.
type LogLevel int

const (
	LogLevelDebug LogLevel = 0
	LogLevelInfo  LogLevel = 1
	LogLevelWarn  LogLevel = 2
	LogLevelError LogLevel = 3
	LogLevelFatal LogLevel = 4
)

// DocOperator represents the document operation type.
type DocOperator int

const (
	DocOpInsert DocOperator = 0
	DocOpUpdate DocOperator = 1
	DocOpUpsert DocOperator = 2
	DocOpDelete DocOperator = 3
)
