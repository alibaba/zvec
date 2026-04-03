use std::fmt;

/// Data type of a field in a zvec collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DataType {
    Undefined = 0,
    Binary = 1,
    String = 2,
    Bool = 3,
    Int32 = 4,
    Int64 = 5,
    Uint32 = 6,
    Uint64 = 7,
    Float = 8,
    Double = 9,
    VectorBinary32 = 20,
    VectorBinary64 = 21,
    VectorFp16 = 22,
    VectorFp32 = 23,
    VectorFp64 = 24,
    VectorInt4 = 25,
    VectorInt8 = 26,
    VectorInt16 = 27,
    SparseVectorFp16 = 30,
    SparseVectorFp32 = 31,
    ArrayBinary = 40,
    ArrayString = 41,
    ArrayBool = 42,
    ArrayInt32 = 43,
    ArrayInt64 = 44,
    ArrayUint32 = 45,
    ArrayUint64 = 46,
    ArrayFloat = 47,
    ArrayDouble = 48,
}

impl From<u32> for DataType {
    fn from(value: u32) -> Self {
        match value {
            0 => DataType::Undefined,
            1 => DataType::Binary,
            2 => DataType::String,
            3 => DataType::Bool,
            4 => DataType::Int32,
            5 => DataType::Int64,
            6 => DataType::Uint32,
            7 => DataType::Uint64,
            8 => DataType::Float,
            9 => DataType::Double,
            20 => DataType::VectorBinary32,
            21 => DataType::VectorBinary64,
            22 => DataType::VectorFp16,
            23 => DataType::VectorFp32,
            24 => DataType::VectorFp64,
            25 => DataType::VectorInt4,
            26 => DataType::VectorInt8,
            27 => DataType::VectorInt16,
            30 => DataType::SparseVectorFp16,
            31 => DataType::SparseVectorFp32,
            40 => DataType::ArrayBinary,
            41 => DataType::ArrayString,
            42 => DataType::ArrayBool,
            43 => DataType::ArrayInt32,
            44 => DataType::ArrayInt64,
            45 => DataType::ArrayUint32,
            46 => DataType::ArrayUint64,
            47 => DataType::ArrayFloat,
            48 => DataType::ArrayDouble,
            _ => DataType::Undefined,
        }
    }
}

impl From<DataType> for u32 {
    fn from(dt: DataType) -> Self {
        dt as u32
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Index type for a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IndexType {
    Undefined = 0,
    Hnsw = 1,
    Ivf = 2,
    Flat = 3,
    Invert = 10,
}

impl From<u32> for IndexType {
    fn from(value: u32) -> Self {
        match value {
            1 => IndexType::Hnsw,
            2 => IndexType::Ivf,
            3 => IndexType::Flat,
            10 => IndexType::Invert,
            _ => IndexType::Undefined,
        }
    }
}

impl From<IndexType> for u32 {
    fn from(it: IndexType) -> Self {
        it as u32
    }
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Distance metric type for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetricType {
    Undefined = 0,
    L2 = 1,
    Ip = 2,
    Cosine = 3,
    MipsL2 = 4,
}

impl From<u32> for MetricType {
    fn from(value: u32) -> Self {
        match value {
            1 => MetricType::L2,
            2 => MetricType::Ip,
            3 => MetricType::Cosine,
            4 => MetricType::MipsL2,
            _ => MetricType::Undefined,
        }
    }
}

impl From<MetricType> for u32 {
    fn from(mt: MetricType) -> Self {
        mt as u32
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Quantization type for vector indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QuantizeType {
    Undefined = 0,
    Fp16 = 1,
    Int8 = 2,
    Int4 = 3,
}

impl From<u32> for QuantizeType {
    fn from(value: u32) -> Self {
        match value {
            1 => QuantizeType::Fp16,
            2 => QuantizeType::Int8,
            3 => QuantizeType::Int4,
            _ => QuantizeType::Undefined,
        }
    }
}

impl From<QuantizeType> for u32 {
    fn from(qt: QuantizeType) -> Self {
        qt as u32
    }
}

impl fmt::Display for QuantizeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Log level for the zvec library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Fatal = 4,
}

impl From<u32> for LogLevel {
    fn from(value: u32) -> Self {
        match value {
            0 => LogLevel::Debug,
            1 => LogLevel::Info,
            2 => LogLevel::Warn,
            3 => LogLevel::Error,
            4 => LogLevel::Fatal,
            _ => LogLevel::Debug,
        }
    }
}

impl From<LogLevel> for u32 {
    fn from(ll: LogLevel) -> Self {
        ll as u32
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Document operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DocOperator {
    Insert = 0,
    Update = 1,
    Upsert = 2,
    Delete = 3,
}

impl From<u32> for DocOperator {
    fn from(value: u32) -> Self {
        match value {
            0 => DocOperator::Insert,
            1 => DocOperator::Update,
            2 => DocOperator::Upsert,
            3 => DocOperator::Delete,
            _ => DocOperator::Insert,
        }
    }
}

impl From<DocOperator> for u32 {
    fn from(op: DocOperator) -> Self {
        op as u32
    }
}

impl fmt::Display for DocOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DataType tests
    // =========================================================================

    #[test]
    fn data_type_from_u32_scalar_types() {
        assert_eq!(DataType::from(0), DataType::Undefined);
        assert_eq!(DataType::from(1), DataType::Binary);
        assert_eq!(DataType::from(2), DataType::String);
        assert_eq!(DataType::from(3), DataType::Bool);
        assert_eq!(DataType::from(4), DataType::Int32);
        assert_eq!(DataType::from(5), DataType::Int64);
        assert_eq!(DataType::from(6), DataType::Uint32);
        assert_eq!(DataType::from(7), DataType::Uint64);
        assert_eq!(DataType::from(8), DataType::Float);
        assert_eq!(DataType::from(9), DataType::Double);
    }

    #[test]
    fn data_type_from_u32_vector_types() {
        assert_eq!(DataType::from(20), DataType::VectorBinary32);
        assert_eq!(DataType::from(21), DataType::VectorBinary64);
        assert_eq!(DataType::from(22), DataType::VectorFp16);
        assert_eq!(DataType::from(23), DataType::VectorFp32);
        assert_eq!(DataType::from(24), DataType::VectorFp64);
        assert_eq!(DataType::from(25), DataType::VectorInt4);
        assert_eq!(DataType::from(26), DataType::VectorInt8);
        assert_eq!(DataType::from(27), DataType::VectorInt16);
    }

    #[test]
    fn data_type_from_u32_sparse_vector_types() {
        assert_eq!(DataType::from(30), DataType::SparseVectorFp16);
        assert_eq!(DataType::from(31), DataType::SparseVectorFp32);
    }

    #[test]
    fn data_type_from_u32_array_types() {
        assert_eq!(DataType::from(40), DataType::ArrayBinary);
        assert_eq!(DataType::from(41), DataType::ArrayString);
        assert_eq!(DataType::from(42), DataType::ArrayBool);
        assert_eq!(DataType::from(43), DataType::ArrayInt32);
        assert_eq!(DataType::from(44), DataType::ArrayInt64);
        assert_eq!(DataType::from(45), DataType::ArrayUint32);
        assert_eq!(DataType::from(46), DataType::ArrayUint64);
        assert_eq!(DataType::from(47), DataType::ArrayFloat);
        assert_eq!(DataType::from(48), DataType::ArrayDouble);
    }

    #[test]
    fn data_type_from_u32_unknown_falls_back_to_undefined() {
        assert_eq!(DataType::from(10), DataType::Undefined);
        assert_eq!(DataType::from(19), DataType::Undefined);
        assert_eq!(DataType::from(100), DataType::Undefined);
        assert_eq!(DataType::from(u32::MAX), DataType::Undefined);
    }

    #[test]
    fn data_type_roundtrip() {
        let all_types = [
            DataType::Undefined, DataType::Binary, DataType::String, DataType::Bool,
            DataType::Int32, DataType::Int64, DataType::Uint32, DataType::Uint64,
            DataType::Float, DataType::Double, DataType::VectorFp32, DataType::VectorFp64,
            DataType::VectorFp16, DataType::VectorInt4, DataType::VectorInt8,
            DataType::VectorInt16, DataType::VectorBinary32, DataType::VectorBinary64,
            DataType::SparseVectorFp16, DataType::SparseVectorFp32,
            DataType::ArrayBinary, DataType::ArrayString, DataType::ArrayBool,
            DataType::ArrayInt32, DataType::ArrayInt64, DataType::ArrayUint32,
            DataType::ArrayUint64, DataType::ArrayFloat, DataType::ArrayDouble,
        ];
        for dt in all_types {
            let numeric: u32 = dt.into();
            let back = DataType::from(numeric);
            assert_eq!(back, dt, "roundtrip failed for {:?} (numeric={})", dt, numeric);
        }
    }

    #[test]
    fn data_type_display() {
        assert_eq!(DataType::VectorFp32.to_string(), "VectorFp32");
        assert_eq!(DataType::String.to_string(), "String");
        assert_eq!(DataType::Undefined.to_string(), "Undefined");
    }

    // =========================================================================
    // IndexType tests
    // =========================================================================

    #[test]
    fn index_type_from_u32() {
        assert_eq!(IndexType::from(0), IndexType::Undefined);
        assert_eq!(IndexType::from(1), IndexType::Hnsw);
        assert_eq!(IndexType::from(2), IndexType::Ivf);
        assert_eq!(IndexType::from(3), IndexType::Flat);
        assert_eq!(IndexType::from(10), IndexType::Invert);
    }

    #[test]
    fn index_type_from_u32_unknown() {
        assert_eq!(IndexType::from(4), IndexType::Undefined);
        assert_eq!(IndexType::from(99), IndexType::Undefined);
    }

    #[test]
    fn index_type_roundtrip() {
        let all = [
            IndexType::Undefined, IndexType::Hnsw, IndexType::Ivf,
            IndexType::Flat, IndexType::Invert,
        ];
        for it in all {
            let numeric: u32 = it.into();
            let back = IndexType::from(numeric);
            assert_eq!(back, it);
        }
    }

    #[test]
    fn index_type_display() {
        assert_eq!(IndexType::Hnsw.to_string(), "Hnsw");
        assert_eq!(IndexType::Flat.to_string(), "Flat");
    }

    // =========================================================================
    // MetricType tests
    // =========================================================================

    #[test]
    fn metric_type_from_u32() {
        assert_eq!(MetricType::from(0), MetricType::Undefined);
        assert_eq!(MetricType::from(1), MetricType::L2);
        assert_eq!(MetricType::from(2), MetricType::Ip);
        assert_eq!(MetricType::from(3), MetricType::Cosine);
        assert_eq!(MetricType::from(4), MetricType::MipsL2);
    }

    #[test]
    fn metric_type_from_u32_unknown() {
        assert_eq!(MetricType::from(5), MetricType::Undefined);
        assert_eq!(MetricType::from(99), MetricType::Undefined);
    }

    #[test]
    fn metric_type_roundtrip() {
        let all = [
            MetricType::Undefined, MetricType::L2, MetricType::Ip,
            MetricType::Cosine, MetricType::MipsL2,
        ];
        for mt in all {
            let numeric: u32 = mt.into();
            let back = MetricType::from(numeric);
            assert_eq!(back, mt);
        }
    }

    #[test]
    fn metric_type_display() {
        assert_eq!(MetricType::Cosine.to_string(), "Cosine");
        assert_eq!(MetricType::L2.to_string(), "L2");
    }

    // =========================================================================
    // QuantizeType tests
    // =========================================================================

    #[test]
    fn quantize_type_from_u32() {
        assert_eq!(QuantizeType::from(0), QuantizeType::Undefined);
        assert_eq!(QuantizeType::from(1), QuantizeType::Fp16);
        assert_eq!(QuantizeType::from(2), QuantizeType::Int8);
        assert_eq!(QuantizeType::from(3), QuantizeType::Int4);
    }

    #[test]
    fn quantize_type_from_u32_unknown() {
        assert_eq!(QuantizeType::from(4), QuantizeType::Undefined);
        assert_eq!(QuantizeType::from(99), QuantizeType::Undefined);
    }

    #[test]
    fn quantize_type_roundtrip() {
        let all = [
            QuantizeType::Undefined, QuantizeType::Fp16,
            QuantizeType::Int8, QuantizeType::Int4,
        ];
        for qt in all {
            let numeric: u32 = qt.into();
            let back = QuantizeType::from(numeric);
            assert_eq!(back, qt);
        }
    }

    #[test]
    fn quantize_type_display() {
        assert_eq!(QuantizeType::Fp16.to_string(), "Fp16");
        assert_eq!(QuantizeType::Int4.to_string(), "Int4");
    }

    // =========================================================================
    // LogLevel tests
    // =========================================================================

    #[test]
    fn log_level_from_u32() {
        assert_eq!(LogLevel::from(0), LogLevel::Debug);
        assert_eq!(LogLevel::from(1), LogLevel::Info);
        assert_eq!(LogLevel::from(2), LogLevel::Warn);
        assert_eq!(LogLevel::from(3), LogLevel::Error);
        assert_eq!(LogLevel::from(4), LogLevel::Fatal);
    }

    #[test]
    fn log_level_from_u32_unknown_defaults_to_debug() {
        assert_eq!(LogLevel::from(5), LogLevel::Debug);
        assert_eq!(LogLevel::from(99), LogLevel::Debug);
    }

    #[test]
    fn log_level_roundtrip() {
        let all = [
            LogLevel::Debug, LogLevel::Info, LogLevel::Warn,
            LogLevel::Error, LogLevel::Fatal,
        ];
        for ll in all {
            let numeric: u32 = ll.into();
            let back = LogLevel::from(numeric);
            assert_eq!(back, ll);
        }
    }

    #[test]
    fn log_level_display() {
        assert_eq!(LogLevel::Info.to_string(), "Info");
        assert_eq!(LogLevel::Error.to_string(), "Error");
    }

    // =========================================================================
    // DocOperator tests
    // =========================================================================

    #[test]
    fn doc_operator_from_u32() {
        assert_eq!(DocOperator::from(0), DocOperator::Insert);
        assert_eq!(DocOperator::from(1), DocOperator::Update);
        assert_eq!(DocOperator::from(2), DocOperator::Upsert);
        assert_eq!(DocOperator::from(3), DocOperator::Delete);
    }

    #[test]
    fn doc_operator_from_u32_unknown_defaults_to_insert() {
        assert_eq!(DocOperator::from(4), DocOperator::Insert);
        assert_eq!(DocOperator::from(99), DocOperator::Insert);
    }

    #[test]
    fn doc_operator_roundtrip() {
        let all = [
            DocOperator::Insert, DocOperator::Update,
            DocOperator::Upsert, DocOperator::Delete,
        ];
        for op in all {
            let numeric: u32 = op.into();
            let back = DocOperator::from(numeric);
            assert_eq!(back, op);
        }
    }

    #[test]
    fn doc_operator_display() {
        assert_eq!(DocOperator::Insert.to_string(), "Insert");
        assert_eq!(DocOperator::Delete.to_string(), "Delete");
    }

    // =========================================================================
    // Cross-cutting tests
    // =========================================================================

    #[test]
    fn all_enums_implement_copy() {
        let dt = DataType::VectorFp32;
        let dt2 = dt;
        assert_eq!(dt, dt2);

        let it = IndexType::Hnsw;
        let it2 = it;
        assert_eq!(it, it2);

        let mt = MetricType::Cosine;
        let mt2 = mt;
        assert_eq!(mt, mt2);

        let qt = QuantizeType::Int8;
        let qt2 = qt;
        assert_eq!(qt, qt2);

        let ll = LogLevel::Info;
        let ll2 = ll;
        assert_eq!(ll, ll2);

        let op = DocOperator::Upsert;
        let op2 = op;
        assert_eq!(op, op2);
    }

    #[test]
    fn all_enums_implement_debug() {
        assert_eq!(format!("{:?}", DataType::VectorFp32), "VectorFp32");
        assert_eq!(format!("{:?}", IndexType::Hnsw), "Hnsw");
        assert_eq!(format!("{:?}", MetricType::Cosine), "Cosine");
        assert_eq!(format!("{:?}", QuantizeType::Int8), "Int8");
        assert_eq!(format!("{:?}", LogLevel::Warn), "Warn");
        assert_eq!(format!("{:?}", DocOperator::Delete), "Delete");
    }

    #[test]
    fn repr_u32_values_match_discriminants() {
        assert_eq!(DataType::VectorFp32 as u32, 23);
        assert_eq!(IndexType::Hnsw as u32, 1);
        assert_eq!(IndexType::Invert as u32, 10);
        assert_eq!(MetricType::Cosine as u32, 3);
        assert_eq!(QuantizeType::Int8 as u32, 2);
        assert_eq!(LogLevel::Fatal as u32, 4);
        assert_eq!(DocOperator::Delete as u32, 3);
    }
}
