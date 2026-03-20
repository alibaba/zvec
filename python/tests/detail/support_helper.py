from zvec import (
    CollectionOption,
    IndexOption,
    OptimizeOption,
    InvertIndexParam,
    HnswIndexParam,
    IVFIndexParam,
    FlatIndexParam,
    DataType,
    IndexType,
    QuantizeType,
)

SUPPORT_SCALAR_DATA_TYPES = [
    DataType.BOOL,
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT32,
    DataType.UINT64,
    DataType.STRING,
    DataType.ARRAY_BOOL,
    DataType.ARRAY_FLOAT,
    DataType.ARRAY_DOUBLE,
    DataType.ARRAY_INT32,
    DataType.ARRAY_INT64,
    DataType.ARRAY_UINT32,
    DataType.ARRAY_UINT64,
    DataType.ARRAY_STRING,
]

DEFAULT_SCALAR_FIELD_NAME = {
    DataType.BOOL: "bool_field",
    DataType.FLOAT: "float_field",
    DataType.DOUBLE: "double_field",
    DataType.INT32: "int32_field",
    DataType.INT64: "int64_field",
    DataType.UINT32: "uint32_field",
    DataType.UINT64: "uint64_field",
    DataType.STRING: "string_field",
    DataType.ARRAY_BOOL: "array_bool_field",
    DataType.ARRAY_FLOAT: "array_float_field",
    DataType.ARRAY_DOUBLE: "array_double_field",
    DataType.ARRAY_INT32: "array_int32_field",
    DataType.ARRAY_INT64: "array_int64_field",
    DataType.ARRAY_UINT32: "array_uint32_field",
    DataType.ARRAY_UINT64: "array_uint64_field",
    DataType.ARRAY_STRING: "array_string_field",
}

SUPPORT_SCALAR_INDEX_TYPES = [
    IndexType.INVERT,
]

SUPPORT_VECTOR_DATA_TYPES = [
    DataType.VECTOR_FP16,
    DataType.VECTOR_FP32,
    DataType.VECTOR_INT8,
    DataType.SPARSE_VECTOR_FP32,
    DataType.SPARSE_VECTOR_FP16,
]

SUPPORT_VECTOR_INDEX_TYPES = [
    IndexType.FLAT,
    IndexType.HNSW,
    IndexType.IVF,
]

DEFAULT_VECTOR_FIELD_NAME = {
    DataType.VECTOR_FP16: "vector_fp16_field",
    DataType.VECTOR_FP32: "vector_fp32_field",
    DataType.VECTOR_INT8: "vector_int8_field",
    DataType.SPARSE_VECTOR_FP32: "sparse_vector_fp32_field",
    DataType.SPARSE_VECTOR_FP16: "sparse_vector_fp16_field",
}

DEFAULT_VECTOR_DIMENSION = 128
VECTOR_DIMENSION_1024 = 4
SUPPORT_VECTOR_DATA_TYPE_INDEX_MAP = {
    DataType.VECTOR_FP16: [IndexType.FLAT, IndexType.HNSW, IndexType.IVF],
    DataType.VECTOR_FP32: [IndexType.FLAT, IndexType.HNSW, IndexType.IVF],
    DataType.VECTOR_INT8: [IndexType.FLAT, IndexType.HNSW],
    DataType.SPARSE_VECTOR_FP32: [IndexType.FLAT, IndexType.HNSW],
    DataType.SPARSE_VECTOR_FP16: [IndexType.FLAT, IndexType.HNSW],
}

SUPPORT_VECTOR_DATA_TYPE_INDEX_MAP_PARAMS = [
    (data_type, index_type)
    for data_type, index_types in SUPPORT_VECTOR_DATA_TYPE_INDEX_MAP.items()
    for index_type in index_types
]

DEFAULT_INDEX_PARAMS = {
    IndexType.FLAT: FlatIndexParam(),
    IndexType.HNSW: HnswIndexParam(),
    IndexType.IVF: IVFIndexParam(),
    IndexType.INVERT: InvertIndexParam(),
}

SUPPORT_VECTOR_DATA_TYPE_QUANT_MAP = {
    DataType.VECTOR_FP32: [QuantizeType.FP16, QuantizeType.INT8, QuantizeType.INT4],
    DataType.SPARSE_VECTOR_FP32: [QuantizeType.FP16],
}

SUPPORT_ADD_COLUMN_DATA_TYPE = [
    DataType.INT32,
    DataType.UINT32,
    DataType.INT64,
    DataType.UINT64,
    DataType.FLOAT,
    DataType.DOUBLE,
]

NOT_SUPPORT_ADD_COLUMN_DATA_TYPE = [
    DataType.BOOL,
    DataType.STRING,
    DataType.ARRAY_BOOL,
    DataType.ARRAY_INT32,
    DataType.ARRAY_INT64,
    DataType.ARRAY_UINT32,
    DataType.ARRAY_UINT64,
    DataType.ARRAY_FLOAT,
    DataType.ARRAY_DOUBLE,
    DataType.ARRAY_STRING,
]
