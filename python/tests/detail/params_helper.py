from zvec import (
    CollectionOption,
    IndexOption,
    OptimizeOption,
    InvertIndexParam,
    HnswIndexParam,
    IVFIndexParam,
    FlatIndexParam,
    AlterColumnOption,
    AddColumnOption,
    DataType,
    MetricType,
    QuantizeType,
)


VALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP = {
    DataType.VECTOR_FP32: [
        HnswIndexParam(),
        HnswIndexParam(
            metric_type=MetricType.IP,
            m=16,
            ef_construction=100,
            quantize_type=QuantizeType.INT8,
        ),
        HnswIndexParam(
            metric_type=MetricType.COSINE,
            m=24,
            ef_construction=150,
            quantize_type=QuantizeType.INT4,
        ),
        HnswIndexParam(
            metric_type=MetricType.L2,
            m=32,
            ef_construction=200,
            quantize_type=QuantizeType.FP16,
        ),
        FlatIndexParam(),
        FlatIndexParam(metric_type=MetricType.IP),
        FlatIndexParam(metric_type=MetricType.L2),
        FlatIndexParam(metric_type=MetricType.COSINE),
        IVFIndexParam(),
        IVFIndexParam(
            metric_type=MetricType.IP,
            quantize_type=QuantizeType.INT4,
            n_list=100,
            n_iters=10,
            use_soar=False,
        ),
        IVFIndexParam(
            metric_type=MetricType.L2,
            quantize_type=QuantizeType.INT8,
            n_list=200,
            n_iters=20,
            use_soar=True,
        ),
        IVFIndexParam(
            metric_type=MetricType.COSINE,
            quantize_type=QuantizeType.FP16,
            n_list=150,
            n_iters=15,
            use_soar=False,
        ),
    ],
    DataType.VECTOR_FP16: [
        HnswIndexParam(),
        HnswIndexParam(metric_type=MetricType.IP, m=16, ef_construction=100),
        HnswIndexParam(metric_type=MetricType.COSINE, m=24, ef_construction=150),
        HnswIndexParam(metric_type=MetricType.L2, m=32, ef_construction=200),
        FlatIndexParam(),
        FlatIndexParam(metric_type=MetricType.IP),
        FlatIndexParam(metric_type=MetricType.L2),
        FlatIndexParam(metric_type=MetricType.COSINE),
        IVFIndexParam(),
        IVFIndexParam(metric_type=MetricType.IP),
        IVFIndexParam(metric_type=MetricType.L2),
        IVFIndexParam(metric_type=MetricType.COSINE),
    ],
    DataType.VECTOR_INT8: [
        HnswIndexParam(),
        FlatIndexParam(),
        HnswIndexParam(metric_type=MetricType.IP, m=16, ef_construction=100),
        FlatIndexParam(metric_type=MetricType.IP),
        IVFIndexParam(metric_type=MetricType.L2),
        HnswIndexParam(metric_type=MetricType.L2),
        FlatIndexParam(metric_type=MetricType.L2),
    ],
    DataType.SPARSE_VECTOR_FP32: [
        HnswIndexParam(),
        FlatIndexParam(),
        HnswIndexParam(
            metric_type=MetricType.IP,
            m=16,
            ef_construction=100,
            quantize_type=QuantizeType.FP16,
        ),
    ],
    DataType.SPARSE_VECTOR_FP16: [
        HnswIndexParam(),
        FlatIndexParam(),
        HnswIndexParam(
            metric_type=MetricType.IP,
            m=16,
            ef_construction=100,
        ),
    ],
}

VALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP_PARAMS = [
    (data_type, param)
    for data_type, params in VALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP.items()
    for param in params
]

INVALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP = {
    DataType.VECTOR_FP32: [
        InvertIndexParam(),
    ],
    DataType.VECTOR_FP16: [
        InvertIndexParam(),
    ],
    DataType.VECTOR_INT8: [
        InvertIndexParam(),
        IVFIndexParam(metric_type=MetricType.IP),
        FlatIndexParam(metric_type=MetricType.COSINE),
        IVFIndexParam(metric_type=MetricType.COSINE),
        HnswIndexParam(metric_type=MetricType.COSINE),
    ],
    DataType.SPARSE_VECTOR_FP32: [
        HnswIndexParam(metric_type=MetricType.L2),
        FlatIndexParam(metric_type=MetricType.COSINE),
        IVFIndexParam(),
        InvertIndexParam(),
    ],
    DataType.SPARSE_VECTOR_FP16: [
        HnswIndexParam(metric_type=MetricType.L2),
        FlatIndexParam(metric_type=MetricType.COSINE),
        IVFIndexParam(),
        InvertIndexParam(),
    ],
}

INVALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP_PARAMS = [
    (data_type, param)
    for data_type, params in INVALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP.items()
    for param in params
]

COLLECTION_NAME_MAX_LENGTH = 64

COLLECTION_NAME_VALID_LIST = [
    "col",
    "C0llECTION",
    "Collection1",
    "collection_2",
    "123collection-",
    "a" * COLLECTION_NAME_MAX_LENGTH,
]

COLLECTION_NAME_INVALID_LIST = [
    "l",
    "1C",
    "",
    " ",
    None,
    "abcdefghijklmnopqrstuvwxzy123456abcdefghijklmnopqrstuvwxzy1234561",
    "test/",
    "!@#$%^&*()test",
]

FIELD_NAME_VALID_LIST = [
    "1",
    "12",
    "col",
    "ID",
    "name1",
    "Weigt_12-",
    "123age",
    "name_with_underscores",
    "123numeric_start",
    "name-with-dashes",
]

FIELD_NAME_INVALID_LIST = [
    "",
    " ",
    None,
    "abcdefghijklmnopqrstuvwxzy1234561",
    "test/",
    "!@#$%^&*()test",
    "name@with#special$chars",
    "name with spaces",
]

FIELD_LIST_MAX_LENGTH = 1024
VECTOR_LIST_MAX_LENGTH = 5
DENSE_VECTOR_MAX_DIMENSION = 20000
SPARSE_VECTOR_MAX_DIMENSION = 4096

FIELD_VECTOR_LIST_DIMENSION_VALID_LIST = [
    # field_list_len, vector_list_len, dimension
    (1, 1, 1),
    (2, 2, 512),
    (512, 3, 1024),
    (1024, 4, 20000),
]

FIELD_VECTOR_LIST_DIMENSION_INVALID_LIST = [
    # field_list_len, vector_list_len, dimension
    (1, 1, 0),
    (1, 1, -1),
    (1, 1, "1"),
    (1, 1, 20001),
]


INCOMPATIBLE_CONSTRUCTOR_ERROR_MSG = "incompatible constructor arguments"
SCHEMA_VALIDATE_ERROR_MSG = "schema validate failed"
CREATE_READ_ONLY_ERROR_MSG = "Unable to create collection with read-only mode"
INCOMPATIBLE_FUNCTION_ERROR_MSG = "incompatible function arguments"
INVALID_PATH_ERROR_MSG = "path validate failed"
INDEX_NON_EXISTENT_COLUMN_ERROR_MSG = "not found in schema"
ACCESS_DESTROYED_COLLECTION_ERROR_MSG = "is already destroyed"
COLLECTION_PATH_NOT_EXIST_ERROR_MSG = "not exist"
NOT_SUPPORT_ADD_COLUMN_ERROR_MSG = "Only support basic numeric data type"
NOT_EXIST_COLUMN_TO_DROP_ERROR_MSG = "Column not exists"
