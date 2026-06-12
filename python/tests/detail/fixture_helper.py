import pytest
import logging
import platform

DISKANN_SUPPORTED = platform.system() == "Linux" and platform.machine() in (
    "x86_64",
    "AMD64",
    "i686",
    "i386",
)

from typing import Any, Generator
from zvec.typing import DataType, StatusCode, MetricType, QuantizeType
import zvec


# Cache the DiskAnn plugin preload status so we pay the load cost once per
# test session. The plugin normally auto-loads on first DiskAnn use, but we
# preload it explicitly here so a missing libaio / misplaced plugin .so
# surfaces as a clear pytest skip instead of a confusing
# "Create vector column indexer failed" deep inside the collection code path.
_DISKANN_PRELOAD_REASON: str | None = None
_DISKANN_PRELOAD_DONE: bool = False


def _ensure_diskann_runtime_or_reason() -> str | None:
    """Preload the DiskAnn plugin and return None on success or a human-readable
    skip reason on failure. Idempotent across calls."""
    global _DISKANN_PRELOAD_DONE, _DISKANN_PRELOAD_REASON
    if _DISKANN_PRELOAD_DONE:
        return _DISKANN_PRELOAD_REASON
    _DISKANN_PRELOAD_DONE = True

    if not DISKANN_SUPPORTED:
        _DISKANN_PRELOAD_REASON = "DiskAnn only supported on Linux x86_64"
        return _DISKANN_PRELOAD_REASON

    if not zvec.is_libaio_available():
        _DISKANN_PRELOAD_REASON = (
            "libaio is not available on this host; DiskAnn cannot run. "
            "Install libaio1 (or libaio1t64 on Ubuntu 24.04+) and retry."
        )
        return _DISKANN_PRELOAD_REASON

    status = zvec.load_diskann_plugin()
    if status != zvec.DISKANN_PLUGIN_OK:
        _DISKANN_PRELOAD_REASON = (
            f"Failed to load DiskAnn plugin (status={status}); "
            "check that libzvec_diskann_plugin.so is installed alongside "
            "_zvec.so in the Python site-packages directory."
        )
        return _DISKANN_PRELOAD_REASON

    _DISKANN_PRELOAD_REASON = None
    return None


from zvec import (
    CollectionOption,
    InvertIndexParam,
    HnswIndexParam,
    FlatIndexParam,
    IVFIndexParam,
    FieldSchema,
    VectorSchema,
    CollectionSchema,
    Collection,
    Doc,
    Query,
)

from support_helper import *


@pytest.fixture(scope="session")
def basic_schema(collection_name="test_collection") -> CollectionSchema:
    return CollectionSchema(
        name=collection_name if len(collection_name) > 0 else "test_collection",
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "name", DataType.STRING, nullable=False, index_param=InvertIndexParam()
            ),
            FieldSchema("weight", DataType.FLOAT, nullable=True),
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=HnswIndexParam(),
            ),
            VectorSchema(
                "sparse", DataType.SPARSE_VECTOR_FP32, index_param=HnswIndexParam()
            ),
        ],
    )


@pytest.fixture(scope="session")
def full_schema(
    nullable: bool = False,
    has_index: bool = False,
) -> CollectionSchema:
    scalar_index_param = None
    vector_index_param = None
    if has_index:
        scalar_index_param = InvertIndexParam(enable_range_optimization=True)
        vector_index_param = HnswIndexParam()

    fields = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        fields.append(
            FieldSchema(
                v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    vetors = []
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        vetors.append(
            VectorSchema(
                v,
                k,
                dimension=DEFAULT_VECTOR_DIMENSION,
                index_param=vector_index_param,
            )
        )

    return CollectionSchema(
        name="full_collection",
        fields=fields,
        vectors=vetors,
    )


@pytest.fixture(scope="function")
def full_schema_new(request) -> CollectionSchema:
    if hasattr(request, "param"):
        nullable, has_index, vector_index = request.param
    else:
        nullable, has_index, vector_index = True, False, HnswIndexParam()

    # Skip DiskAnn tests on unsupported platforms or when the runtime cannot
    # be brought up (missing libaio, plugin .so not installed, etc.).
    from zvec.model.param import DiskAnnIndexParam

    if isinstance(vector_index, DiskAnnIndexParam):
        skip_reason = _ensure_diskann_runtime_or_reason()
        if skip_reason is not None:
            pytest.skip(skip_reason)

    scalar_index_param = None
    vector_index_param = None
    if has_index:
        scalar_index_param = InvertIndexParam(enable_range_optimization=True)
        vector_index_param = vector_index

    fields = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        fields.append(
            FieldSchema(
                v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    vectors = []

    if vector_index_param in [
        HnswIndexParam(),
        FlatIndexParam(),
        HnswIndexParam(
            metric_type=MetricType.IP,
            m=16,
            ef_construction=100,
        ),
        FlatIndexParam(
            metric_type=MetricType.IP,
        ),
    ]:
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            vectors.append(
                VectorSchema(
                    v,
                    k,
                    dimension=DEFAULT_VECTOR_DIMENSION,
                    index_param=vector_index_param,
                )
            )
    elif vector_index_param in [
        IVFIndexParam(),
        IVFIndexParam(
            metric_type=MetricType.IP,
            n_list=100,
            n_iters=10,
            use_soar=False,
        ),
        IVFIndexParam(
            metric_type=MetricType.L2,
            n_list=200,
            n_iters=20,
            use_soar=True,
        ),
        (
            IVFIndexParam(
                metric_type=MetricType.COSINE,
                n_list=150,
                n_iters=15,
                use_soar=False,
            )
        ),
        (
            HnswIndexParam(
                metric_type=MetricType.COSINE,
                m=24,
                ef_construction=150,
            )
        ),
        (
            HnswIndexParam(
                metric_type=MetricType.L2,
                m=32,
                ef_construction=200,
            )
        ),
        (
            FlatIndexParam(
                metric_type=MetricType.COSINE,
            )
        ),
        (
            FlatIndexParam(
                metric_type=MetricType.L2,
            )
        ),
    ]:
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            if v in ["vector_fp16_field", "vector_fp32_field"]:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=DEFAULT_VECTOR_DIMENSION,
                        index_param=vector_index_param,
                    )
                )
            elif v in ["vector_int8_field"] and vector_index_param in [
                IVFIndexParam(
                    metric_type=MetricType.L2,
                    n_list=200,
                    n_iters=20,
                    use_soar=True,
                ),
                (
                    HnswIndexParam(
                        metric_type=MetricType.L2,
                        m=32,
                        ef_construction=200,
                    )
                ),
                (
                    FlatIndexParam(
                        metric_type=MetricType.L2,
                    )
                ),
            ]:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=DEFAULT_VECTOR_DIMENSION,
                        index_param=vector_index_param,
                    )
                )
            else:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=DEFAULT_VECTOR_DIMENSION,
                        index_param=HnswIndexParam(),
                    )
                )
    else:
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            if v in ["vector_fp16_field", "vector_fp32_field"]:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=DEFAULT_VECTOR_DIMENSION,
                        index_param=vector_index_param,
                    )
                )
            else:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=DEFAULT_VECTOR_DIMENSION,
                        index_param=HnswIndexParam(),
                    )
                )

    return CollectionSchema(
        name="full_collection_new",
        fields=fields,
        vectors=vectors,
    )


@pytest.fixture(scope="function")
def full_schema_ivf(request) -> CollectionSchema:
    if hasattr(request, "param"):
        nullable, has_index, vector_index = request.param
    else:
        nullable, has_index, vector_index = True, False, IVFIndexParam()

    scalar_index_param = None
    vector_index_param = None
    if has_index:
        scalar_index_param = InvertIndexParam(enable_range_optimization=True)
        vector_index_param = vector_index

    fields = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        fields.append(
            FieldSchema(
                v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    vectors = []
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        if v in ["vector_fp16_field", "vector_fp32_field"]:
            vectors.append(
                VectorSchema(
                    v,
                    k,
                    dimension=DEFAULT_VECTOR_DIMENSION,
                    index_param=vector_index_param,
                )
            )

    return CollectionSchema(
        name="full_collection_ivf",
        fields=fields,
        vectors=vectors,
    )


@pytest.fixture(scope="function")
def full_schema_1024(request) -> CollectionSchema:
    if hasattr(request, "param"):
        nullable, has_index, vector_index = request.param
    else:
        nullable, has_index, vector_index = True, False, HnswIndexParam()

    scalar_index_param = None
    vector_index_param = None
    if has_index:
        scalar_index_param = InvertIndexParam(enable_range_optimization=True)
        vector_index_param = vector_index

    fields = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        fields.append(
            FieldSchema(
                v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    vectors = []

    if vector_index_param in [
        HnswIndexParam(),
        FlatIndexParam(),
        HnswIndexParam(
            metric_type=MetricType.IP,
            m=16,
            ef_construction=100,
        ),
        FlatIndexParam(
            metric_type=MetricType.IP,
        ),
    ]:
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            vectors.append(
                VectorSchema(
                    v,
                    k,
                    dimension=VECTOR_DIMENSION_1024,
                    index_param=vector_index_param,
                )
            )
    elif vector_index_param in [
        IVFIndexParam(),
        IVFIndexParam(
            metric_type=MetricType.IP,
            n_list=100,
            n_iters=10,
            use_soar=False,
        ),
        IVFIndexParam(
            metric_type=MetricType.L2,
            n_list=200,
            n_iters=20,
            use_soar=True,
        ),
        IVFIndexParam(
            metric_type=MetricType.COSINE,
            n_list=150,
            n_iters=15,
            use_soar=False,
        ),
    ]:
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            if v in ["vector_fp16_field", "vector_fp32_field"]:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=VECTOR_DIMENSION_1024,
                        index_param=vector_index_param,
                    )
                )
            elif v in ["vector_int8_field"] and vector_index_param in [
                IVFIndexParam(
                    metric_type=MetricType.L2,
                    n_list=200,
                    n_iters=20,
                    use_soar=True,
                ),
                IVFIndexParam(
                    metric_type=MetricType.COSINE,
                    n_list=150,
                    n_iters=15,
                    use_soar=False,
                ),
            ]:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=DVECTOR_DIMENSION_1024,
                        index_param=vector_index_param,
                    )
                )
            else:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=VECTOR_DIMENSION_1024,
                        index_param=HnswIndexParam(),
                    )
                )
    else:
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            if v in ["vector_fp16_field", "vector_fp32_field", "vector_int8_field"]:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=VECTOR_DIMENSION_1024,
                        index_param=vector_index_param,
                    )
                )
            else:
                vectors.append(
                    VectorSchema(
                        v,
                        k,
                        dimension=VECTOR_DIMENSION_1024,
                        index_param=HnswIndexParam(),
                    )
                )

    return CollectionSchema(
        name="full_collection_new",
        fields=fields,
        vectors=vectors,
    )


@pytest.fixture(scope="function")
def single_vector_schema(
    data_type: DataType,
) -> CollectionSchema:
    vector_schema = [
        VectorSchema(
            DEFAULT_VECTOR_FIELD_NAME[data_type],
            data_type,
            DEFAULT_VECTOR_DIMENSION,
        )
    ]

    return CollectionSchema(
        name="full_collection",
        vectors=vector_schema,
    )


@pytest.fixture(scope="function")
def single_vector_schema_with_index_param(
    data_type: DataType, index_param
) -> CollectionSchema:
    vector_schema = [
        VectorSchema(
            DEFAULT_VECTOR_FIELD_NAME[data_type],
            data_type,
            DEFAULT_VECTOR_DIMENSION,
            index_param,
        )
    ]

    return CollectionSchema(
        name="full_collection",
        vectors=vector_schema,
    )


def create_collection_fixture(
    collection_temp_dir, schema: CollectionSchema, collection_option: CollectionOption
) -> Generator[Any, Any, Collection]:
    """Common helper function to create and manage collection fixtures."""
    coll = zvec.create_and_open(
        path=str(collection_temp_dir),
        schema=schema,
        option=collection_option,
    )

    assert coll is not None, "Failed to create and open collection"
    assert coll.path == str(collection_temp_dir)
    assert coll.schema.name == schema.name
    assert list(coll.schema.fields) == list(schema.fields)
    assert list(coll.schema.vectors) == list(schema.vectors)
    assert coll.option.read_only == collection_option.read_only
    assert coll.option.enable_mmap == collection_option.enable_mmap

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                logging.warning(f"Warning: failed to destroy collection: {e}")


@pytest.fixture(scope="function")
def basic_collection(
    collection_temp_dir, basic_schema, collection_option
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, basic_schema, collection_option
    )


@pytest.fixture(scope="function")
def collection_option():
    return CollectionOption(read_only=False, enable_mmap=True)


@pytest.fixture(scope="function")
def collection_temp_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("zvec")
    collection_path = temp_dir / "test_collection_path"
    return str(collection_path)


@pytest.fixture(scope="function")
def full_collection(
    collection_temp_dir,
    full_schema,
    collection_option,
    nullable: bool = True,
    has_index: bool = False,
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, full_schema, collection_option
    )


@pytest.fixture(scope="function")
def full_collection_new(
    collection_temp_dir, full_schema_new, collection_option
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, full_schema_new, collection_option
    )


@pytest.fixture(scope="function")
def full_collection_ivf(
    collection_temp_dir, full_schema_ivf, collection_option
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, full_schema_ivf, collection_option
    )


@pytest.fixture(scope="function")
def full_collection_1024(
    collection_temp_dir, full_schema_1024, collection_option
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, full_schema_1024, collection_option
    )


@pytest.fixture
def sample_field_list(nullable: bool = True, scalar_index_param=None, name_prefix=""):
    field_list = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        field_list.append(
            FieldSchema(
                f"{name_prefix}_{v}" if len(name_prefix) > 0 else v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    return field_list


@pytest.fixture
def sample_vector_list(vector_index_param=None, name_prefix=""):
    vector_list = []
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        vector_list.append(
            VectorSchema(
                f"{name_prefix}_{v}" if len(name_prefix) > 0 else v,
                k,
                dimension=DEFAULT_VECTOR_DIMENSION,
                index_param=vector_index_param,
            )
        )
    return vector_list
