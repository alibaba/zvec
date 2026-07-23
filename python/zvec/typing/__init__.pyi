"""
This module contains the basic data types of Zvec
"""

from __future__ import annotations

import typing

__all__: list[str] = [
    "DataType",
    "IOBackendType",
    "IndexType",
    "MetricType",
    "QuantizeType",
    "Status",
    "StatusCode",
]

class DataType:
    """

    Enumeration of supported data types in Zvec.

    Includes scalar types, dense/sparse vector types, and array types.

    Examples:
        >>> import zvec
        >>> print(zvec.DataType.FLOAT)
        DataType.FLOAT
        >>> print(zvec.DataType.VECTOR_FP32)
        DataType.VECTOR_FP32


    Members:

      STRING

      BOOL

      INT32

      INT64

      FLOAT

      DOUBLE

      UINT32

      UINT64

      VECTOR_FP16

      VECTOR_FP32

      VECTOR_FP64

      VECTOR_INT8

      SPARSE_VECTOR_FP32

      SPARSE_VECTOR_FP16

      ARRAY_STRING

      ARRAY_INT32

      ARRAY_INT64

      ARRAY_FLOAT

      ARRAY_DOUBLE

      ARRAY_BOOL

      ARRAY_UINT32

      ARRAY_UINT64
    """

    ARRAY_BOOL: typing.ClassVar[DataType]  # value = <DataType.ARRAY_BOOL: 42>
    ARRAY_DOUBLE: typing.ClassVar[DataType]  # value = <DataType.ARRAY_DOUBLE: 48>
    ARRAY_FLOAT: typing.ClassVar[DataType]  # value = <DataType.ARRAY_FLOAT: 47>
    ARRAY_INT32: typing.ClassVar[DataType]  # value = <DataType.ARRAY_INT32: 43>
    ARRAY_INT64: typing.ClassVar[DataType]  # value = <DataType.ARRAY_INT64: 44>
    ARRAY_STRING: typing.ClassVar[DataType]  # value = <DataType.ARRAY_STRING: 41>
    ARRAY_UINT32: typing.ClassVar[DataType]  # value = <DataType.ARRAY_UINT32: 45>
    ARRAY_UINT64: typing.ClassVar[DataType]  # value = <DataType.ARRAY_UINT64: 46>
    BOOL: typing.ClassVar[DataType]  # value = <DataType.BOOL: 3>
    DOUBLE: typing.ClassVar[DataType]  # value = <DataType.DOUBLE: 9>
    FLOAT: typing.ClassVar[DataType]  # value = <DataType.FLOAT: 8>
    INT32: typing.ClassVar[DataType]  # value = <DataType.INT32: 4>
    INT64: typing.ClassVar[DataType]  # value = <DataType.INT64: 5>
    SPARSE_VECTOR_FP16: typing.ClassVar[
        DataType
    ]  # value = <DataType.SPARSE_VECTOR_FP16: 30>
    SPARSE_VECTOR_FP32: typing.ClassVar[
        DataType
    ]  # value = <DataType.SPARSE_VECTOR_FP32: 31>
    STRING: typing.ClassVar[DataType]  # value = <DataType.STRING: 2>
    UINT32: typing.ClassVar[DataType]  # value = <DataType.UINT32: 6>
    UINT64: typing.ClassVar[DataType]  # value = <DataType.UINT64: 7>
    VECTOR_FP16: typing.ClassVar[DataType]  # value = <DataType.VECTOR_FP16: 22>
    VECTOR_FP32: typing.ClassVar[DataType]  # value = <DataType.VECTOR_FP32: 23>
    VECTOR_FP64: typing.ClassVar[DataType]  # value = <DataType.VECTOR_FP64: 24>
    VECTOR_INT8: typing.ClassVar[DataType]  # value = <DataType.VECTOR_INT8: 26>
    __members__: typing.ClassVar[
        dict[str, DataType]
    ]  # value = {'STRING': <DataType.STRING: 2>, 'BOOL': <DataType.BOOL: 3>, 'INT32': <DataType.INT32: 4>, 'INT64': <DataType.INT64: 5>, 'FLOAT': <DataType.FLOAT: 8>, 'DOUBLE': <DataType.DOUBLE: 9>, 'UINT32': <DataType.UINT32: 6>, 'UINT64': <DataType.UINT64: 7>, 'VECTOR_FP16': <DataType.VECTOR_FP16: 22>, 'VECTOR_FP32': <DataType.VECTOR_FP32: 23>, 'VECTOR_FP64': <DataType.VECTOR_FP64: 24>, 'VECTOR_INT8': <DataType.VECTOR_INT8: 26>, 'SPARSE_VECTOR_FP32': <DataType.SPARSE_VECTOR_FP32: 31>, 'SPARSE_VECTOR_FP16': <DataType.SPARSE_VECTOR_FP16: 30>, 'ARRAY_STRING': <DataType.ARRAY_STRING: 41>, 'ARRAY_INT32': <DataType.ARRAY_INT32: 43>, 'ARRAY_INT64': <DataType.ARRAY_INT64: 44>, 'ARRAY_FLOAT': <DataType.ARRAY_FLOAT: 47>, 'ARRAY_DOUBLE': <DataType.ARRAY_DOUBLE: 48>, 'ARRAY_BOOL': <DataType.ARRAY_BOOL: 42>, 'ARRAY_UINT32': <DataType.ARRAY_UINT32: 45>, 'ARRAY_UINT64': <DataType.ARRAY_UINT64: 46>}

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class IOBackendType:
    """

    Enumeration of supported I/O backend types for DiskAnn async disk reads.

    - PREAD: Synchronous pread() — no async I/O.
    - LIBAIO: libaio loaded at runtime via dlopen().

    Examples:
        >>> from zvec.typing import IOBackendType
        >>> print(IOBackendType.LIBAIO)
        IOBackendType.LIBAIO


    Members:

      PREAD

      LIBAIO
    """

    LIBAIO: typing.ClassVar[IOBackendType]  # value = <IOBackendType.LIBAIO: 1>
    PREAD: typing.ClassVar[IOBackendType]  # value = <IOBackendType.PREAD: 0>
    __members__: typing.ClassVar[
        dict[str, IOBackendType]
    ]  # value = {'PREAD': <IOBackendType.PREAD: 0>, 'LIBAIO': <IOBackendType.LIBAIO: 1>}

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class IndexType:
    """

    Enumeration of supported index types in Zvec.

    Examples:
        >>> import zvec
        >>> print(zvec.IndexType.HNSW)
        IndexType.HNSW


    Members:

      UNDEFINED

      HNSW

      IVF

      FLAT

      HNSW_RABITQ

      DISKANN

      VAMANA

      INVERT

      FTS
    """

    DISKANN: typing.ClassVar[IndexType]  # value = <IndexType.DISKANN: 5>
    FLAT: typing.ClassVar[IndexType]  # value = <IndexType.FLAT: 3>
    FTS: typing.ClassVar[IndexType]  # value = <IndexType.FTS: 11>
    HNSW: typing.ClassVar[IndexType]  # value = <IndexType.HNSW: 1>
    HNSW_RABITQ: typing.ClassVar[IndexType]  # value = <IndexType.HNSW_RABITQ: 4>
    INVERT: typing.ClassVar[IndexType]  # value = <IndexType.INVERT: 10>
    IVF: typing.ClassVar[IndexType]  # value = <IndexType.IVF: 2>
    UNDEFINED: typing.ClassVar[IndexType]  # value = <IndexType.UNDEFINED: 0>
    VAMANA: typing.ClassVar[IndexType]  # value = <IndexType.VAMANA: 6>
    __members__: typing.ClassVar[
        dict[str, IndexType]
    ]  # value = {'UNDEFINED': <IndexType.UNDEFINED: 0>, 'HNSW': <IndexType.HNSW: 1>, 'IVF': <IndexType.IVF: 2>, 'FLAT': <IndexType.FLAT: 3>, 'HNSW_RABITQ': <IndexType.HNSW_RABITQ: 4>, 'DISKANN': <IndexType.DISKANN: 5>, 'VAMANA': <IndexType.VAMANA: 6>, 'INVERT': <IndexType.INVERT: 10>, 'FTS': <IndexType.FTS: 11>}

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class MetricType:
    """

    Enumeration of supported distance/similarity metrics.

    - COSINE: Cosine similarity.
    - IP: Inner product (dot product).
    - L2: Euclidean distance (L2 norm).

    Examples:
        >>> import zvec
        >>> print(zvec.MetricType.COSINE)
        MetricType.COSINE


    Members:

      COSINE

      IP

      L2
    """

    COSINE: typing.ClassVar[MetricType]  # value = <MetricType.COSINE: 3>
    IP: typing.ClassVar[MetricType]  # value = <MetricType.IP: 2>
    L2: typing.ClassVar[MetricType]  # value = <MetricType.L2: 1>
    __members__: typing.ClassVar[
        dict[str, MetricType]
    ]  # value = {'COSINE': <MetricType.COSINE: 3>, 'IP': <MetricType.IP: 2>, 'L2': <MetricType.L2: 1>}

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class QuantizeType:
    """

    Enumeration of supported quantization types for vector compression.

    Examples:
        >>> import zvec
        >>> print(zvec.QuantizeType.INT8)
        QuantizeType.INT8


    Members:

      UNDEFINED

      FP16

      INT8

      INT4

      RABITQ
    """

    FP16: typing.ClassVar[QuantizeType]  # value = <QuantizeType.FP16: 1>
    INT4: typing.ClassVar[QuantizeType]  # value = <QuantizeType.INT4: 3>
    INT8: typing.ClassVar[QuantizeType]  # value = <QuantizeType.INT8: 2>
    RABITQ: typing.ClassVar[QuantizeType]  # value = <QuantizeType.RABITQ: 4>
    UNDEFINED: typing.ClassVar[QuantizeType]  # value = <QuantizeType.UNDEFINED: 0>
    __members__: typing.ClassVar[
        dict[str, QuantizeType]
    ]  # value = {'UNDEFINED': <QuantizeType.UNDEFINED: 0>, 'FP16': <QuantizeType.FP16: 1>, 'INT8': <QuantizeType.INT8: 2>, 'INT4': <QuantizeType.INT4: 3>, 'RABITQ': <QuantizeType.RABITQ: 4>}

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Status:
    """

    Represents the outcome of a Zvec operation.

    A `Status` object is either OK (success) or carries an error code and message.

    Examples:
        >>> from zvec.typing import Status, StatusCode
        >>> s = Status()
        >>> print(s.ok())
        True
        >>> s = Status(StatusCode.INVALID_ARGUMENT, "Field not found")
        >>> print(s.code() == StatusCode.INVALID_ARGUMENT)
        True
        >>> print(s.message())
        Field not found
    """

    __hash__: typing.ClassVar[None] = None

    @staticmethod
    def AlreadyExists(message: str) -> Status: ...
    @staticmethod
    def InternalError(message: str) -> Status: ...
    @staticmethod
    def InvalidArgument(message: str) -> Status: ...
    @staticmethod
    def NotFound(message: str) -> Status: ...
    @staticmethod
    def OK() -> Status:
        """
        Create an OK status.
        """

    @staticmethod
    def PermissionDenied(message: str) -> Status: ...
    def __eq__(self, arg0: Status) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, code: StatusCode, message: str = "") -> None:
        """
        Construct a status with the given code and optional message.

        Args:
            code (StatusCode): The status code.
            message (str, optional): Error message. Defaults to empty string.
        """

    def __ne__(self, arg0: Status) -> bool: ...
    def __repr__(self) -> str: ...
    def code(self) -> StatusCode:
        """
        StatusCode: Returns the status code.
        """

    def message(self) -> str:
        """
        str: Returns the error message (may be empty).
        """

    def ok(self) -> bool:
        """
        bool: Returns True if the status is OK.
        """

class StatusCode:
    """

    Enumeration of possible status codes for Zvec operations.

    Used by the `Status` class to indicate success or failure reason.


    Members:

      OK

      NOT_FOUND

      ALREADY_EXISTS

      INVALID_ARGUMENT

      PERMISSION_DENIED

      FAILED_PRECONDITION

      RESOURCE_EXHAUSTED

      UNAVAILABLE

      INTERNAL_ERROR

      NOT_SUPPORTED

      UNKNOWN
    """

    ALREADY_EXISTS: typing.ClassVar[
        StatusCode
    ]  # value = <StatusCode.ALREADY_EXISTS: 2>
    FAILED_PRECONDITION: typing.ClassVar[
        StatusCode
    ]  # value = <StatusCode.FAILED_PRECONDITION: 5>
    INTERNAL_ERROR: typing.ClassVar[
        StatusCode
    ]  # value = <StatusCode.INTERNAL_ERROR: 8>
    INVALID_ARGUMENT: typing.ClassVar[
        StatusCode
    ]  # value = <StatusCode.INVALID_ARGUMENT: 3>
    NOT_FOUND: typing.ClassVar[StatusCode]  # value = <StatusCode.NOT_FOUND: 1>
    NOT_SUPPORTED: typing.ClassVar[StatusCode]  # value = <StatusCode.NOT_SUPPORTED: 9>
    OK: typing.ClassVar[StatusCode]  # value = <StatusCode.OK: 0>
    PERMISSION_DENIED: typing.ClassVar[
        StatusCode
    ]  # value = <StatusCode.PERMISSION_DENIED: 4>
    RESOURCE_EXHAUSTED: typing.ClassVar[
        StatusCode
    ]  # value = <StatusCode.RESOURCE_EXHAUSTED: 6>
    UNAVAILABLE: typing.ClassVar[StatusCode]  # value = <StatusCode.UNAVAILABLE: 7>
    UNKNOWN: typing.ClassVar[StatusCode]  # value = <StatusCode.UNKNOWN: 10>
    __members__: typing.ClassVar[
        dict[str, StatusCode]
    ]  # value = {'OK': <StatusCode.OK: 0>, 'NOT_FOUND': <StatusCode.NOT_FOUND: 1>, 'ALREADY_EXISTS': <StatusCode.ALREADY_EXISTS: 2>, 'INVALID_ARGUMENT': <StatusCode.INVALID_ARGUMENT: 3>, 'PERMISSION_DENIED': <StatusCode.PERMISSION_DENIED: 4>, 'FAILED_PRECONDITION': <StatusCode.FAILED_PRECONDITION: 5>, 'RESOURCE_EXHAUSTED': <StatusCode.RESOURCE_EXHAUSTED: 6>, 'UNAVAILABLE': <StatusCode.UNAVAILABLE: 7>, 'INTERNAL_ERROR': <StatusCode.INTERNAL_ERROR: 8>, 'NOT_SUPPORTED': <StatusCode.NOT_SUPPORTED: 9>, 'UNKNOWN': <StatusCode.UNKNOWN: 10>}

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
