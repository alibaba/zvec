"""
This module contains the params of Zvec
"""

from __future__ import annotations

import collections
import typing

import _zvec.typing

__all__: list[str] = [
    "AddColumnOption",
    "AlterColumnOption",
    "CollectionOption",
    "FlatIndexParam",
    "HnswIndexParam",
    "HnswQueryParam",
    "HnswRabitqIndexParam",
    "HnswRabitqQueryParam",
    "IVFIndexParam",
    "IVFQueryParam",
    "IndexOption",
    "IndexParam",
    "InvertIndexParam",
    "OptimizeOption",
    "QueryParam",
    "SegmentOption",
    "VectorIndexParam",
]

class AddColumnOption:
    """

    Options for adding a new column to a collection.

    Attributes:
        concurrency (int): Number of threads to use when backfilling data
            for the new column. If 0, auto-detect is used. Default is 0.

    Examples:
        >>> opt = AddColumnOption(concurrency=1)
        >>> print(opt.concurrency)
        1
    """

    def __getstate__(self) -> tuple: ...
    def __init__(self, concurrency: typing.SupportsInt = 0) -> None:
        """
        Constructs an AddColumnOption instance.

        Args:
            concurrency (int, optional): Number of threads for data backfill.
                0 means auto-detect. Defaults to 0.
        """

    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def concurrency(self) -> int:
        """
        int: Number of threads used when adding a column (0 = auto).
        """

class AlterColumnOption:
    """

    Options for altering an existing column (e.g., changing index settings).

    Attributes:
        concurrency (int): Number of threads to use during the alteration process.
            If 0, the system will choose an optimal value automatically.
            Default is 0.

    Examples:
        >>> opt = AlterColumnOption(concurrency=1)
        >>> print(opt.concurrency)
        1
    """

    def __getstate__(self) -> tuple: ...
    def __init__(self, concurrency: typing.SupportsInt = 0) -> None:
        """
        Constructs an AlterColumnOption instance.

        Args:
            concurrency (int, optional): Number of threads for column alteration.
                0 means auto-detect. Defaults to 0.
        """

    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def concurrency(self) -> int:
        """
        int: Number of threads used when altering a column (0 = auto).
        """

class CollectionOption:
    """

    Options for opening or creating a collection.

    Attributes:
        read_only (bool): Whether the collection is opened in read-only mode.
            Default is False.
        enable_mmap (bool): Whether to use memory-mapped I/O for data files.
            Default is True.

    Examples:
        >>> opt = CollectionOption(read_only=True, enable_mmap=False)
        >>> print(opt.read_only)
        True
    """

    def __getstate__(self) -> tuple: ...
    def __init__(self, read_only: bool = False, enable_mmap: bool = True) -> None:
        """
        Constructs a CollectionOption instance.

        Args:
            read_only (bool, optional): Open collection in read-only mode.
                Defaults to False.
            enable_mmap (bool, optional): Enable memory-mapped I/O.
                Defaults to True.
        """

    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def enable_mmap(self) -> bool: ...
    @property
    def read_only(self) -> bool: ...

class FlatIndexParam(VectorIndexParam):
    """

    Parameters for configuring a flat (brute-force) index.

    A flat index performs exact nearest neighbor search by comparing the query
    vector against all vectors in the collection. It is simple, accurate, and
    suitable for small to medium datasets or as a baseline.

    Attributes:
        metric_type (MetricType): Distance metric used for similarity computation.
            Default is ``MetricType.IP`` (inner product).
        quantize_type (QuantizeType): Optional quantization type for vector
            compression (e.g., FP16, INT8). Use ``QuantizeType.UNDEFINED`` to
            disable quantization. Default is ``QuantizeType.UNDEFINED``.

    Examples:
        >>> from zvec.typing import MetricType, QuantizeType
        >>> params = FlatIndexParam(
        ...     metric_type=MetricType.L2,
        ...     quantize_type=QuantizeType.FP16
        ... )
        >>> print(params)
        {'metric_type': 'L2', 'quantize_type': 'FP16'}
    """

    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        metric_type: _zvec.typing.MetricType = ...,
        quantize_type: _zvec.typing.QuantizeType = ...,
    ) -> None:
        """
        Constructs a FlatIndexParam instance.

        Args:
            metric_type (MetricType, optional): Distance metric. Defaults to MetricType.IP.
            quantize_type (QuantizeType, optional): Vector quantization type.
                Defaults to QuantizeType.UNDEFINED (no quantization).
        """

    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def to_dict(self) -> dict:
        """
        Convert to dictionary with all fields
        """

class HnswIndexParam(VectorIndexParam):
    """

    Parameters for configuring an HNSW (Hierarchical Navigable Small World) index.

    HNSW is a graph-based approximate nearest neighbor search index. This class
    encapsulates its construction hyperparameters.

    Attributes:
        metric_type (MetricType): Distance metric used for similarity computation.
            Default is ``MetricType.IP`` (inner product).
        m (int): Number of bi-directional links created for every new element
            during construction. Higher values improve accuracy but increase
            memory usage and construction time. Default is 50.
        ef_construction (int): Size of the dynamic candidate list for nearest
            neighbors during index construction. Larger values yield better
            graph quality at the cost of slower build time. Default is 500.
        quantize_type (QuantizeType): Optional quantization type for vector
            compression (e.g., FP16, INT8). Default is `QuantizeType.UNDEFINED` to
            disable quantization.

    Examples:
        >>> from zvec.typing import MetricType, QuantizeType
        >>> params = HnswIndexParam(
        ...     metric_type=MetricType.COSINE,
        ...     m=16,
        ...     ef_construction=200,
        ...     quantize_type=QuantizeType.INT8
        ... )
        >>> print(params)
        {'metric_type': 'IP', 'm': 16, 'ef_construction': 200, 'quantize_type': 'INT8'}
    """

    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        metric_type: _zvec.typing.MetricType = ...,
        m: typing.SupportsInt = 50,
        ef_construction: typing.SupportsInt = 500,
        quantize_type: _zvec.typing.QuantizeType = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def to_dict(self) -> dict:
        """
        Convert to dictionary with all fields
        """

    @property
    def ef_construction(self) -> int:
        """
        int: Candidate list size during index construction.
        """

    @property
    def m(self) -> int:
        """
        int: Maximum number of neighbors per node in upper layers.
        """

class HnswQueryParam(QueryParam):
    """

    Query parameters for HNSW (Hierarchical Navigable Small World) index.

    Controls the trade-off between search speed and accuracy via the `ef` parameter.

    Attributes:
        type (IndexType): Always ``IndexType.HNSW``.
        ef (int): Size of the dynamic candidate list during search.
            Larger values improve recall but slow down search.
            Default is 300.
        radius (float): Search radius for range queries. Default is 0.0.
        is_linear (bool): Force linear search. Default is False.
        is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.

    Examples:
        >>> params = HnswQueryParam(ef=300)
        >>> print(params.ef)
        300
        >>> print(params.to_dict() if hasattr(params, 'to_dict') else params)
        {"type":"HNSW", "ef":300}
    """
    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        ef: typing.SupportsInt = 300,
        radius: typing.SupportsFloat = 0.0,
        is_linear: bool = False,
        is_using_refiner: bool = False,
    ) -> None:
        """
        Constructs an HnswQueryParam instance.

        Args:
            ef (int, optional): Search-time candidate list size.
                Higher values improve accuracy. Defaults to 300.
            radius (float, optional): Search radius for range queries. Default is 0.0.
            is_linear (bool, optional): Force linear search. Default is False.
            is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.
        """
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def ef(self) -> int:
        """
        int: Size of the dynamic candidate list during HNSW search.
        """

class HnswRabitqIndexParam(VectorIndexParam):
    """

    Parameters for configuring an HNSW (Hierarchical Navigable Small World) index with RabitQ quantization.

    HNSW is a graph-based approximate nearest neighbor search index. RabitQ is a
    quantization method that provides high compression with minimal accuracy loss.

    Attributes:
        metric_type (MetricType): Distance metric used for similarity computation.
            Default is ``MetricType.IP`` (inner product).
        total_bits (int): Total bits for RabitQ quantization. Default is 7.
        num_clusters (int): Number of clusters for RabitQ. Default is 16.
        m (int): Number of bi-directional links created for every new element
            during construction. Higher values improve accuracy but increase
            memory usage and construction time. Default is 50.
        ef_construction (int): Size of the dynamic candidate list for nearest
            neighbors during index construction. Larger values yield better
            graph quality at the cost of slower build time. Default is 500.
        sample_count (int): Sample count for RabitQ training. Default is 0.

    Examples:
        >>> from zvec.typing import MetricType
        >>> params = HnswRabitqIndexParam(
        ...     metric_type=MetricType.COSINE,
        ...     total_bits=8,
        ...     num_clusters=256,
        ...     m=16,
        ...     ef_construction=200,
        ...     sample_count=10000
        ... )
        >>> print(params)
        {'metric_type': 'COSINE', 'total_bits': 8, 'num_clusters': 256, 'm': 16, 'ef_construction': 200, 'sample_count': 10000}
    """

    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        metric_type: _zvec.typing.MetricType = ...,
        total_bits: typing.SupportsInt = 7,
        num_clusters: typing.SupportsInt = 16,
        m: typing.SupportsInt = 50,
        ef_construction: typing.SupportsInt = 500,
        sample_count: typing.SupportsInt = 0,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def to_dict(self) -> dict:
        """
        Convert to dictionary with all fields
        """

    @property
    def ef_construction(self) -> int:
        """
        int: Candidate list size during index construction.
        """

    @property
    def m(self) -> int:
        """
        int: Maximum number of neighbors per node.
        """

    @property
    def total_bits(self) -> int:
        """
        int: Total bits for RabitQ quantization.
        """

    @property
    def num_clusters(self) -> int:
        """
        int: Number of clusters for RabitQ.
        """

    @property
    def sample_count(self) -> int:
        """
        int: Sample count for RabitQ training.
        """

class HnswRabitqQueryParam(QueryParam):
    """

    Query parameters for HNSW index with RabitQ quantization.

    Controls the trade-off between search speed and accuracy via the `ef` parameter.

    Attributes:
        type (IndexType): Always ``IndexType.HNSW_RABITQ``.
        ef (int): Size of the dynamic candidate list during search.
            Larger values improve recall but slow down search.
            Default is 300.
        radius (float): Search radius for range queries. Default is 0.0.
        is_linear (bool): Force linear search. Default is False.
        is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.

    Examples:
        >>> params = HnswRabitqQueryParam(ef=300)
        >>> print(params.ef)
        300
    """
    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        ef: typing.SupportsInt = 300,
        radius: typing.SupportsFloat = 0.0,
        is_linear: bool = False,
        is_using_refiner: bool = False,
    ) -> None:
        """
        Constructs an HnswRabitqQueryParam instance.

        Args:
            ef (int, optional): Search-time candidate list size.
                Higher values improve accuracy. Defaults to 300.
            radius (float, optional): Search radius for range queries. Default is 0.0.
            is_linear (bool, optional): Force linear search. Default is False.
            is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.
        """
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def ef(self) -> int:
        """
        int: Size of the dynamic candidate list during HNSW search.
        """

class IVFIndexParam(VectorIndexParam):
    """

    Parameters for configuring an IVF (Inverted File Index) index.

    IVF partitions the vector space into clusters (inverted lists). At query time,
    only a subset of clusters is searched, providing a trade-off between speed
    and accuracy.

    Attributes:
        metric_type (MetricType): Distance metric used for similarity computation.
            Default is ``MetricType.IP`` (inner product).
        n_list (int): Number of clusters (inverted lists) to partition the dataset into.
            If set to 0, the system will auto-select a reasonable value based on data size.
            Default is 0 (auto).
        n_iters (int): Number of iterations for k-means clustering during index training.
            Higher values yield more stable centroids. Default is 10.
        use_soar (bool): Whether to enable SOAR (Scalable Optimized Adaptive Routing)
            for improved IVF search performance. Default is False.
        quantize_type (QuantizeType): Optional quantization type for vector
            compression (e.g., FP16, INT8). Default is ``QuantizeType.UNDEFINED``.

    Examples:
        >>> from zvec.typing import MetricType, QuantizeType
        >>> params = IVFIndexParam(
        ...     metric_type=MetricType.COSINE,
        ...     n_list=100,
        ...     n_iters=15,
        ...     use_soar=True,
        ...     quantize_type=QuantizeType.INT8
        ... )
        >>> print(params.n_list)
        100
    """

    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        metric_type: _zvec.typing.MetricType = ...,
        n_list: typing.SupportsInt = 0,
        n_iters: typing.SupportsInt = 10,
        use_soar: bool = False,
        quantize_type: _zvec.typing.QuantizeType = ...,
    ) -> None:
        """
        Constructs an IVFIndexParam instance.

        Args:
            metric_type (MetricType, optional): Distance metric. Defaults to MetricType.IP.
            n_list (int, optional): Number of inverted lists (clusters). Set to 0 for auto.
                Defaults to 0.
            n_iters (int, optional): Number of k-means iterations during training.
                Defaults to 10.
            use_soar (bool, optional): Enable SOAR optimization. Defaults to False.
            quantize_type (QuantizeType, optional): Vector quantization type.
                Defaults to QuantizeType.UNDEFINED.
        """

    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def to_dict(self) -> dict:
        """
        Convert to dictionary with all fields
        """

    @property
    def n_iters(self) -> int:
        """
        int: Number of k-means iterations during training.
        """

    @property
    def n_list(self) -> int:
        """
        int: Number of inverted lists (0 = auto).
        """

    @property
    def use_soar(self) -> bool:
        """
        bool: Whether SOAR optimization is enabled.
        """

class IVFQueryParam(QueryParam):
    """

    Query parameters for IVF (Inverted File Index) index.

    Controls how many inverted lists (`nprobe`) to visit during search.

    Attributes:
        type (IndexType): Always ``IndexType.IVF``.
        nprobe (int): Number of closest clusters (inverted lists) to search.
            Higher values improve recall but increase latency.
            Default is 10.
        radius (float): Search radius for range queries. Default is 0.0.
        is_linear (bool): Force linear search. Default is False.

    Examples:
        >>> params = IVFQueryParam(nprobe=20)
        >>> print(params.nprobe)
        20
    """
    def __getstate__(self) -> tuple: ...
    def __init__(self, nprobe: typing.SupportsInt = 10) -> None:
        """
        Constructs an IVFQueryParam instance.

        Args:
            nprobe (int, optional): Number of inverted lists to probe during search.
                Higher values improve accuracy. Defaults to 10.
        """
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def nprobe(self) -> int:
        """
        int: Number of inverted lists to search during IVF query.
        """

class IndexOption:
    """

    Options for creating an index.

    Attributes:
        concurrency (int): Number of threads to use during index creation.
            If 0, the system will choose an optimal value automatically.
            Default is 0.

    Examples:
        >>> opt = IndexOption(concurrency=4)
        >>> print(opt.concurrency)
        4
    """

    def __getstate__(self) -> tuple: ...
    def __init__(self, concurrency: typing.SupportsInt = 0) -> None:
        """
        Constructs an IndexOption instance.

        Args:
            concurrency (int, optional): Number of concurrent threads.
                0 means auto-detect. Defaults to 0.
        """

    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def concurrency(self) -> int:
        """
        int: Number of threads used for index creation (0 = auto).
        """

class IndexParam:
    """

    Base class for all index parameter configurations.

    This abstract base class defines the common interface for index types.
    It should not be instantiated directly; use derived classes instead.

    Attributes:
        type (IndexType): The type of the index (e.g., HNSW, FLAT, INVERT).
    """

    __hash__: typing.ClassVar[None] = None

    def __eq__(self, arg0: typing.Any) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def clone(self) -> IndexParam: ...
    def to_dict(self) -> dict:
        """
        Convert to dictionary with all fields
        """

    @property
    def type(self) -> _zvec.typing.IndexType:
        """
        IndexType: The type of the index.
        """

class InvertIndexParam(IndexParam):
    """

    Parameters for configuring an invert index.

    This class controls whether range query
    optimization is enabled for invert index structures.

    Attributes:
        type (IndexType): Always `IndexType.INVERTED`.
        enable_range_optimization (bool): Whether range optimization is enabled.
        enable_extended_wildcard (bool): Whether extended wildcard (suffix and infix) search is enabled.

    Examples:
        >>> params = InvertIndexParam(enable_range_optimization=True, enable_extended_wildcard=False)
        >>> print(params.enable_range_optimization)
        True
        >>> print(params.enable_extended_wildcard)
        False
        >>> config = params.to_dict()
        >>> print(config)
        {'enable_range_optimization': True, 'enable_extended_wildcard': False}
    """
    def __getstate__(self) -> tuple: ...
    def __init__(
        self,
        enable_range_optimization: bool = False,
        enable_extended_wildcard: bool = False,
    ) -> None:
        """
        Constructs an InvertIndexParam instance.

        Args:
            enable_range_optimization (bool, optional): If True, enables range query
                optimization for the invert index. Defaults to False.
            enable_extended_wildcard (bool, optional): If True, enables extended wildcard
                search including suffix and infix patterns. Defaults to False.
        """
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def to_dict(self) -> dict:
        """
        Convert to dictionary with all fields
        """
    @property
    def enable_extended_wildcard(self) -> bool:
        """
        bool: Whether extended wildcard (suffix and infix) search is enabled.
        Note: Prefix search is always enabled regardless of this setting.
        """
    @property
    def enable_range_optimization(self) -> bool:
        """
        bool: Whether range optimization is enabled for this inverted index.
        """

class OptimizeOption:
    """

    Options for optimizing a collection (e.g., merging segments).

    Attributes:
        concurrency (int): Number of threads to use during optimization.
            If 0, the system will choose an optimal value automatically.
            Default is 0.

    Examples:
        >>> opt = OptimizeOption(concurrency=2)
        >>> print(opt.concurrency)
        2
    """

    def __getstate__(self) -> tuple: ...
    def __init__(self, concurrency: typing.SupportsInt = 0) -> None:
        """
        Constructs an OptimizeOption instance.

        Args:
            concurrency (int, optional): Number of concurrent threads.
                0 means auto-detect. Defaults to 0.
        """

    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def concurrency(self) -> int:
        """
        int: Number of threads used for optimization (0 = auto).
        """

class QueryParam:
    """

    Base class for all query parameter configurations.

    This abstract base class defines common query settings such as search radius
    and whether to force linear (brute-force) search. It should not be instantiated
    directly; use derived classes like `HnswQueryParam` or `IVFQueryParam`.

    Attributes:
        type (IndexType): The index type this query is configured for.
        radius (float): Search radius for range queries. Used in combination with
            top-k to filter results. Default is 0.0 (disabled).
        is_linear (bool): If True, forces brute-force linear search instead of
            using the index. Useful for debugging or small datasets. Default is False.
        is_using_refiner (bool, optional): Whether to use refiner for the query. Default is False.
    """
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def is_linear(self) -> bool:
        """
        bool: Whether to bypass the index and use brute-force linear search.
        """
    @property
    def is_using_refiner(self) -> bool:
        """
        bool: Whether to use refiner for the query.
        """
    @property
    def radius(self) -> float:
        """
        IndexType: The type of index this query targets.
        """
    @property
    def type(self) -> _zvec.typing.IndexType:
        """
        IndexType: The type of index this query targets.
        """

class SegmentOption:
    """

    Options for segment-level operations.

    Currently, this class mirrors CollectionOption and is used internally.
    It supports read-only mode, memory mapping, and buffer configuration.

    Note:
        This class is primarily for internal use. Most users should use
        CollectionOption instead.

    Examples:
        >>> opt = SegmentOption()
        >>> print(opt.enable_mmap)
        True
    """

    def __getstate__(self) -> tuple: ...
    def __init__(self) -> None:
        """
        Constructs a SegmentOption with default settings.
        """

    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def enable_mmap(self) -> bool:
        """
        bool: Whether memory-mapped I/O is enabled.
        """

    @property
    def max_buffer_size(self) -> int:
        """
        int: Maximum buffer size in bytes (internal use).
        """

    @property
    def read_only(self) -> bool:
        """
        bool: Whether the segment is read-only.
        """

class VectorIndexParam(IndexParam):
    """

    Base class for vector index parameter configurations.

    Encapsulates common settings for all vector index types.

    Attributes:
        type (IndexType): The specific vector index type (e.g., HNSW, FLAT).
        metric_type (MetricType): Distance metric used for similarity search.
        quantize_type (QuantizeType): Optional vector quantization type.
    """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def to_dict(self) -> dict:
        """
        Convert to dictionary with all fields
        """

    @property
    def metric_type(self) -> _zvec.typing.MetricType:
        """
        MetricType: Distance metric (e.g., IP, COSINE, L2).
        """

    @property
    def quantize_type(self) -> _zvec.typing.QuantizeType:
        """
        QuantizeType: Vector quantization type (e.g., FP16, INT8).
        """

class _VectorQuery:
    field_name: str
    filter: str
    include_vector: bool
    query_params: QueryParam

    def __getstate__(self) -> tuple: ...
    def __init__(self) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def set_vector(self, arg0: ..., arg1: typing.Any) -> None: ...
    @property
    def output_fields(self) -> list[str] | None: ...
    @output_fields.setter
    def output_fields(self, arg0: collections.abc.Sequence[str] | None) -> None: ...
    @property
    def topk(self) -> int: ...
    @topk.setter
    def topk(self, arg0: typing.SupportsInt) -> None: ...
