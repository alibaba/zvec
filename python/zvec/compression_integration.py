"""
Compression integration utilities for zvec.

This module provides utilities to integrate compression with zvec collections
at the Python level. Full C++ integration would require modifying the core
storage layer, but this provides a practical solution using pre/post processing.

Usage:
    from zvec.compression_integration import compress_for_storage, decompress_from_storage

    # Pre-compress vectors before adding to collection
    compressed_vectors = compress_for_storage(vectors, method="gzip")
    collection.add(vectors=compressed_vectors)

    # Post-process after querying
    results = decompress_from_storage(results, method="gzip")
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from .compression import (
    Z85_AVAILABLE,
    ZSTD_AVAILABLE,
    compress_vector,
    decompress_vector,
)

# Export compression availability
__all__ = [
    "Z85_AVAILABLE",
    "ZSTD_AVAILABLE",
    "compress_for_storage",
    "decompress_from_storage",
    "get_optimal_compression",
]


def get_optimal_compression(vector_size: int) -> str:
    """
    Determine optimal compression method based on vector size.

    Args:
        vector_size: Size of vector data in bytes

    Returns:
        Recommended compression method

    Examples:
        >>> get_optimal_compression(1000)
        'gzip'
        >>> get_optimal_compression(100000)
        'zstd'
    """
    if ZSTD_AVAILABLE and vector_size > 10000:
        return "zstd"
    if vector_size > 50000:
        return "gzip"
    return "none"


def compress_for_storage(
    data: Union[np.ndarray, bytes],
    method: Literal["zstd", "gzip", "lzma", "auto", "none"] = "auto",
) -> bytes:
    """
    Compress vector data for storage.

    This function compresses vector data before storing in zvec.
    Use decompress_from_storage() to decompress after retrieval.

    Args:
        data: Numpy array or bytes to compress
        method: Compression method. "auto" selects based on size.

    Returns:
        Compressed bytes (ready for storage)

    Examples:
        >>> import numpy as np
        >>> vectors = np.random.rand(1000, 128).astype(np.float32)
        >>> compressed = compress_for_storage(vectors, method="auto")
        >>> # Store compressed bytes in zvec document
    """
    # Convert numpy array to bytes if needed
    data_bytes = data.tobytes() if isinstance(data, np.ndarray) else data

    # Auto-select compression method
    if method == "auto":
        method = get_optimal_compression(len(data_bytes))

    # No compression requested
    if method == "none":
        return data_bytes

    return compress_vector(data_bytes, method=method)


def decompress_from_storage(
    data: bytes,
    original_shape: tuple,
    dtype: np.dtype,
    method: Literal["zstd", "gzip", "lzma", "none"] = "none",
) -> np.ndarray:
    """
    Decompress vector data retrieved from storage.

    Args:
        data: Compressed bytes from storage
        original_shape: Original shape of vector array (e.g., (1000, 128))
        dtype: NumPy dtype (e.g., np.float32)
        method: Compression method used ("none" if not compressed)

    Returns:
        Decompressed numpy array

    Examples:
        >>> # After retrieving compressed bytes from zvec
        >>> vectors = decompress_from_storage(
        ...     compressed_bytes,
        ...     original_shape=(1000, 128),
        ...     dtype=np.float32,
        ...     method="gzip"
        ... )
    """
    # No compression to remove
    if method == "none":
        return np.frombuffer(data, dtype=dtype).reshape(original_shape)

    decompressed = decompress_vector(data, method=method)
    return np.frombuffer(decompressed, dtype=dtype).reshape(original_shape)


class CompressedVectorField:
    """
    Wrapper for compressed vector fields in zvec documents.

    This provides a convenient way to handle compressed vectors
    in zvec documents without modifying the core storage.

    Examples:
        >>> # Define a compressed vector field
        >>> cvf = CompressedVectorField(
        ...     name="embedding",
        ...     compression="gzip"
        ... )
        >>>
        >>> # Add to document
        >>> doc = zvec.Doc()
        >>> doc[cvf] = vectors
    """

    def __init__(
        self,
        name: str,
        compression: Literal["zstd", "gzip", "lzma", "auto", "none"] = "none",
    ):
        self.name = name
        self.compression = compression

    def __repr__(self) -> str:
        return (
            f"CompressedVectorField(name={self.name}, compression={self.compression})"
        )
