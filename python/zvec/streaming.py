"""
Streaming compression utilities for zvec.

This module provides streaming compression for large datasets that don't fit in memory.
Supports chunked compression and decompression for efficient memory usage.

Usage:
    from zvec.streaming import StreamCompressor, StreamDecompressor

    # Streaming compression
    with StreamCompressor("output.gz", method="gzip") as compressor:
        for batch in large_dataset_batches:
            compressor.write(batch)

    # Streaming decompression
    with StreamDecompressor("output.gz") as decompressor:
        for chunk in decompressor:
            process(chunk)
"""

from __future__ import annotations

import gzip
import lzma
from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, Literal, Optional, TypedDict

if TYPE_CHECKING:
    import numpy as np

# Check for Python 3.13+ features
try:
    import base64

    Z85_AVAILABLE = hasattr(base64, "z85encode")
except ImportError:
    Z85_AVAILABLE = False

# Check for Python 3.14+ features (for future use)
# compression.zstd will be available in Python 3.14+
ZSTD_AVAILABLE = False  # Will be True when Python 3.14 is widely available

__all__ = [
    "Z85_AVAILABLE",
    "ZSTD_AVAILABLE",
    "StreamCompressor",
    "StreamDecompressor",
    "StreamingConfig",
    "chunked_compress",
    "chunked_decompress",
]


class StreamingConfig(TypedDict):
    """Configuration for streaming compression."""

    chunk_size: int
    compression: str


class StreamCompressor:
    """
    Streaming compressor for large datasets.

    Writes compressed data in chunks to avoid loading entire dataset in memory.

    Examples:
        >>> with StreamCompressor("data.gz", method="gzip") as comp:
        ...     for batch in batches:
        ...         comp.write(batch)
    """

    def __init__(
        self,
        file_path: str,
        method: Literal["gzip", "lzma"] = "gzip",
        chunk_size: int = 8192,
        compression_level: int = 6,
    ):
        """
        Initialize streaming compressor.

        Args:
            file_path: Output file path
            method: Compression method ("gzip" or "lzma")
            chunk_size: Size of chunks in bytes
            compression_level: Compression level (1-9)
        """
        self.file_path = file_path
        self.method = method
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        self._file = None
        self._compressor = None

    def __enter__(self):
        """Context manager entry."""
        if self.method == "gzip":
            self._file = gzip.open(
                self.file_path, "wb", compresslevel=self.compression_level
            )
        elif self.method == "lzma":
            self._file = lzma.open(self.file_path, "wb", preset=self.compression_level)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file:
            self._file.close()

    def write(self, data: bytes) -> int:
        """
        Write compressed data.

        Args:
            data: Bytes to compress

        Returns:
            Number of bytes written
        """
        if self._file is None:
            raise RuntimeError("Compressor not opened. Use 'with' statement.")
        self._file.write(data)
        return len(data)

    def write_iterable(self, iterable: Iterable[bytes]) -> int:
        """
        Write from iterable of bytes.

        Args:
            iterable: Iterable yielding byte chunks

        Returns:
            Total bytes written
        """
        total = 0
        for chunk in iterable:
            total += self.write(chunk)
        return total


class StreamDecompressor:
    """
    Streaming decompressor for large compressed files.

    Reads compressed data in chunks to avoid loading entire file in memory.

    Examples:
        >>> with StreamDecompressor("data.gz") as decomp:
        ...     for chunk in decomp:
        ...         process(chunk)
    """

    def __init__(
        self,
        file_path: str,
        method: Optional[Literal["gzip", "lzma"]] = None,
        chunk_size: int = 8192,
    ):
        """
        Initialize streaming decompressor.

        Args:
            file_path: Input file path
            method: Compression method (auto-detected if None)
            chunk_size: Size of chunks in bytes
        """
        self.file_path = file_path
        self.method = method
        self.chunk_size = chunk_size
        self._file = None

    def __enter__(self):
        """Context manager entry."""
        # Auto-detect compression method from file extension
        method = self.method
        if method is None:
            if self.file_path.endswith(".gz"):
                method = "gzip"
            elif self.file_path.endswith(".xz") or self.file_path.endswith(".lzma"):
                method = "lzma"
            else:
                # Try gzip first
                method = "gzip"

        if method == "gzip":
            self._file = gzip.open(self.file_path, "rb")
        elif method == "lzma":
            self._file = lzma.open(self.file_path, "rb")
        else:
            raise ValueError(f"Unsupported method: {method}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file:
            self._file.close()

    def __iter__(self) -> Generator[bytes]:
        """Iterate over decompressed chunks."""
        if self._file is None:
            raise RuntimeError("Decompressor not opened. Use 'with' statement.")

        while True:
            chunk = self._file.read(self.chunk_size)
            if not chunk:
                break
            yield chunk

    def read_all(self) -> bytes:
        """
        Read all decompressed data.

        Note: For large files, prefer using iteration.

        Returns:
            All decompressed bytes
        """
        return b"".join(self)


def chunked_compress(
    data: bytes,
    method: Literal["gzip", "lzma"] = "gzip",
    chunk_size: int = 8192,
) -> Generator[bytes]:
    """
    Compress data in chunks.

    Note: Due to how gzip/lzma work, this yields the full compressed data
    after each chunk_size bytes. For true streaming, use StreamCompressor.

    Args:
        data: Data to compress
        method: Compression method
        chunk_size: Size of input chunks (not output)

    Yields:
        Compressed bytes (full compressed result)

    Examples:
        >>> # For true streaming, use StreamCompressor instead
        >>> for chunk in chunked_compress(large_data, method="gzip"):
        ...     output_file.write(chunk)
    """
    if method == "gzip":
        compressed = gzip.compress(data)
    elif method == "lzma":
        compressed = lzma.compress(data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Yield in chunks
    for i in range(0, len(compressed), chunk_size):
        yield compressed[i : i + chunk_size]


def chunked_decompress(
    compressed_data: bytes,
    method: Literal["gzip", "lzma"] = "gzip",
) -> bytes:
    """
    Decompress data.

    Args:
        compressed_data: Compressed bytes
        method: Compression method

    Returns:
        Decompressed bytes
    """
    if method == "gzip":
        return gzip.decompress(compressed_data)
    if method == "lzma":
        return lzma.decompress(compressed_data)
    raise ValueError(f"Unsupported method: {method}")


class VectorStreamCompressor:
    """
    Specialized compressor for vector data.

    Optimized for numpy arrays with metadata tracking.

    Examples:
        >>> import numpy as np  # noqa: PLC0415
        >>> comp = VectorStreamCompressor("vectors.gz", dtype=np.float32)
        >>>
        >>> # Write multiple batches
        >>> comp.write_batch(np.random.rand(100, 128).astype(np.float32))
        >>> comp.write_batch(np.random.rand(200, 128).astype(np.float32))
        >>>
        >>> # Finalize and get metadata
        >>> metadata = comp.close()
        >>> print(f"Total vectors: {metadata['count']}")
    """

    def __init__(
        self,
        file_path: str,
        dtype: str = "float32",
        method: Literal["gzip", "lzma"] = "gzip",
    ):
        """
        Initialize vector stream compressor.

        Args:
            file_path: Output file path
            dtype: NumPy dtype string (e.g., "float32", "int8")
            method: Compression method
        """
        self.file_path = file_path
        self.dtype = dtype
        self.method = method
        self.vector_count = 0
        self.dimension = None
        self._compressor = StreamCompressor(file_path, method=method)

    def __enter__(self):
        self._compressor.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._compressor.__exit__(exc_type, exc_val, exc_tb)

    def write_batch(self, vectors: np.ndarray) -> None:
        """
        Write a batch of vectors.

        Args:
            vectors: NumPy array of vectors
        """
        import numpy as np  # noqa: PLC0415

        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors must be a numpy array")

        # Track metadata
        if self.dimension is None:
            self.dimension = vectors.shape[1] if len(vectors.shape) > 1 else 1
        self.vector_count += len(vectors)

        # Write as bytes
        self._compressor.write(vectors.tobytes())

    def close(self) -> dict:
        """
        Close compressor and return metadata.

        Returns:
            Dictionary with metadata (count, dimension, dtype, method)
        """
        self._compressor.__exit__(None, None, None)
        return {
            "count": self.vector_count,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "method": self.method,
            "file_path": self.file_path,
        }
