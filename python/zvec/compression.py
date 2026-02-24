"""
Compression utilities for zvec.

This module provides compression and encoding utilities for zvec vectors,
leveraging Python 3.13+ features when available.

Usage:
    from zvec.compression import compress_vector, decompress_vector

    # Compress a vector for storage
    compressed = compress_vector(vector_bytes, method="zstd")

    # Decompress when reading
    decompressed = decompress_vector(compressed, method="zstd")
"""

from __future__ import annotations

import gzip
import lzma
from typing import Literal

# Check for Python 3.13+ features
try:
    import base64

    Z85_AVAILABLE = hasattr(base64, "z85encode")
except ImportError:
    Z85_AVAILABLE = False

# Check for Python 3.14+ features
try:
    import compression.zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


def compress_vector(
    data: bytes, method: Literal["zstd", "gzip", "lzma"] = "zstd"
) -> bytes:
    """
    Compress vector data.

    Args:
        data: Raw vector bytes (e.g., numpy.tobytes())
        method: Compression method

    Returns:
        Compressed bytes

    Examples:
        >>> import numpy as np
        >>> vectors = np.random.rand(1000, 128).astype(np.float32)
        >>> compressed = compress_vector(vectors.tobytes(), method="zstd")
    """
    if method == "zstd":
        if ZSTD_AVAILABLE:
            return compression.zstd.compress(data)
        # Fallback to gzip if zstd not available
        return gzip.compress(data)
    if method == "gzip":
        return gzip.compress(data)
    if method == "lzma":
        return lzma.compress(data)
    raise ValueError(f"Unknown compression method: {method}")


def decompress_vector(
    data: bytes, method: Literal["zstd", "gzip", "lzma"] = "zstd"
) -> bytes:
    """
    Decompress vector data.

    Args:
        data: Compressed vector bytes
        method: Compression method used

    Returns:
        Decompressed bytes

    Examples:
        >>> decompressed = decompress_vector(compressed, method="zstd")
        >>> vectors = np.frombuffer(decompressed, dtype=np.float32).reshape(1000, 128)
    """
    if method == "zstd":
        if ZSTD_AVAILABLE:
            return compression.zstd.decompress(data)
        # Fallback to gzip
        return gzip.decompress(data)
    if method == "gzip":
        return gzip.decompress(data)
    if method == "lzma":
        return lzma.decompress(data)
    raise ValueError(f"Unknown compression method: {method}")


def encode_vector(
    data: bytes, encoding: Literal["z85", "base64", "urlsafe"] = "z85"
) -> str:
    """
    Encode vector data as string.

    Args:
        data: Raw vector bytes
        encoding: Encoding method

    Returns:
        Encoded string

    Examples:
        >>> encoded = encode_vector(vector_bytes, encoding="z85")
    """
    if encoding == "z85":
        if Z85_AVAILABLE:
            return base64.z85encode(data).decode("ascii")
        # Fallback to base64
        return base64.b64encode(data).decode("ascii")
    if encoding == "base64":
        return base64.b64encode(data).decode("ascii")
    if encoding == "urlsafe":
        return base64.urlsafe_b64encode(data).decode("ascii")
    raise ValueError(f"Unknown encoding: {encoding}")


def decode_vector(
    encoded: str, encoding: Literal["z85", "base64", "urlsafe"] = "z85"
) -> bytes:
    """
    Decode vector data from string.

    Args:
        encoded: Encoded string
        encoding: Encoding method used

    Returns:
        Decoded bytes

    Examples:
        >>> vector_bytes = decode_vector(encoded, encoding="z85")
    """
    if encoding == "z85":
        if Z85_AVAILABLE:
            return base64.z85decode(encoded.encode("ascii"))
        return base64.b64decode(encoded)
    if encoding == "base64":
        return base64.b64decode(encoded)
    if encoding == "urlsafe":
        return base64.urlsafe_b64decode(encoded)
    raise ValueError(f"Unknown encoding: {encoding}")


# Export availability status
__all__ = [
    "Z85_AVAILABLE",
    "ZSTD_AVAILABLE",
    "compress_vector",
    "decode_vector",
    "decompress_vector",
    "encode_vector",
]
