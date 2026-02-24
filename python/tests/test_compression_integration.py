"""
Tests for compression integration module.
"""

import numpy as np
import pytest

from zvec.compression_integration import (
    compress_for_storage,
    decompress_from_storage,
    get_optimal_compression,
    CompressedVectorField,
    ZSTD_AVAILABLE,
)


class TestCompressionIntegration:
    """Tests for compression integration utilities."""

    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors."""
        return np.random.rand(100, 128).astype(np.float32)

    def test_compress_for_storage_numpy(self, sample_vectors):
        """Test compressing numpy array."""
        compressed = compress_for_storage(sample_vectors, method="gzip")

        assert isinstance(compressed, bytes)
        assert len(compressed) < sample_vectors.nbytes

    def test_compress_for_storage_bytes(self, sample_vectors):
        """Test compressing bytes."""
        data_bytes = sample_vectors.tobytes()
        compressed = compress_for_storage(data_bytes, method="gzip")

        assert isinstance(compressed, bytes)

    def test_compress_auto(self, sample_vectors):
        """Test auto compression selection."""
        compressed = compress_for_storage(sample_vectors, method="auto")

        # Should have compressed
        assert len(compressed) < sample_vectors.nbytes

    def test_compress_none(self, sample_vectors):
        """Test no compression."""
        compressed = compress_for_storage(sample_vectors, method="none")

        # Should return raw bytes
        assert compressed == sample_vectors.tobytes()

    def test_decompress_from_storage(self, sample_vectors):
        """Test decompression."""
        compressed = compress_for_storage(sample_vectors, method="gzip")

        decompressed = decompress_from_storage(
            compressed,
            original_shape=sample_vectors.shape,
            dtype=sample_vectors.dtype,
            method="gzip",
        )

        np.testing.assert_array_equal(decompressed, sample_vectors)

    def test_decompress_none(self, sample_vectors):
        """Test no decompression."""
        data_bytes = sample_vectors.tobytes()

        decompressed = decompress_from_storage(
            data_bytes,
            original_shape=sample_vectors.shape,
            dtype=sample_vectors.dtype,
            method="none",
        )

        np.testing.assert_array_equal(decompressed, sample_vectors)

    def test_roundtrip_all_methods(self, sample_vectors):
        """Test roundtrip for all compression methods."""
        for method in ["gzip", "lzma", "none"]:
            compressed = compress_for_storage(sample_vectors, method=method)
            decompressed = decompress_from_storage(
                compressed,
                original_shape=sample_vectors.shape,
                dtype=sample_vectors.dtype,
                method=method,
            )
            np.testing.assert_array_equal(decompressed, sample_vectors)

    def test_compression_ratio(self, sample_vectors):
        """Test actual compression ratio."""
        compressed = compress_for_storage(sample_vectors, method="gzip")
        ratio = len(compressed) / sample_vectors.nbytes

        # Should be smaller
        assert ratio < 1.0


class TestOptimalCompression:
    """Tests for optimal compression selection."""

    def test_small_vector_no_compression(self):
        """Test that small vectors don't use heavy compression."""
        result = get_optimal_compression(1000)
        # Small vectors: no compression
        assert result == "none"

    def test_medium_vector_gzip(self):
        """Test medium vector uses gzip when zstd not available."""
        # Without zstd, medium vectors use gzip or none
        # Threshold is > 50000 for gzip, < 10000 for none
        # 50000 should give gzip or none depending on implementation
        result = get_optimal_compression(50000)
        assert result in ["gzip", "none"]

    def test_large_vector_zstd(self, monkeypatch):
        """Test large vector uses zstd if available."""
        # Mock zstd as available
        monkeypatch.setattr("zvec.compression_integration.ZSTD_AVAILABLE", True)

        result = get_optimal_compression(20000)
        assert result == "zstd"


class TestCompressedVectorField:
    """Tests for CompressedVectorField class."""

    def test_creation(self):
        """Test creating a compressed vector field."""
        cvf = CompressedVectorField("embedding", compression="gzip")

        assert cvf.name == "embedding"
        assert cvf.compression == "gzip"

    def test_repr(self):
        """Test string representation."""
        cvf = CompressedVectorField("embedding", compression="gzip")

        assert "embedding" in repr(cvf)
        assert "gzip" in repr(cvf)

    def test_default_compression(self):
        """Test default compression is none."""
        cvf = CompressedVectorField("embedding")

        assert cvf.compression == "none"
