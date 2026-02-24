"""
Tests for zvec.compression module.
"""

import numpy as np
import pytest

from zvec.compression import (
    compress_vector,
    decompress_vector,
    encode_vector,
    decode_vector,
    Z85_AVAILABLE,
    ZSTD_AVAILABLE,
)


class TestCompression:
    """Tests for vector compression."""

    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors for testing."""
        return np.random.rand(100, 128).astype(np.float32)

    def test_compress_decompress_zstd(self, sample_vectors):
        """Test zstd compression and decompression."""
        data = sample_vectors.tobytes()

        compressed = compress_vector(data, method="zstd")
        decompressed = decompress_vector(compressed, method="zstd")

        assert decompressed == data
        assert len(compressed) < len(data)  # Should be smaller

    def test_compress_decompress_gzip(self, sample_vectors):
        """Test gzip compression and decompression."""
        data = sample_vectors.tobytes()

        compressed = compress_vector(data, method="gzip")
        decompressed = decompress_vector(compressed, method="gzip")

        assert decompressed == data

    def test_compress_decompress_lzma(self, sample_vectors):
        """Test lzma compression and decompression."""
        data = sample_vectors.tobytes()

        compressed = compress_vector(data, method="lzma")
        decompressed = decompress_vector(compressed, method="lzma")

        assert decompressed == data

    def test_compress_decompress_invalid_method(self, sample_vectors):
        """Test that invalid compression method raises ValueError."""
        data = sample_vectors.tobytes()

        with pytest.raises(ValueError, match="Unknown compression method"):
            compress_vector(data, method="invalid")

        with pytest.raises(ValueError, match="Unknown compression method"):
            decompress_vector(data, method="invalid")

    def test_compression_ratio(self, sample_vectors):
        """Test that compression actually reduces size."""
        data = sample_vectors.tobytes()
        original_size = len(data)

        # Test all methods
        for method in ["zstd", "gzip", "lzma"]:
            compressed = compress_vector(data, method=method)
            ratio = len(compressed) / original_size
            assert ratio < 1.0, f"{method} should compress"

    def test_unknown_method(self, sample_vectors):
        """Test that unknown method raises error."""
        data = sample_vectors.tobytes()

        with pytest.raises(ValueError):
            compress_vector(data, method="unknown")

    def test_zstd_fallback(self, sample_vectors):
        """Test that zstd falls back to gzip if not available."""
        data = sample_vectors.tobytes()

        if ZSTD_AVAILABLE:
            # If available, zstd should work
            compressed = compress_vector(data, method="zstd")
            decompressed = decompress_vector(compressed, method="zstd")
            assert decompressed == data
        else:
            # Should fall back to gzip
            compressed = compress_vector(data, method="zstd")
            # Should work with gzip decompression
            decompressed = decompress_vector(compressed, method="gzip")
            assert decompressed == data


class TestEncoding:
    """Tests for vector encoding."""

    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors for testing."""
        return np.random.rand(10, 128).astype(np.float32)

    def test_encode_decode_z85(self, sample_vectors):
        """Test Z85 encoding and decoding."""
        if not Z85_AVAILABLE:
            pytest.skip("Z85 not available (requires Python 3.13+)")

        data = sample_vectors.tobytes()

        encoded = encode_vector(data, encoding="z85")
        decoded = decode_vector(encoded, encoding="z85")

        assert decoded == data
        assert isinstance(encoded, str)

    def test_encode_decode_base64(self, sample_vectors):
        """Test base64 encoding and decoding."""
        data = sample_vectors.tobytes()

        encoded = encode_vector(data, encoding="base64")
        decoded = decode_vector(encoded, encoding="base64")

        assert decoded == data
        assert isinstance(encoded, str)

    def test_encode_decode_urlsafe(self, sample_vectors):
        """Test urlsafe base64 encoding and decoding."""
        data = sample_vectors.tobytes()

        encoded = encode_vector(data, encoding="urlsafe")
        decoded = decode_vector(encoded, encoding="urlsafe")

        assert decoded == data
        assert isinstance(encoded, str)

    def test_z85_smaller_than_base64(self, sample_vectors):
        """Test that Z85 produces smaller output than base64."""
        if not Z85_AVAILABLE:
            pytest.skip("Z85 not available (requires Python 3.13+)")

        data = sample_vectors.tobytes()

        z85_encoded = encode_vector(data, encoding="z85")
        base64_encoded = encode_vector(data, encoding="base64")

        # Z85 should be ~10% smaller
        assert len(z85_encoded) < len(base64_encoded)

    def test_unknown_encoding(self, sample_vectors):
        """Test that unknown encoding raises error."""
        data = sample_vectors.tobytes()

        with pytest.raises(ValueError):
            encode_vector(data, encoding="unknown")

    def test_z85_fallback(self, sample_vectors):
        """Test that Z85 falls back to base64 if not available."""
        data = sample_vectors.tobytes()

        if Z85_AVAILABLE:
            encoded = encode_vector(data, encoding="z85")
            decoded = decode_vector(encoded, encoding="z85")
            assert decoded == data
        else:
            # Should fall back to base64
            encoded = encode_vector(data, encoding="z85")
            decoded = decode_vector(encoded, encoding="base64")
            assert decoded == data


class TestIntegration:
    """Integration tests for compression + encoding."""

    def test_compress_then_encode(self):
        """Test compressing then encoding a vector."""
        vectors = np.random.rand(10, 128).astype(np.float32)
        data = vectors.tobytes()

        # Compress
        compressed = compress_vector(data, method="gzip")

        # Encode
        encoded = encode_vector(compressed, encoding="base64")

        # Decode
        decoded = decode_vector(encoded, encoding="base64")

        # Decompress
        final = decompress_vector(decoded, method="gzip")

        assert final == data
