"""
Tests for streaming compression module.
"""

import gzip
import io
import lzma
import os
import tempfile
import numpy as np
import pytest

from zvec.streaming import (
    StreamCompressor,
    StreamDecompressor,
    chunked_compress,
    chunked_decompress,
    VectorStreamCompressor,
    ZSTD_AVAILABLE,
)


class TestStreamCompressor:
    """Tests for StreamCompressor."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        return b"Hello World! " * 1000

    @pytest.fixture
    def temp_file(self):
        """Create temporary file."""
        fd, path = tempfile.mkstemp(suffix=".gz")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_gzip_compression(self, sample_data, temp_file):
        """Test gzip streaming compression."""
        with StreamCompressor(temp_file, method="gzip") as comp:
            comp.write(sample_data)

        # Verify
        with gzip.open(temp_file, "rb") as f:
            decompressed = f.read()

        assert decompressed == sample_data

    def test_lzma_compression(self, sample_data):
        """Test lzma streaming compression."""
        with tempfile.NamedTemporaryFile(suffix=".lzma", delete=False) as f:
            path = f.name

        try:
            with StreamCompressor(path, method="lzma") as comp:
                comp.write(sample_data)

            with lzma.open(path, "rb") as f:
                decompressed = f.read()

            assert decompressed == sample_data
        finally:
            os.remove(path)

    def test_compression_levels(self, sample_data):
        """Test different compression levels."""
        for level in [1, 6, 9]:
            with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
                path = f.name

            try:
                with StreamCompressor(
                    path, method="gzip", compression_level=level
                ) as comp:
                    comp.write(sample_data)

                file_size = os.path.getsize(path)
                assert file_size > 0
            finally:
                os.remove(path)

    def test_multiple_writes(self, sample_data):
        """Test multiple write calls."""
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name

        try:
            with StreamCompressor(path, method="gzip") as comp:
                # Write in chunks
                for i in range(0, len(sample_data), 100):
                    comp.write(sample_data[i : i + 100])

            with gzip.open(path, "rb") as f:
                decompressed = f.read()

            assert decompressed == sample_data
        finally:
            os.remove(path)


class TestStreamDecompressor:
    """Tests for StreamDecompressor."""

    @pytest.fixture
    def sample_data(self):
        return b"Test Data " * 500

    @pytest.fixture
    def gz_file(self, sample_data):
        """Create temp gzip file."""
        fd, path = tempfile.mkstemp(suffix=".gz")
        os.close(fd)
        with gzip.open(path, "wb") as f:
            f.write(sample_data)
        yield path
        os.remove(path)

    @pytest.fixture
    def lzma_file(self, sample_data):
        """Create temp lzma file."""
        fd, path = tempfile.mkstemp(suffix=".lzma")
        os.close(fd)
        with lzma.open(path, "wb") as f:
            f.write(sample_data)
        yield path
        os.remove(path)

    def test_gzip_decompression(self, sample_data, gz_file):
        """Test gzip streaming decompression."""
        with StreamDecompressor(gz_file) as decomp:
            result = b"".join(decomp)

        assert result == sample_data

    def test_lzma_decompression(self, sample_data, lzma_file):
        """Test lzma streaming decompression."""
        with StreamDecompressor(lzma_file) as decomp:
            result = b"".join(decomp)

        assert result == sample_data

    def test_iteration(self, sample_data, gz_file):
        """Test iteration yields chunks."""
        chunks = []
        with StreamDecompressor(gz_file) as decomp:
            for chunk in decomp:
                chunks.append(chunk)

        result = b"".join(chunks)
        assert result == sample_data


class TestChunkedCompress:
    """Tests for chunked_compress."""

    def test_gzip_chunked(self):
        """Test chunked gzip compression."""
        data = b"Test data " * 100

        # This now yields compressed chunks
        chunks = list(chunked_compress(data, method="gzip"))

        # Verify we get chunks
        assert len(chunks) > 0

        # Decompress the full result
        decompressed = gzip.decompress(b"".join(chunks))
        assert decompressed == data

    def test_lzma_chunked(self):
        """Test chunked lzma compression."""
        data = b"Test data " * 100

        chunks = list(chunked_compress(data, method="lzma"))

        assert len(chunks) > 0
        decompressed = lzma.decompress(b"".join(chunks))
        assert decompressed == data

    def test_multiple_chunks(self):
        """Test data yields multiple chunks."""
        data = b"X" * 10000

        chunks = list(chunked_compress(data, method="gzip", chunk_size=100))

        # Should have multiple chunks due to small chunk_size
        assert len(chunks) >= 1

        # Verify decompression
        decompressed = gzip.decompress(b"".join(chunks))
        assert decompressed == data


class TestVectorStreamCompressor:
    """Tests for VectorStreamCompressor."""

    def test_vector_batch_write(self):
        """Test writing vector batches."""
        vectors1 = np.random.rand(100, 128).astype(np.float32)
        vectors2 = np.random.rand(50, 128).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name

        try:
            with VectorStreamCompressor(path, dtype="float32", method="gzip") as comp:
                comp.write_batch(vectors1)
                comp.write_batch(vectors2)
                metadata = comp.close()

            assert metadata["count"] == 150
            assert metadata["dimension"] == 128
            assert metadata["dtype"] == "float32"

            # Verify compressed data
            with gzip.open(path, "rb") as f:
                data = f.read()
                restored = np.frombuffer(data, dtype=np.float32).reshape(150, 128)

            np.testing.assert_array_equal(restored[:100], vectors1)
            np.testing.assert_array_equal(restored[100:], vectors2)
        finally:
            os.remove(path)

    def test_metadata_tracking(self):
        """Test metadata is tracked correctly."""
        vectors = np.random.rand(42, 64).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name

        try:
            with VectorStreamCompressor(path, dtype="float32", method="gzip") as comp:
                comp.write_batch(vectors)
                metadata = comp.close()

            assert metadata["count"] == 42
            assert metadata["dimension"] == 64
        finally:
            os.remove(path)

    def test_context_manager(self):
        """Test proper context manager usage."""
        vectors = np.random.rand(10, 32).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name

        with VectorStreamCompressor(path, method="gzip") as comp:
            comp.write_batch(vectors)

        # Verify file exists and has content
        assert os.path.getsize(path) > 0


class TestStreamingIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete compress-decompress pipeline."""
        # Create sample vectors
        original = np.random.rand(500, 256).astype(np.float32)

        # Compress
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            comp_path = f.name

        try:
            with VectorStreamCompressor(comp_path, method="gzip") as comp:
                comp.write_batch(original)

            # Decompress
            with StreamDecompressor(comp_path) as decomp:
                decompressed = b"".join(decomp)

            restored = np.frombuffer(decompressed, dtype=np.float32).reshape(500, 256)

            np.testing.assert_array_equal(restored, original)
        finally:
            os.remove(comp_path)

    def test_multiple_batches(self):
        """Test writing multiple batches over time."""
        batches = [np.random.rand(100, 64).astype(np.float32) for _ in range(5)]

        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name

        try:
            # Write batches
            with VectorStreamCompressor(path, method="gzip") as comp:
                for batch in batches:
                    comp.write_batch(batch)

            # Read back
            with StreamDecompressor(path) as decomp:
                data = b"".join(decomp)

            total_vectors = np.frombuffer(data, dtype=np.float32)
            restored = total_vectors.reshape(-1, 64)

            expected = np.vstack(batches)
            np.testing.assert_array_equal(restored, expected)
        finally:
            os.remove(path)
