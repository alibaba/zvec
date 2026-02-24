"""
Tests for compression support in CollectionSchema.
"""

import pytest
from zvec import CollectionSchema, VectorSchema, DataType


class TestCollectionSchemaCompression:
    """Tests for compression parameter in CollectionSchema."""

    def test_default_compression(self):
        """Test that default compression is 'none'."""
        schema = CollectionSchema(
            name="test",
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
        )
        assert schema.compression == "none"

    def test_gzip_compression(self):
        """Test gzip compression setting."""
        schema = CollectionSchema(
            name="test",
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
            compression="gzip",
        )
        assert schema.compression == "gzip"

    def test_zstd_compression(self):
        """Test zstd compression setting."""
        schema = CollectionSchema(
            name="test",
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
            compression="zstd",
        )
        assert schema.compression == "zstd"

    def test_lzma_compression(self):
        """Test lzma compression setting."""
        schema = CollectionSchema(
            name="test",
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
            compression="lzma",
        )
        assert schema.compression == "lzma"

    def test_auto_compression(self):
        """Test auto compression setting."""
        schema = CollectionSchema(
            name="test",
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
            compression="auto",
        )
        assert schema.compression == "auto"

    def test_invalid_compression(self):
        """Test that invalid compression raises error."""
        with pytest.raises(ValueError) as exc_info:
            CollectionSchema(
                name="test",
                vectors=VectorSchema(
                    "emb", dimension=128, data_type=DataType.VECTOR_FP32
                ),
                compression="invalid",
            )
        assert "compression must be one of" in str(exc_info.value)

    def test_compression_in_repr(self):
        """Test that compression appears in repr."""
        schema = CollectionSchema(
            name="test",
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
            compression="gzip",
        )
        repr_str = repr(schema)
        assert '"compression": "gzip"' in repr_str

    def test_compression_none_explicit(self):
        """Test that explicitly setting 'none' works."""
        schema = CollectionSchema(
            name="test",
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
            compression="none",
        )
        assert schema.compression == "none"

    def test_compression_with_fields(self):
        """Test compression with scalar fields."""
        from zvec import FieldSchema

        schema = CollectionSchema(
            name="test",
            fields=FieldSchema("id", DataType.INT64),
            vectors=VectorSchema("emb", dimension=128, data_type=DataType.VECTOR_FP32),
            compression="gzip",
        )
        assert schema.compression == "gzip"
        assert len(schema.fields) == 1
        assert schema.fields[0].name == "id"
