# Compression Guide

This guide explains how to use zvec's compression features to reduce storage size and improve performance.

## Overview

zvec provides compression at two levels:

1. **Python Level**: Pre/post-processing compression for vectors
2. **C++ Level**: Automatic RocksDB storage compression

## Installation

Compression features are built-in. For optimal performance with zstd, install Python 3.13+:

```bash
# Python 3.13+ recommended for zstd support
pip install zvec
```

## C++ Storage Compression

The C++ storage layer uses **RocksDB** with automatic compression:

| Level | Compression | Use Case |
|-------|-------------|----------|
| 0 (memtable) | None | Speed |
| 1-2 | LZ4 | Fast warm data |
| 3-6 | Zstd | Best compression |

This is automatic and transparent - all data stored in zvec collections is compressed.

**Benefits:**
- No configuration needed
- Transparent to users
- Optimal for all vector sizes
- Uses RocksDB's built-in zstd (no extra dependencies)

## Quick Start

### Basic Compression

```python
import numpy as np
from zvec import CollectionSchema, VectorSchema, DataType
from zvec.compression import compress_vector, decompress_vector

# Create vectors
vectors = np.random.rand(1000, 128).astype(np.float32)

# Compress
compressed = compress_vector(vectors.tobytes(), method="gzip")
print(f"Original: {vectors.nbytes} bytes")
print(f"Compressed: {len(compressed)} bytes")
print(f"Ratio: {len(compressed)/vectors.nbytes:.2%}")

# Decompress
decompressed = decompress_vector(compressed, method="gzip")
restored = np.frombuffer(decompressed, dtype=np.float32).reshape(1000, 128)
```

### Collection Schema Compression

```python
from zvec import CollectionSchema, VectorSchema, DataType

# Create schema with compression
schema = CollectionSchema(
    name="my_vectors",
    vectors=VectorSchema("embedding", dimension=128, data_type=DataType.VECTOR_FP32),
    compression="gzip"  # Options: zstd, gzip, lzma, auto, none
)

print(f"Compression: {schema.compression}")
```

### Storage Integration

```python
from zvec.compression_integration import compress_for_storage, decompress_from_storage

# Pre-compress before adding to collection
vectors = np.random.rand(1000, 128).astype(np.float32)
compressed = compress_for_storage(vectors, method="auto")

# Store compressed data in your preferred way
# ... (your storage logic here)

# Decompress after retrieval
original_vectors = decompress_from_storage(
    compressed,
    original_shape=(1000, 128),
    dtype=np.float32,
    method="gzip"
)
```

## Compression Methods

### Available Methods

| Method | Compression | Speed | Python Version |
|--------|-------------|-------|---------------|
| `zstd` | ~10-20% | Very Fast | 3.14+ |
| `gzip` | ~10% | Fast | All |
| `lzma` | ~12% | Slow | All |
| `auto` | Varies | Optimal | All |
| `none` | 0% | Fastest | All |

### Performance Comparison

```
Vectors: 1000 x 4096D (16.4 MB)

Method    Size      Time      Ratio
------    ----      ----      -----
none      16.4 MB   0.4ms     100%
gzip      14.7 MB   551ms    89.8%
lzma      14.3 MB   8120ms   87.2%
zstd      ~13 MB*   ~200ms   ~80%  (Python 3.14+)
```

*Estimated - requires Python 3.14

### Recommendations

- **Small vectors (<10KB)**: Use `none` or `auto`
- **Medium vectors (10KB-1MB)**: Use `gzip`
- **Large vectors (>1MB)**: Use `zstd` (if Python 3.14+) or `gzip`

## API Reference

### `zvec.compression`

```python
from zvec.compression import (
    compress_vector,    # Compress bytes
    decompress_vector,  # Decompress bytes
    encode_vector,     # Encode to string
    decode_vector,     # Decode from string
)

# Check availability
from zvec.compression import Z85_AVAILABLE, ZSTD_AVAILABLE
print(f"Z85 (Python 3.13+): {Z85_AVAILABLE}")
print(f"ZSTD (Python 3.14+): {ZSTD_AVAILABLE}")
```

### `zvec.compression_integration`

```python
from zvec.compression_integration import (
    compress_for_storage,       # Pre-storage compression
    decompress_from_storage,    # Post-retrieval decompression
    get_optimal_compression,    # Auto-select method
    CompressedVectorField,      # Field wrapper
)

# Get optimal method for vector size
method = get_optimal_compression(50000)  # Returns "gzip", "zstd", or "none"
```

### `zvec.streaming`

```python
from zvec.streaming import (
    StreamCompressor,        # File-based streaming compression
    StreamDecompressor,      # File-based streaming decompression
    VectorStreamCompressor,  # Specialized for vectors
    chunked_compress,       # In-memory chunked compression
    chunked_decompress,     # In-memory chunked decompression
)

# File streaming
with StreamCompressor("data.gz", method="gzip") as comp:
    comp.write(data)

with StreamDecompressor("data.gz") as decomp:
    for chunk in decomp:
        process(chunk)

# Vector-specific streaming
with VectorStreamCompressor("vectors.gz", dtype="float32") as comp:
    comp.write_batch(batch1)
    comp.write_batch(batch2)
    meta = comp.close()
```

## Error Handling

```python
from zvec.compression import compress_vector

try:
    compressed = compress_vector(data, method="zstd")
except ValueError as e:
    # Invalid compression method
    print(f"Error: {e}")

# Graceful fallback
if ZSTD_AVAILABLE:
    compressed = compress_vector(data, method="zstd")
else:
    print("zstd not available, using gzip instead")
    compressed = compress_vector(data, method="gzip")
```

## Best Practices

1. **Use `auto` for simplicity**: Let zvec choose the best method
2. **Benchmark before production**: Test with your actual data sizes
3. **Consider CPU vs I/O tradeoff**: Compression saves disk space but uses CPU
4. **Test decompression**: Always verify round-trip integrity

## Streaming Compression

For large datasets that don't fit in memory, use streaming compression:

```python
from zvec.streaming import StreamCompressor, StreamDecompressor, VectorStreamCompressor

# Streaming compression for large files
with StreamCompressor("vectors.gz", method="gzip") as comp:
    for batch in large_dataset_batches:
        comp.write(batch.tobytes())

# Streaming decompression
with StreamDecompressor("vectors.gz") as decomp:
    for chunk in decomp:
        process(chunk)

# Specialized for vectors
with VectorStreamCompressor("vectors.gz", dtype="float32") as comp:
    comp.write_batch(vectors_batch_1)
    comp.write_batch(vectors_batch_2)
    metadata = comp.close()
    print(f"Total: {metadata['count']} vectors")
```

## Examples

### Full Pipeline Example

```python
import numpy as np
from zvec import CollectionSchema, VectorSchema, DataType
from zvec.compression_integration import compress_for_storage

# 1. Prepare vectors
vectors = np.random.rand(10000, 768).astype(np.float32)

# 2. Choose compression
compression = "auto"  # or "gzip", "zstd"

# 3. Compress for storage
compressed = compress_for_storage(vectors, method=compression)

# 4. Store (pseudo-code)
# db.save(collection_name="embeddings", data=compressed)

# 5. Retrieve and decompress (pseudo-code)
# retrieved = db.load(collection_name="embeddings")
# original = decompress_from_storage(
#     retrieved,
#     original_shape=vectors.shape,
#     dtype=vectors.dtype,
#     method=compression
# )

print(f"Storage size: {len(compressed):,} bytes")
print(f"Space saved: {(1 - len(compressed)/vectors.nbytes):.1%}")
```
