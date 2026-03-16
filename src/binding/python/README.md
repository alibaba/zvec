# zvec — Python Bindings

Python (ctypes FFI) bindings for [zvec](https://zvec.org), an open-source in-process vector database.

## Installation

```bash
pip install zvec
```

> **Note:** Pre-built wheels are available for Linux (x86_64, ARM64) and macOS (ARM64). See [PyPI](https://pypi.org/project/zvec/).

## Quick Start

```python
import zvec

schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

collection = zvec.create_and_open(path="./data", schema=schema)

collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]}),
])

results = collection.query(
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)
print(results)
```

## Building from Source

```bash
# Build zvec C library first
cd /path/to/zvec && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Build Python wheel
cd /path/to/zvec
pip install -e .
```

## Supported Platforms

- Linux (x86_64, ARM64)
- macOS (ARM64)
- Python 3.10 — 3.12

## License

Apache-2.0
