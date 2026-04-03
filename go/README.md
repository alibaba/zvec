# ZVec Go SDK

Go bindings for the [zvec](https://github.com/alibaba/zvec) vector database, powered by cgo wrapping the zvec C-API.

## Prerequisites

- **Go** ≥ 1.21
- **zvec C-API library** (`libzvec_c_api.so` / `libzvec_c_api.dylib`) built from source
- **C compiler** (gcc/clang) for cgo

### Building the C-API Library

```bash
git clone --recursive https://github.com/alibaba/zvec.git
cd zvec
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_C_BINDINGS=ON
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) --target zvec_c_api
```

The shared library will be at `build/lib/libzvec_c_api.{so,dylib}`.

## Installation

```bash
go get github.com/alibaba/zvec/go/zvec
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/alibaba/zvec/go/zvec"
)

func main() {
    // Initialize
    if err := zvec.Initialize(nil); err != nil {
        log.Fatal(err)
    }
    defer zvec.Shutdown()

    // Create schema
    schema := zvec.NewCollectionSchema("example")
    defer schema.Destroy()

    idField := zvec.NewFieldSchema("id", zvec.DataTypeString, false, 0)
    idField.SetIndexParams(zvec.NewInvertIndexParams(true, false))
    schema.AddField(idField)

    embField := zvec.NewFieldSchema("embedding", zvec.DataTypeVectorFP32, false, 4)
    embField.SetIndexParams(zvec.NewHNSWIndexParams(zvec.MetricTypeCosine, 16, 200))
    schema.AddField(embField)

    // Create collection
    collection, err := zvec.CreateAndOpen("./my_data", schema, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer collection.Close()

    // Insert documents
    doc := zvec.NewDoc()
    doc.SetPK("doc1")
    doc.AddStringField("id", "doc1")
    doc.AddVectorFP32Field("embedding", []float32{0.1, 0.2, 0.3, 0.4})

    if _, err := collection.Insert([]*zvec.Doc{doc}); err != nil {
        log.Fatal(err)
    }
    doc.Destroy()

    // Query
    query := zvec.NewVectorQuery()
    query.SetFieldName("embedding")
    query.SetQueryVector([]float32{0.4, 0.3, 0.3, 0.1})
    query.SetTopK(10)

    results, err := collection.Query(query)
    if err != nil {
        log.Fatal(err)
    }
    query.Destroy()
    defer zvec.FreeDocs(results)

    for _, r := range results {
        fmt.Printf("PK=%s Score=%.4f\n", r.GetPK(), r.GetScore())
    }
}
```

## Running Examples

```bash
cd go/examples/basic

# Set library path (adjust to your build directory)
# macOS:
export DYLD_LIBRARY_PATH=/path/to/zvec/build/lib:$DYLD_LIBRARY_PATH
# Linux:
export LD_LIBRARY_PATH=/path/to/zvec/build/lib:$LD_LIBRARY_PATH

go run main.go
```

## API Reference

### Core Types

| Go Type | Description |
|---------|-------------|
| `Collection` | A vector collection (create, open, insert, query, close) |
| `CollectionSchema` | Schema definition for a collection |
| `FieldSchema` | Schema definition for a single field |
| `IndexParams` | Index configuration (HNSW, IVF, Flat, Invert) |
| `Doc` | A document with typed fields |
| `VectorQuery` | Vector similarity search query |
| `GroupByVectorQuery` | Grouped vector search query |
| `CollectionOptions` | Options for creating/opening collections |
| `ConfigData` | Global library configuration |

### Data Types

| Constant | Description |
|----------|-------------|
| `DataTypeString` | String field |
| `DataTypeBool` | Boolean field |
| `DataTypeInt32` / `Int64` | Integer fields |
| `DataTypeFloat` / `Double` | Floating point fields |
| `DataTypeVectorFP32` | Dense float32 vector |
| `DataTypeVectorFP16` | Dense float16 vector |
| `DataTypeSparseVectorFP32` | Sparse float32 vector |

### Index Types

| Constant | Description |
|----------|-------------|
| `IndexTypeHNSW` | HNSW graph index (recommended for most use cases) |
| `IndexTypeIVF` | IVF index |
| `IndexTypeFlat` | Flat (brute-force) index |
| `IndexTypeInvert` | Inverted index (for scalar fields) |

### Metric Types

| Constant | Description |
|----------|-------------|
| `MetricTypeCosine` | Cosine similarity |
| `MetricTypeL2` | Euclidean distance |
| `MetricTypeIP` | Inner product |

## Supported Platforms

- Linux (x86_64, ARM64)
- macOS (ARM64)

## License

Apache License 2.0
