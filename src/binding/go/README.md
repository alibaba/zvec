# zvec — Go Bindings

Go (cgo) bindings for [zvec](https://zvec.org), an open-source in-process vector database.

## Installation

```bash
go get github.com/alibaba/zvec/src/binding/go
```

> **Note:** Requires a pre-built zvec C library. Build zvec from source first (see [Building from Source](https://zvec.org/en/docs/build/)).

## Quick Start

```go
package main

import (
    "fmt"
    zvec "github.com/alibaba/zvec/src/binding/go"
)

func main() {
    schema := zvec.NewSchema("example")
    schema.AddField("embedding", zvec.TypeVectorFp32, 4)

    col, _ := zvec.CreateAndOpenCollection("./data", schema)

    doc := zvec.NewDoc()
    doc.SetPK("doc_1")
    doc.SetVector("embedding", []float32{0.1, 0.2, 0.3, 0.4})

    col.Upsert([]*zvec.Doc{doc})
    col.Flush()

    count, _ := col.Stats()
    fmt.Printf("Documents: %d\n", count)
}
```

## API

| Type         | Description |
|--------------|-------------|
| `Schema`     | Collection schema (field name, type, dimension) |
| `Doc`        | Document with PK, vector, string, and int32 fields |
| `Collection` | Create, open, upsert, fetch, flush, stats |

## Building

```bash
# Build zvec C library first (from repo root)
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Then test Go bindings
cd src/binding/go
go test -v ./...
```

## License

Apache-2.0
