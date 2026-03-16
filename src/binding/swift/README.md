# ZVec — Swift Bindings

Swift Package Manager bindings for [zvec](https://zvec.org), an open-source in-process vector database.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/alibaba/zvec.git", from: "0.1.0")
]
```

> **Note:** Requires a pre-built zvec C++ library. Set `ZVEC_LIB_DIR` and `ZVEC_EXT_LIB_DIR` environment variables pointing to the build artifacts.

## Quick Start

```swift
import ZVec

let schema = ZVecSchema(name: "example")
try schema.addField("embedding", type: .vectorFp32, dimension: 4)

let col = try ZVecCollection.createAndOpen(path: "./data", schema: schema)

let doc = ZVecDoc()
doc.setPK("doc_1")
try doc.setVector("embedding", [0.1, 0.2, 0.3, 0.4])

try col.upsert([doc])
try col.flush()

let count = try col.stats()
print("Documents: \(count)")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ZVEC_LIB_DIR` | Path to zvec static libraries | `../../../build/lib` |
| `ZVEC_EXT_LIB_DIR` | Path to external dependencies | `../../../build/external/usr/local/lib` |
| `ZVEC_ARROW_LIB_DIR` | Path to Arrow libraries | (in-tree default) |

## License

Apache-2.0
