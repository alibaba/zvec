# ZVec — C# Bindings

C# (P/Invoke) bindings for [zvec](https://zvec.org), an open-source in-process vector database.

## Installation

```bash
dotnet add package ZVec
```

> **Note:** Requires the native zvec shared library (`libzvec_c.so` / `libzvec_c.dylib`) on the library search path.

## Quick Start

```csharp
using ZVec;

var schema = new ZVecSchema("example");
schema.AddField("embedding", DataType.VectorFp32, 4);

using var col = ZVecCollection.CreateAndOpen("./data", schema);

var doc = new ZVecDoc();
doc.SetPK("doc_1");
doc.SetVector("embedding", new float[] { 0.1f, 0.2f, 0.3f, 0.4f });

col.Upsert(doc);
col.Flush();

var count = col.Stats();
Console.WriteLine($"Documents: {count}");
```

## Building

```bash
# Build zvec C library first
cd /path/to/zvec && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Build and test C# bindings
cd src/binding/csharp
dotnet build
dotnet test
```

## License

Apache-2.0
