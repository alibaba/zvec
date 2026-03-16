# @zvec/zvec — Node.js Bindings

Native Node.js (N-API) bindings for [zvec](https://zvec.org), an open-source in-process vector database.

## Installation

```bash
npm install @zvec/zvec
```

> **Note:** Requires a pre-built zvec C++ library. See [Building from Source](https://zvec.org/en/docs/build/).

## Quick Start

```javascript
const zvec = require('@zvec/zvec');

// Create schema
const schema = new zvec.Schema('example');
schema.addField('embedding', zvec.DataType.VectorFp32, 4);

// Create collection
const col = zvec.Collection.createAndOpen('./data', schema);

// Insert
const doc = new zvec.Doc();
doc.setPK('doc_1');
doc.setVector('embedding', [0.1, 0.2, 0.3, 0.4]);
col.upsert([doc]);
col.flush();

// Query
const results = col.query('embedding', [0.4, 0.3, 0.2, 0.1], 10);
console.log(results);
```

## API

| Class        | Methods |
|--------------|---------|
| `Schema`     | `addField(name, type, dim)` |
| `Doc`        | `setPK`, `setVector`, `setString`, `setInt32` |
| `Collection` | `createAndOpen`, `open`, `upsert`, `fetch`, `query`, `flush`, `stats` |

## License

Apache-2.0
